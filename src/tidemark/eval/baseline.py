"""The long-horizon seed set and the resumable naive-baseline runner.

**Tasks** (``evaluation/data/tidemark_long_horizon_v1.jsonl``). A fact (a file
path, an exact error code, or a user constraint) is planted in the opening
conversation, ``distance_N`` steps before the step that needs it. The first
``needed_at_step`` entries of ``scripted_steps`` are read-only tool calls that
build up history. The rest are the reference solution: they show the gold
diff is satisfiable, and the CPU scripted model replays them. A task scores on
the final store (Steadyhand's store diff, plus required note text), never on
prose. Separately, it records whether the exact evidence was in the rendered
request when it was needed. A model can fail with the evidence present, and
the two are reported apart.

**Live runs** use ``ForcedPrefixClient``. Before ``needed_at_step``, each step
sends the real rendered request (so llama-server's prefill and cache counts
are real), but the scripted calls replace the model's output. That gives
every context mode the same history, which paired comparisons need. From
``needed_at_step`` on, the model acts freely. Forced and free steps are
labeled in every record.

**The runner** claims and resumes episodes through ``run_claimed_episode``
(so replay, resume, and claims are Steadyhand's own). It appends one score per
unit to ``<run_id>.scores.jsonl``, refuses a run directory whose spec differs
(dataset, context config, model, server, scheduler, mode), reruns
infrastructure aborts under a new label, and keeps infrastructure and
``did_not_fit`` outcomes visible and out of task-success denominators.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import steadyhand
import toolsmith
from steadyhand.context import ContextSession, RenderedContext
from steadyhand.controller import Budgets, call_hash
from steadyhand.eval.suite import GOLD_KEYS as STORE_GOLD_KEYS
from steadyhand.eval.suite import INFRASTRUCTURE_ABORTS, score_diff, shared_registry, store_diff
from steadyhand.executor import SequentialExecutor
from steadyhand.journal import JOURNAL_VERSION, manifest_path_for, read_journal
from steadyhand.manifest import load_manifest
from steadyhand.messages import SYSTEM_PROMPT
from steadyhand.policy import Effect
from steadyhand.replay import replay_episode, run_claimed_episode
from steadyhand.store import Store, load_fixture
from steadyhand.types import EpisodeStatus, ModelTurn, ToolCall, TurnKind
from steadyhand.validate import validate_call
from toolsmith.evaluation.models import canonical_hash
from toolsmith.evaluation.runner import atomic_write_json
from toolsmith.json_utils import strict_json_loads
from toolsmith.recording import JsonlRecorder, trim_partial_line

from .. import __version__ as TIDEMARK_VERSION

DATASET_PATH = Path(__file__).resolve().parents[3] / "evaluation" / "data" / "tidemark_long_horizon_v1.jsonl"
DATASET_VERSION = "tidemark_long_horizon_v1"
CATEGORIES = ("file_path", "error_code", "user_constraint")
AVAILABILITY = ("inline", "blob_handle", "memory", "any_restorable")
GOLD_KEYS = STORE_GOLD_KEYS | {"note_contains"}
TASK_FIELDS = frozenset({
    "task_id", "version", "category", "goal", "initial_memory", "scripted_steps", "planted_fact",
    "needed_at_step", "distance_N", "distractors", "observation_ids", "expected_tools", "gold_store_diff",
    "required_availability", "possible", "seed",
})
FACT_FIELDS = frozenset({"kind", "text", "planted_at_step", "source"})
TAIL_ALLOWANCE = 6  # free steps the model may take after the forced prefix, beyond the reference tail
GUARD_LIMIT = 3
MODES = ("scripted", "forced_prefix")
OUTCOMES = ("success", "failure", "infrastructure", "did_not_fit")
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"), re.compile(r"sk-[A-Za-z0-9]{16,}"), re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{8,}"), re.compile(r"(?i)\b(api[_-]?key|password|local_key|hf_token)\b"),
)


class DatasetError(ValueError):
    """The long-horizon file breaks a rule; every problem found is listed."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = tuple(problems)


class BaselineMismatch(RuntimeError):
    """The run directory belongs to a different baseline spec. The run stops."""

    def __init__(self, field_name: str, recorded: Any, current: Any) -> None:
        super().__init__(f"Baseline spec field {field_name!r} was {recorded!r}; this run has {current!r}")
        self.field = field_name


# Tasks


@dataclass(frozen=True)
class LongHorizonTask:
    task_id: str
    version: str
    category: str
    goal: str
    initial_memory: tuple[dict[str, Any], ...]
    scripted_steps: tuple[tuple[dict[str, Any], ...], ...]
    planted_fact: dict[str, Any]
    needed_at_step: int
    distance_N: int
    distractors: tuple[str, ...]
    observation_ids: tuple[str, ...]
    expected_tools: tuple[str, ...]
    gold_store_diff: dict[str, Any]
    required_availability: str
    possible: bool
    seed: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LongHorizonTask:
        missing, unexpected = TASK_FIELDS - set(value), set(value) - TASK_FIELDS
        if missing or unexpected:
            raise DatasetError([f"{value.get('task_id')}: fields missing {sorted(missing)} unexpected {sorted(unexpected)}"])
        return cls(
            task_id=value["task_id"], version=value["version"], category=value["category"], goal=value["goal"],
            initial_memory=tuple(dict(message) for message in value["initial_memory"]),
            scripted_steps=tuple(tuple(dict(call) for call in step) for step in value["scripted_steps"]),
            planted_fact=dict(value["planted_fact"]), needed_at_step=value["needed_at_step"],
            distance_N=value["distance_N"], distractors=tuple(value["distractors"]),
            observation_ids=tuple(value["observation_ids"]), expected_tools=tuple(value["expected_tools"]),
            gold_store_diff=dict(value["gold_store_diff"]), required_availability=value["required_availability"],
            possible=value["possible"], seed=value["seed"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "version": self.version, "category": self.category, "goal": self.goal,
            "initial_memory": [dict(m) for m in self.initial_memory],
            "scripted_steps": [[dict(c) for c in step] for step in self.scripted_steps],
            "planted_fact": dict(self.planted_fact), "needed_at_step": self.needed_at_step,
            "distance_N": self.distance_N, "distractors": list(self.distractors),
            "observation_ids": list(self.observation_ids), "expected_tools": list(self.expected_tools),
            "gold_store_diff": dict(self.gold_store_diff), "required_availability": self.required_availability,
            "possible": self.possible, "seed": self.seed,
        }

    @property
    def forced_steps(self) -> tuple[tuple[dict[str, Any], ...], ...]:
        return self.scripted_steps[: self.needed_at_step]

    @property
    def reference_tail(self) -> tuple[tuple[dict[str, Any], ...], ...]:
        return self.scripted_steps[self.needed_at_step:]

    def messages(self) -> list[dict[str, Any]]:
        """The opening conversation: system prompt, the planted prior exchange, then the current request."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[dict(message) for message in self.initial_memory],
            {"role": "user", "content": self.goal},
        ]

    def budgets(self, max_tokens: int = 512) -> Budgets:
        """Room for the prefix, the reference tail, and a few free steps; prompt tokens effectively unbounded."""
        steps = self.needed_at_step + len(self.reference_tail) + TAIL_ALLOWANCE
        return Budgets(max_steps=steps, max_prompt_tokens=8_000_000, max_completion_tokens=(steps + 1) * max_tokens)


def load_long_horizon(path: str | Path = DATASET_PATH) -> list[LongHorizonTask]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [LongHorizonTask.from_dict(strict_json_loads(line)) for line in lines if line.strip()]


def dataset_sha256(tasks: Sequence[LongHorizonTask]) -> str:
    return canonical_hash([task.as_dict() for task in tasks])


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [s for item in value.values() for s in _strings(item)]
    if isinstance(value, (list, tuple)):
        return [s for item in value for s in _strings(item)]
    return []


def _fixture_ids(fixture: Store) -> set[str]:
    """Order, ticket, and customer ids (customers are the ones orders and tickets reference)."""
    return (
        {order.order_id for order in fixture.orders}
        | {order.customer_id for order in fixture.orders}
        | {ticket.ticket_id for ticket in fixture.tickets}
        | {ticket.customer_id for ticket in fixture.tickets}
    )


def validate_long_horizon(
    tasks: Sequence[LongHorizonTask],
    fixture: Store | None = None,
    *,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    """Check every rule and raise ``DatasetError`` listing all problems; return a summary otherwise.

    Unique ids, the version, distances, known tools, read-only forced steps, the
    repeated-call guard, valid source references (``store:<id>`` that a forced call reads), a planted
    fact that is in the droppable opening exchange but not in the pinned request, distractors that
    differ from it, a satisfiable gold diff (all scripted steps run on a fresh fixture), and no secrets.
    """
    fixture = fixture or load_fixture()
    registry = shared_registry()
    known = set(registry.names())
    ids = _fixture_ids(fixture)
    environment_secrets = [os.environ.get(name, "") for name in ("HF_TOKEN", "LOCAL_KEY")]
    problems: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        where = task.task_id
        if task.task_id in seen:
            problems.append(f"{where}: duplicate task_id")
        seen.add(task.task_id)
        if task.version != DATASET_VERSION:
            problems.append(f"{where}: version {task.version!r} is not {DATASET_VERSION}")
        if task.category not in CATEGORIES:
            problems.append(f"{where}: unknown category {task.category!r}")
        if task.required_availability not in AVAILABILITY:
            problems.append(f"{where}: unknown required_availability {task.required_availability!r}")
        if not task.goal.strip():
            problems.append(f"{where}: empty goal")
        fact = task.planted_fact
        if set(fact) != FACT_FIELDS:
            problems.append(f"{where}: planted_fact needs exactly {sorted(FACT_FIELDS)}")
            continue
        text = fact["text"]
        if fact["kind"] != task.category:
            problems.append(f"{where}: planted_fact kind {fact['kind']!r} differs from category {task.category!r}")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"{where}: empty planted fact")
            continue
        if fact["source"] != "initial_memory":
            problems.append(f"{where}: part 1 plants facts in initial_memory only, not {fact['source']!r}")
        memory_text = "\n".join(str(m.get("content") or "") for m in task.initial_memory)
        if text not in memory_text:
            problems.append(f"{where}: the planted fact is not in initial_memory")
        if text in task.goal:
            problems.append(f"{where}: the planted fact is in the pinned request, so it could never scroll out")
        if any(m.get("role") not in ("user", "assistant") for m in task.initial_memory):
            problems.append(f"{where}: initial_memory holds only user and assistant messages")
        planted = fact["planted_at_step"]
        if not (isinstance(task.needed_at_step, int) and task.needed_at_step >= 1):
            problems.append(f"{where}: needed_at_step must be at least 1")
        elif task.needed_at_step >= len(task.scripted_steps) and task.possible:
            problems.append(f"{where}: a possible task needs a reference tail after needed_at_step")
        if not isinstance(planted, int) or planted < 0 or planted >= task.needed_at_step:
            problems.append(f"{where}: planted_at_step must be in [0, needed_at_step)")
        elif task.distance_N != task.needed_at_step - planted:
            problems.append(f"{where}: distance_N {task.distance_N} is not needed_at_step - planted_at_step")
        for distractor in task.distractors:
            if not distractor.strip() or distractor == text or text in distractor or distractor in text:
                problems.append(f"{where}: distractor {distractor!r} is empty or overlaps the fact")
            elif distractor not in memory_text:
                problems.append(f"{where}: distractor {distractor!r} does not appear in initial_memory")
        counts: Counter[str] = Counter()
        read_ids: set[str] = set()
        for index, step in enumerate(task.scripted_steps):
            if not step:
                problems.append(f"{where}: scripted step {index} is empty")
            for call in step:
                name = call.get("name")
                if name not in known:
                    problems.append(f"{where}: unknown tool {name!r}")
                    continue
                if index < task.needed_at_step and registry.policy(name).effect is not Effect.READ:
                    problems.append(f"{where}: forced step {index} calls {name}, which is not read-only")
                counts[call_hash(name, json.dumps(call.get("arguments", {})))] += 1
                if index < task.needed_at_step:
                    read_ids.update(s for s in _strings(call.get("arguments", {})) if s in ids)
        if counts and max(counts.values()) >= GUARD_LIMIT:
            problems.append(f"{where}: a scripted call repeats {max(counts.values())} times; the guard blocks at {GUARD_LIMIT}")
        for reference in task.observation_ids:
            kind, _, entity = reference.partition(":")
            if kind != "store" or entity not in ids:
                problems.append(f"{where}: observation id {reference!r} is not a store:<fixture id>")
            elif entity not in read_ids:
                problems.append(f"{where}: observation id {reference!r} is never read by a forced step")
        tail_tools = {call.get("name") for step in task.reference_tail for call in step}
        for tool in task.expected_tools:
            if tool not in known:
                problems.append(f"{where}: unknown expected tool {tool!r}")
            elif tool not in tail_tools:
                problems.append(f"{where}: expected tool {tool!r} is not in the reference tail")
        unknown_gold = set(task.gold_store_diff) - GOLD_KEYS
        if unknown_gold:
            problems.append(f"{where}: unknown gold keys {sorted(unknown_gold)}")
        elif known >= {call.get("name") for step in task.scripted_steps for call in step}:
            problems.extend(_check_reference(task, fixture))
        for value in _strings(task.as_dict()):
            if any(pattern.search(value) for pattern in SECRET_PATTERNS) or any(
                secret and secret in value for secret in [*secrets, *environment_secrets]
            ):
                problems.append(f"{where}: a string looks like a secret or credential")
                break
    if problems:
        raise DatasetError(problems)
    return {
        "tasks": len(tasks),
        "categories": dict(sorted(Counter(t.category for t in tasks).items())),
        "distances": sorted({t.distance_N for t in tasks}),
        "availability": dict(sorted(Counter(t.required_availability for t in tasks).items())),
        "dataset_sha256": dataset_sha256(tasks),
    }


def _check_reference(task: LongHorizonTask, fixture: Store) -> list[str]:
    """Run every scripted step through validation on a fresh fixture; the result must match the gold."""
    registry = shared_registry()
    store = fixture.clone()
    executor = SequentialExecutor()
    for step_index, step in enumerate(task.scripted_steps):
        for call_index, call in enumerate(step):
            tool_call = ToolCall(f"ref_{step_index}_{call_index}", call["name"], json.dumps(call["arguments"]))
            if validate_call(tool_call, registry, store).ok:
                executor.run(tool_call, registry, store)
            else:
                return [f"{task.task_id}: scripted step {step_index} call {call_index} fails validation"]
    problems = gold_problems(fixture, store, task.gold_store_diff)
    if task.possible:
        return [f"{task.task_id}: the reference steps miss the gold: {problem}" for problem in problems]
    return [f"{task.task_id}: an impossible task's reference changed the store"] if store_diff(fixture, store).writes() else []


def gold_problems(before: Store, after: Store, gold: Mapping[str, Any]) -> list[str]:
    """Steadyhand's store-diff check plus ``note_contains``: every listed text in a new note on that order."""
    store_gold = {key: value for key, value in gold.items() if key != "note_contains"}
    problems = list(score_diff(store_diff(before, after), store_gold).problems)
    old_notes = {o.order_id: {n.note_id for n in o.notes} for o in before.orders}
    for order_id, texts in (gold.get("note_contains") or {}).items():
        order = next((o for o in after.orders if o.order_id == order_id), None)
        new_text = [n.text for n in (order.notes if order else ()) if n.note_id not in old_notes.get(order_id, set())]
        for text in texts:
            if not any(text in note for note in new_text):
                problems.append(f"no new note on {order_id} contains {text!r}")
    return problems


# Evidence availability


BLOB_HANDLE_IN_TEXT = re.compile(r"blob://sha256:[0-9a-f]{64}")


def fact_available(messages: Sequence[Mapping[str, Any]], fact: str, required: str, *, blobs: Any = None) -> dict[str, Any]:
    """Whether ``fact`` was reachable in the rendered request. Inline evidence always counts.

    ``blob_handle`` and ``any_restorable`` also accept a blob whose handle the
    request names and whose bytes (read through ``blobs``, the session's
    namespaced reader) contain the fact. ``memory`` waits for part 5, so only
    inline evidence counts for it.
    """
    handles: list[str] = []
    for message in messages:
        texts = [message.get("content") or ""] + [
            (call.get("function") or {}).get("arguments") or "" for call in message.get("tool_calls") or []
        ]
        if any(isinstance(text, str) and fact in text for text in texts):
            return {"available": True, "via": "inline", "required": required}
        for text in texts:
            if isinstance(text, str):
                handles.extend(BLOB_HANDLE_IN_TEXT.findall(text))
    if blobs is not None and required in ("blob_handle", "any_restorable"):
        for handle in dict.fromkeys(handles):
            try:
                payload = blobs.read_bytes(handle).decode("utf-8")
            except Exception:  # an unreadable handle is not evidence
                continue
            if fact in payload or json.dumps(fact, ensure_ascii=False)[1:-1] in payload:
                return {"available": True, "via": "blob_handle", "required": required, "handle": handle}
    return {"available": False, "via": None, "required": required}


class EvidenceProbe:
    """Wraps a context session and notes, at the step a task needs its fact, whether it was rendered.

    It changes nothing it delegates: identity, rendering, shaping, and timings are the inner session's.
    """

    def __init__(self, inner: ContextSession, task: LongHorizonTask) -> None:
        self.inner = inner
        self.task = task
        self.strategy = inner.strategy
        self.version = inner.version
        self.by_step: dict[int, bool] = {}
        self.at_needed: dict[str, Any] | None = None

    def identity(self) -> dict[str, Any]:
        return self.inner.identity()

    def render(self, history: Any, *, step: int, tools: Any, tool_choice: Any, initial_count: int,
               **options: Any) -> RenderedContext:
        rendered = self.inner.render(history, step=step, tools=tools, tool_choice=tool_choice,
                                     initial_count=initial_count, **options)
        if tool_choice != "none":
            found = fact_available(rendered.messages, self.task.planted_fact["text"], self.task.required_availability,
                                   blobs=getattr(self.inner, "blob_reader", None))
            self.by_step[step] = found["available"]
            if step == self.task.needed_at_step and self.at_needed is None:
                self.at_needed = found
        return rendered

    def shape_observation(self, message: Mapping[str, Any], *, step: int) -> dict[str, Any]:
        return self.inner.shape_observation(message, step=step)

    def shape_result(self, source: Any) -> Any:
        shaper = getattr(self.inner, "shape_result", None)
        if shaper is not None:
            return shaper(source)
        from steadyhand.context import ShapedObservation

        message = {"role": "tool", "tool_call_id": source.tool_call_id, "content": source.content}
        return ShapedObservation(dict(self.inner.shape_observation(message, step=source.step)))

    def execution_context(self, environment: Any) -> Any:
        provider = getattr(self.inner, "execution_context", None)
        return provider(environment) if provider is not None else None

    def check_journal(self, records: Any) -> None:
        checker = getattr(self.inner, "check_journal", None)
        if checker is not None:
            checker(records)

    @property
    def blob_reader(self) -> Any:
        return getattr(self.inner, "blob_reader", None)

    def backend_timings(self, turn: ModelTurn, rendered: RenderedContext) -> dict[str, Any]:
        return self.inner.backend_timings(turn, rendered)


# Model clients


def _scripted_turn(task: LongHorizonTask, step: int, calls: Sequence[Mapping[str, Any]], *, prompt_tokens: int = 100,
                   completion_tokens: int = 20, metrics: Any = None, elapsed_s: float = 0.0) -> ModelTurn:
    tool_calls = [
        {"id": f"{task.task_id}_s{step}_{index}", "type": "function",
         "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}
        for index, call in enumerate(calls)
    ]
    return ModelTurn(
        kind=TurnKind.TOOL_CALLS,
        assistant_message={"role": "assistant", "content": None, "tool_calls": tool_calls},
        tool_calls=tuple(ToolCall.from_openai(call) for call in tool_calls),
        content=None, reasoning_content=None, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        error_code=None, elapsed_s=elapsed_s, metrics=metrics,
    )


def _text_turn(content: str) -> ModelTurn:
    return ModelTurn(
        kind=TurnKind.TEXT, assistant_message={"role": "assistant", "content": content}, tool_calls=(),
        content=content, reasoning_content=None, prompt_tokens=100, completion_tokens=12, error_code=None,
    )


def _replace_text(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, Mapping):
        return {key: _replace_text(item, old, new) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_text(item, old, new) for item in value]
    return value


class LongHorizonScriptedModel:
    """CPU model: every scripted step in order, then a final answer.

    ``evidence_sensitive``: at the first reference-tail step, if the planted fact
    is not in the messages it was sent, it writes from the distractor instead (the
    store check then fails), which makes context loss visible on CPU.
    """

    def __init__(self, task: LongHorizonTask, *, evidence_sensitive: bool = False, max_tokens: int = 512) -> None:
        self.task = task
        self.evidence_sensitive = evidence_sensitive
        self.max_tokens = max_tokens
        self._step = 0

    def settings(self) -> dict[str, Any]:
        behavior = "evidence_sensitive" if self.evidence_sensitive else "reference"
        return {
            "profile": "scripted", "model": f"scripted/long_horizon/{behavior}", "protocol": None, "use_grammars": True,
            "enable_thinking": False, "max_tokens": self.max_tokens, "temperature": 0.0, "seed": self.task.seed,
            "slot_context": None,
        }

    def complete(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]], *,
                 tool_choice: Any = "auto", **options: Any) -> ModelTurn:
        """``id_slot`` and ``allowed_tool_names`` are accepted and ignored: nothing is served."""
        if tool_choice == "none":
            return _text_turn("I could not finish this request.")
        step = self._step
        self._step += 1
        if step >= len(self.task.scripted_steps):
            return _text_turn("Done: the review is complete.")
        calls = list(self.task.scripted_steps[step])
        if self.evidence_sensitive and step == self.task.needed_at_step:
            fact = self.task.planted_fact["text"]
            if not fact_available(messages, fact, "inline")["available"]:
                replacement = self.task.distractors[0] if self.task.distractors else "unknown"
                calls = [_replace_text(call, fact, replacement) for call in calls]
        return _scripted_turn(self.task, step, calls)


class ForcedPrefixClient:
    """Live client for the forced prefix (see the module docstring); free from ``needed_at_step`` on.

    A forced step keeps the live turn's usage and metrics (the real request was
    evaluated) but returns the scripted calls. A backend failure during a forced
    step is returned as is, so the episode aborts as infrastructure.
    """

    def __init__(self, live: Any, task: LongHorizonTask) -> None:
        self.live = live
        self.task = task
        self._step = 0
        self.sources: list[str] = []

    def settings(self) -> dict[str, Any]:
        return self.live.settings()

    def render_prompt(self, *args: Any, **kwargs: Any) -> str:
        return self.live.render_prompt(*args, **kwargs)

    def complete(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]], *,
                 tool_choice: Any = "auto", **options: Any) -> ModelTurn:
        """``id_slot`` (a slot lease) and ``allowed_tool_names`` go to the live client with the request."""
        turn = self.live.complete(messages, tools, tool_choice=tool_choice, **options)
        if tool_choice == "none":
            self.sources.append("paraphrase")
            return turn
        step = self._step
        self._step += 1
        if step < self.task.needed_at_step and turn.kind is not TurnKind.FATAL_ERROR:
            self.sources.append("forced")
            return _scripted_turn(
                self.task, step, self.task.forced_steps[step], prompt_tokens=turn.prompt_tokens,
                completion_tokens=turn.completion_tokens, metrics=turn.metrics, elapsed_s=turn.elapsed_s,
            )
        self.sources.append("model")
        return turn


# Scoring


@dataclass(frozen=True)
class LongHorizonScore:
    task_id: str
    category: str
    distance_N: int
    required_availability: str
    mode: str
    status: str
    outcome: str
    task_success: bool | None  # None for infrastructure and did_not_fit (out of the denominators)
    evidence_available: bool | None
    problems: tuple[str, ...]
    episode_id: str
    journal_path: str
    steps: int
    prompt_tokens: int
    completion_tokens: int
    error_code: str | None = None
    attempt: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "problems": list(self.problems)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LongHorizonScore:
        names = set(cls.__dataclass_fields__)
        data = {key: item for key, item in value.items() if key in names}
        data["problems"] = tuple(data.get("problems", ()))
        return cls(**data)


def score_long_horizon(
    task: LongHorizonTask,
    *,
    mode: str,
    before: Store,
    after: Store,
    records: Sequence[Mapping[str, Any]],
    journal_path: str | Path,
    evidence: Mapping[str, Any] | None,
) -> LongHorizonScore:
    started, finished = records[0], records[-1]
    status = finished["status"]
    error_code = finished.get("error_code")
    if status == EpisodeStatus.ABORTED.value and error_code in INFRASTRUCTURE_ABORTS:
        outcome = "infrastructure"
    elif status == EpisodeStatus.ABORTED.value and error_code == "context_budget_exhausted":
        outcome = "did_not_fit"
    else:
        outcome = ""
    problems = gold_problems(before, after, task.gold_store_diff)
    succeeded = {r["tool"] for r in records if r["event"] == "tool_result" and r["ok"]}
    missing = [tool for tool in task.expected_tools if tool not in succeeded]
    if missing and task.possible:
        problems.append(f"expected tools never used successfully: {missing}")
    allowed = {EpisodeStatus.DONE.value} if task.possible else {EpisodeStatus.DONE.value, EpisodeStatus.ESCALATED.value}
    if status not in allowed:
        problems.append(f"ended {status}" + (f" ({error_code})" if error_code else ""))
    if not outcome:
        outcome = "success" if not problems else "failure"
    return LongHorizonScore(
        task_id=task.task_id, category=task.category, distance_N=task.distance_N,
        required_availability=task.required_availability, mode=mode, status=status, outcome=outcome,
        task_success=None if outcome in ("infrastructure", "did_not_fit") else outcome == "success",
        evidence_available=None if evidence is None else bool(evidence["available"]),
        problems=tuple(problems), episode_id=started["episode_id"], journal_path=str(journal_path),
        steps=finished["steps"], prompt_tokens=finished["prompt_tokens"], completion_tokens=finished["completion_tokens"],
        error_code=error_code if status == EpisodeStatus.ABORTED.value else None,
    )


# Runs


@dataclass(frozen=True)
class BaselineSpec:
    """Everything a baseline run's records depend on. Resuming needs an identical spec."""

    run_id: str
    task_ids: tuple[str, ...]
    dataset_sha256: str
    context_config: dict[str, Any]
    mode: str
    model: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    gpu: str | None = None
    code: dict[str, Any] = field(default_factory=lambda: {
        "tidemark": TIDEMARK_VERSION, "steadyhand": steadyhand.__version__, "toolsmith": toolsmith.__version__,
        "journal": JOURNAL_VERSION,
    })

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.run_id):
            raise ValueError("run_id must be short and filename-safe")

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "task_ids": list(self.task_ids)}


def run_long_horizon_task(
    task: LongHorizonTask,
    *,
    spec: BaselineSpec,
    client: Any,
    session: ContextSession,
    directory: str | Path,
    fixture: Store | None = None,
    secrets: Sequence[str] = (),
    run_label: str | None = None,
    context_server: Mapping[str, Any] | None = None,
    replay_session: Callable[[], ContextSession] | None = None,
) -> LongHorizonScore | None:
    """Run (or resume, or re-score) one task. ``None``: another worker holds its claim."""
    fixture = fixture or load_fixture()
    registry = shared_registry()
    before, store = fixture.clone(), fixture.clone()
    probe = EvidenceProbe(session, task)
    max_tokens = int(client.settings().get("max_tokens") or 512)
    result = run_claimed_episode(
        directory, client, registry, store, task.messages(), budgets=task.budgets(max_tokens), guard_limit=GUARD_LIMIT,
        secrets=secrets, run_label=run_label or f"{spec.run_id}/{spec.mode}", context=probe,
        context_server=context_server,
    )
    if result is None:
        # Completed earlier but never scored, or claimed elsewhere: find its journal from a dry manifest.
        from steadyhand.manifest import build_manifest

        manifest = build_manifest(
            client.settings(), registry, fixture.clone(), task.messages(), budgets=task.budgets(max_tokens).as_dict(),
            guard_limit=GUARD_LIMIT, run_label=run_label or f"{spec.run_id}/{spec.mode}",
            context_config=session.identity(), context_version=session.version, context_server=context_server,
        )
        journal_path = Path(directory) / f"{manifest.episode_id}.jsonl"
        manifest_file = manifest_path_for(journal_path)
        if not (manifest_file.exists() and load_manifest(manifest_file).completed):
            return None
        store = fixture.clone()
        probe = EvidenceProbe(replay_session() if replay_session is not None else session, task)
        replay_episode(journal_path, store=store, context=probe)
    else:
        journal_path = result.journal_path
    records = read_journal(journal_path)
    return score_long_horizon(task, mode=spec.mode, before=before, after=store, records=records,
                              journal_path=journal_path, evidence=probe.at_needed)


def load_baseline_scores(path: str | Path) -> dict[str, list[LongHorizonScore]]:
    """Every scored attempt per task, oldest first (a torn last line is dropped)."""
    path = Path(path)
    history: dict[str, list[LongHorizonScore]] = {}
    if not path.exists():
        return history
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = strict_json_loads(line)
        except ValueError:
            if index == len(lines) - 1 and not text.endswith("\n"):
                break
            raise
        if record.get("event") == "task_scored":
            history.setdefault(record["task_id"], []).append(LongHorizonScore.from_dict(record))
    return history


def run_baseline(
    spec: BaselineSpec,
    tasks: Sequence[LongHorizonTask],
    *,
    client_factory: Callable[[LongHorizonTask, str], Any],
    session_factory: Callable[[LongHorizonTask, Any], ContextSession],
    run_dir: str | Path,
    scheduler: Any = None,
    workers: int = 1,
    secrets: Sequence[str] = (),
    retries: int = 2,
    context_server: Mapping[str, Any] | None = None,
    on_scored: Callable[[LongHorizonScore], None] | None = None,
    prepare: Callable[[LongHorizonTask], Any] | None = None,
) -> list[LongHorizonScore]:
    """Run every task in ``spec`` not already scored (claims, resume, infrastructure reruns).

    ``client_factory(task, owner)`` builds the episode's model client (wrapped in a slot lease when a
    ``scheduler`` runs the episodes); ``session_factory(task, client)`` builds a fresh context session.
    ``prepare(task)`` runs ahead on the scheduler's bounded prefetch thread (for example, rendering a task's
    frozen and pinned segments into a shared cache); it never touches a running episode's state.
    """
    selected = [task for task in tasks if task.task_id in set(spec.task_ids)]
    if [task.task_id for task in selected] != list(spec.task_ids):
        raise BaselineMismatch("task_ids", list(spec.task_ids), [task.task_id for task in selected])
    if dataset_sha256(selected) != spec.dataset_sha256:
        raise BaselineMismatch("dataset_sha256", spec.dataset_sha256, dataset_sha256(selected))
    run_dir = Path(run_dir)
    journal_dir = run_dir / "journals"
    manifest_path = run_dir / f"{spec.run_id}.json"
    scores_path = run_dir / f"{spec.run_id}.scores.jsonl"
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = spec.as_dict()
        for key in sorted(set(recorded) | set(current)):
            if recorded.get(key) != current.get(key):
                raise BaselineMismatch(key, recorded.get(key), current.get(key))
    else:
        atomic_write_json(manifest_path, spec.as_dict())
    trim_partial_line(scores_path)
    history = load_baseline_scores(scores_path)
    done, labels = {}, {}
    for task_id, attempts in history.items():
        if attempts[-1].outcome == "infrastructure" and len(attempts) <= retries:
            labels[task_id] = f"{spec.run_id}/{spec.mode}/retry-{len(attempts)}"
        else:
            done[task_id] = attempts[-1]
    recorder = JsonlRecorder(scores_path, secrets=secrets)
    lock = threading.Lock()
    fixture = load_fixture()

    def work(task: LongHorizonTask, owner: str) -> LongHorizonScore | None:
        inner = client_factory(task, owner)
        client = scheduler.client(inner, owner) if scheduler is not None else inner
        score = run_long_horizon_task(
            task, spec=spec, client=client, session=session_factory(task, client), directory=journal_dir,
            fixture=fixture, secrets=secrets, run_label=labels.get(task.task_id), context_server=context_server,
            replay_session=lambda: session_factory(task, client),
        )
        if score is None:
            return None
        score = replace(score, attempt=len(history.get(task.task_id, ())) + 1)
        with lock:
            recorder.append({"event": "task_scored", **score.as_dict()})
        if on_scored is not None:
            on_scored(score)
        return score

    pending = [task for task in selected if task.task_id not in done]
    if pending:
        with lock:
            recorder.append({"event": "run_segment", "pending": len(pending), "done": len(done),
                             "scheduler": spec.scheduler, "server": spec.server})
    if scheduler is not None:
        warm = None
        if prepare is not None:
            def warm(task: LongHorizonTask) -> LongHorizonTask:
                prepare(task)
                return task
        fresh = scheduler.run_episodes(pending, work, key=lambda task: task.task_id, prepare=warm)
    elif workers <= 1:
        fresh = [work(task, task.task_id) for task in pending]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tidemark-baseline") as pool:
            fresh = list(pool.map(lambda task: work(task, task.task_id), pending))
    return [*done.values(), *[score for score in fresh if score is not None]]


# Per-step records

STEP_COLUMNS = (
    "run_id", "gpu", "mode", "task_id", "category", "distance_N", "episode_id", "step", "purpose", "turn_source",
    "step_kind", "render_status", "rendered_prompt_tokens", "prompt_evaluated_tokens", "reused_tokens",
    "cache_reuse_fraction", "cache_status", "completion_tokens", "prompt_ms", "predicted_ms", "response_elapsed_ms",
    "ttft_ms", "measurement_path", "slot_id", "repair_attempts", "dropped_messages", "slot_headroom",
    "evidence_available", "outcome", "task_success",
)


def step_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    task: LongHorizonTask,
    score: LongHorizonScore,
    run_id: str,
    gpu: str | None,
) -> list[dict[str, Any]]:
    """One row per rendered model request, joined from ``context_render``, ``model_turn``, and ``backend_timings``.

    ``step_kind``: ``cold`` (the episode's first request), ``truncation`` (turns
    were dropped), ``paraphrase``, or ``ordinary``. Missing values stay empty.
    """
    rows: list[dict[str, Any]] = []
    agent_requests = 0
    index = 0
    while index < len(records):
        record = records[index]
        if record["event"] != "context_render":
            index += 1
            continue
        turn = records[index + 1] if index + 1 < len(records) and records[index + 1]["event"] == "model_turn" else None
        timings = records[index + 2] if turn is not None and index + 2 < len(records) and records[index + 2]["event"] == "backend_timings" else None
        purpose = "paraphrase" if turn is not None and turn["tool_choice"] == "none" else "agent"
        dropped = sum(end - start for start, end in record["dropped_ranges"])
        if purpose == "paraphrase":
            kind, source = "paraphrase", "model"
        else:
            kind = "cold" if agent_requests == 0 else "truncation" if dropped else "ordinary"
            source = "scripted" if score.mode == "scripted" else "forced" if agent_requests < task.needed_at_step else "model"
            agent_requests += 1
        rows.append({
            "run_id": run_id, "gpu": gpu, "mode": score.mode, "task_id": task.task_id, "category": task.category,
            "distance_N": task.distance_N, "episode_id": score.episode_id, "step": record["step"], "purpose": purpose,
            "turn_source": source, "step_kind": kind, "render_status": record["status"],
            "rendered_prompt_tokens": record["rendered_prompt_tokens"],
            "prompt_evaluated_tokens": timings["prompt_evaluated_tokens"] if timings else None,
            "reused_tokens": timings["reused_tokens"] if timings else None,
            "cache_reuse_fraction": timings["cache_reuse_fraction"] if timings else None,
            "cache_status": timings["cache_status"] if timings else None,
            "completion_tokens": turn["usage"]["completion_tokens"] if turn else None,
            "prompt_ms": timings["prompt_ms"] if timings else None,
            "predicted_ms": timings["predicted_ms"] if timings else None,
            "response_elapsed_ms": timings["response_elapsed_ms"] if timings else None,
            "ttft_ms": timings["ttft_ms"] if timings else None,
            "measurement_path": timings["measurement_path"] if timings else None,
            "slot_id": timings["slot_id"] if timings else None,
            "repair_attempts": turn["repair_attempts"] if turn else None,
            "dropped_messages": dropped, "slot_headroom": record["slot_headroom"],
            "evidence_available": score.evidence_available if record["step"] == task.needed_at_step and purpose == "agent" else None,
            "outcome": score.outcome, "task_success": score.task_success,
        })
        index += 3 if timings is not None else 2 if turn is not None else 1
    return rows


def write_steps_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """The per-step CSV (atomically replaced). Unknown values are empty cells, never zeros."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=STEP_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in STEP_COLUMNS})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(buffer.getvalue(), encoding="utf-8")
    os.replace(temporary, path)
    return path


__all__ = [
    "AVAILABILITY",
    "BaselineMismatch",
    "BaselineSpec",
    "CATEGORIES",
    "DATASET_PATH",
    "DATASET_VERSION",
    "DatasetError",
    "EvidenceProbe",
    "ForcedPrefixClient",
    "LongHorizonScore",
    "LongHorizonScriptedModel",
    "LongHorizonTask",
    "STEP_COLUMNS",
    "dataset_sha256",
    "fact_available",
    "gold_problems",
    "load_baseline_scores",
    "load_long_horizon",
    "run_baseline",
    "run_long_horizon_task",
    "score_long_horizon",
    "step_rows",
    "validate_long_horizon",
]

"""The Steadyhand own suite: tasks, gold store diffs, scoring, a scripted model, and sweeps.

A task is scored on the final store, never on the wording of the answer. Gold
is a declarative diff against the fixture:

    {"refunds_for_order": {"ord_1001": [{"amount_inr": "120.00", "count": 1}]},
     "tickets_created": 0, "notes_added": {"ord_1002": 1},
     "ticket_status": {"tkt_9001": "pending"}, "order_status": {"ord_1003": "cancelled"}}

Anything the gold does not list must stay unchanged, so a correct sentence
over two refunds is a failure with one ``duplicate_side_effect``. Ledger
refunds (the switch strategy's alternative) count as refunds.

Each task runs clean and under each fault variant it declares. The fault seed
is the task seed. Infrastructure aborts are kept in the log but never enter a
success denominator.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import toolsmith
from toolsmith.artifacts import MirroredFile
from toolsmith.evaluation.models import canonical_hash
from toolsmith.evaluation.runner import atomic_write_json
from toolsmith.json_utils import strict_json_loads
from toolsmith.recording import JsonlRecorder, trim_partial_line

from .. import __version__ as STEADYHAND_VERSION
from ..controller import Budgets, call_hash
from ..executor import SequentialExecutor
from ..faults import FaultConfigError, FaultMode, FaultSchedule, FaultSpec, build_schedule
from ..journal import JOURNAL_VERSION, JournalCopier, manifest_path_for, read_journal
from ..manifest import build_manifest, load_manifest
from ..messages import initial_messages
from ..recovery import STRATEGIES, no_sleep, strategy_for
from ..registry import SteadyhandRegistry
from ..replay import replay_episode, run_claimed_episode
from ..store import Store, load_fixture
from ..tools import build_registry
from ..tools_ledger import build_alternatives
from ..types import EpisodeStatus, ModelTurn, ToolCall, TurnKind
from ..validate import validate_call

DATASET_PATH = Path(__file__).resolve().parents[3] / "evaluation" / "data" / "steadyhand_tasks_v1.jsonl"
LOCAL_SWEEP_DIR = Path(__file__).resolve().parents[3] / "results" / "steadyhand" / "sweeps"
CATEGORIES = ("single_step", "multi_step", "missing_information")
GOLD_KEYS = frozenset({"refunds_for_order", "tickets_created", "notes_added", "ticket_status", "order_status"})
RECOVERY_LABELS = ("none", "retry", "verify", "switch", "stop")
CLEAN = "clean"
TASK_FIELDS = frozenset({
    "task_id", "category", "goal", "possible", "seed", "expected_tools", "reference",
    "final_answer", "gold", "faults",
})


@lru_cache(maxsize=1)
def shared_registry() -> SteadyhandRegistry:
    """The frozen model-facing registry. Read-only, so every worker can share it."""
    return build_registry()


class DatasetError(ValueError):
    """The task file breaks a rule; every problem found is listed."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = tuple(problems)


class SweepMismatch(RuntimeError):
    """A sweep manifest on disk describes a different sweep. The sweep stops."""

    def __init__(self, field: str, recorded: Any, current: Any) -> None:
        super().__init__(f"Sweep manifest field {field!r} was {recorded!r}; this run has {current!r}")
        self.field = field


# Tasks


@dataclass(frozen=True)
class FaultVariant:
    """One way a task is run under faults: which faults, and what the right recovery and outcome are."""

    name: str
    pins: tuple[dict[str, Any], ...]
    gold_recovery: str
    gold: dict[str, Any] | None = None  # overrides the task gold when the fault changes the right outcome
    possible: bool | None = None

    def schedule(self, seed: int) -> FaultSchedule:
        """The fault schedule for this variant, seeded by the task."""
        return build_schedule(seed, [FaultSpec.from_dict(pin) for pin in self.pins])

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"pins": [dict(pin) for pin in self.pins], "gold_recovery": self.gold_recovery}
        if self.gold is not None:
            value["gold"] = self.gold
        if self.possible is not None:
            value["possible"] = self.possible
        return value


@dataclass(frozen=True)
class Task:
    """One mock-store task: the goal, a scripted reference solution, the gold store diff, and fault variants."""

    task_id: str
    category: str
    goal: str
    possible: bool
    seed: int
    expected_tools: tuple[str, ...]
    reference: tuple[tuple[dict[str, Any], ...], ...]  # steps of calls; one step is one assistant message
    final_answer: str
    gold: dict[str, Any]
    faults: dict[str, FaultVariant] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Task:
        unexpected = set(value) - TASK_FIELDS
        missing = TASK_FIELDS - set(value)
        if unexpected or missing:
            raise DatasetError([f"{value.get('task_id')}: fields missing {sorted(missing)} unexpected {sorted(unexpected)}"])
        return cls(
            task_id=value["task_id"],
            category=value["category"],
            goal=value["goal"],
            possible=value["possible"],
            seed=value["seed"],
            expected_tools=tuple(value["expected_tools"]),
            reference=tuple(tuple(step) for step in value["reference"]),
            final_answer=value["final_answer"],
            gold=dict(value["gold"]),
            faults={
                name: FaultVariant(
                    name=name,
                    pins=tuple(variant["pins"]),
                    gold_recovery=variant["gold_recovery"],
                    gold=variant.get("gold"),
                    possible=variant.get("possible"),
                )
                for name, variant in value["faults"].items()
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "goal": self.goal,
            "possible": self.possible,
            "seed": self.seed,
            "expected_tools": list(self.expected_tools),
            "reference": [list(step) for step in self.reference],
            "final_answer": self.final_answer,
            "gold": self.gold,
            "faults": {name: variant.as_dict() for name, variant in self.faults.items()},
        }

    def variants(self) -> list[str]:
        """``clean`` plus every fault variant the task declares."""
        return [CLEAN, *self.faults]

    def gold_for(self, variant: str) -> dict[str, Any]:
        """The expected store changes; a fault variant may override them (a lost write, an escalation)."""
        override = self.faults[variant].gold if variant != CLEAN else None
        return self.gold if override is None else override

    def possible_for(self, variant: str) -> bool:
        """Whether the task can be completed under this variant (a permanent fault may make it impossible)."""
        override = self.faults[variant].possible if variant != CLEAN else None
        return self.possible if override is None else override

    def gold_recovery_for(self, variant: str) -> str:
        """The recovery a good policy shows here (switch, verify, retry, stop, or none)."""
        return "none" if variant == CLEAN else self.faults[variant].gold_recovery

    def schedule_for(self, variant: str) -> FaultSchedule | None:
        """The variant's fault schedule (``None`` for clean)."""
        return None if variant == CLEAN else self.faults[variant].schedule(self.seed)

    def writes(self) -> bool:
        """Whether the reference solution changes the store."""
        registry = shared_registry()
        return any(
            registry.policy(call["name"]).effect.value != "read"
            for step in self.reference
            for call in step
            if call["name"] in registry
        )


def load_tasks(path: str | Path = DATASET_PATH) -> list[Task]:
    """All tasks from the JSONL dataset (strict JSON, every field checked)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Task.from_dict(strict_json_loads(line)) for line in lines if line.strip()]


def dataset_sha256(tasks: Iterable[Task]) -> str:
    """Hash of the tasks; a sweep manifest records it, so a changed dataset is caught."""
    return canonical_hash([task.as_dict() for task in tasks])


# Store diffs and scoring


@dataclass(frozen=True)
class StoreDiff:
    """What an episode changed in the store, in the terms the gold uses."""

    refunds: dict[str, list[str]]  # order -> amounts of new refunds and ledger entries
    tickets_created: int
    notes_added: dict[str, int]
    ticket_status: dict[str, str]  # tickets that existed before and changed status
    order_status: dict[str, str]

    def writes(self) -> bool:
        # Whether the episode changed anything at all.
        return bool(self.refunds or self.tickets_created or self.notes_added or self.ticket_status or self.order_status)


def store_diff(before: Store, after: Store) -> StoreDiff:
    """Compare the store before and after: new refunds (and ledger entries), tickets, notes, status changes."""
    old_refunds = {refund.refund_id for refund in before.refunds}
    old_ledger = {entry.entry_id for entry in before.ledger}
    refunds: dict[str, list[str]] = {}
    for refund in after.refunds:
        if refund.refund_id not in old_refunds:
            refunds.setdefault(refund.order_id, []).append(refund.amount_inr)
    for entry in after.ledger:
        if entry.entry_id not in old_ledger:
            refunds.setdefault(entry.order_id, []).append(entry.amount_inr)
    old_tickets = {ticket.ticket_id: ticket.status for ticket in before.tickets}
    old_orders = {order.order_id: order for order in before.orders}
    notes = {}
    for order in after.orders:
        previous = {note.note_id for note in old_orders[order.order_id].notes} if order.order_id in old_orders else set()
        added = sum(1 for note in order.notes if note.note_id not in previous)
        if added:
            notes[order.order_id] = added
    return StoreDiff(
        refunds={order: sorted(amounts) for order, amounts in refunds.items()},
        tickets_created=sum(1 for ticket in after.tickets if ticket.ticket_id not in old_tickets),
        notes_added=notes,
        ticket_status={
            ticket.ticket_id: ticket.status
            for ticket in after.tickets
            if ticket.ticket_id in old_tickets and old_tickets[ticket.ticket_id] != ticket.status
        },
        order_status={
            order.order_id: order.status
            for order in after.orders
            if order.order_id in old_orders and old_orders[order.order_id].status != order.status
        },
    )


@dataclass(frozen=True)
class DiffScore:
    """Whether the diff equals the gold, how many effects were duplicated, and every mismatch."""

    matches: bool
    duplicate_side_effects: int
    problems: tuple[str, ...]


def score_diff(diff: StoreDiff, gold: Mapping[str, Any]) -> DiffScore:
    """Score a diff against the gold. Extra refunds, tickets, or notes also count as duplicate effects."""
    problems: list[str] = []
    duplicates = 0
    gold_refunds = gold.get("refunds_for_order", {})
    for order in sorted(set(diff.refunds) | set(gold_refunds)):
        actual = Counter(diff.refunds.get(order, []))
        expected = Counter({item["amount_inr"]: item["count"] for item in gold_refunds.get(order, [])})
        if actual != expected:
            problems.append(f"refunds on {order}: expected {dict(expected)}, got {dict(actual)}")
        duplicates += max(0, sum(actual.values()) - sum(expected.values()))
    expected_tickets = gold.get("tickets_created", 0)
    if diff.tickets_created != expected_tickets:
        problems.append(f"tickets created: expected {expected_tickets}, got {diff.tickets_created}")
    duplicates += max(0, diff.tickets_created - expected_tickets)
    gold_notes = gold.get("notes_added", {})
    for order in sorted(set(diff.notes_added) | set(gold_notes)):
        actual_notes, expected_notes = diff.notes_added.get(order, 0), gold_notes.get(order, 0)
        if actual_notes != expected_notes:
            problems.append(f"notes on {order}: expected {expected_notes}, got {actual_notes}")
        duplicates += max(0, actual_notes - expected_notes)
    for key, actual in (("ticket_status", diff.ticket_status), ("order_status", diff.order_status)):
        expected = gold.get(key, {})
        if actual != expected:
            problems.append(f"{key}: expected {expected}, got {actual}")
    return DiffScore(not problems, duplicates, tuple(problems))


@dataclass(frozen=True)
class EpisodeScore:
    """One scored episode: task, variant, policy, outcome, duplicates, cost, and where its journal is."""

    task_id: str
    category: str
    variant: str
    policy: str
    prompt_variant: str
    profile: str | None
    status: str
    possible: bool
    task_success: bool | None  # None when the episode aborted on infrastructure
    duplicate_side_effects: int
    steps: int
    prompt_tokens: int
    completion_tokens: int
    escalated: bool
    aborted: bool
    recovery_class: str
    gold_recovery: str
    problems: tuple[str, ...]
    episode_id: str
    journal_path: str
    error_code: str | None = None  # why an aborted episode stopped (backend_timeout, completion_truncated, ...)
    attempt: int = 1  # 2 or more when the sweep reran the unit after an infrastructure abort

    @property
    def recovery_match(self) -> bool:
        """Whether the episode recovered the way the gold says it should."""
        return self.recovery_class == self.gold_recovery

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["problems"] = list(self.problems)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EpisodeScore:
        known = {name for name in cls.__dataclass_fields__}
        data = {key: item for key, item in value.items() if key in known}
        data["problems"] = tuple(data.get("problems", ()))
        return cls(**data)


def recovery_class(records: Sequence[Mapping[str, Any]]) -> str:
    """How the episode responded to failure: switch > verify > retry > stop > none."""
    purposes = {record["purpose"] for record in records if record["event"] == "policy_attempt"}
    if "switch" in purposes:
        return "switch"
    if "verify" in purposes:
        return "verify"
    failed: set[str] = set()
    retried = "retry" in purposes
    any_failure = False
    for record in records:
        if record["event"] == "tool_result":
            key = call_hash(record["tool"], record["arguments"])
            if record["ok"]:
                retried = retried or key in failed
            else:
                retried = retried or key in failed
                failed.add(key)
                any_failure = True
        elif record["event"] == "observation" and (record.get("observation") or {}).get("error_type"):
            any_failure = True
    if retried:
        return "retry"
    finished = records[-1] if records and records[-1]["event"] == "episode_finished" else {}
    if any_failure or finished.get("status") == EpisodeStatus.ESCALATED.value:
        return "stop"
    return "none"


def score_episode(
    task: Task,
    variant: str,
    *,
    policy: str,
    prompt_variant: str,
    before: Store,
    after: Store,
    records: Sequence[Mapping[str, Any]],
    journal_path: str | Path,
) -> EpisodeScore:
    started, finished = records[0], records[-1]
    status = finished["status"]
    aborted = status == EpisodeStatus.ABORTED.value
    possible = task.possible_for(variant)
    diff_score = score_diff(store_diff(before, after), task.gold_for(variant))
    problems = list(diff_score.problems)
    # A read must have returned data. A write only has to have been called: the
    # store diff already decides whether its effect is right.
    results = [record for record in records if record["event"] == "tool_result"]
    succeeded = {record["tool"] for record in results if record["ok"]}
    called = {record["tool"] for record in results}
    registry = shared_registry()
    missing_tools = [
        tool for tool in task.expected_tools
        if tool not in succeeded and not (tool in called and registry.policy(tool).effect.value != "read")
    ]
    if missing_tools and possible:
        problems.append(f"expected tools never used successfully: {missing_tools}")
    allowed = {EpisodeStatus.DONE.value} if possible else {EpisodeStatus.DONE.value, EpisodeStatus.ESCALATED.value}
    if status not in allowed:
        problems.append(f"ended {status}")
    return EpisodeScore(
        task_id=task.task_id,
        category=task.category,
        variant=variant,
        policy=policy,
        prompt_variant=prompt_variant,
        profile=(started.get("settings") or {}).get("profile"),
        status=status,
        possible=possible,
        task_success=None if aborted else not problems,
        duplicate_side_effects=diff_score.duplicate_side_effects,
        steps=finished["steps"],
        prompt_tokens=finished["prompt_tokens"],
        completion_tokens=finished["completion_tokens"],
        escalated=status == EpisodeStatus.ESCALATED.value,
        aborted=aborted,
        recovery_class=recovery_class(records),
        gold_recovery=task.gold_recovery_for(variant),
        problems=tuple(problems),
        episode_id=started["episode_id"],
        journal_path=str(journal_path),
        error_code=finished.get("error_code") if aborted else None,
    )


# Validation


def validate_tasks(tasks: Sequence[Task], fixture: Store | None = None) -> dict[str, Any]:
    """Check the task file; raise ``DatasetError`` listing every problem.

    Beyond ids and names, each possible task's reference calls are executed on
    a fresh fixture and must produce exactly its gold diff, and each impossible
    task's reference calls must leave the store unchanged.
    """
    fixture = fixture or load_fixture()
    registry = shared_registry()
    known_tools = set(registry.names()) | set(build_alternatives().names())
    problems: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        where = task.task_id
        if task.task_id in seen:
            problems.append(f"{where}: duplicate task_id")
        seen.add(task.task_id)
        if task.category not in CATEGORIES:
            problems.append(f"{where}: unknown category {task.category!r}")
        if not task.goal.strip():
            problems.append(f"{where}: empty goal")
        unknown_gold = set(task.gold) - GOLD_KEYS
        if unknown_gold:
            problems.append(f"{where}: unknown gold keys {sorted(unknown_gold)}")
        called = {call["name"] for step in task.reference for call in step}
        for tool in sorted((called | set(task.expected_tools)) - set(registry.names())):
            problems.append(f"{where}: unknown tool {tool!r}")
        if task.category == "missing_information":
            if task.gold or task.reference:
                problems.append(f"{where}: a missing-information task cannot write or call tools in its gold")
        if not task.possible and task.gold:
            problems.append(f"{where}: an impossible task must have an empty gold diff")
        for name, variant in task.faults.items():
            if variant.gold_recovery not in RECOVERY_LABELS:
                problems.append(f"{where}/{name}: unknown gold_recovery {variant.gold_recovery!r}")
            try:
                schedule = variant.schedule(task.seed)
            except (FaultConfigError, KeyError, ValueError) as exc:
                problems.append(f"{where}/{name}: bad pin: {exc}")
                continue
            for pin in schedule.pins:
                if pin.tool not in known_tools:
                    problems.append(f"{where}/{name}: pin on unknown tool {pin.tool!r}")
                if pin.mode is FaultMode.LOST_ACKNOWLEDGMENT and not task.writes():
                    problems.append(f"{where}/{name}: a task without a write cannot take a lost-ack pin")
        if set(registry.names()) >= called:
            problems.extend(_check_reference(task, registry, fixture))
    if problems:
        raise DatasetError(problems)
    categories = Counter(task.category for task in tasks)
    return {
        "tasks": len(tasks),
        "categories": dict(sorted(categories.items())),
        "possible": sum(task.possible for task in tasks),
        "fault_variants": sum(len(task.faults) for task in tasks),
        "dataset_sha256": dataset_sha256(tasks),
    }


# Run a task's reference calls (through validation, like the loop) and check they produce the gold;
# an impossible task's reference must change nothing.
def _check_reference(task: Task, registry: SteadyhandRegistry, fixture: Store) -> list[str]:
    store = fixture.clone()
    executor = SequentialExecutor()
    for step_index, step in enumerate(task.reference):
        for call_index, call in enumerate(step):
            tool_call = ToolCall(f"ref_{step_index}_{call_index}", call["name"], json.dumps(call["arguments"]))
            if validate_call(tool_call, registry, store).ok:  # the same gate the loop applies
                executor.run(tool_call, registry, store)
    diff = store_diff(fixture, store)
    if task.possible:
        score = score_diff(diff, task.gold)
        return [f"{task.task_id}: reference calls miss the gold: {problem}" for problem in score.problems]
    return [f"{task.task_id}: an impossible task's reference calls changed the store"] if diff.writes() else []


# A CPU model that follows the task's reference calls


class ScriptedTaskModel:
    """Emits a task's reference calls step by step, and reacts to failures by ``behavior``.

    ``follow_hints``: re-send failed calls once when the observation says
    ``retryable`` (or gives no hint at all, as with ``raw_error``); stop and
    answer when any failure is not retryable or the retry also failed.
    ``blind_retry``: re-send every failed call once, whatever it says.
    """

    def __init__(self, task: Task, *, behavior: str = "follow_hints", profile: str = "scripted") -> None:
        if behavior not in ("follow_hints", "blind_retry"):
            raise ValueError(f"Unknown scripted behavior {behavior!r}")
        self.task = task
        self.behavior = behavior
        self.profile = profile
        self._step = 0
        self._retried_step = -1
        self._last_calls: list[dict[str, Any]] = []
        self._turns = 0

    def settings(self) -> dict[str, Any]:
        """Settings shaped like a real client's, so manifests and journals look the same."""
        return {
            "profile": self.profile,
            "model": f"scripted/{self.behavior}",
            "protocol": None,
            "use_grammars": True,
            "enable_thinking": False,
            "max_tokens": 256,
            "temperature": 0.0,
            "seed": self.task.seed,
            "slot_context": None,
        }

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ModelTurn:
        """The next scripted turn: the next reference step, a retry of failed calls, or the final answer."""
        self._turns += 1
        if tool_choice == "none":
            return self._text("I could not finish this request.")
        failures = self._failures(messages) if self._last_calls else []
        if failures:
            retry = self._retry_choice(failures)
            if retry:
                self._retried_step = self._step
                return self._calls(retry)
            if self.behavior == "follow_hints" or self._retried_step == self._step:
                return self._text(self.task.final_answer)
        if self._last_calls:
            self._step += 1
        if self._step >= len(self.task.reference):
            return self._text(self.task.final_answer)
        return self._calls([dict(call) for call in self.task.reference[self._step]])

    # The last turn's calls that came back with ok=false, paired with their observations.
    def _failures(self, messages: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        observations = {}
        for message in messages:
            if message.get("role") == "tool":
                observations[message["tool_call_id"]] = json.loads(message["content"])
        return [
            (call, observations[call["id"]])
            for call in self._last_calls
            if call["id"] in observations and not observations[call["id"]].get("ok")
        ]

    # Which failed calls to send again (each step is retried at most once).
    def _retry_choice(self, failures: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
        if self._retried_step == self._step:
            return []
        if self.behavior == "blind_retry":
            return [{"name": call["name"], "arguments": call["arguments"]} for call, _ in failures]
        if any(observation.get("retryable") is False for _, observation in failures):
            return []
        return [{"name": call["name"], "arguments": call["arguments"]} for call, _ in failures]

    # A tool-call turn with fixed token counts, so budgets behave the same on every run.
    def _calls(self, calls: list[dict[str, Any]]) -> ModelTurn:
        tool_calls = []
        self._last_calls = []
        for index, call in enumerate(calls):
            call_id = f"{self.task.task_id}_t{self._turns}_{index}"
            arguments = json.dumps(call["arguments"], ensure_ascii=False)
            tool_calls.append({"id": call_id, "type": "function", "function": {"name": call["name"], "arguments": arguments}})
            self._last_calls.append({"id": call_id, "name": call["name"], "arguments": call["arguments"]})
        message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        return ModelTurn(
            kind=TurnKind.TOOL_CALLS,
            assistant_message=message,
            tool_calls=tuple(ToolCall.from_openai(call) for call in tool_calls),
            content=None,
            reasoning_content=None,
            prompt_tokens=100,
            completion_tokens=20,
            error_code=None,
        )

    # A final text turn.
    def _text(self, content: str) -> ModelTurn:
        return ModelTurn(
            kind=TurnKind.TEXT,
            assistant_message={"role": "assistant", "content": content},
            tool_calls=(),
            content=content,
            reasoning_content=None,
            prompt_tokens=100,
            completion_tokens=12,
            error_code=None,
        )


# Running tasks and sweeps


def run_task(
    task: Task,
    variant: str,
    *,
    policy: str,
    directory: str | Path,
    client: Any = None,
    prompt_variant: str = "plain",
    behavior: str = "follow_hints",
    fixture: Store | None = None,
    budgets: Budgets | None = None,
    sleep: Callable[[float], None] = no_sleep,
    secrets: Sequence[str] = (),
    run_label: str | None = None,
) -> EpisodeScore | None:
    """Run (or re-score) one task variant under one policy. ``None``: another worker holds it."""
    fixture = fixture or load_fixture()
    client = client or ScriptedTaskModel(task, behavior=behavior)
    registry = shared_registry()
    before = fixture.clone()
    store = fixture.clone()
    # Two clones: ``before`` stays untouched for the diff, ``store`` is the one the episode changes.
    messages = initial_messages(task.goal, variant=prompt_variant)
    faults = task.schedule_for(variant)
    budgets = budgets or Budgets()
    manifest = build_manifest(
        client.settings(), registry, store, messages,
        budgets=budgets.as_dict(), guard_limit=3, run_label=run_label,
        fault_schedule=faults, recovery_policy=policy, system_prompt_variant=prompt_variant,
    )
    journal_path = Path(directory) / f"{manifest.episode_id}.jsonl"
    result = run_claimed_episode(
        directory, client, registry, store, messages,
        budgets=budgets, secrets=secrets, run_label=run_label, faults=faults,
        recovery=strategy_for(policy), system_prompt_variant=prompt_variant, sleep=sleep,
    )
    if result is None:
        manifest_file = manifest_path_for(journal_path)
        if not (manifest_file.exists() and load_manifest(manifest_file).completed):
            return None
        store = fixture.clone()  # completed earlier but never scored: rebuild its store by replay
        replay_episode(journal_path, store=store)
    records = read_journal(journal_path)
    return score_episode(
        task, variant, policy=policy, prompt_variant=prompt_variant,
        before=before, after=store, records=records, journal_path=journal_path,
    )


@dataclass(frozen=True)
class SweepUnit:
    """One episode of a sweep: a task, a fault variant, a policy, and a prompt variant."""

    task: Task
    variant: str
    policy: str
    prompt_variant: str

    @property
    def key(self) -> str:
        """The unit's id in the scores file."""
        return f"{self.task.task_id}|{self.variant}|{self.policy}|{self.prompt_variant}"


@dataclass(frozen=True)
class SweepSpec:
    """Everything a sweep's results depend on. Resuming requires an identical spec."""

    sweep_id: str
    task_ids: tuple[str, ...]
    policies: tuple[str, ...]
    prompt_variants: tuple[str, ...]
    variants: tuple[str, ...] | None  # None: clean plus every declared fault variant
    profile: str
    dataset_sha256: str
    model: dict[str, Any] = field(default_factory=dict)  # model, use_grammars, enable_thinking, temperature
    server: dict[str, Any] = field(default_factory=dict)  # gpu, context_size, parallel, per_slot_context, kv cache, batch
    seed_rule: str = "task_seed"
    code: dict[str, Any] = field(default_factory=lambda: {
        "steadyhand": STEADYHAND_VERSION, "toolsmith": toolsmith.__version__, "journal": JOURNAL_VERSION,
    })

    def __post_init__(self) -> None:
        unknown = set(self.policies) - set(STRATEGIES)
        if unknown:
            raise ValueError(f"Unknown policies {sorted(unknown)}")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("task_ids", "policies", "prompt_variants"):
            value[key] = list(value[key])
        value["variants"] = list(self.variants) if self.variants is not None else None
        return value

    def units(self, tasks: Sequence[Task]) -> list[SweepUnit]:
        """Every (task, variant, policy, prompt) the sweep covers, in a fixed order."""
        by_id = {task.task_id: task for task in tasks}
        units = []
        for task_id in self.task_ids:
            task = by_id[task_id]
            variants = [v for v in task.variants() if self.variants is None or v in self.variants]
            for variant in variants:
                for policy in self.policies:
                    for prompt_variant in self.prompt_variants:
                        units.append(SweepUnit(task, variant, policy, prompt_variant))
        return units


# Episodes that stopped because the server failed (not the model, not a token limit) are rerun,
# under a new run label so they get a new episode id, up to SWEEP_RETRIES more times.
INFRASTRUCTURE_ABORTS = frozenset({
    "backend_error", "backend_timeout", "backend_connection_error", "invalid_backend_response",
})
SWEEP_RETRIES = 2


class SweepInterrupted(RuntimeError):
    """Too many episodes in a row aborted on infrastructure: the server is probably down."""


def infrastructure_abort(score: EpisodeScore) -> bool:
    """Whether an episode aborted because the server failed (so a rerun may succeed)."""
    return score.aborted and score.error_code in INFRASTRUCTURE_ABORTS


# The slot count may differ between resumes (the probe shrinks it when VRAM is short); episodes
# do not depend on it. Per-slot context, KV cache type, GPU and model still must match.
SWEEP_SHAPE_KEYS = ("parallel", "context_size")


def sweep_identity(spec_dict: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest fields a resumed sweep must match: everything but the slot count."""
    value = dict(spec_dict)
    if isinstance(value.get("server"), Mapping):
        value["server"] = {key: item for key, item in value["server"].items() if key not in SWEEP_SHAPE_KEYS}
    return value


def load_scores(path: str | Path) -> dict[str, EpisodeScore]:
    """The latest score of each unit in a sweep's scores file."""
    return {unit: attempts[-1] for unit, attempts in score_history(path).items()}


def score_history(path: str | Path) -> dict[str, list[EpisodeScore]]:
    """Every scored attempt of each unit, oldest first, skipping segment events.

    A copy taken mid-write (from Drive after a disconnect) can end in a partial
    line; that line is dropped and its episode simply runs again.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    done: dict[str, list[EpisodeScore]] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = strict_json_loads(line)
        except ValueError:
            if index == len(lines) - 1 and not text.endswith("\n"):
                break
            raise
        if record.get("event", "episode_scored") != "episode_scored":
            continue
        done.setdefault(record["unit"], []).append(EpisodeScore.from_dict(record))
    return done


def run_sweep(
    spec: SweepSpec,
    tasks: Sequence[Task],
    client_factory: Callable[[SweepUnit], Any],
    *,
    journal_dir: str | Path,
    sweep_dir: str | Path = LOCAL_SWEEP_DIR,
    workers: int = 1,
    copier: JournalCopier | None = None,
    sleep: Callable[[float], None] = no_sleep,
    budgets: Budgets | None = None,
    secrets: Sequence[str] = (),
    mirror_dir: str | Path | None = None,
    mirror_seconds: float = 30.0,
    retries: int = SWEEP_RETRIES,
    abort_limit: int | None = None,
) -> list[EpisodeScore]:
    """Run every unit of ``spec`` not already scored, ``workers`` at a time.

    Scores are appended to ``<sweep_id>.scores.jsonl`` as each episode ends, so
    a rerun resumes. A sweep manifest that differs from ``spec`` (other than in
    slot count) stops the run.

    With ``mirror_dir`` (on Drive), a missing local manifest and scores file
    are restored from it first, so a sweep resumes on a fresh runtime after a
    disconnect, and the scores file is copied back at most every
    ``mirror_seconds`` (through ``copier`` when given) and once at the end.

    A unit whose latest episode aborted on infrastructure is rerun (``retries``
    times at most) under ``run_label="retry-<n>"``. After ``abort_limit``
    infrastructure aborts in a row (default ``max(4, 2 * workers)``) no new
    episode starts and :class:`SweepInterrupted` is raised: restart the server
    and run again.
    """
    if spec.dataset_sha256 != dataset_sha256([task for task in tasks if task.task_id in set(spec.task_ids)]):
        raise SweepMismatch("dataset_sha256", spec.dataset_sha256, "the tasks given")
    sweep_dir = Path(sweep_dir)
    manifest_path = sweep_dir / f"{spec.sweep_id}.json"
    scores_path = sweep_dir / f"{spec.sweep_id}.scores.jsonl"
    mirror = Path(mirror_dir) if mirror_dir is not None else None

    # A local file with its Drive copy (if any); copies go through the journal copier when given.
    def mirrored(path: Path) -> MirroredFile:
        return MirroredFile(
            path, mirror / path.name if mirror is not None else None, every_seconds=mirror_seconds,
            submit=copier.submit_call if copier is not None else None,
        )

    manifest_store, scores_store = mirrored(manifest_path), mirrored(scores_path)
    if not manifest_path.exists() and manifest_store.restore():
        scores_store.restore()
    trim_partial_line(scores_path)  # a copy cut mid-record must not glue onto the next append
    if manifest_path.exists():
        recorded = sweep_identity(json.loads(manifest_path.read_text(encoding="utf-8")))
        current = sweep_identity(spec.as_dict())
        for key in sorted(set(recorded) | set(current)):
            if recorded.get(key) != current.get(key):
                raise SweepMismatch(key, recorded.get(key), current.get(key))
    else:
        atomic_write_json(manifest_path, spec.as_dict())

    history = score_history(scores_path) if scores_path.exists() else {}
    done: dict[str, EpisodeScore] = {}
    retry_labels: dict[str, str] = {}
    for unit_key, attempts in history.items():
        if infrastructure_abort(attempts[-1]) and len(attempts) <= retries:
            retry_labels[unit_key] = f"retry-{len(attempts)}"
        else:
            done[unit_key] = attempts[-1]
    recorder = JsonlRecorder(scores_path, secrets=secrets)
    lock = threading.Lock()
    limit = abort_limit or max(4, 2 * workers)
    aborts_in_a_row = [0]
    stop = threading.Event()

    # Run and score one unit, record it, and count infrastructure aborts in a row.
    def work(unit: SweepUnit) -> EpisodeScore | None:
        if stop.is_set():
            return None
        score = run_task(
            unit.task, unit.variant, policy=unit.policy, prompt_variant=unit.prompt_variant,
            directory=journal_dir, client=client_factory(unit), budgets=budgets, sleep=sleep, secrets=secrets,
            run_label=retry_labels.get(unit.key),
        )
        if score is None:
            return None
        score = replace(score, attempt=len(history.get(unit.key, ())) + 1)
        with lock:
            recorder.append({"event": "episode_scored", "unit": unit.key, **score.as_dict()})
            aborts_in_a_row[0] = aborts_in_a_row[0] + 1 if infrastructure_abort(score) else 0
            if aborts_in_a_row[0] >= limit:
                stop.set()
        scores_store.sync()
        if copier is not None:
            copier.submit(score.journal_path)
        return score

    pending = [unit for unit in spec.units(tasks) if unit.key not in done]
    if pending:
        with lock:
            recorder.append({
                "event": "sweep_segment", "workers": workers, "pending": len(pending), "done": len(done),
                **{key: spec.server.get(key) for key in SWEEP_SHAPE_KEYS},
            })
    manifest_store.sync()
    try:
        if workers <= 1:
            fresh = [work(unit) for unit in pending]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="steadyhand-sweep") as pool:
                fresh = list(pool.map(work, pending))
    finally:
        with lock:
            scores_store.sync(force=True)
    if stop.is_set():
        raise SweepInterrupted(
            f"{aborts_in_a_row[0]} episodes in a row aborted on infrastructure (last: "
            f"{next((s.error_code for s in reversed(fresh) if s is not None), None)}); no new episode was started. "
            "Restart llama-server and run the sweep again: aborted episodes are rerun."
        )
    return [*done.values(), *[score for score in fresh if score is not None]]


def run_cpu_suite(
    tasks: Sequence[Task],
    policies: Sequence[str],
    *,
    directory: str | Path,
    behavior: str = "follow_hints",
    variants: Sequence[str] | None = None,
    workers: int = 1,
) -> list[EpisodeScore]:
    """Every task variant under every policy with the scripted model. No GPU, no sleeping.

    ``workers > 1`` spreads tasks over a process pool (at most 8 processes), each
    loading its own fixture. Use it while llama-server loads, then stop it.
    """
    jobs = [(task.as_dict(), list(policies), str(directory), behavior, list(variants) if variants else None)
            for task in tasks]
    if workers <= 1:
        groups = [_cpu_task(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        count = min(workers, 8, os.cpu_count() or 1, len(jobs))
        with ProcessPoolExecutor(max_workers=count, mp_context=context) as pool:
            groups = list(pool.map(_cpu_task, jobs))
    return [EpisodeScore.from_dict(score) for group in groups for score in group]


# Runs in a pool process: every variant and policy for one task, on the scripted model.
def _cpu_task(job: tuple[dict[str, Any], list[str], str, str, list[str] | None]) -> list[dict[str, Any]]:
    task_value, policies, directory, behavior, variants = job
    task = Task.from_dict(task_value)
    fixture = load_fixture()
    scores = []
    for variant in task.variants():
        if variants is not None and variant not in variants:
            continue
        for policy in policies:
            score = run_task(task, variant, policy=policy, directory=directory, behavior=behavior,
                             fixture=fixture, run_label=f"cpu/{behavior}")
            if score is not None:
                scores.append(score.as_dict())
    return scores


__all__ = [
    "CATEGORIES",
    "CLEAN",
    "DATASET_PATH",
    "DatasetError",
    "DiffScore",
    "EpisodeScore",
    "FaultVariant",
    "LOCAL_SWEEP_DIR",
    "RECOVERY_LABELS",
    "ScriptedTaskModel",
    "StoreDiff",
    "INFRASTRUCTURE_ABORTS",
    "SWEEP_RETRIES",
    "SweepInterrupted",
    "SweepMismatch",
    "SweepSpec",
    "SweepUnit",
    "Task",
    "dataset_sha256",
    "infrastructure_abort",
    "load_scores",
    "load_tasks",
    "recovery_class",
    "run_cpu_suite",
    "run_sweep",
    "run_task",
    "score_diff",
    "score_history",
    "score_episode",
    "store_diff",
    "sweep_identity",
    "validate_tasks",
]

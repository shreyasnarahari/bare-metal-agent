"""The observation corpus gate, its must-keep scoring, and the live shaping benchmark.

**Corpus** (``evaluation/data/tidemark_observations_v1.jsonl``, built by
``tidemark.eval.observation_corpus``): 30–50 sanitized, real-shaped tool
outputs, each with labeled must-keep strings and the availability the task
needs (``inline`` or ``blob_handle``). ``validate_corpus`` checks the index,
payload hashes, labels, and that each entry's reducer is the one the frozen
policy assigns to its tool.

**Scoring** shapes every entry once through the real ``ObservationShaper``
and reports token reduction and restorability *separately*. A label passes
when it appears exactly in the preview (``inline``), or, when its
availability allows a handle, when a ``read_blob``-style range read of a
handle the preview names returns it exactly. Evidence that exists only on a
host path or in a raw journal does not count, a broken handle is a failure,
and no "nothing lost" claim is made: omitted ranges are reported as such.

**Benchmark** (live, ``tidemark-observation-benchmark``): one episode per
entry. The first step is forced (the real request is sent, then the scripted
corpus call replaces the model's output, as ``ForcedPrefixClient`` does), so
every episode receives its large observation; after that the model acts
freely and may call ``read_blob``. Episodes run through the shared scheduler,
so shaping one episode's output overlaps other episodes' model calls. The
shaping pool's workers and in-flight byte bound are autotuned on an unscored
subset, then frozen. Every journal is replayed on CPU afterwards with a
read-only blob store, which proves event order and blob references did not
diverge under concurrency.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from bulkhead.manifest import SecuritySpec, Trust
from steadyhand.context import ObservationSource
from steadyhand.controller import DEFAULT_GUARD_LIMIT, Budgets
from steadyhand.errors import FieldProblem
from steadyhand.journal import manifest_path_for, read_journal
from steadyhand.manifest import build_manifest, load_manifest
from steadyhand.messages import initial_messages
from steadyhand.registry import SteadyhandRegistry
from steadyhand.replay import replay_episode, run_claimed_episode
from steadyhand.store import Store, load_fixture
from steadyhand.tools import READ_POLICY
from steadyhand.tools_context import register_context_tools
from steadyhand.types import ModelTurn, ToolCall, TurnKind, observation_content
from toolsmith.evaluation.models import canonical_hash
from toolsmith.evaluation.runner import atomic_write_json
from toolsmith.json_utils import strict_json_loads

from ..blobs import BlobError, BlobStore, NamespacedBlobs, sha256_hex
from ..observations import DEFAULT_POLICY, ObservationConfig, ObservationPolicy, ObservationShaper, ReducerCache, ShapingPool
from ..reducers import page_text, replace_pointer, resolve_pointer
from ..tokenizer import RegexTokenizer, Tokenizer
from .observation_corpus import DATA_DIR, INDEX_PATH, VERSION

CATEGORIES = ("pytest", "compiler", "log", "text", "exec", "json", "html")
AVAILABILITY = ("inline", "blob_handle")
MIN_ENTRIES, MAX_ENTRIES = 30, 50
CORPUS_SECURITY = SecuritySpec(sandbox=None, output_trust=Trust.UNTRUSTED, sinks=())
MARKER_OFFSET = re.compile(r"read_blob offset (\d+)")
HANDLE = re.compile(r"blob://sha256:[0-9a-f]{64}")


class CorpusError(ValueError):
    """The corpus index or a payload is not well formed."""


@dataclass(frozen=True)
class CorpusEntry:
    observation_id: str
    version: str
    category: str
    tool: str
    reducer: str
    media_type: str
    description: str
    payload_file: str
    payload_encoding: str
    must_keep: tuple[dict[str, str], ...]
    optional_forbidden_preview: tuple[str, ...]
    source_license_or_origin: str
    sha256: str
    value_wrapper: dict[str, Any] | None = None
    text_pointer: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CorpusEntry:
        known = {name for name in cls.__dataclass_fields__}
        unexpected = set(value) - known
        if unexpected:
            raise CorpusError(f"{value.get('observation_id')}: unexpected fields {sorted(unexpected)}")
        data = dict(value)
        data["must_keep"] = tuple(dict(item) for item in data["must_keep"])
        data["optional_forbidden_preview"] = tuple(data.get("optional_forbidden_preview", ()))
        return cls(**data)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["must_keep"] = [dict(item) for item in self.must_keep]
        value["optional_forbidden_preview"] = list(self.optional_forbidden_preview)
        if self.value_wrapper is None:
            value.pop("value_wrapper")
            value.pop("text_pointer")
        return value

    def payload_path(self, root: str | Path = DATA_DIR) -> Path:
        return Path(root) / self.payload_file

    def load_value(self, root: str | Path = DATA_DIR) -> Any:
        """The tool result value, after checking the payload's sha256."""
        data = self.payload_path(root).read_bytes()
        if sha256_hex(data) != self.sha256:
            raise CorpusError(f"{self.observation_id}: payload sha256 differs from the index")
        if self.payload_encoding == "json":
            return strict_json_loads(data.decode("utf-8"))
        text = data.decode("utf-8")
        if self.value_wrapper is None:
            return text
        return replace_pointer(self.value_wrapper, self.text_pointer, text)


def load_corpus(path: str | Path = INDEX_PATH) -> list[CorpusEntry]:
    entries = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                entries.append(CorpusEntry.from_dict(strict_json_loads(line)))
            except (ValueError, TypeError, KeyError) as exc:
                raise CorpusError(f"line {number}: {exc}") from None
    return entries


def corpus_sha256(entries: Sequence[CorpusEntry]) -> str:
    return canonical_hash([entry.as_dict() for entry in entries])


def strings_in(value: Any) -> list[str]:
    """Every string leaf (and object key) of a JSON value, in order."""
    found: list[str] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found.append(item)
        elif isinstance(item, Mapping):
            for key, child in reversed(list(item.items())):
                stack.append(child)
                stack.append(key)
        elif isinstance(item, list):
            stack.extend(reversed(item))
    return found


def payload_text(value: Any) -> str:
    """Every string in the value plus its canonical JSON (so numeric and key/value labels can be found)."""
    return "\n".join(strings_in(value)) + "\n" + json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def page_visible_text(entry: CorpusEntry, value: Any) -> str:
    body = resolve_pointer(value, entry.text_pointer)
    return "\n".join(line for line, _ in page_text(body)[1]) if isinstance(body, str) else ""


def validate_corpus(entries: Sequence[CorpusEntry], *, policy: ObservationPolicy = DEFAULT_POLICY,
                    root: str | Path = DATA_DIR) -> dict[str, Any]:
    """Raise ``CorpusError`` listing every problem; return the corpus summary when there is none."""
    problems: list[str] = []
    if not MIN_ENTRIES <= len(entries) <= MAX_ENTRIES:
        problems.append(f"the corpus needs {MIN_ENTRIES}-{MAX_ENTRIES} entries, has {len(entries)}")
    ids = [entry.observation_id for entry in entries]
    if len(set(ids)) != len(ids):
        problems.append("observation ids must be unique")
    for entry in entries:
        where = entry.observation_id
        if entry.version != VERSION:
            problems.append(f"{where}: version {entry.version!r} is not {VERSION}")
        if entry.category not in CATEGORIES:
            problems.append(f"{where}: unknown category {entry.category!r}")
        if policy.rule_for(entry.tool).reducer != entry.reducer:
            problems.append(f"{where}: the policy shapes {entry.tool} with {policy.rule_for(entry.tool).reducer}, "
                            f"the entry says {entry.reducer}")
        if not entry.must_keep:
            problems.append(f"{where}: no must-keep labels")
        if not entry.source_license_or_origin:
            problems.append(f"{where}: no origin")
        try:
            value = entry.load_value(root)
        except (OSError, CorpusError, ValueError) as exc:
            problems.append(f"{where}: {exc}")
            continue
        raw = payload_text(value)
        visible = raw + "\n" + page_visible_text(entry, value) if entry.reducer == "html_text_v1" else raw
        for label in entry.must_keep:
            if set(label) != {"exact", "pointer_or_line", "availability"}:
                problems.append(f"{where}: labels have exact, pointer_or_line, and availability")
                continue
            if label["availability"] not in AVAILABILITY:
                problems.append(f"{where}: availability must be one of {AVAILABILITY}")
            # A restorable label must be in the raw payload (the blob); an inline one may be a page's visible text.
            haystack = raw if label["availability"] == "blob_handle" else visible
            if not label["exact"] or label["exact"] not in haystack:
                problems.append(f"{where}: must-keep {label['exact'][:60]!r} is not in the payload")
    if problems:
        raise CorpusError("; ".join(problems))
    labels = [label for entry in entries for label in entry.must_keep]
    return {
        "version": VERSION, "entries": len(entries), "corpus_sha256": corpus_sha256(entries),
        "categories": {name: sum(1 for e in entries if e.category == name) for name in CATEGORIES},
        "labels": len(labels), "inline_labels": sum(1 for label in labels if label["availability"] == "inline"),
        "payload_bytes": sum(entry.payload_path(root).stat().st_size for entry in entries),
    }


def source_for(entry: CorpusEntry, value: Any, *, episode_id: str = "corpus", source_seq: int = 1,
               verifying: bool = False) -> ObservationSource:
    observation = {"ok": True, "value": value}
    return ObservationSource(
        episode_id=episode_id, step=0, source_seq=source_seq, tool_call_id=f"call_{entry.observation_id}", tool=entry.tool,
        ok=True, handler_called=True, observation=observation, content=observation_content(observation), verifying=verifying,
    )


def range_returns(blobs: NamespacedBlobs, handle: str, exact: str) -> dict[str, Any] | None:
    """Prove a range read returns ``exact``: locate it in the verified object, then read that range."""
    record = blobs.verify(handle)
    payload = blobs.read_bytes(handle)
    candidates = [exact]
    if record.media_type.startswith("application/json"):
        candidates.append(json.dumps(exact, ensure_ascii=False)[1:-1])
    for wanted in candidates:
        needle = wanted.encode("utf-8")
        start = payload.find(needle)
        if start < 0:
            continue
        collected, offset = "", start
        while len(collected.encode("utf-8")) < len(needle):
            window = blobs.read(handle, offset=offset, max_bytes=min(blobs.max_read_bytes, len(needle)))
            collected += window["content"]
            if window["next_offset"] is None:
                break
            offset = window["next_offset"]
        if wanted in collected:
            return {"handle": handle, "offset": start, "bytes": len(needle)}
    return None


def score_entry(entry: CorpusEntry, shaper: ObservationShaper, *, root: str | Path = DATA_DIR,
                episode_id: str = "corpus") -> dict[str, Any]:
    """Shape one entry and check every label, the forbidden strings, the blobs, and the token counts."""
    value = entry.load_value(root)
    source = source_for(entry, value, episode_id=episode_id)
    try:
        shaped = shaper.shape(source)
    except BlobError as exc:  # a store that cannot keep the payload is a failure, never a compression win
        return {
            "observation_id": entry.observation_id, "category": entry.category, "tool": entry.tool, "reducer": None,
            "expected_reducer": entry.reducer, "shaped": False, "raw_tokens": shaper.count(source.content),
            "shaped_tokens": shaper.count(source.content), "cap": shaper.policy.rule_for(entry.tool).preview_tokens,
            "over_cap": True, "raw_bytes": len(source.content.encode("utf-8")), "shaped_bytes": None,
            "blob_handles": [], "omitted_ranges": 0,
            "labels": [{**label, "inline": False, "restorable": False, "passed": False, "via": None, "range": None}
                       for label in entry.must_keep],
            "forbidden_in_preview": [], "blob_failures": [str(exc)], "shaped_content_sha256": None, "passed": False,
        }
    event = shaped.event
    parsed = json.loads(shaped.message["content"])
    preview = strings_in(parsed)
    handles = [part["handle"] for part in event["blob_parts"]] if event is not None else []
    blobs = shaper.blobs()
    blob_failures = []
    for handle in handles:
        try:
            blobs.verify(handle)
        except BlobError as exc:
            blob_failures.append(f"{handle}: {exc}")
    labels, needed = [], {}
    for label in entry.must_keep:
        if label["availability"] == "inline":
            needed[label["exact"]] = needed.get(label["exact"], 0) + 1
    content = shaped.message["content"]
    for label in entry.must_keep:
        exact = label["exact"]
        # Strings are searched decoded; numbers and "key": value pairs in the serialized content.
        occurrences = max(sum(text.count(exact) for text in preview), content.count(exact))
        inline = occurrences >= max(1, needed.get(exact, 0)) if label["availability"] == "inline" else occurrences > 0
        restored = None
        if not inline and not blob_failures:
            for handle in handles:
                restored = range_returns(blobs, handle, exact)
                if restored is not None:
                    break
        passed = inline or (label["availability"] == "blob_handle" and restored is not None)
        labels.append({**label, "inline": inline, "restorable": restored is not None or inline, "passed": passed,
                       "via": "inline" if inline else ("blob_handle" if restored is not None else None),
                       "range": restored})
    forbidden = [text for text in entry.optional_forbidden_preview if any(text in item for item in preview)]
    raw_tokens = event["raw_tokens"] if event is not None else shaper.count(source.content)
    shaped_tokens = event["shaped_tokens"] if event is not None else raw_tokens
    cap = shaper.policy.rule_for(entry.tool).preview_tokens
    return {
        "observation_id": entry.observation_id, "category": entry.category, "tool": entry.tool,
        "reducer": event["reducer"] if event is not None else "none", "expected_reducer": entry.reducer,
        "shaped": event is not None, "raw_tokens": raw_tokens, "shaped_tokens": shaped_tokens, "cap": cap,
        "over_cap": shaped_tokens > cap, "raw_bytes": len(source.content.encode("utf-8")),
        "shaped_bytes": len(shaped.message["content"].encode("utf-8")), "blob_handles": handles,
        "omitted_ranges": len(event["omitted_ranges"]) if event is not None else 0, "labels": labels,
        "forbidden_in_preview": forbidden, "blob_failures": blob_failures,
        "shaped_content_sha256": sha256_hex(shaped.message["content"].encode("utf-8")),
        "passed": all(item["passed"] for item in labels) and not forbidden and not blob_failures and shaped_tokens <= cap,
    }


def summarize_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [label for row in rows for label in row["labels"]]
    raw = sum(row["raw_tokens"] for row in rows)
    shaped = sum(row["shaped_tokens"] for row in rows)
    ratios = [1 - row["shaped_tokens"] / row["raw_tokens"] for row in rows if row["raw_tokens"]]
    passed_labels = sum(1 for label in labels if label["passed"])
    return {
        "entries": len(rows), "shaped_entries": sum(1 for row in rows if row["shaped"]),
        "raw_tokens": raw, "shaped_tokens": shaped, "token_reduction": round(1 - shaped / raw, 6) if raw else None,
        "median_entry_reduction": round(statistics.median(ratios), 6) if ratios else None,
        "must_keep": {
            "labels": len(labels), "passed": passed_labels,
            "inline": sum(1 for label in labels if label["via"] == "inline"),
            "restorable_only": sum(1 for label in labels if label["via"] == "blob_handle"),
            "inline_required": sum(1 for label in labels if label["availability"] == "inline"),
            "inline_required_met": sum(1 for label in labels if label["availability"] == "inline" and label["inline"]),
            "failed": [f"{row['observation_id']}: {label['exact'][:80]}" for row in rows for label in row["labels"]
                       if not label["passed"]],
        },
        "survival_or_restorability": round(passed_labels / len(labels), 6) if labels else None,
        "forbidden_violations": [row["observation_id"] for row in rows if row["forbidden_in_preview"]],
        "blob_failures": [failure for row in rows for failure in row["blob_failures"]],
        "over_cap": [row["observation_id"] for row in rows if row["over_cap"]],
        "by_category": {
            category: {
                "entries": sum(1 for row in rows if row["category"] == category),
                "raw_tokens": sum(row["raw_tokens"] for row in rows if row["category"] == category),
                "shaped_tokens": sum(row["shaped_tokens"] for row in rows if row["category"] == category),
            } for category in sorted({row["category"] for row in rows})
        },
        "passed": bool(rows) and all(row["passed"] for row in rows),
    }


def round_trip(blobs: NamespacedBlobs, handle: str, *, max_bytes: int = 997) -> dict[str, Any]:
    """Read a whole blob in small ranges (an odd size, to cross UTF-8 boundaries) and compare with its bytes."""
    payload = blobs.read_bytes(handle)
    collected, offset, reads, adjusted = [], 0, 0, 0
    while True:
        window = blobs.read(handle, offset=offset, max_bytes=max_bytes)
        collected.append(window["content"])
        reads += 1
        adjusted += int(window["boundary_adjusted"])
        if window["next_offset"] is None:
            break
        offset = window["next_offset"]
    joined = "".join(collected).encode("utf-8")
    return {"handle": handle, "bytes": len(payload), "reads": reads, "boundary_adjusted": adjusted, "equal": joined == payload}


def observation_gate(
    store_root: str | Path,
    *,
    tokenizer: Tokenizer | None = None,
    policy: ObservationPolicy = DEFAULT_POLICY,
    namespace: str = "corpus-gate",
    repeats: int = 2,
    root: str | Path = DATA_DIR,
    entries: Sequence[CorpusEntry] | None = None,
) -> dict[str, Any]:
    """The CPU gate: validate, score every entry, check determinism, read-only verification, and round trips."""
    tokenizer = tokenizer or RegexTokenizer()
    entries = list(entries) if entries is not None else load_corpus()
    summary = validate_corpus(entries, policy=policy, root=root)
    store = BlobStore(store_root)
    config = ObservationConfig.for_store(store, namespace, policy)
    shaper = ObservationShaper(config, tokenizer, store)
    started = time.perf_counter()
    rows = [score_entry(entry, shaper, root=root) for entry in entries]
    elapsed = time.perf_counter() - started
    scores = summarize_scores(rows)
    # Determinism: fresh shapers (no cache) give byte-identical messages; a read-only store verifies, never writes.
    unstable = []
    readonly = ObservationShaper(config, tokenizer, BlobStore(store_root, read_only=True))
    for entry, row in zip(entries, rows):
        for _ in range(repeats):
            again = ObservationShaper(config, tokenizer, store).shape(source_for(entry, entry.load_value(root)))
            if sha256_hex(again.message["content"].encode("utf-8")) != row["shaped_content_sha256"]:
                unstable.append(entry.observation_id)
                break
        replayed = readonly.shape(source_for(entry, entry.load_value(root), verifying=True))
        if sha256_hex(replayed.message["content"].encode("utf-8")) != row["shaped_content_sha256"]:
            unstable.append(f"{entry.observation_id} (read-only)")
    trips = [round_trip(shaper.blobs(), handle) for row in rows[:6] for handle in row["blob_handles"][:1]]
    revision_before = store.catalog.revision
    passed = scores["passed"] and not unstable and all(trip["equal"] for trip in trips)
    return {
        "event": "tidemark_observation_gate", "passed": passed, "corpus": summary, "scores": scores,
        "deterministic": not unstable, "unstable": unstable, "round_trips": trips,
        "catalog_revision": revision_before, "catalog_id": store.catalog_id, "namespace": namespace,
        "tokenizer": dict(tokenizer.identity()), "policy_sha256": policy.sha256(), "shaping_s": round(elapsed, 3),
        "rows": rows,
    }


# Live benchmark: corpus tools, a forced first call, and the runner


def build_corpus_registry(entries: Sequence[CorpusEntry], *, root: str | Path = DATA_DIR) -> SteadyhandRegistry:
    """One read tool per corpus tool name (``observation_id`` -> that entry's value), plus ``read_blob``."""
    by_tool: dict[str, dict[str, CorpusEntry]] = {}
    for entry in entries:
        by_tool.setdefault(entry.tool, {})[entry.observation_id] = entry

    @lru_cache(maxsize=8)
    def value_of(observation_id: str) -> str:
        entry = next(e for e in entries if e.observation_id == observation_id)
        return json.dumps(entry.load_value(root), ensure_ascii=False)

    registry = SteadyhandRegistry()
    for tool in sorted(by_tool):
        known = by_tool[tool]

        def handler(store: Store, observation_id: Annotated[str, Field(description="The corpus observation id.")],
                    _known: dict[str, CorpusEntry] = known) -> Any:
            return json.loads(value_of(observation_id))

        def check(store: Store, arguments: Mapping[str, Any], _known: dict[str, CorpusEntry] = known) -> list[FieldProblem]:
            ident = arguments.get("observation_id")
            return [] if ident in _known else [FieldProblem("/observation_id", "is not a known observation for this tool", ident)]

        handler.__name__ = tool
        handler.__qualname__ = tool
        registry.register(handler, READ_POLICY, check, security=CORPUS_SECURITY,
                          parameters={"type": "object", "properties": {"observation_id": {"type": "string",
                                      "description": "The corpus observation id."}}, "required": ["observation_id"],
                                      "additionalProperties": False},
                          description=f"Return the output of the {tool} observation with this id.")
    register_context_tools(registry)
    return registry.freeze()


def benchmark_messages(entry: CorpusEntry) -> list[dict[str, Any]]:
    goal = (f"Call {entry.tool} with observation_id \"{entry.observation_id}\". It returns {entry.description}. "
            "Then answer in one or two sentences, quoting the exact line or value that explains the result. "
            "If the evidence was shortened, use read_blob with the handle and offset shown.")
    return initial_messages(goal)


def _forced_turn(entry: CorpusEntry, turn: ModelTurn) -> ModelTurn:
    call = {"id": f"call_{entry.observation_id}", "type": "function",
            "function": {"name": entry.tool, "arguments": json.dumps({"observation_id": entry.observation_id})}}
    return ModelTurn(
        kind=TurnKind.TOOL_CALLS, assistant_message={"role": "assistant", "content": None, "tool_calls": [call]},
        tool_calls=(ToolCall.from_openai(call),), content=None, reasoning_content=None, prompt_tokens=turn.prompt_tokens,
        completion_tokens=turn.completion_tokens, error_code=None, elapsed_s=turn.elapsed_s, metrics=turn.metrics,
    )


class ForcedFirstCallClient:
    """Live client whose first turn sends the real request, then returns the scripted corpus call."""

    def __init__(self, live: Any, entry: CorpusEntry) -> None:
        self.live = live
        self.entry = entry
        self._first = True
        self.sources: list[str] = []

    def settings(self) -> dict[str, Any]:
        return self.live.settings()

    def render_prompt(self, *args: Any, **kwargs: Any) -> str:
        return self.live.render_prompt(*args, **kwargs)

    def complete(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]], *,
                 tool_choice: Any = "auto", **options: Any) -> ModelTurn:
        turn = self.live.complete(messages, tools, tool_choice=tool_choice, **options)
        if self._first and tool_choice != "none" and turn.kind is not TurnKind.FATAL_ERROR:
            self._first = False
            self.sources.append("forced")
            return _forced_turn(self.entry, turn)
        self.sources.append("model")
        return turn


class CorpusScriptedModel:
    """CPU model for the pipeline: the corpus call, then (if the evidence was cut) a ``read_blob``, then an answer."""

    def __init__(self, entry: CorpusEntry, *, max_tokens: int = 256, delay_s: float = 0.0) -> None:
        self.entry = entry
        self.max_tokens = max_tokens
        self.delay_s = delay_s
        self._step = 0

    def settings(self) -> dict[str, Any]:
        return {"profile": "scripted", "model": "scripted/observations", "protocol": None, "use_grammars": True,
                "enable_thinking": False, "max_tokens": self.max_tokens, "temperature": 0.0, "seed": 0,
                "slot_context": None}

    def complete(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]], *,
                 tool_choice: Any = "auto", **options: Any) -> ModelTurn:
        if self.delay_s:
            time.sleep(self.delay_s)
        step, self._step = self._step, self._step + 1
        base = ModelTurn(kind=TurnKind.TEXT, assistant_message=None, tool_calls=(), content=None, reasoning_content=None,
                         prompt_tokens=100, completion_tokens=10, error_code=None)
        if step == 0:
            return _forced_turn(self.entry, base)
        exact = self.entry.must_keep[0]["exact"]
        tool_messages = [str(m.get("content") or "") for m in messages if m.get("role") == "tool"]
        seen = "\n".join(strings_in([json.loads(text) for text in tool_messages if text.startswith("{")]))
        if exact in seen or step >= 2 or tool_choice == "none":
            answer = f"Evidence: {exact}" if exact in seen else "I could not find the evidence."
            return ModelTurn(kind=TurnKind.TEXT, assistant_message={"role": "assistant", "content": answer}, tool_calls=(),
                             content=answer, reasoning_content=None, prompt_tokens=100, completion_tokens=10, error_code=None)
        handles = HANDLE.findall(tool_messages[-1]) if tool_messages else []
        offsets = [int(item) for item in MARKER_OFFSET.findall(tool_messages[-1])] if tool_messages else []
        if not handles:
            answer = "I could not find the evidence."
            return ModelTurn(kind=TurnKind.TEXT, assistant_message={"role": "assistant", "content": answer}, tool_calls=(),
                             content=answer, reasoning_content=None, prompt_tokens=100, completion_tokens=10, error_code=None)
        call = {"id": f"call_read_{step}", "type": "function", "function": {"name": "read_blob", "arguments": json.dumps(
            {"handle": handles[0], "offset": offsets[0] if offsets else 0, "max_bytes": 16384})}}
        return ModelTurn(kind=TurnKind.TOOL_CALLS, assistant_message={"role": "assistant", "content": None, "tool_calls": [call]},
                         tool_calls=(ToolCall.from_openai(call),), content=None, reasoning_content=None, prompt_tokens=100,
                         completion_tokens=10, error_code=None)


@dataclass(frozen=True)
class ShapingProfile:
    """The shaping pipeline's tunable shape: reducer workers and the in-flight input byte bound."""

    workers: int
    max_inflight_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def shaping_candidates(cpu_count: int, ram_budget_mib: int, *, llama_threads: int = 1) -> list[ShapingProfile]:
    """Candidates from the detected envelope (never the GPU name): workers up to the CPUs left after llama.cpp."""
    spare = max(1, cpu_count - llama_threads)
    workers = sorted({1, max(1, spare // 2), spare})
    bound = max(4, min(256, ram_budget_mib // 64)) * 1024 * 1024
    sizes = sorted({max(4 * 1024 * 1024, bound // 4), bound})
    return [ShapingProfile(w, size) for w in workers for size in sizes]


@dataclass(frozen=True)
class BenchmarkRow:
    observation_id: str
    category: str
    status: str
    outcome: str
    steps: int
    shaped: int
    raw_tokens: int
    shaped_tokens: int
    read_blob_calls: int
    answer_has_evidence: bool
    episode_id: str
    journal_path: str
    turn_sources: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "turn_sources": list(self.turn_sources)}


def benchmark_row(entry: CorpusEntry, records: Sequence[Mapping[str, Any]], journal_path: str | Path,
                  sources: Sequence[str] = ()) -> BenchmarkRow:
    finished = records[-1]
    shaped = [r for r in records if r["event"] == "observation_shaped"]
    final = finished.get("final_content") or ""
    labels = [label["exact"] for label in entry.must_keep]
    status = finished["status"]
    outcome = "infrastructure" if status == "aborted" and finished.get("error_code") not in (None, "context_budget_exhausted") else (
        "did_not_fit" if finished.get("error_code") == "context_budget_exhausted" else "completed")
    return BenchmarkRow(
        observation_id=entry.observation_id, category=entry.category, status=status, outcome=outcome,
        steps=finished["steps"], shaped=len(shaped), raw_tokens=sum(r["raw_tokens"] for r in shaped),
        shaped_tokens=sum(r["shaped_tokens"] for r in shaped),
        read_blob_calls=sum(1 for r in records if r["event"] == "tool_result" and r["tool"] == "read_blob"),
        answer_has_evidence=any(label in final for label in labels), episode_id=records[0]["episode_id"],
        journal_path=str(journal_path), turn_sources=tuple(sources),
    )


def run_observation_benchmark(
    entries: Sequence[CorpusEntry],
    *,
    run_dir: str | Path,
    run_label: str,
    client_factory: Callable[[CorpusEntry, str], Any],
    session_factory: Callable[[CorpusEntry, Any], Any],
    scheduler: Any = None,
    registry: SteadyhandRegistry | None = None,
    budgets: Budgets | None = None,
    secrets: Sequence[str] = (),
    context_server: Mapping[str, Any] | None = None,
    root: str | Path = DATA_DIR,
    fixture: Store | None = None,
) -> dict[str, Any]:
    """One claimed, resumable episode per entry (through ``scheduler`` when given); rows plus throughput."""
    registry = registry or build_corpus_registry(entries, root=root)
    budgets = budgets or Budgets(max_steps=4)
    fixture = fixture or load_fixture()
    journal_dir = Path(run_dir) / "journals"

    def work(entry: CorpusEntry, owner: str) -> BenchmarkRow | None:
        inner = client_factory(entry, owner)
        client = scheduler.client(inner, owner) if scheduler is not None else inner
        session = session_factory(entry, client)
        messages = benchmark_messages(entry)
        result = run_claimed_episode(journal_dir, client, registry, fixture.clone(), messages, budgets=budgets,
                                     secrets=secrets, run_label=run_label, context=session,
                                     context_server=context_server)
        if result is None:
            manifest = build_manifest(client.settings(), registry, fixture.clone(), messages, budgets=budgets.as_dict(),
                                      guard_limit=DEFAULT_GUARD_LIMIT, run_label=run_label, context_config=session.identity(),
                                      context_version=session.version, context_server=context_server)
            path = journal_dir / f"{manifest.episode_id}.jsonl"
            if not (manifest_path_for(path).exists() and load_manifest(manifest_path_for(path)).completed):
                return None
        else:
            path = result.journal_path
        return benchmark_row(entry, read_journal(path), path, getattr(inner, "sources", ()))

    started = time.monotonic()
    if scheduler is not None:
        rows = scheduler.run_episodes(list(entries), work, key=lambda entry: entry.observation_id)
    else:
        rows = [work(entry, entry.observation_id) for entry in entries]
    elapsed = time.monotonic() - started
    completed = [row for row in rows if row is not None]
    return {
        "rows": completed, "elapsed_s": round(elapsed, 3),
        "episodes_per_minute": round(len(completed) / (elapsed / 60), 3) if elapsed > 0 else None,
        "shaped_observations": sum(row.shaped for row in completed),
        "answers_with_evidence": sum(1 for row in completed if row.answer_has_evidence),
        "read_blob_calls": sum(row.read_blob_calls for row in completed),
    }


def replay_benchmark(rows: Iterable[BenchmarkRow], *, registry: SteadyhandRegistry,
                     session_factory: Callable[[str], Any]) -> dict[str, Any]:
    """Replay every journal on CPU with a fresh, read-only session (``session_factory(observation_id)``)."""
    failures, replayed = [], 0
    for row in rows:
        try:
            replay_episode(row.journal_path, registry=registry, context=session_factory(row.observation_id))
            replayed += 1
        except Exception as exc:  # reported, never hidden
            failures.append({"observation_id": row.observation_id, "error": f"{type(exc).__name__}: {exc}"})
    return {"replayed": replayed, "failures": failures, "passed": not failures}


def select_profile(measurements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The correct profile (replays clean, nothing lost) with the most episodes per minute; ties keep fewer workers."""
    eligible = [m for m in measurements if m.get("correct")]
    if not eligible:
        raise RuntimeError("no shaping profile completed correctly; see the measurements")
    return max(eligible, key=lambda m: (m["episodes_per_minute"] or 0.0, -m["profile"]["workers"],
                                        -m["profile"]["max_inflight_bytes"]))


def autotune_shaping(profiles: Sequence[ShapingProfile], evaluate: Callable[[ShapingProfile], Mapping[str, Any]]) -> dict[str, Any]:
    """Measure each profile (unscored), then freeze the best. ``evaluate`` returns throughput and correctness."""
    measurements = []
    for profile in profiles:
        measured = dict(evaluate(profile))
        measurements.append({"profile": profile.as_dict(), **measured})
    selected = select_profile(measurements)
    return {"selected": selected["profile"], "measurements": measurements,
            "selected_sha256": canonical_hash(selected["profile"])}


def write_benchmark(path: str | Path, report: Mapping[str, Any]) -> Path:
    serial = {key: ([row.as_dict() if isinstance(row, BenchmarkRow) else row for row in value] if key == "rows" else value)
              for key, value in report.items()}
    return atomic_write_json(path, serial)


__all__ = [
    "AVAILABILITY", "BenchmarkRow", "CATEGORIES", "CorpusEntry", "CorpusError", "CorpusScriptedModel",
    "ForcedFirstCallClient", "ShapingProfile", "autotune_shaping", "benchmark_messages", "benchmark_row",
    "build_corpus_registry", "corpus_sha256", "load_corpus", "observation_gate", "range_returns", "replay_benchmark",
    "round_trip", "run_observation_benchmark", "score_entry", "select_profile", "shaping_candidates", "source_for",
    "strings_in", "summarize_scores", "validate_corpus", "write_benchmark",
]

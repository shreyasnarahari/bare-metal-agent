"""Observation shaping: large tool results are shrunk once, when they arrive, and stay restorable.

The loop journals the raw ``tool_result`` first; that record stays the source
of truth for replay and Bulkhead taint. The session then asks
``ObservationShaper.shape`` for the message to append:

1. A typed error (validation, guard, denial, ASK, timeout, parse, handler
   error) is never shaped: its bytes reach the model unchanged.
2. The frozen ``ObservationPolicy`` picks the tool's rule (an unknown tool gets
   deterministic bounded JSON, never a guess from its name). A result whose
   exact content is within the rule's real-token cap is left as it is.
3. Otherwise the rule's reducer (``tidemark.reducers``) plans the complete
   payload(s) and preview candidates, and the first candidate that fits the cap
   (counted with the parity-checked tokenizer on the final message content) is
   kept. The complete payload goes to the blob store (``tidemark.blobs``) before
   anything is journaled, so every omitted byte stays reachable by handle.
4. The shaped message keeps the ordinary contract, ``{"ok": true, "value": ...}``,
   plus a bounded ``context`` member naming the shape, the handle(s), the
   complete size, the preview's token count, and how to read a range. The
   ``observation_shaped`` event records hashes, counts, ranges, and the handle,
   never the payload.

Shaping happens once and its message is appended to the immutable history; an
old observation is never recomputed to fit a later budget. Replay recomputes
the same message from the recorded raw result and verifies (never writes) the
blob. ``context`` is provenance, not a security label: trust still comes from
Bulkhead's ``SecuritySpec``, and taint from the raw result.

**Throughput.** Reductions for different calls and episodes may run together
on a ``ShapingPool`` (bounded by in-flight input bytes, admitted in arrival
order, so an earlier observation is never delayed for a later one), while each
episode still commits its raw result, shape event, message, and taint in call
order. ``ReducerCache`` reuses a reduction only for identical raw content,
policy rule, reducer version, and tokenizer; blob catalog entries are still
written per namespace. Local blob writes are synchronous and atomic; Drive
copies run on ``tidemark.blobs.BlobMirror``'s background thread. Timings are
volatile metrics, never journaled.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from steadyhand.context import CallEnvironment, ObservationSource, ShapedObservation
from steadyhand.journal import MAX_OMITTED_RANGES, MAX_PREVIEW_METADATA_BYTES, ReplayMismatch
from steadyhand.types import observation_content
from toolsmith.evaluation.models import canonical_hash

from .blobs import BLOB_FORMAT_VERSION, BlobError, BlobStore, NamespacedBlobs, check_namespace, verify_references
from .reducers import (
    EXEC_STREAMS,
    HTML_TEXT,
    JSON_BOUNDED,
    JSON_PROJECTION,
    LOG_HEAD_TAIL,
    NO_REDUCER,
    REDUCER_VERSION,
    REDUCERS,
    TEXT_HEAD_TAIL,
    BlobPart,
    Candidate,
    choose,
    plan_for,
)
from .resources import ToolExecutionContext
from .tokenizer import InvalidText, Tokenizer

POLICY_VERSION = 1
OBSERVATION_CONFIG_VERSION = 1
RESTORE_HINT = "read_blob with this handle; offset is a byte offset (see the [tidemark: ...] markers)"


@dataclass(frozen=True)
class ReducerRule:
    """How one tool's results are shaped. Part of the context configuration (an identity field)."""

    reducer: str
    preview_tokens: int = 1024
    text_pointer: str = ""
    pointers: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    head_lines: int = 30
    tail_lines: int = 40
    max_line_chars: int = 800
    must_keep_patterns: tuple[str, ...] = ()
    marks: bool = True
    media_type: str | None = None

    def __post_init__(self) -> None:
        if self.reducer not in REDUCERS:
            raise ValueError(f"unknown reducer {self.reducer!r}; reducers are {REDUCERS}")
        for name in ("preview_tokens", "head_lines", "tail_lines", "max_line_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name != "head_lines" else 0):
                raise ValueError(f"{name} must be a positive integer")
        for pointer in (*self.pointers, *self.required, *((self.text_pointer,) if self.text_pointer else ())):
            if not pointer.startswith("/"):
                raise ValueError(f"JSON Pointers start with '/': {pointer!r}")
        if self.reducer == JSON_PROJECTION and not self.pointers:
            raise ValueError("json_projection_v1 needs an allowlist of pointers")
        object.__setattr__(self, "pointers", tuple(self.pointers))
        object.__setattr__(self, "required", tuple(self.required))
        object.__setattr__(self, "must_keep_patterns", tuple(self.must_keep_patterns))

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["pointers"] = list(self.pointers)
        value["required"] = list(self.required)
        value["must_keep_patterns"] = list(self.must_keep_patterns)
        return value

    def sha256(self) -> str:
        return canonical_hash(self.as_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReducerRule:
        data = dict(value)
        for name in ("pointers", "required", "must_keep_patterns"):
            data[name] = tuple(data.get(name, ()))
        return cls(**data)


@dataclass(frozen=True)
class ObservationPolicy:
    """The frozen side table: tool name -> rule, a bounded-JSON default, and the audit switch."""

    rules: Mapping[str, ReducerRule]
    default: ReducerRule = field(default_factory=lambda: ReducerRule(JSON_BOUNDED, preview_tokens=768))
    audit: bool = False
    version: int = POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", dict(sorted(dict(self.rules).items())))

    def rule_for(self, tool: str) -> ReducerRule:
        return self.rules.get(tool, self.default)

    def with_rules(self, **rules: ReducerRule) -> ObservationPolicy:
        return ObservationPolicy({**self.rules, **rules}, default=self.default, audit=self.audit, version=self.version)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "audit": self.audit, "default": self.default.as_dict(),
            "rules": {name: rule.as_dict() for name, rule in self.rules.items()},
        }

    def sha256(self) -> str:
        return canonical_hash(self.as_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ObservationPolicy:
        return cls(
            {name: ReducerRule.from_dict(rule) for name, rule in value["rules"].items()},
            default=ReducerRule.from_dict(value["default"]), audit=bool(value["audit"]), version=int(value["version"]),
        )


EXEC_RULE = ReducerRule(EXEC_STREAMS, preview_tokens=1200, head_lines=20, tail_lines=40)
LOG_RULE = ReducerRule(LOG_HEAD_TAIL, preview_tokens=1200, head_lines=20, tail_lines=40)
HTML_RULE = ReducerRule(HTML_TEXT, preview_tokens=1000, text_pointer="/body", head_lines=40, tail_lines=20, marks=False)
# Projections were reviewed against each store tool's result schema (``check_projections``). Free text that
# outside parties write (note text, refund and ledger reasons) is left out of the preview and kept in the blob.
STORE_PROJECTIONS = {
    "get_order": ReducerRule(JSON_PROJECTION, preview_tokens=768, pointers=(
        "/order_id", "/customer_id", "/status", "/placed_at", "/total_inr", "/cancelled_at", "/items/*/sku",
        "/items/*/name", "/items/*/quantity", "/items/*/unit_price_inr", "/notes/*/note_id", "/notes/*/created_at",
    ), required=("/order_id", "/customer_id", "/status", "/total_inr")),
    "get_ticket": ReducerRule(JSON_PROJECTION, preview_tokens=768, pointers=(
        "/ticket_id", "/customer_id", "/order_id", "/subject", "/status", "/created_at", "/transitions/*/status",
        "/transitions/*/at",
    ), required=("/ticket_id", "/customer_id", "/status")),
    "list_payments": ReducerRule(JSON_PROJECTION, preview_tokens=768, pointers=(
        "/order_id", "/captured_total_inr", "/refunded_total_inr", "/payments/*/payment_id", "/payments/*/method",
        "/payments/*/status", "/payments/*/amount_inr", "/payments/*/created_at", "/refunds/*/refund_id",
        "/refunds/*/amount_inr", "/refunds/*/status", "/refunds/*/created_at", "/ledger_entries/*/entry_id",
        "/ledger_entries/*/amount_inr", "/ledger_entries/*/created_at",
    ), required=("/order_id", "/captured_total_inr", "/refunded_total_inr")),
    "search_orders": ReducerRule(JSON_PROJECTION, preview_tokens=768, pointers=(
        "/customer_id", "/status", "/orders/*/order_id", "/orders/*/status", "/orders/*/placed_at",
        "/orders/*/total_inr",
    ), required=("/customer_id", "/status")),
}
DEFAULT_RULES: dict[str, ReducerRule] = {
    # Steadyhand's sandboxed exec tools: gated stdout and stderr, each reduced separately.
    "run_shell": EXEC_RULE,
    "run_python": EXEC_RULE,
    # The observation corpus's tools (tidemark.eval.observations).
    "run_tests": EXEC_RULE,
    "build_project": EXEC_RULE,
    "run_command": EXEC_RULE,
    "read_log": LOG_RULE,
    "read_text": ReducerRule(TEXT_HEAD_TAIL, preview_tokens=1000, marks=False),
    "fetch_page": HTML_RULE,
    **STORE_PROJECTIONS,
    # Reading a blob back is never shaped again (its size is bounded by read_blob's byte ceiling).
    "read_blob": ReducerRule(NO_REDUCER, preview_tokens=1 << 20),
}
DEFAULT_POLICY = ObservationPolicy(DEFAULT_RULES)


def check_projections(policy: ObservationPolicy, result_schemas: Mapping[str, Mapping[str, Any] | None]) -> list[str]:
    """Every projection pointer must name a property in its tool's result schema (``*`` = array items)."""
    problems = []
    for tool, rule in policy.rules.items():
        if rule.reducer != JSON_PROJECTION:
            continue
        schema = result_schemas.get(tool)
        if schema is None:
            problems.append(f"{tool}: no result schema to review the projection against")
            continue
        for pointer in (*rule.pointers, *rule.required):
            if not _schema_has(schema, schema, pointer.lstrip("/").split("/")):
                problems.append(f"{tool}: {pointer} is not in the result schema")
    return problems


def _schema_has(root: Mapping[str, Any], schema: Mapping[str, Any], tokens: list[str]) -> bool:
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        schema = root.get("$defs", {}).get(name, {})
    if not tokens:
        return True
    for option in schema.get("anyOf", ()):
        if _schema_has(root, option, tokens):
            return True
    head, rest = tokens[0], tokens[1:]
    if head == "*":
        items = schema.get("items")
        return isinstance(items, Mapping) and _schema_has(root, items, rest)
    child = schema.get("properties", {}).get(head)
    return isinstance(child, Mapping) and _schema_has(root, child, rest)


@dataclass(frozen=True)
class ObservationConfig:
    """Everything that decides a shaped message: policy, namespace, catalog, formats (``ContextConfig.observations``)."""

    policy: ObservationPolicy
    namespace: str
    catalog_id: str
    max_read_bytes: int
    blob_format_version: int = BLOB_FORMAT_VERSION
    reducer_version: str = REDUCER_VERSION
    version: int = OBSERVATION_CONFIG_VERSION

    def __post_init__(self) -> None:
        check_namespace(self.namespace)

    @classmethod
    def for_store(cls, store: BlobStore, namespace: str, policy: ObservationPolicy = DEFAULT_POLICY) -> ObservationConfig:
        return cls(policy=policy, namespace=namespace, catalog_id=store.catalog_id, max_read_bytes=store.max_read_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "policy": self.policy.as_dict(), "namespace": self.namespace,
            "catalog_id": self.catalog_id, "max_read_bytes": self.max_read_bytes,
            "blob_format_version": self.blob_format_version, "reducer_version": self.reducer_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ObservationConfig:
        return cls(
            policy=ObservationPolicy.from_dict(value["policy"]), namespace=value["namespace"],
            catalog_id=value["catalog_id"], max_read_bytes=value["max_read_bytes"],
            blob_format_version=value["blob_format_version"], reducer_version=value["reducer_version"],
            version=value["version"],
        )


@dataclass(frozen=True)
class Reduced:
    """A finished reduction: the envelope content and the evidence. Cacheable by raw content and rule."""

    reducer: str
    content: str
    shaped_tokens: int
    preview_tokens: int
    parts: tuple[BlobPart, ...]
    omitted_ranges: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    @property
    def weight(self) -> int:
        return len(self.content) + sum(part.size for part in self.parts)


class ReducerCache:
    """A byte-bounded LRU of reductions keyed by raw content, rule, reducer version, and tokenizer identity."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self._items: OrderedDict[str, Reduced] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Reduced | None:
        with self._lock:
            found = self._items.get(key)
            if found is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return found

    def put(self, key: str, value: Reduced) -> None:
        with self._lock:
            if key in self._items or value.weight > self.max_bytes:
                return
            self._items[key] = value
            self._bytes += value.weight
            while self._bytes > self.max_bytes:
                _, evicted = self._items.popitem(last=False)
                self._bytes -= evicted.weight

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._items), "bytes": self._bytes, "hits": self.hits, "misses": self.misses}


class ShapingPool:
    """Bounded reducer workers shared by every episode of a run.

    ``run`` blocks its caller (the observation is on that episode's critical
    path) until a worker has finished. Admission is in arrival order and bounded
    by estimated in-flight input bytes; one input larger than the bound runs
    alone rather than never. Counters feed the resource sampler and the
    benchmark; they are volatile.
    """

    def __init__(self, workers: int = 1, *, max_inflight_bytes: int = 32 * 1024 * 1024,
                 executor: Executor | None = None, clock: Callable[[], float] = time.perf_counter) -> None:
        if workers < 1 or max_inflight_bytes < 1:
            raise ValueError("workers and max_inflight_bytes must be positive")
        self.workers = workers
        self.max_inflight_bytes = max_inflight_bytes
        self._executor = executor or ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tidemark-shape")
        self._owns = executor is None
        self._clock = clock
        self._cond = threading.Condition()
        self._queue: deque[int] = deque()
        self._tickets = 0
        self._inflight = 0
        self._inflight_bytes = 0
        self.stats = {"submitted": 0, "queue_wait_s": 0.0, "run_s": 0.0, "peak_inflight_bytes": 0, "peak_waiting": 0,
                      "oversized": 0}

    def run(self, fn: Callable[..., Any], *args: Any, weight: int) -> Any:
        return self.run_timed(fn, *args, weight=weight)[0]

    def run_timed(self, fn: Callable[..., Any], *args: Any, weight: int) -> tuple[Any, float]:
        """``fn(*args)`` on a worker, and how long this call waited for admission."""
        weight = max(1, int(weight))
        started = self._clock()
        with self._cond:
            ticket = self._tickets
            self._tickets += 1
            self._queue.append(ticket)
            self.stats["peak_waiting"] = max(self.stats["peak_waiting"], len(self._queue))
            while not (self._queue[0] == ticket and self._inflight < self.workers and (
                self._inflight_bytes + weight <= self.max_inflight_bytes or self._inflight == 0
            )):
                self._cond.wait()
            self._queue.popleft()
            self._inflight += 1
            self._inflight_bytes += weight
            self.stats["submitted"] += 1
            self.stats["oversized"] += int(weight > self.max_inflight_bytes)
            self.stats["peak_inflight_bytes"] = max(self.stats["peak_inflight_bytes"], self._inflight_bytes)
            waited = self._clock() - started
            self.stats["queue_wait_s"] += waited
            self._cond.notify_all()
        began = self._clock()
        try:
            return self._executor.submit(fn, *args).result(), waited
        finally:
            with self._cond:
                self._inflight -= 1
                self._inflight_bytes -= weight
                self.stats["run_s"] += self._clock() - began
                self._cond.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            return {"shaping_running": self._inflight, "shaping_waiting": len(self._queue),
                    "shaping_inflight_bytes": self._inflight_bytes, "shaping_capacity": self.workers}

    def close(self) -> None:
        if self._owns:
            self._executor.shutdown(wait=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def envelope(reducer: str, candidate: Candidate, parts: Sequence[BlobPart], preview_tokens: int) -> dict[str, Any]:
    """The shaped observation: the ordinary success contract plus a bounded ``context`` member."""
    context: dict[str, Any] = {"shape": reducer, "omitted": True, "preview_tokens": preview_tokens}
    if len(parts) == 1:
        context.update(blob=parts[0].handle, bytes=parts[0].size, media_type=parts[0].media_type)
    else:
        context["blobs"] = {part.part: {"blob": part.handle, "bytes": part.size, "media_type": part.media_type}
                            for part in parts}
    context["restore"] = RESTORE_HINT
    context.update(candidate.context)
    return {"ok": True, "value": candidate.value, "context": context}


class ObservationShaper:
    """Shapes one episode's (or one run's) observations against a blob store. Thread-safe."""

    def __init__(
        self,
        config: ObservationConfig,
        tokenizer: Tokenizer,
        store: BlobStore,
        *,
        cache: ReducerCache | None = None,
        pool: ShapingPool | None = None,
    ) -> None:
        if store.catalog_id != config.catalog_id:
            raise ValueError("the blob store's catalog is not the one this configuration names")
        if store.max_read_bytes != config.max_read_bytes:
            raise ValueError("the blob store's read ceiling differs from the configuration")
        self.config = config
        self.policy = config.policy
        self.tokenizer = tokenizer
        self.store = store
        self.cache = cache
        self.pool = pool
        self._tokenizer_key = canonical_hash(dict(tokenizer.identity()))
        self._lock = threading.Lock()
        self.metrics: dict[str, float] = {
            "shaped": 0, "untouched": 0, "typed_errors": 0, "reducer_s": 0.0, "io_s": 0.0, "queue_wait_s": 0.0,
            "raw_tokens": 0, "shaped_tokens": 0, "raw_bytes": 0, "shaped_bytes": 0, "blob_bytes": 0, "cache_hits": 0,
        }

    @property
    def namespace(self) -> str:
        return self.config.namespace

    def blobs(self) -> NamespacedBlobs:
        return self.store.namespace(self.config.namespace)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _add(self, **values: float) -> None:
        with self._lock:
            for name, value in values.items():
                self.metrics[name] += value

    def shape(self, source: ObservationSource) -> ShapedObservation:
        """The message for one raw result, and the ``observation_shaped`` fields when it was shaped."""
        message = {"role": "tool", "tool_call_id": source.tool_call_id, "content": source.content}
        rule = self.policy.rule_for(source.tool)
        if not source.ok:
            self._add(typed_errors=1)
            return self._untouched(source, message, "typed_error")
        if rule.reducer == NO_REDUCER:
            return self._untouched(source, message, "unshaped_tool")
        try:
            raw_tokens = self.count(source.content)
        except InvalidText:
            return self._untouched(source, message, "invalid_text")  # the upstream gate's bytes, left as they are
        if raw_tokens <= rule.preview_tokens:
            return self._untouched(source, message, "under_cap", raw_tokens)
        key = canonical_hash({"raw": _sha256(source.content), "rule": rule.sha256(), "reducer": REDUCER_VERSION,
                              "tokenizer": self._tokenizer_key})
        reduced = self.cache.get(key) if self.cache is not None else None
        if reduced is None:
            weight = len(source.content.encode("utf-8"))
            started = time.perf_counter()
            if self.pool is not None:
                reduced, waited = self.pool.run_timed(self._reduce, source.observation["value"], rule, weight=weight)
                self._add(queue_wait_s=waited)
            else:
                reduced = self._reduce(source.observation["value"], rule)
            self._add(reducer_s=time.perf_counter() - started)
            if self.cache is not None:
                self.cache.put(key, reduced)
        else:
            self._add(cache_hits=1)
        if reduced.shaped_tokens >= raw_tokens:  # a cap below the envelope's own size: the raw message is smaller
            return self._untouched(source, message, "no_gain", raw_tokens)
        revision = self._persist(source, reduced)
        shaped_bytes = len(reduced.content.encode("utf-8"))
        raw_bytes = len(source.content.encode("utf-8"))
        parts = [{"part": part.part, "handle": part.handle, "bytes": part.size} for part in reduced.parts]
        ranges = list(reduced.omitted_ranges)
        metadata = dict(reduced.metadata)
        if len(ranges) > MAX_OMITTED_RANGES:
            metadata["omitted_ranges_total"] = len(ranges)
            ranges = ranges[: MAX_OMITTED_RANGES - 1] + ranges[-1:]
        event = {
            "source_seq": source.source_seq, "tool_call_id": source.tool_call_id, "tool": source.tool,
            "reducer": reduced.reducer, "raw_content_sha256": _sha256(source.content),
            "shaped_content_sha256": _sha256(reduced.content), "raw_bytes": raw_bytes, "shaped_bytes": shaped_bytes,
            "raw_tokens": raw_tokens, "shaped_tokens": reduced.shaped_tokens, "omitted_ranges": ranges,
            "blob_handle": parts[0]["handle"] if parts else None, "blob_bytes": parts[0]["bytes"] if parts else None,
            "blob_parts": parts, "catalog_revision": revision if parts else None,
            "preview_metadata": _bounded_metadata(metadata),
        }
        self._add(shaped=1, raw_tokens=raw_tokens, shaped_tokens=reduced.shaped_tokens, raw_bytes=raw_bytes,
                  shaped_bytes=shaped_bytes, blob_bytes=sum(part.size for part in reduced.parts))
        return ShapedObservation({**message, "content": reduced.content}, event)

    def _untouched(self, source: ObservationSource, message: dict[str, Any], reason: str,
                   raw_tokens: int | None = None) -> ShapedObservation:
        self._add(untouched=1)
        if not self.policy.audit:
            return ShapedObservation(message)
        tokens = raw_tokens if raw_tokens is not None else self._safe_count(source.content)
        digest = _sha256(source.content)
        size = len(source.content.encode("utf-8", "surrogatepass"))
        return ShapedObservation(message, {
            "source_seq": source.source_seq, "tool_call_id": source.tool_call_id, "tool": source.tool,
            "reducer": NO_REDUCER, "raw_content_sha256": digest, "shaped_content_sha256": digest, "raw_bytes": size,
            "shaped_bytes": size, "raw_tokens": tokens, "shaped_tokens": tokens, "omitted_ranges": [],
            "blob_handle": None, "blob_bytes": None, "blob_parts": [], "catalog_revision": None,
            "preview_metadata": {"reason": reason},
        })

    def _safe_count(self, text: str) -> int:
        try:
            return self.count(text)
        except InvalidText:
            return 0

    def _reduce(self, value: Any, rule: ReducerRule) -> Reduced:
        """Plan, then choose the first candidate that fits the cap on the exact final content (pure CPU work)."""
        plan = plan_for(rule.reducer, value, rule)
        values: dict[int, int] = {}

        def measure(candidate: Candidate) -> int:
            # The preview's own tokens go into the context member; the cap is checked on the whole message content.
            preview = values[id(candidate)] = self.count(observation_content({"value": candidate.value}))
            return self.count(observation_content(envelope(plan.reducer, candidate, plan.parts, preview)))

        choice = choose(plan, measure, rule.preview_tokens)
        preview = values[id(choice.candidate)]
        content = observation_content(envelope(plan.reducer, choice.candidate, plan.parts, preview))
        metadata = {**choice.candidate.metadata, "level": choice.level, "levels": choice.levels,
                    "cap": rule.preview_tokens, "over_cap": choice.over_cap, "value_tokens": preview}
        return Reduced(plan.reducer, content, choice.tokens, preview, plan.parts, choice.candidate.omitted_ranges,
                       metadata)

    def _persist(self, source: ObservationSource, reduced: Reduced) -> int | None:
        """Write (or, while the journal verifies a recorded prefix, verify) every blob part; the newest revision."""
        started = time.perf_counter()
        revision = None
        try:
            for part in reduced.parts:
                if source.verifying or self.store.read_only:
                    record = self.store.verify_put(self.namespace, part.payload, media_type=part.media_type)
                else:
                    record = self.store.put(
                        self.namespace, part.payload, media_type=part.media_type, source_episode_id=source.episode_id,
                        source_event_seq=source.source_seq, source_tool=source.tool,
                        created_seq=source.source_seq + 1, reducer_version=REDUCER_VERSION,
                    ).record
                revision = record.catalog_revision if revision is None else max(revision, record.catalog_revision)
        except BlobError as exc:
            if source.verifying or self.store.read_only:
                raise ReplayMismatch(f"A recorded blob cannot be verified: {exc}", seq=source.source_seq + 1,
                                     field="blob_handle") from None
            raise
        finally:
            self._add(io_s=time.perf_counter() - started)
        return revision

    def verify_references(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Resume and replay check: every blob the journal references is cataloged here and intact."""
        try:
            return verify_references(self.store, self.namespace, records)
        except BlobError as exc:
            raise ReplayMismatch(f"A blob this journal references is missing or corrupt: {exc}",
                                 field="blob_handle") from None

    def execution_context(self, environment: CallEnvironment) -> ToolExecutionContext:
        return ToolExecutionContext.for_call(environment, namespace=self.namespace, blobs=self.blobs())

    def summary(self) -> dict[str, Any]:
        """Volatile counters for reports (never journaled)."""
        with self._lock:
            metrics = dict(self.metrics)
        metrics["cache"] = self.cache.stats() if self.cache is not None else None
        metrics["pool"] = dict(self.pool.stats) if self.pool is not None else None
        metrics["store"] = dict(self.store.stats)
        return metrics


def _bounded_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(dict(metadata), sort_keys=True, default=str))
    while len(json.dumps(value, sort_keys=True).encode("utf-8")) > MAX_PREVIEW_METADATA_BYTES and value:
        value.pop(sorted(value)[-1])
    return value


class ShapingSupport:
    """Mixed into a context session: part 3's hooks on top of its ``shape_observation``.

    ``observations`` is the session's shaper (None leaves every message as it
    arrived and writes no event). The session's configuration must carry the
    shaper's ``ObservationConfig`` as ``observations``.
    """

    observations: ObservationShaper | None = None
    _handles: list[str]

    def _init_shaping(self, observations: ObservationShaper | None, config_observations: Mapping[str, Any] | None) -> None:
        expected = observations.config.as_dict() if observations is not None else None
        if (dict(config_observations) if config_observations is not None else None) != expected:
            raise ValueError("ContextConfig.observations must equal the shaper's ObservationConfig.as_dict()")
        self.observations = observations
        self._handles = []

    def shape_result(self, source: ObservationSource) -> ShapedObservation:
        if self.observations is None:
            message = {"role": "tool", "tool_call_id": source.tool_call_id, "content": source.content}
            return ShapedObservation(dict(self.shape_observation(message, step=source.step)))  # type: ignore[attr-defined]
        shaped = self.observations.shape(source)
        if shaped.event is not None:
            for part in shaped.event["blob_parts"]:
                if part["handle"] in self._handles:
                    self._handles.remove(part["handle"])
                self._handles.append(part["handle"])
        return shaped

    @property
    def open_blob_handles(self) -> tuple[str, ...]:
        """Handles of this episode's shaped observations, oldest first (the recitation keeps the newest)."""
        return tuple(getattr(self, "_handles", ()))

    def execution_context(self, environment: CallEnvironment) -> ToolExecutionContext | None:
        if self.observations is None:
            return None
        return self.observations.execution_context(environment)

    def check_journal(self, records: Sequence[Mapping[str, Any]]) -> None:
        if self.observations is not None:
            self.observations.verify_references(records)
        elif any(record.get("event") == "observation_shaped" for record in records):
            raise ReplayMismatch("This journal has shaped observations; the session has no shaper", field="event")

    @property
    def blob_reader(self) -> NamespacedBlobs | None:
        return self.observations.blobs() if self.observations is not None else None


__all__ = [
    "DEFAULT_POLICY", "DEFAULT_RULES", "EXEC_RULE", "HTML_RULE", "LOG_RULE", "ObservationConfig", "ObservationPolicy",
    "ObservationShaper", "Reduced", "ReducerCache", "ReducerRule", "STORE_PROJECTIONS", "ShapingPool",
    "ShapingSupport", "check_projections", "envelope",
]

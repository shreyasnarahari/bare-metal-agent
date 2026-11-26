"""Prefix-stable context assembly: an immutable state, a pure assembler, and the session that drives them.

``ContextState`` holds the five segments' inputs. Its transitions are explicit
and return new states: ``append`` adds complete history messages to the recent
segment, ``with_recitation`` replaces the tail, and ``advance_epoch`` (used by
part 4) replaces the summary and resets the recent segment. An old recent
message is never rewritten.

``ContextAssembler.render(state)`` is pure: no model call, file, clock,
random id, database, or environment read. The same state and span renderer
always give byte-identical prompt text, token ids, spans, and messages. It
maps the segments to legal OpenAI-shaped messages (see ``tidemark.segments``)
and renders them through ``SpanRenderer``, which uses Toolsmith's own template
path (``render_with_token_spans``). The prompt the linter measures is the
prompt the shim sends.

``PrefixStableSession`` is the ``steadyhand.context.ContextSession`` for the
``prefix_stable`` strategy. On each request it appends the history messages
that arrived since the last render, rebuilds and bounds the tail, renders,
applies the budget rules, lints against the previous render, and reports the
transitions. Over the hard limit it stops with ``context_budget_exhausted``:
part 2 has no compactor and never falls back to a different algorithm.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from steadyhand.context import CONTEXT_BUDGET_EXHAUSTED, RenderedContext
from steadyhand.types import ModelTurn
from toolsmith.evaluation.models import canonical_hash
from toolsmith.prompting import SPANS_OK, TokenSpans, normalize_history, render_with_token_spans, request_tools

from . import __version__
from .budgets import EXHAUSTED, COMPACTION_PENDING, BudgetConfigError, check_static, evaluate
from .cache import ContentCache
from .prefix_lint import Fingerprint, FingerprintStore, lint
from .recitation import PlanItem, RecitationState, bound
from .segments import (
    FROZEN,
    PINNED,
    RECENT,
    SEGMENTS,
    SUMMARY,
    TAIL,
    SegmentError,
    canonical,
    exact_sha256,
    freeze_tools,
    message_sha256,
    pinned_message,
    summary_message,
    system_message,
    tool_names,
    tools_sha256,
)
from .telemetry import backend_timing_fields
from .tokenizer import Tokenizer
from .types import ContextConfig


@dataclass(frozen=True)
class ContextState:
    """Everything the assembler needs for one render. Immutable; transitions return new states."""

    frozen_system: str
    frozen_tools: tuple[dict[str, Any], ...]
    task_charter: str
    pinned_memory: tuple[str, ...] = ()
    epoch: int = 0
    summary: Mapping[str, Any] | None = None
    recent: tuple[dict[str, Any], ...] = ()
    recent_seqs: tuple[int | None, ...] = ()
    recitation: RecitationState = field(default_factory=RecitationState)
    previous: Fingerprint | None = None

    def append(self, messages: Sequence[Mapping[str, Any]], seqs: Sequence[int | None] | None = None) -> ContextState:
        """Add complete history messages to the recent segment (canonical, so never re-serialized later)."""
        added = tuple(canonical(dict(message)) for message in messages)
        added_seqs = tuple(seqs) if seqs is not None else (None,) * len(added)
        if len(added_seqs) != len(added):
            raise ValueError("seqs must align with the appended messages")
        return replace(self, recent=self.recent + added, recent_seqs=self.recent_seqs + added_seqs)

    def with_recitation(self, recitation: RecitationState) -> ContextState:
        return replace(self, recitation=recitation)

    def with_previous(self, fingerprint: Fingerprint | None) -> ContextState:
        return replace(self, previous=fingerprint)

    def advance_epoch(self, summary: Mapping[str, Any], keep_from: int) -> ContextState:
        """Replace the summary, advance the epoch, and keep only recent messages from ``keep_from`` on."""
        return replace(self, epoch=self.epoch + 1, summary=canonical(dict(summary)),
                       recent=self.recent[keep_from:], recent_seqs=self.recent_seqs[keep_from:])

    def segment_hashes(self) -> dict[str, str | None]:
        return {
            FROZEN: exact_sha256({"system": self.frozen_system, "tools": list(self.frozen_tools)}),
            PINNED: exact_sha256({"charter": self.task_charter, "memory": list(self.pinned_memory)}),
            SUMMARY: exact_sha256({"epoch": self.epoch, "summary": self.summary}) if self.summary is not None else None,
            RECENT: exact_sha256(list(self.recent)),
            TAIL: self.recitation.sha256(),
        }

    def sha256(self) -> str:
        """The logical state (not the previous fingerprint's ids), for ``context_transition.state_sha256``."""
        return canonical_hash({"epoch": self.epoch, "segments": self.segment_hashes(), "recent_seqs": list(self.recent_seqs)})

    def layout(self) -> tuple[list[dict[str, Any]], tuple[tuple[str, int], ...]]:
        """The request messages in segment order and each segment's end index."""
        messages: list[dict[str, Any]] = [system_message(self.frozen_system)]
        boundaries = [(FROZEN, 1)]
        messages.append(pinned_message(self.task_charter, self.pinned_memory))
        boundaries.append((PINNED, len(messages)))
        summary = summary_message(self.epoch, self.summary)
        if summary is not None:
            messages.append(summary)
        boundaries.append((SUMMARY, len(messages)))
        messages.extend(dict(message) for message in self.recent)
        boundaries.append((RECENT, len(messages)))
        messages.append(self.recitation.message())
        boundaries.append((TAIL, len(messages)))
        return messages, tuple(boundaries)


class PromptTemplate(Protocol):
    """The ``ChatPromptRenderer`` interface: the model's own chat template."""

    def render(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] = (), *,
               enable_thinking: bool | None = None, add_generation_prompt: bool = True) -> str: ...


class SpanRenderer:
    """Renders and tokenizes a request exactly as the shim does, with exact segment spans.

    Intermediate renders of a message prefix are cached by content (the
    prefix's canonical hash, tools, protocol, thinking flag, and template
    revision), so the unchanged segments are rendered and tokenized once.
    """

    def __init__(
        self,
        template: PromptTemplate,
        tokenizer: Tokenizer,
        *,
        protocol: Any,
        enable_thinking: bool | None = False,
        template_sha256: str | None = None,
        cache: ContentCache | None = None,
    ) -> None:
        self.template = template
        self.tokenizer = tokenizer
        self.protocol = protocol
        self.enable_thinking = enable_thinking
        self.template_sha256 = template_sha256
        self.cache = cache if cache is not None else ContentCache("rendered_prefixes", max_entries=512)
        self.render_ns = 0
        self.tokenize_ns = 0
        self.renders = 0

    def _encode(self, text: str) -> tuple[int, ...]:
        started = time.perf_counter_ns()
        try:
            return tuple(self.tokenizer.encode(text))
        finally:
            self.tokenize_ns += time.perf_counter_ns() - started

    def render(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]],
               boundaries: Sequence[tuple[str, int]]) -> TokenSpans:
        normalized = normalize_history(messages, self.protocol)
        prepared = request_tools(tools)
        # Exact, order-sensitive keys: a reordered schema is a different prompt and must miss the cache.
        tools_key = exact_sha256(prepared)

        def prefix(end: int) -> str:
            if end == 0:
                return ""
            key = exact_sha256({
                "messages": [dict(m) for m in messages[:end]], "tools": tools_key, "protocol": str(self.protocol),
                "thinking": self.enable_thinking, "template": self.template_sha256,
            })
            cached = self.cache.get(key)
            if cached is None:
                started = time.perf_counter_ns()
                cached = self.template.render(normalized[:end], prepared, enable_thinking=self.enable_thinking,
                                              add_generation_prompt=False)
                self.render_ns += time.perf_counter_ns() - started
                self.cache.put(key, cached)
            return cached

        started = time.perf_counter_ns()
        self.renders += 1
        spans = render_with_token_spans(
            self.template, messages, tools, boundaries=boundaries, encode=self._encode, protocol=self.protocol,
            enable_thinking=self.enable_thinking, render_prefix=prefix,
        )
        self.render_ns += time.perf_counter_ns() - started
        return spans

    def performance(self) -> dict[str, Any]:
        return {"renders": self.renders, "render_us": self.render_ns // 1000, "tokenize_us": self.tokenize_ns // 1000,
                "segment_cache": self.cache.stats()}


@dataclass(frozen=True)
class AssembledPrompt:
    """One render: the exact request messages, tools, token spans, and the fingerprint it leaves."""

    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    spans: TokenSpans
    segment_tokens: dict[str, int] | None
    fingerprint: Fingerprint

    @property
    def total(self) -> int:
        return len(self.spans.token_ids)


class ContextAssembler:
    """``render(state)`` is a pure function of the state and the span renderer."""

    def __init__(self, renderer: SpanRenderer) -> None:
        self.renderer = renderer

    def render(self, state: ContextState) -> AssembledPrompt:
        messages, boundaries = state.layout()
        spans = self.renderer.render(messages, state.frozen_tools, boundaries)
        segment_tokens = None
        if spans.status == SPANS_OK:
            segment_tokens = {name: end - start for name, start, end in spans.spans if name in SEGMENTS}
            segment_tokens["generation_prompt"] = next(end - start for name, start, end in spans.spans
                                                       if name == "generation_prompt")
        tail = spans.span(TAIL)
        pinned = spans.span(PINNED)
        fingerprint = Fingerprint(
            token_ids=spans.token_ids,
            prompt_sha256=spans.text_sha256,
            stable_prefix_tokens=tail[0] if tail is not None else None,
            pinned_end_tokens=pinned[1] if pinned is not None else None,
            segment_hashes=state.segment_hashes(),
            recent_hashes=tuple(message_sha256(m) for m in state.recent),
            epoch=state.epoch,
            spans=spans.spans,
        )
        return AssembledPrompt(tuple(messages), tuple(state.frozen_tools), spans, segment_tokens, fingerprint)


def initial_state(
    history: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    initial_count: int,
    pinned_memory: Sequence[str] = (),
    seqs: Sequence[int | None] | None = None,
) -> tuple[ContextState, int]:
    """The state for an episode's opening messages; returns it and how many history messages it consumed.

    The first message must be the system prompt; the task charter is the last
    user message of the opening input. Opening messages between them (an
    earlier exchange) start the recent segment, in order.
    """
    if not history or history[0].get("role") != "system":
        raise SegmentError("A prefix-stable session needs the system prompt as the first message")
    request = max((i for i in range(1, min(initial_count, len(history))) if history[i].get("role") == "user"),
                  default=None)
    if request is None:
        raise SegmentError("The opening messages have no user request to use as the task charter")
    earlier = [dict(m) for m in history[1:request]]
    earlier_seqs = list(seqs[1:request]) if seqs is not None else None
    state = ContextState(
        frozen_system=str(history[0].get("content") or ""),
        frozen_tools=freeze_tools(tools),
        task_charter=str(history[request].get("content") or ""),
        pinned_memory=tuple(pinned_memory),
    ).append(earlier, earlier_seqs)
    rest = history[request + 1: initial_count]
    if rest:
        state = state.append(rest, list(seqs[request + 1: initial_count]) if seqs is not None else None)
    return state, initial_count


class PrefixStableSession:
    """The ``prefix_stable`` context strategy (see the module docstring). One instance per episode."""

    strategy = "prefix_stable"
    version = __version__

    def __init__(
        self,
        config: ContextConfig,
        renderer: SpanRenderer,
        *,
        plan: Sequence[PlanItem] = (),
        fingerprint_store: FingerprintStore | None = None,
        checkpoint_key: str | None = None,
    ) -> None:
        if config.strategy != self.strategy:
            raise ValueError(f"PrefixStableSession needs strategy {self.strategy!r}, not {config.strategy!r}")
        self.config = config
        self.assembler = ContextAssembler(renderer)
        self.renderer = renderer
        self.plan = tuple(plan)
        self.store = fingerprint_store
        self.checkpoint_key = checkpoint_key
        self.state: ContextState | None = None
        self._consumed = 0
        self._tools_sha256: str | None = None
        self.last_table: dict[str, Any] | None = None

    def identity(self) -> dict[str, Any]:
        return self.config.as_dict()

    def shape_observation(self, message: Mapping[str, Any], *, step: int) -> dict[str, Any]:
        """Part 2 keeps observations as they arrived (part 3 shapes them once, at arrival)."""
        return dict(message)

    def backend_timings(self, turn: ModelTurn, rendered: RenderedContext) -> dict[str, Any]:
        return backend_timing_fields(turn.metrics, rendered.rendered_prompt_tokens, self.config.cache_semantics)

    def validate(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Check a configuration before an episode starts; raises ``BudgetConfigError`` or ``SegmentError``."""
        state, _ = initial_state(messages, tools, initial_count=len(messages), pinned_memory=self.config.pinned_memory)
        self._check_allowed(state.frozen_tools)
        assembled = self.assembler.render(state)
        tokens = assembled.segment_tokens or {}
        check_static(self.config, tokens.get(FROZEN), tokens.get(PINNED))
        return {"frozen_tokens": tokens.get(FROZEN), "pinned_tokens": tokens.get(PINNED), "total": assembled.total}

    def _check_allowed(self, tools: Sequence[Mapping[str, Any]]) -> None:
        allowed = self.config.allowed_tool_names
        if allowed is not None and set(allowed) - set(tool_names(tools)):
            raise SegmentError(f"allowed_tool_names names unknown tools: {sorted(set(allowed) - set(tool_names(tools)))}")

    def _recitation(self, step: int, budget_state: Mapping[str, Any] | None) -> RecitationState:
        previous = self.state.previous if self.state is not None else None
        remaining = None
        if budget_state is not None and budget_state.get("max_steps") is not None:
            remaining = max(0, int(budget_state["max_steps"]) - int(budget_state.get("steps", step)))
        return RecitationState(
            current_plan_items=self.plan,
            completed_step_count=int(budget_state.get("steps", step)) if budget_state else step,
            prompt_tokens_used=len(previous.token_ids) if previous is not None else None,
            remaining_step_budget=remaining,
        )

    def _exhausted(self, detail: str, transitions: tuple[dict[str, Any], ...] = ()) -> RenderedContext:
        return RenderedContext(messages=(), tools=(), strategy=self.strategy, status=CONTEXT_BUDGET_EXHAUSTED,
                               detail=detail, max_tokens=self.config.max_tokens, transitions=transitions)

    def render(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        step: int,
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        initial_count: int = 0,
        seqs: Sequence[int | None] | None = None,
        budget_state: Mapping[str, Any] | None = None,
    ) -> RenderedContext:
        transitions: list[dict[str, Any]] = []
        epoch_before = self.state.epoch if self.state is not None else 0
        try:
            if self.state is None:
                self.state, self._consumed = initial_state(
                    history, tools, initial_count=initial_count, pinned_memory=self.config.pinned_memory, seqs=seqs,
                )
                self._tools_sha256 = tools_sha256(freeze_tools(tools))
                self._check_allowed(self.state.frozen_tools)
                transitions.append(self._transition("start", epoch_before, list(seqs or ()), "opening messages"))
            else:
                if tools_sha256(freeze_tools(tools)) != self._tools_sha256:
                    return self._exhausted("the tool list changed mid-session; the frozen list cannot change")
                new = history[self._consumed:]
                if new:
                    new_seqs = list(seqs[self._consumed:]) if seqs is not None else None
                    self.state = self.state.append(new, new_seqs)
                    self._consumed = len(history)
                    transitions.append(self._transition("append", epoch_before, new_seqs or [],
                                                        f"{len(new)} history message(s) appended"))
        except SegmentError as exc:
            return self._exhausted(f"configuration: {exc}")
        # Rebuild the tail, then bound it to its cap by exact token span.
        recitation = self._recitation(step, budget_state)
        cap = self.config.cap(TAIL)
        cache: dict[str, Any] = {}

        def fits(candidate: RecitationState) -> bool:
            assembled = self.assembler.render(self.state.with_recitation(candidate))
            cache["last"] = assembled
            tail = (assembled.segment_tokens or {}).get(TAIL)
            return cap is None or tail is None or tail <= cap

        recitation, _evicted = bound(
            recitation, max_items=self.config.recitation_setting("max_items"),
            max_item_chars=self.config.recitation_setting("max_item_chars"),
            max_handles=self.config.recitation_setting("max_handles"), fits=fits,
        )
        state = self.state.with_recitation(recitation)
        assembled = cache.get("last")
        if assembled is None or assembled.fingerprint.segment_hashes != state.segment_hashes():
            assembled = self.assembler.render(state)
        table = evaluate(self.config, assembled.total, assembled.segment_tokens,
                         (assembled.segment_tokens or {}).get("generation_prompt"))
        if self.config.cap(FROZEN) is not None and table.segment(FROZEN).used is not None:
            try:
                check_static(self.config, table.segment(FROZEN).used)
            except BudgetConfigError as exc:
                return self._exhausted(f"configuration: {exc}", tuple(transitions))
        result = lint(state.previous, assembled.fingerprint)
        self.state = state.with_previous(assembled.fingerprint)
        self.last_table = table.as_dict()
        if self.store is not None and self.checkpoint_key is not None:
            self.store.save(self.checkpoint_key, assembled.fingerprint)
        rendered = RenderedContext(
            messages=assembled.messages,
            tools=assembled.tools,
            strategy=self.strategy,
            status=CONTEXT_BUDGET_EXHAUSTED if table.decision == EXHAUSTED else "ok",
            detail="; ".join(table.reasons) if table.decision == EXHAUSTED else None,
            epoch=state.epoch,
            prompt_sha256=assembled.spans.text_sha256,
            prompt_token_ids_sha256=_ids_sha256(assembled.spans.token_ids),
            rendered_prompt_tokens=assembled.total,
            segment_budgets=table.budgets(self.config),
            segment_tokens=table.tokens(),
            max_tokens=self.config.max_tokens,
            slot_headroom=table.hard_headroom,
            recitation_sha256=recitation.sha256(),
            compaction_pending=table.decision == COMPACTION_PENDING,
            prefix=result.fields(),
            transitions=tuple(transitions),
            allowed_tool_names=self.config.allowed_tool_names,
            lint={"spans": assembled.spans.status} if assembled.spans.status != SPANS_OK else {},
        )
        return rendered

    def _transition(self, kind: str, epoch_before: int, seqs: Sequence[int | None], reason: str) -> dict[str, Any]:
        assert self.state is not None
        known = [seq for seq in self.state.recent_seqs if seq is not None]
        appended = sorted({seq for seq in seqs if seq is not None})
        return {
            "transition": kind, "epoch_before": epoch_before, "epoch_after": self.state.epoch,
            "history_start_seq": min(known) if known else (min(appended) if appended else None),
            "history_end_seq": max(known) if known else (max(appended) if appended else None),
            "appended_seqs": appended, "dropped_seqs": [], "reason": reason, "state_sha256": self.state.sha256(),
        }

    def performance(self) -> dict[str, Any]:
        """Volatile overhead counters (never journaled, never KV-cache hits)."""
        return self.renderer.performance()


def _ids_sha256(ids: Sequence[int]) -> str:
    return canonical_hash(list(ids))


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "AssembledPrompt", "ContextAssembler", "ContextState", "PrefixStableSession", "PromptTemplate", "SpanRenderer",
    "initial_state", "prompt_sha256",
]

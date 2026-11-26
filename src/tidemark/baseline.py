"""The naive baseline: append every message, drop the oldest complete turns only when the request cannot fit.

``NaiveAppendTruncate`` is a ``steadyhand.context.ContextSession``. It is
deliberately lossy and is not the prefix-stable design (part 2). For each
model request it:

1. renders the full logical history with the model's own chat template
   (through Toolsmith, so the text is exactly what ``chat()`` sends) and counts
   its tokens with the real tokenizer;
2. sends it unchanged if ``tokens + max_tokens + safety_margin <= slot_context``;
3. otherwise drops the oldest *complete turns* until it fits. A turn is atomic:
   an assistant message together with every tool result answering its calls, a
   standalone user message, or a standalone assistant text. The system message,
   the task's current request (the last user message of the opening input), and
   the newest turn (the current tool-call/result group) are never dropped, so no
   tool result is orphaned and no call is left unanswered;
4. returns ``status="context_budget_exhausted"`` when even the pinned messages
   cannot fit. It never deletes the current request or lowers ``max_tokens``.

Every dropped ``[start, end)`` range of the logical history and the resulting
token count go into ``context_render``. The render is a pure function of the
history, so replay re-derives it exactly. Messages are sent with their keys in
sorted order: a journaled message comes back with sorted keys, so a live run
and its replay render the same structure even if a template were sensitive to
key order. Templates read message fields by name, so the model's prompt is
unchanged; tool schemas keep their registry order, which the model does see.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from steadyhand.context import CONTEXT_BUDGET_EXHAUSTED, RenderedContext
from steadyhand.types import ModelTurn

from . import __version__
from .telemetry import backend_timing_fields
from .tokenizer import Tokenizer, token_ids_sha256
from .types import ContextConfig

# (messages, tools, tool_choice) -> the exact prompt text. Live: SteadyhandClient.render_prompt.
PromptRender = Callable[[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], Any], str]


@dataclass(frozen=True)
class TurnGroup:
    """``history[start:end]``, kept or dropped as one unit."""

    start: int
    end: int
    pinned: bool


def complete_turn_groups(history: Sequence[Mapping[str, Any]], initial_count: int) -> list[TurnGroup]:
    """Split the history into atomic turns and mark the ones that must never be dropped."""
    groups: list[tuple[int, int]] = []
    index = 0
    while index < len(history):
        message = history[index]
        end = index + 1
        if message.get("role") == "assistant" and message.get("tool_calls"):
            ids = {call.get("id") for call in message["tool_calls"]}
            while end < len(history) and history[end].get("role") == "tool" and history[end].get("tool_call_id") in ids:
                end += 1
        groups.append((index, end))
        index = end
    request = max(
        (i for i in range(min(initial_count, len(history))) if history[i].get("role") == "user"),
        default=None,
    )
    marked = []
    for position, (start, end) in enumerate(groups):
        message = history[start]
        pinned = (
            (start == 0 and message.get("role") == "system")
            or (request is not None and start <= request < end)
            or position == len(groups) - 1
            # A tool message with no assistant call before it cannot be fixed by dropping; keep it visible.
            or message.get("role") == "tool"
        )
        marked.append(TurnGroup(start, end, pinned))
    return marked


def canonical_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """The message with every mapping's keys in sorted order (how the journal stores it)."""
    return json.loads(json.dumps(dict(message), ensure_ascii=False, sort_keys=True, allow_nan=False))


def _ranges(indexes: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Contiguous ``[start, end)`` ranges covering sorted ``indexes``."""
    ranges: list[list[int]] = []
    for index in indexes:
        if ranges and ranges[-1][1] == index:
            ranges[-1][1] = index + 1
        else:
            ranges.append([index, index + 1])
    return tuple((start, end) for start, end in ranges)


class NaiveAppendTruncate:
    """Part 1's measurement baseline (see the module docstring). One instance per episode."""

    strategy = "naive_append_truncate"
    version = __version__

    def __init__(self, config: ContextConfig, *, render: PromptRender, tokenizer: Tokenizer) -> None:
        if config.strategy != self.strategy:
            raise ValueError(f"NaiveAppendTruncate needs strategy {self.strategy!r}, not {config.strategy!r}")
        self.config = config
        self._render = render
        self._tokenizer = tokenizer
        self.renders = 0  # prompt renders, for the cost of the drop search

    def identity(self) -> dict[str, Any]:
        return self.config.as_dict()

    def shape_observation(self, message: Mapping[str, Any], *, step: int) -> dict[str, Any]:
        """The baseline keeps every observation exactly as it arrived."""
        return dict(message)

    def backend_timings(self, turn: ModelTurn, rendered: RenderedContext) -> dict[str, Any]:
        return backend_timing_fields(turn.metrics, rendered.rendered_prompt_tokens, self.config.cache_semantics)

    def _measure(self, messages: list[dict[str, Any]], tools: Sequence[Mapping[str, Any]], tool_choice: Any) -> tuple[str, tuple[int, ...]]:
        self.renders += 1
        prompt = self._render(messages, tools, tool_choice)
        return prompt, self._tokenizer.encode(prompt)

    def render(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        step: int,
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        initial_count: int = 0,
    ) -> RenderedContext:
        config = self.config
        groups = complete_turn_groups(history, initial_count)
        droppable = [group for group in groups if not group.pinned]

        def without(count: int) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
            dropped = {i for group in droppable[:count] for i in range(group.start, group.end)}
            kept = [canonical_message(message) for i, message in enumerate(history) if i not in dropped]
            return kept, tuple(sorted(dropped))

        def fits(tokens: int) -> bool:
            return tokens <= config.prompt_limit

        # Smallest number of oldest droppable turns whose removal makes the prompt fit. Dropping more
        # never adds tokens, so a binary search needs only O(log n) renders; the answer is re-measured.
        best: tuple[int, list[dict[str, Any]], tuple[int, ...], str, tuple[int, ...]] | None = None
        messages, dropped = without(0)
        prompt, ids = self._measure(messages, tools, tool_choice)
        if fits(len(ids)):
            best = (0, messages, dropped, prompt, ids)
        elif droppable:
            low, high = 1, len(droppable)
            while low <= high:
                middle = (low + high) // 2
                candidate, candidate_dropped = without(middle)
                candidate_prompt, candidate_ids = self._measure(candidate, tools, tool_choice)
                if fits(len(candidate_ids)):
                    best = (middle, candidate, candidate_dropped, candidate_prompt, candidate_ids)
                    high = middle - 1
                else:
                    low = middle + 1
                    messages, dropped, prompt, ids = candidate, candidate_dropped, candidate_prompt, candidate_ids
        budgets = {
            "slot_context": config.slot_context,
            "completion_reserve": config.max_tokens,
            "safety_margin": config.safety_margin,
            "prompt_limit": config.prompt_limit,
        }
        if best is None:
            # Even with every droppable turn gone, the pinned request does not fit (messages/ids hold
            # the smallest prompt measured). Nothing is sent.
            return RenderedContext(
                messages=tuple(messages),
                tools=tuple(dict(tool) for tool in tools),
                strategy=self.strategy,
                status=CONTEXT_BUDGET_EXHAUSTED,
                detail=(
                    f"the required current turn needs {len(ids)} prompt tokens; the slot allows "
                    f"{config.prompt_limit} ({config.slot_context} minus max_tokens {config.max_tokens} "
                    f"and margin {config.safety_margin}) after dropping {len(droppable)} older turns"
                ),
                prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                prompt_token_ids_sha256=token_ids_sha256(ids),
                rendered_prompt_tokens=len(ids),
                segment_budgets=budgets,
                segment_tokens={"prompt": len(ids)},
                max_tokens=config.max_tokens,
                slot_headroom=config.prompt_limit - len(ids),
                dropped_ranges=_ranges(dropped),
            )
        _, messages, dropped, prompt, ids = best
        return RenderedContext(
            messages=tuple(messages),
            tools=tuple(dict(tool) for tool in tools),
            strategy=self.strategy,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            prompt_token_ids_sha256=token_ids_sha256(ids),
            rendered_prompt_tokens=len(ids),
            segment_budgets=budgets,
            segment_tokens={"prompt": len(ids)},
            max_tokens=config.max_tokens,
            slot_headroom=config.prompt_limit - len(ids),
            dropped_ranges=_ranges(dropped),
        )


__all__ = ["NaiveAppendTruncate", "PromptRender", "TurnGroup", "canonical_message", "complete_turn_groups"]

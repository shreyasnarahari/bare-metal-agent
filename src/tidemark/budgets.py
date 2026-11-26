"""Exact prompt budgets: the hard and soft limits, and per-segment use from exact token spans.

    hard_limit = slot_context - max_tokens - safety_margin
    soft_limit = 70% of hard_limit unless configured

The exact final render decides admission: a prompt over the hard limit never
reaches llama-server. Per-segment counts come from exact spans only. When the
template makes spans ambiguous, segments are ``None``, ``spans_available`` is
false, and no per-segment compliance can be claimed. Decisions, in order:
``exhausted`` (over the hard limit, or a non-tail segment over its cap) beats
``compaction_pending`` (over the soft limit, or recent history over its cap)
beats ``ok``. The frozen segment's cap is a configuration rule
(``check_static``): a tool catalog too large for the slot fails before any
episode starts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .segments import FROZEN, PINNED, RECENT, SEGMENTS, TAIL
from .types import ContextConfig

OK, COMPACTION_PENDING, EXHAUSTED = "ok", "compaction_pending", "exhausted"


class BudgetConfigError(ValueError):
    """The configuration cannot work for this session (for example, the frozen segment alone is too large)."""


@dataclass(frozen=True)
class SegmentUsage:
    name: str
    used: int | None
    cap: int | None

    @property
    def remaining(self) -> int | None:
        return None if self.used is None or self.cap is None else self.cap - self.used


@dataclass(frozen=True)
class BudgetTable:
    segments: tuple[SegmentUsage, ...]
    total: int
    generation_prompt: int | None
    generation_reserve: int
    hard_limit: int
    soft_limit: int
    spans_available: bool
    decision: str
    reasons: tuple[str, ...] = ()

    @property
    def hard_headroom(self) -> int:
        return self.hard_limit - self.total

    def segment(self, name: str) -> SegmentUsage:
        return next(item for item in self.segments if item.name == name)

    def budgets(self, config: ContextConfig) -> dict[str, int | None]:
        """Caps and limits for ``context_render.segment_budgets``."""
        return {
            **{item.name: item.cap for item in self.segments},
            "slot_context": config.slot_context, "completion_reserve": self.generation_reserve,
            "safety_margin": config.safety_margin, "prompt_limit": self.hard_limit, "soft_limit": self.soft_limit,
        }

    def tokens(self) -> dict[str, int | None]:
        """Used tokens per segment, the generation marker, and the exact total for ``segment_tokens``."""
        return {**{item.name: item.used for item in self.segments}, "generation_prompt": self.generation_prompt,
                "prompt": self.total}

    def as_dict(self) -> dict[str, Any]:
        return {
            "segments": [{"name": s.name, "used": s.used, "cap": s.cap, "remaining": s.remaining} for s in self.segments],
            "total": self.total, "generation_reserve": self.generation_reserve, "hard_limit": self.hard_limit,
            "soft_limit": self.soft_limit, "hard_headroom": self.hard_headroom, "spans_available": self.spans_available,
            "decision": self.decision, "reasons": list(self.reasons),
        }


def evaluate(config: ContextConfig, total: int, segment_tokens: Mapping[str, int] | None,
             generation_prompt: int | None = None) -> BudgetTable:
    """The budget table and decision for one exact render (``segment_tokens`` None: spans unavailable)."""
    hard, soft = config.prompt_limit, config.effective_soft_limit
    usage = tuple(
        SegmentUsage(name, segment_tokens.get(name) if segment_tokens is not None else None, config.cap(name))
        for name in SEGMENTS
    )
    exhausted, pending = [], []
    if total > hard:
        exhausted.append(f"the prompt needs {total} tokens; the hard limit is {hard}")
    for item in usage:
        if item.used is None or item.cap is None or item.used <= item.cap:
            continue
        message = f"{item.name} uses {item.used} tokens over its cap of {item.cap}"
        (pending if item.name == RECENT else exhausted).append(message)
    if total > soft:
        pending.append(f"the prompt passed the soft limit of {soft} tokens")
    decision = EXHAUSTED if exhausted else COMPACTION_PENDING if pending else OK
    return BudgetTable(usage, total, generation_prompt, config.max_tokens, hard, soft, segment_tokens is not None,
                       decision, tuple(exhausted + pending))


def check_static(config: ContextConfig, frozen_tokens: int | None, pinned_tokens: int | None = None) -> None:
    """Refuse a session whose frozen (and pinned) segments cannot fit with room for history."""
    frozen_cap = config.cap(FROZEN)
    if frozen_tokens is not None and frozen_cap is not None and frozen_tokens > frozen_cap:
        raise BudgetConfigError(f"the frozen system and tools need {frozen_tokens} tokens; the cap is {frozen_cap}")
    pinned_cap = config.cap(PINNED)
    if pinned_tokens is not None and pinned_cap is not None and pinned_tokens > pinned_cap:
        raise BudgetConfigError(f"the pinned block needs {pinned_tokens} tokens; the cap is {pinned_cap}")
    fixed = (frozen_tokens or 0) + (pinned_tokens or 0) + (config.cap(TAIL) or 0)
    if fixed >= config.prompt_limit:
        raise BudgetConfigError(
            f"the frozen and pinned segments plus the tail cap need {fixed} tokens, leaving no history room "
            f"under the {config.prompt_limit}-token prompt limit"
        )


__all__ = [
    "BudgetConfigError", "BudgetTable", "COMPACTION_PENDING", "EXHAUSTED", "OK", "SegmentUsage", "check_static",
    "evaluate",
]

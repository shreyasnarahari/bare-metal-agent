"""Tidemark value types: the context configuration, segment budgets, and turn metrics.

``TurnMetrics`` and ``AttemptMetrics`` live in ``steadyhand.types`` because
``ModelTurn`` carries them, and Steadyhand never imports Tidemark. They are
re-exported here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from steadyhand.context import context_config_sha256
from steadyhand.types import AttemptMetrics, TurnMetrics

CONFIG_VERSION = 1
STRATEGIES = ("naive_append_truncate",)
GENERATION_PATHS = ("non_streaming", "buffered_stream")
# Cache metrics need the meaning of llama.cpp's counters confirmed on the pinned build first
# (tidemark.telemetry.probe_cache_semantics). Until then they stay null.
UNCONFIRMED_CACHE = {"confirmed": False, "new_tokens_field": None, "total_tokens_field": None}


@dataclass(frozen=True)
class SegmentBudget:
    """A named token allowance and what one render used of it (``None`` when not measured)."""

    name: str
    cap: int | None
    used: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "cap": self.cap, "used": self.used}


@dataclass(frozen=True)
class ContextConfig:
    """Everything that decides what a context session sends. Its canonical form is a manifest identity field.

    ``slot_context`` is the tokens one llama-server slot holds. A render fits
    when ``rendered_prompt_tokens + max_tokens + safety_margin <= slot_context``,
    measured with the real tokenizer on the exact rendered prompt.
    ``max_tokens`` is the client's completion limit, never lowered mid-run.
    ``tokenizer`` and ``template_sha256`` identify the tokenizer and chat
    template behind the counts (with their parity status). ``generation_path``
    is the TTFT path every paired row must share.
    """

    slot_context: int
    max_tokens: int
    strategy: str = "naive_append_truncate"
    safety_margin: int = 64
    soft_limit: int | None = None
    segment_caps: Mapping[str, int] = field(default_factory=dict)
    tokenizer: Mapping[str, Any] = field(default_factory=dict)
    template_sha256: str | None = None
    generation_path: str = "non_streaming"
    cache_semantics: Mapping[str, Any] = field(default_factory=lambda: dict(UNCONFIRMED_CACHE))
    version: int = CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown context strategy {self.strategy!r}; part 1 provides {STRATEGIES}")
        if self.generation_path not in GENERATION_PATHS:
            raise ValueError(f"generation_path must be one of {GENERATION_PATHS}")
        for name in ("slot_context", "max_tokens", "safety_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.prompt_limit < 1:
            raise ValueError("slot_context must leave room for a prompt after max_tokens and the safety margin")
        if self.soft_limit is not None and not 0 < self.soft_limit <= self.prompt_limit:
            raise ValueError("soft_limit must be positive and no larger than the prompt limit")

    @property
    def prompt_limit(self) -> int:
        """The largest rendered prompt that fits: the slot minus the completion reserve and margin."""
        return self.slot_context - self.max_tokens - self.safety_margin

    def as_dict(self) -> dict[str, Any]:
        """The canonical configuration (sorted mappings, plain JSON values)."""
        return {
            "version": self.version,
            "strategy": self.strategy,
            "slot_context": self.slot_context,
            "max_tokens": self.max_tokens,
            "safety_margin": self.safety_margin,
            "soft_limit": self.soft_limit,
            "segment_caps": dict(sorted(self.segment_caps.items())),
            "tokenizer": dict(sorted(self.tokenizer.items())),
            "template_sha256": self.template_sha256,
            "generation_path": self.generation_path,
            "cache_semantics": dict(sorted(self.cache_semantics.items())),
        }

    def sha256(self) -> str:
        return context_config_sha256(self.as_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContextConfig:
        known = {
            "version", "strategy", "slot_context", "max_tokens", "safety_margin", "soft_limit", "segment_caps",
            "tokenizer", "template_sha256", "generation_path", "cache_semantics",
        }
        unexpected = set(value) - known
        if unexpected:
            raise ValueError(f"Unknown context config fields {sorted(unexpected)}")
        return cls(**dict(value))


__all__ = [
    "AttemptMetrics",
    "CONFIG_VERSION",
    "ContextConfig",
    "GENERATION_PATHS",
    "STRATEGIES",
    "SegmentBudget",
    "TurnMetrics",
    "UNCONFIRMED_CACHE",
]

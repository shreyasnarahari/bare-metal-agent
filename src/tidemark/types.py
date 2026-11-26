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
STRATEGIES = ("naive_append_truncate", "prefix_stable")
SEGMENT_NAMES = ("frozen", "pinned", "summary", "recent", "tail")
# What a strategy does when the exact prompt would pass the hard limit: part 2 has no compactor,
# so the prefix-stable strategy stops with context_budget_exhausted (part 4 adds "compact").
OVERFLOW_POLICIES = ("exhaust",)
DEFAULT_TAIL_CAP = 512
DEFAULT_RECITATION = {"max_items": 5, "max_item_chars": 120, "max_handles": 4}
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
    # Part 2 (prefix_stable): pinned memory text, a request-local callable-tool subset (None means all),
    # recitation bounds, and the overflow policy. The naive strategy ignores them.
    pinned_memory: tuple[str, ...] = ()
    allowed_tool_names: tuple[str, ...] | None = None
    recitation: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_RECITATION))
    overflow: str = "exhaust"
    # Part 3: the observation shaping configuration (tidemark.observations.ObservationConfig.as_dict: reducer
    # policy, blob namespace, catalog id, formats). None disables shaping and leaves the canonical form unchanged.
    observations: Mapping[str, Any] | None = None

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
        unknown = set(self.segment_caps) - set(SEGMENT_NAMES)
        if unknown:
            raise ValueError(f"Unknown segment caps {sorted(unknown)}; segments are {SEGMENT_NAMES}")
        if any(isinstance(cap, bool) or not isinstance(cap, int) or cap < 1 for cap in self.segment_caps.values()):
            raise ValueError("segment caps must be positive integers")
        if self.overflow not in OVERFLOW_POLICIES:
            raise ValueError(f"overflow must be one of {OVERFLOW_POLICIES}")
        if self.allowed_tool_names is not None and (
            not self.allowed_tool_names or len(set(self.allowed_tool_names)) != len(self.allowed_tool_names)
        ):
            raise ValueError("allowed_tool_names must be None or a non-empty tuple of unique names")
        if set(self.recitation) - set(DEFAULT_RECITATION):
            raise ValueError(f"recitation settings are {sorted(DEFAULT_RECITATION)}")

    @property
    def prompt_limit(self) -> int:
        """The largest rendered prompt that fits: the slot minus the completion reserve and margin."""
        return self.slot_context - self.max_tokens - self.safety_margin

    @property
    def effective_soft_limit(self) -> int:
        """``soft_limit``, or 70% of the prompt limit when unset (where compaction becomes pending)."""
        return self.soft_limit if self.soft_limit is not None else int(0.70 * self.prompt_limit)

    def cap(self, segment: str) -> int | None:
        """The token cap for one segment (the tail defaults to ``DEFAULT_TAIL_CAP``; others uncapped)."""
        if segment == "tail":
            return self.segment_caps.get("tail", DEFAULT_TAIL_CAP)
        return self.segment_caps.get(segment)

    def recitation_setting(self, name: str) -> int:
        return int(self.recitation.get(name, DEFAULT_RECITATION[name]))

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
            "pinned_memory": list(self.pinned_memory),
            "allowed_tool_names": list(self.allowed_tool_names) if self.allowed_tool_names is not None else None,
            "recitation": dict(sorted(self.recitation.items())),
            "overflow": self.overflow,
            # Present only with shaping, so part 1 and 2 configuration hashes are unchanged.
            **({"observations": dict(self.observations)} if self.observations is not None else {}),
        }

    def sha256(self) -> str:
        return context_config_sha256(self.as_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContextConfig:
        known = {
            "version", "strategy", "slot_context", "max_tokens", "safety_margin", "soft_limit", "segment_caps",
            "tokenizer", "template_sha256", "generation_path", "cache_semantics", "pinned_memory",
            "allowed_tool_names", "recitation", "overflow", "observations",
        }
        unexpected = set(value) - known
        if unexpected:
            raise ValueError(f"Unknown context config fields {sorted(unexpected)}")
        data = dict(value)
        data["pinned_memory"] = tuple(data.get("pinned_memory", ()))
        if data.get("allowed_tool_names") is not None:
            data["allowed_tool_names"] = tuple(data["allowed_tool_names"])
        return cls(**data)


__all__ = [
    "AttemptMetrics",
    "CONFIG_VERSION",
    "ContextConfig",
    "DEFAULT_TAIL_CAP",
    "GENERATION_PATHS",
    "OVERFLOW_POLICIES",
    "SEGMENT_NAMES",
    "STRATEGIES",
    "SegmentBudget",
    "TurnMetrics",
    "UNCONFIRMED_CACHE",
]

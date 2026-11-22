"""Per-tool policy metadata: effect class, timeout, retry, verify probe, alternatives.

``Tool`` has no field for these, so they live in a side table on
``SteadyhandRegistry``. Part 2 stores and validates them. The executor reads
``timeout_s`` now. Part 6 consumes ``retry``, ``verify``, and ``alternatives``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

BACKOFF_STRATEGIES = ("exp",)


class PolicyError(ValueError):
    """Policy metadata that contradicts itself or the registry."""


class Effect(str, Enum):
    READ = "read"
    IDEMPOTENT_WRITE = "idempotent_write"
    NON_IDEMPOTENT_WRITE = "non_idempotent_write"

    @property
    def idempotent(self) -> bool:
        return self is not Effect.NON_IDEMPOTENT_WRITE


@dataclass(frozen=True)
class RetrySpec:
    on: tuple[str, ...]  # error types, e.g. timeout, rate_limit, server_error
    max_attempts: int  # extra attempts after the first call
    backoff: str = "exp"  # applied in part 6

    def __post_init__(self) -> None:
        object.__setattr__(self, "on", tuple(self.on))
        if self.max_attempts < 0:
            raise PolicyError("retry.max_attempts must not be negative")
        if self.max_attempts > 0 and not self.on:
            raise PolicyError("retry.on must name at least one error type when retries are allowed")
        if any(not isinstance(kind, str) or not kind for kind in self.on):
            raise PolicyError("retry.on entries must be non-empty strings")
        if len(set(self.on)) != len(self.on):
            raise PolicyError("retry.on must not repeat an error type")
        if self.backoff not in BACKOFF_STRATEGIES:
            raise PolicyError(f"retry.backoff must be one of {list(BACKOFF_STRATEGIES)}")

    def as_dict(self) -> dict[str, Any]:
        return {"on": list(self.on), "max_attempts": self.max_attempts, "backoff": self.backoff}


@dataclass(frozen=True)
class ToolPolicy:
    effect: Effect
    timeout_s: float
    idempotent: bool
    retry: RetrySpec | None = None
    verify: str | None = None  # read-only tool name, used in part 6
    alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "alternatives", tuple(self.alternatives))
        if not isinstance(self.timeout_s, (int, float)) or not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise PolicyError("timeout_s must be a positive finite number")
        if self.idempotent != self.effect.idempotent:
            raise PolicyError(
                f"idempotent={self.idempotent} contradicts effect {self.effect.value!r}"
            )
        if not self.idempotent and self.retry is not None and self.retry.max_attempts > 0:
            raise PolicyError("A non-idempotent tool cannot have automatic retries")
        if len(set(self.alternatives)) != len(self.alternatives):
            raise PolicyError("alternatives must not repeat a tool name")

    def as_dict(self) -> dict[str, Any]:
        return {
            "effect": self.effect.value,
            "timeout_s": self.timeout_s,
            "idempotent": self.idempotent,
            "retry": self.retry.as_dict() if self.retry is not None else None,
            "verify": self.verify,
            "alternatives": list(self.alternatives),
        }


__all__ = ["BACKOFF_STRATEGIES", "Effect", "PolicyError", "RetrySpec", "ToolPolicy"]

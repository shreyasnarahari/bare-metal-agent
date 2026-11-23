"""Errors raised by the Bulkhead probe, report parser, and tier checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import MissingRequirement, SandboxTier


class BulkheadError(Exception):
    """Base class for every Bulkhead error."""


class ProbeReportError(BulkheadError, ValueError):
    """A probe report is malformed, uses unknown values, or contradicts its own evidence."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(f"{field}: {message}" if field else message)
        self.field = field


class CapabilityError(BulkheadError, RuntimeError):
    """A capability the caller needs has not been proven on this host."""


class UnsupportedTierError(CapabilityError):
    """The required sandbox tier is not proven here.

    There is never an automatic fallback to a lower tier: running at another
    tier needs a new configuration and manifest, not a caught exception.
    """

    def __init__(self, tier: SandboxTier, missing: Sequence[MissingRequirement], *, reason: str | None = None) -> None:
        self.tier = tier
        self.missing = tuple(missing)
        if reason is None:
            listed = "; ".join(item.summary() for item in self.missing) or "no evidence"
            reason = f"{tier.value} ({tier.label}) is not supported on this host: {listed}"
        super().__init__(reason)


__all__ = ["BulkheadError", "CapabilityError", "ProbeReportError", "UnsupportedTierError"]

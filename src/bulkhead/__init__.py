"""Bulkhead: sandboxed tool execution and permissions for the Steadyhand agent loop.

Part 1 provides the threat model (``Bulkhead/THREAT_MODEL.md``) and the
capability probe (``python -m bulkhead.probe``). Nothing in this package
executes a tool yet. The probe layer imports neither ``toolsmith`` nor
``steadyhand``, so it can be tested and run on its own.
"""

from .errors import BulkheadError, CapabilityError, ProbeReportError, UnsupportedTierError
from .types import (
    CAPABILITY_NAMES,
    SCHEMA_VERSION,
    TIER_REQUIREMENTS,
    Capability,
    CapabilityStatus,
    ProbeReport,
    SandboxTier,
)

__version__ = "0.1.0"

__all__ = [
    "CAPABILITY_NAMES",
    "SCHEMA_VERSION",
    "TIER_REQUIREMENTS",
    "BulkheadError",
    "Capability",
    "CapabilityError",
    "CapabilityStatus",
    "ProbeReport",
    "ProbeReportError",
    "SandboxTier",
    "UnsupportedTierError",
    "__version__",
]

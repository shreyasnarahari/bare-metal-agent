"""Bulkhead: sandboxed tool execution and permissions for the Steadyhand agent loop.

Part 1 provides the threat model (``Bulkhead/THREAT_MODEL.md``) and the
capability probe (``python -m bulkhead.probe``); parts 2 and 3 the policy
engine, approvals, taint, audit events, and recorded replay; part 4 the
execution backends (``bulkhead.backends``: the bubblewrap namespace/seccomp
phase), the seccomp filter compiler (``bulkhead.seccomp``), the process
supervisor (``bulkhead.supervisor``), and workspaces (``bulkhead.workspace``);
part 5 cgroups, rlimits, Landlock, and the pre-exec helper; part 6 egress
decisions (``bulkhead.egress``), the per-call CONNECT proxy
(``bulkhead.proxy``), the in-sandbox bridge (``bulkhead.bridge``), and the
output gate (``bulkhead.output``); part 7 the evaluation layer
(``bulkhead.eval``): the versioned attack corpus and its resumable runner,
overhead and usability benchmarks, and the results report.
Nothing outside ``bulkhead.eval`` imports ``toolsmith`` or ``steadyhand``, so
the sandbox layer can be tested and run on its own. ``bulkhead.eval`` sits
above both: it drives Steadyhand episodes and reuses Toolsmith's metrics.
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

"""Execution backends: the contract every backend implements, and the sandbox backends themselves.

``base`` holds the contract (requests, programs, observed results, audit
outcomes, resource-exhaustion classification). The live backends are imported
explicitly, since constructing one needs a probe report proving its features:

- ``bwrap.BwrapBackend``: tier 2, ``bwrap-t2`` (with a delegated cgroup parent)
  or the namespace-only ``bwrap-namespace-phase``;
- ``landlock.LandlockBackend``: tier 1, ``landlock-t1-cgroup`` or ``landlock-t1``.

Both run through ``common.run_governed`` (cgroup leaf, helper with start
barrier, governor). ``profiles`` describes what each profile enforces.
"""

from .base import (
    AUDIT_FIELDS,
    EXEC_RESULT_FIELDS,
    MAX_PROGRAM_BYTES,
    BackendIdentity,
    ExecResult,
    ExecutionBackend,
    ExecutionOutcome,
    ExecutionRequest,
    Program,
    ProgramError,
    ProgramKind,
    ProgramSpec,
    build_outcome,
    violation,
)

__all__ = [
    "AUDIT_FIELDS",
    "EXEC_RESULT_FIELDS",
    "MAX_PROGRAM_BYTES",
    "BackendIdentity",
    "ExecResult",
    "ExecutionBackend",
    "ExecutionOutcome",
    "ExecutionRequest",
    "Program",
    "ProgramError",
    "ProgramKind",
    "ProgramSpec",
    "build_outcome",
    "violation",
]

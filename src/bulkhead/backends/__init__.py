"""Execution backends: the contract every backend implements, and the sandbox backends themselves.

``base`` holds the contract (requests, programs, observed results, audit
outcomes). ``bwrap`` is the namespace/seccomp phase of tier 2 (part 4); import
it explicitly, since constructing it needs a probe report proving its features.
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

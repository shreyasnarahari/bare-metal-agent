"""Execution backends, and the recorded backend replay uses instead of re-running effects.

Tools that run processes or reach the filesystem or network never run
in-process: they go to an ``ExecutionBackend`` (the bubblewrap and Landlock
backends arrive in parts 4 and 5). Each backend run is journaled as a
``sandbox_execution`` event carrying its audit fields and its typed result.

Default replay must never repeat such a side effect. ``RecordedExecutionBackend``
serves the recorded runs back in order, after checking that the replayed
request (tool, call id, caller, request hash, sandbox config hash) is exactly
the recorded one. It never spawns a process or opens a socket. This proves the
control plane reproduces its decisions, observations, and taint; it does not
re-prove kernel enforcement, which only a live run under the real backend can.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import BulkheadError
from .manifest import SecuritySpec, check_keys
from .taint import Sink
from .types import SandboxTier, canonical_hash

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# The sandbox_execution fields a backend reports (the journal adds step, call id, caller, tool, and result).
AUDIT_FIELDS = (
    "backend", "tier", "config_sha256", "request_sha256", "started", "exit_code", "signal", "timed_out",
    "cleanup_status", "stdout_bytes", "stderr_bytes", "stdout_truncated", "stderr_truncated", "violations",
    "resource_usage", "elapsed_s",
)
# Sinks whose effects live outside the in-memory store and so must never be re-executed by replay.
EXTERNAL_SINKS = frozenset({Sink.NETWORK, Sink.FILESYSTEM, Sink.FILESYSTEM_SENSITIVE})


class RecordedExecutionMismatch(BulkheadError, RuntimeError):
    """Replay asked for a backend run the recording does not hold, or a different one."""

    def __init__(self, message: str, *, seq: int | None = None, field: str | None = None) -> None:
        super().__init__(message)
        self.seq = seq
        self.field = field


def requires_backend(spec: SecuritySpec | None, *, exec_effect: bool) -> bool:
    """Whether a tool must run through a backend: exec, a sandbox, or an external filesystem/network sink."""
    if exec_effect:
        return True
    if spec is None:
        return False
    return spec.sandboxed or bool(set(spec.sinks) & EXTERNAL_SINKS)


@dataclass(frozen=True)
class BackendIdentity:
    """Which backend ran the sandboxed calls: name, build version, tier, and the probe that proved the tier."""

    name: str
    version: str
    tier: SandboxTier
    probe_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError(f"bad backend name {self.name!r}")
        if not isinstance(self.version, str) or not self.version or len(self.version) > 64:
            raise ValueError("backend version must be a short string")
        object.__setattr__(self, "tier", SandboxTier(self.tier))
        if not self.tier.is_sandbox:
            raise ValueError("T0 is not a sandbox; no backend runs at T0")
        if self.probe_sha256 is not None and (not isinstance(self.probe_sha256, str) or not _SHA256.fullmatch(self.probe_sha256)):
            raise ValueError("probe_sha256 must be 64 lowercase hex characters")

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "tier": self.tier.value, "probe_sha256": self.probe_sha256}

    @classmethod
    def from_dict(cls, value: Any) -> BackendIdentity:
        check_keys(value, ("name", "version", "tier", "probe_sha256"), "backend")
        return cls(**dict(value))


@dataclass(frozen=True)
class ExecutionRequest:
    """One call handed to a backend. Arguments travel separately and are never journaled."""

    episode_id: str
    tool_call_id: str
    via: str
    tool: str
    arguments_sha256: str
    spec: SecuritySpec

    @property
    def config_sha256(self) -> str:
        """Hash of the tool's sandbox configuration."""
        return self.spec.sha256()

    @property
    def sha256(self) -> str:
        """The request hash: tool, canonical arguments, and sandbox configuration."""
        return canonical_hash({"tool": self.tool, "arguments_sha256": self.arguments_sha256, "config_sha256": self.config_sha256})


@dataclass(frozen=True)
class ExecutionOutcome:
    """A backend run: its typed result and its audit fields (``AUDIT_FIELDS``)."""

    result: Mapping[str, Any]
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if set(self.audit) != set(AUDIT_FIELDS):
            raise ValueError(f"audit fields must be exactly {list(AUDIT_FIELDS)}")


class ExecutionBackend(Protocol):
    """Runs a sandboxed call. Live backends arrive in parts 4 and 5."""

    identity: BackendIdentity

    def execute(self, request: ExecutionRequest, arguments: Mapping[str, Any]) -> ExecutionOutcome: ...


class RecordedExecutionBackend:
    """Serves recorded ``sandbox_execution`` runs in order; never spawns a process or opens a socket.

    ``identity`` is the original backend's identity (from the manifest), so
    replayed records and summaries match the recording exactly.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]], identity: BackendIdentity) -> None:
        self.identity = identity
        self._recorded = [dict(record) for record in records if record.get("event") == "sandbox_execution"]

    @property
    def remaining(self) -> int:
        return len(self._recorded)

    def execute(self, request: ExecutionRequest, arguments: Mapping[str, Any]) -> ExecutionOutcome:
        if not self._recorded:
            raise RecordedExecutionMismatch(
                f"replay asked to run {request.tool} for {request.tool_call_id}, but the recording has no further run",
                field="sandbox_execution",
            )
        recorded = self._recorded[0]
        for field, value in (
            ("tool_call_id", request.tool_call_id),
            ("tool", request.tool),
            ("via", request.via),
            ("config_sha256", request.config_sha256),
            ("request_sha256", request.sha256),
        ):
            if recorded.get(field) != value:
                raise RecordedExecutionMismatch(
                    f"the replayed {request.tool} request differs from the recorded run in {field}",
                    seq=recorded.get("seq"),
                    field=field,
                )
        self._recorded.pop(0)
        return ExecutionOutcome(result=recorded["result"], audit={field: recorded[field] for field in AUDIT_FIELDS})


__all__ = [
    "AUDIT_FIELDS",
    "EXTERNAL_SINKS",
    "BackendIdentity",
    "ExecutionBackend",
    "ExecutionOutcome",
    "ExecutionRequest",
    "RecordedExecutionBackend",
    "RecordedExecutionMismatch",
    "requires_backend",
]

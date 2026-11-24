"""Execution backends, and the recorded backend replay uses instead of re-running effects.

Tools that run processes or reach the filesystem or network never run
in-process: they go to an ``ExecutionBackend`` (``bulkhead.backends``: the
bubblewrap backend from part 4, the Landlock backend in part 5). The backend
contract lives in ``bulkhead.backends.base`` and is re-exported here. Each
backend run is journaled as a ``sandbox_execution`` event carrying its audit
fields and its typed result.

Default replay must never repeat such a side effect. ``RecordedExecutionBackend``
serves the recorded runs back in order, after checking that the replayed
request (tool, call id, caller, request hash, sandbox config hash) is exactly
the recorded one. It never spawns a process or opens a socket. This proves the
control plane reproduces its decisions, observations, and taint; it does not
re-prove kernel enforcement, which only a live run under the real backend can.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .backends.base import (
    AUDIT_FIELDS,
    BackendIdentity,
    ExecutionBackend,
    ExecutionOutcome,
    ExecutionRequest,
)
from .errors import BulkheadError
from .manifest import SecuritySpec
from .taint import Sink

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


class RecordedExecutionBackend:
    """Serves recorded ``sandbox_execution`` runs in order; never spawns a process or opens a socket.

    ``identity`` is the original backend's identity (from the manifest), so
    replayed records and summaries match the recording exactly.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]], identity: BackendIdentity) -> None:
        self.identity = identity
        # Each run with the egress decisions and output-gate record journaled right after it.
        self._recorded: list[dict[str, Any]] = []
        for record in records:
            event = record.get("event")
            if event == "sandbox_execution":
                self._recorded.append({**record, "_egress": [], "_gate": None})
            elif event in ("egress_decision", "output_gate") and self._recorded and record.get("tool_call_id") == self._recorded[-1]["tool_call_id"]:
                fields = {key: value for key, value in record.items() if key not in ("event", "episode_id", "seq", "recorded_at", "step", "tool_call_id")}
                if event == "egress_decision":
                    self._recorded[-1]["_egress"].append(fields)
                else:
                    self._recorded[-1]["_gate"] = fields["streams"]

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
        return ExecutionOutcome(
            result=recorded["result"],
            audit={field: recorded[field] for field in AUDIT_FIELDS},
            egress=tuple(recorded["_egress"]),
            gate=recorded["_gate"],
        )


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

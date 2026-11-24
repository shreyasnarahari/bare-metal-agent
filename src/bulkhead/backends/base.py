"""The backend contract: what a sandboxed call asks for, and what a backend reports back.

Nothing here imports Steadyhand or touches its store. A backend receives an
immutable ``ExecutionRequest`` only after the policy gate allowed the call,
runs it, and returns an ``ExecutionOutcome``: the ``sandbox_execution`` audit
fields plus either

- ``process``, an ``ExecResult`` describing what a live backend observed
  (whether the sandbox was set up and the program started, its exit code or
  signal, bounded output, timeout and truncation flags), which the Steadyhand
  adapter turns into a typed tool result; or
- ``result``, a typed tool result as already recorded (the recorded backend
  replay uses, and scripted test backends).

Program text travels inside the request (``Program``) and reaches the sandbox
over a bounded stdin pipe. It is never journaled, logged, or placed on a host
command line; the request hash covers it through the arguments hash.
"""

from __future__ import annotations

import re
import signal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from ..errors import BulkheadError
from ..manifest import SecuritySpec, check_keys
from ..output import GatedText, render, sanitize
from ..types import SandboxTier, canonical_hash

if TYPE_CHECKING:
    from ..supervisor import ProcessOutcome

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# The sandbox_execution fields a backend reports (the journal adds step, call id, caller, tool, and result).
AUDIT_FIELDS = (
    "backend", "tier", "config_sha256", "request_sha256", "started", "exit_code", "signal", "timed_out",
    "cleanup_status", "stdout_bytes", "stderr_bytes", "stdout_truncated", "stderr_truncated", "violations",
    "resource_usage", "elapsed_s",
)
# Program text a model may hand to an exec tool. Larger programs are refused before anything runs.
MAX_PROGRAM_BYTES = 64 * 1024
# Ceilings whose exhaustion is a resource_exhausted result, distinct from a timeout or a nonzero exit.
EXHAUSTION_LIMITS = ("memory", "pids", "disk", "cpu")
# The typed value of a completed exec run, as the model sees it. Each stream is an output-gate object
# (``bulkhead.output.render``): {"trust": "untrusted", "content": <wrapped, sanitized text>, "truncated": ...}.
EXEC_RESULT_FIELDS = ("exit_code", "signal", "stdout", "stderr")


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


# Programs


class ProgramKind(str, Enum):
    """How the sandbox runs program text: ``python -I -`` or ``sh -c`` inside the sandbox."""

    PYTHON = "python"
    SHELL = "shell"


class ProgramError(BulkheadError, ValueError):
    """Program text that cannot be handed to a sandbox. ``code`` is a stable rule id."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Program:
    """Program text for one call: its kind and its bounded UTF-8 source, with no NUL bytes."""

    kind: ProgramKind
    source: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ProgramKind(self.kind))
        if not isinstance(self.source, bytes):
            raise TypeError("program source must be bytes")
        if len(self.source) > MAX_PROGRAM_BYTES:
            raise ProgramError(f"the program is larger than {MAX_PROGRAM_BYTES} bytes", code="bulkhead.exec.program_too_large")
        if b"\x00" in self.source:
            raise ProgramError("the program contains a NUL byte", code="bulkhead.exec.program_nul")

    @classmethod
    def from_text(cls, kind: ProgramKind | str, text: Any) -> Program:
        if not isinstance(text, str):
            raise ProgramError("the program must be a string", code="bulkhead.exec.program_type")
        try:
            source = text.encode("utf-8")
        except UnicodeEncodeError:
            raise ProgramError("the program is not valid UTF-8", code="bulkhead.exec.program_encoding") from None
        return cls(ProgramKind(kind), source)


@dataclass(frozen=True)
class ProgramSpec:
    """An exec tool's dispatch descriptor: which program kind it runs, and which string argument holds the text.

    The model supplies only that argument. Interpreter paths, launcher flags,
    environment entries, and descriptors are fixed by the backend.
    """

    kind: ProgramKind
    argument: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ProgramKind(self.kind))
        if not isinstance(self.argument, str) or not _NAME.fullmatch(self.argument):
            raise ValueError(f"bad program argument name {self.argument!r}")

    def program(self, arguments: Mapping[str, Any]) -> Program:
        return Program.from_text(self.kind, arguments.get(self.argument))


# Requests and outcomes


@dataclass(frozen=True)
class ExecutionRequest:
    """One allowed call handed to a backend. Never journaled except through its hashes.

    ``program`` is the text an exec tool runs (None for tools a backend serves
    without one). ``timeout_s`` is the tool policy's timeout; the backend
    enforces the smaller of it and the sandbox's ``wall_s`` limit.
    """

    episode_id: str
    tool_call_id: str
    via: str
    tool: str
    arguments_sha256: str
    spec: SecuritySpec
    program: Program | None = None
    timeout_s: float | None = None

    @property
    def config_sha256(self) -> str:
        """Hash of the tool's sandbox configuration."""
        return self.spec.sha256()

    @property
    def sha256(self) -> str:
        """The request hash: tool, canonical arguments, and sandbox configuration."""
        return canonical_hash({"tool": self.tool, "arguments_sha256": self.arguments_sha256, "config_sha256": self.config_sha256})

    @property
    def effective_timeout_s(self) -> float:
        """The wall-clock limit the backend enforces."""
        limits = [self.spec.sandbox.limits.wall_s] if self.spec.sandbox is not None else []
        if self.timeout_s is not None:
            limits.append(self.timeout_s)
        if not limits:
            raise ValueError("a request needs a sandbox or a timeout")
        return float(min(limits))


@dataclass(frozen=True)
class ExecResult:
    """What a live backend observed for one call, before the adapter turns it into a tool result.

    ``started`` means the sandbox was fully set up and the program began; a
    run that never started carries a stable ``setup_error`` code and no output.
    A nonzero exit or a signal is a valid result. ``timed_out`` means the
    supervisor killed the run at its deadline.
    """

    started: bool
    exit_code: int | None = None
    signal: int | None = None
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    setup_error: str | None = None
    exhausted: str | None = None  # memory, pids, disk, or cpu: the ceiling that stopped a started run
    output: Mapping[str, Mapping[str, Any]] | None = None  # the output gate's per-stream statistics

    def __post_init__(self) -> None:
        if not self.started and (self.exit_code is not None or self.signal is not None or self.stdout or self.stderr):
            raise ValueError("a run that never started has no exit status or output")
        if self.exhausted is not None and (not self.started or self.exhausted not in EXHAUSTION_LIMITS):
            raise ValueError("only a started run can exhaust a known ceiling")
        if self.started and self.setup_error is not None:
            raise ValueError("a started run has no setup error")
        if not self.started and self.setup_error is None:
            raise ValueError("a run that never started needs a setup error code")

    def render(self, request_sha256: str, secrets: Sequence[str] = ()) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The model-facing value of a completed run and the ``output_gate`` stream records.

        ``stdout``/``stderr`` are already sanitized (``build_outcome``); here
        they are redacted and wrapped in deterministic untrusted-output
        delimiters derived from ``request_sha256``.
        """
        value: dict[str, Any] = {"exit_code": self.exit_code, "signal": self.signal}
        streams = []
        for name, text, truncated in (("stdout", self.stdout, self.stdout_truncated), ("stderr", self.stderr, self.stderr_truncated)):
            stats = dict((self.output or {}).get(name) or GatedText(text, len(text.encode()), len(text.encode()), False, 0, 0, truncated).stats())
            rendered, redacted = render(GatedText(text, stats["raw_bytes"], stats["kept_bytes"], stats["invalid_utf8"], stats["stripped"], stats["neutralized"], truncated),
                                        request_sha256=request_sha256, stream=name, secrets=secrets)
            value[name] = rendered
            streams.append({"name": name, **stats, "redacted": redacted})
        return value, streams


@dataclass(frozen=True)
class ExecutionOutcome:
    """A backend run: its audit fields (``AUDIT_FIELDS``) plus a typed ``result`` or an observed ``process``."""

    result: Mapping[str, Any] | None
    audit: Mapping[str, Any]
    process: ExecResult | None = None
    egress: tuple[Mapping[str, Any], ...] = ()  # the call's egress_decision records, in request order
    gate: tuple[Mapping[str, Any], ...] | None = None  # recorded output_gate streams that go with a typed result

    def __post_init__(self) -> None:
        if set(self.audit) != set(AUDIT_FIELDS):
            raise ValueError(f"audit fields must be exactly {list(AUDIT_FIELDS)}")
        if (self.result is None) == (self.process is None):
            raise ValueError("an outcome carries exactly one of a typed result or an observed process")
        if self.process is not None and self.process.started != self.audit["started"]:
            raise ValueError("the observed process and the audit disagree about whether it started")


class ExecutionBackend(Protocol):
    """Runs an allowed sandboxed call and reports it. It never imports Steadyhand or touches the store."""

    identity: BackendIdentity

    def execute(self, request: ExecutionRequest, arguments: Mapping[str, Any]) -> ExecutionOutcome: ...


def violation(kind: str, rule_id: str, detail: str) -> dict[str, str]:
    """One audit violation: a stable kind and rule id, and a short single-line detail."""
    return {"kind": kind, "rule_id": rule_id, "detail": detail}


def exhaustion(request: ExecutionRequest, observed: ProcessOutcome) -> str | None:
    """Which ceiling, if any, stopped a started run, from the evidence the run left.

    In order: a cgroup OOM kill (memory); PID-limit rejections in a run that
    then failed or timed out (pids); the governor's disk stop, SIGXFSZ, or a
    failed run whose writable area reached ``disk_mb`` (disk); the governor's
    CPU stop or SIGXCPU (cpu). A run that succeeded despite a rejected fork is
    an ordinary result; the audit still counts the rejections.
    """
    usage = observed.resource_usage
    failed = observed.timed_out or observed.signal is not None or observed.exit_code not in (0, None) or observed.stopped is not None
    if usage.get("oom_kills", 0) > 0 or usage.get("oom_killed") is True:
        return "memory"
    if usage.get("pids_limit_hits", 0) > 0 and failed:
        return "pids"
    disk_kb = request.spec.sandbox.limits.disk_mb * 1024 if request.spec.sandbox is not None else None
    if observed.stopped == "disk" or observed.signal == signal.SIGXFSZ:
        return "disk"
    if failed and disk_kb is not None and usage.get("disk_peak_kb", 0) >= disk_kb:
        return "disk"
    if observed.stopped == "cpu" or observed.signal == signal.SIGXCPU:
        return "cpu"
    return None


def with_egress(outcome: ExecutionOutcome, records: Sequence[Mapping[str, Any]]) -> ExecutionOutcome:
    """Attach the call's proxy records, and an ``egress`` violation for each distinct denial rule."""
    egress = tuple(dict(record) for record in records)
    denied = sorted({record["rule_id"] for record in egress if record["decision"] == "deny"})
    violations = list(outcome.audit["violations"])
    for rule in denied:
        if len(violations) < 16:
            violations.append(violation("egress", rule, "the egress proxy refused a CONNECT"))
    return replace(outcome, audit={**outcome.audit, "violations": violations}, egress=egress)


def build_outcome(request: ExecutionRequest, identity: BackendIdentity, observed: ProcessOutcome) -> ExecutionOutcome:
    """Turn what the supervisor observed into the audit fields and the ``ExecResult`` for one call.

    A run that never started reports no exit status and no output (whatever the
    launcher's stderr said stays out of the record) and a ``setup`` violation.
    A program killed by SIGSYS was stopped by its seccomp filter and is
    recorded as a ``seccomp`` violation; a process that outlived the call's
    cleanup is a ``cleanup`` violation.
    """
    violations: list[dict[str, str]] = []
    if not observed.started:
        code = observed.setup_error or "bulkhead.setup.failed"
        violations.append(violation("setup", code, "the sandbox was not set up, so the program did not run"))
        process = ExecResult(started=False, timed_out=observed.timed_out, setup_error=code)
        cleanup = observed.cleanup_status if observed.cleanup_status in ("killed", "leaked") else "not_started"
        counts = {"stdout_bytes": 0, "stderr_bytes": 0, "stdout_truncated": False, "stderr_truncated": False}
    else:
        if observed.signal == signal.SIGSYS:
            violations.append(violation("seccomp", "bulkhead.seccomp.kill", "the program was killed by its seccomp filter (SIGSYS)"))
        exhausted = exhaustion(request, observed)
        if exhausted is not None:
            violations.append(violation("resource", f"bulkhead.limit.{exhausted}", f"the {exhausted} ceiling stopped the program"))
        stdout = sanitize(observed.stdout, raw_bytes=observed.stdout_bytes, truncated=observed.stdout_truncated)
        stderr = sanitize(observed.stderr, raw_bytes=observed.stderr_bytes, truncated=observed.stderr_truncated)
        process = ExecResult(
            started=True,
            exit_code=observed.exit_code,
            signal=observed.signal,
            timed_out=observed.timed_out,
            stdout=stdout.text,
            stderr=stderr.text,
            stdout_truncated=observed.stdout_truncated,
            stderr_truncated=observed.stderr_truncated,
            exhausted=exhausted,
            output={"stdout": stdout.stats(), "stderr": stderr.stats()},
        )
        cleanup = observed.cleanup_status
        counts = {
            "stdout_bytes": observed.stdout_bytes,
            "stderr_bytes": observed.stderr_bytes,
            "stdout_truncated": observed.stdout_truncated,
            "stderr_truncated": observed.stderr_truncated,
        }
    if observed.cleanup_status == "leaked":
        violations.append(violation("cleanup", "bulkhead.cleanup.leaked", "a process outlived the call's cleanup"))
    audit = {
        "backend": identity.name,
        "tier": identity.tier.value,
        "config_sha256": request.config_sha256,
        "request_sha256": request.sha256,
        "started": process.started,
        "exit_code": process.exit_code,
        "signal": process.signal,
        "timed_out": observed.timed_out,
        "cleanup_status": cleanup,
        **counts,
        "violations": violations,
        "resource_usage": dict(observed.resource_usage),
        "elapsed_s": round(observed.elapsed_s, 6),
    }
    return ExecutionOutcome(None, audit, process=process)


__all__ = [
    "AUDIT_FIELDS",
    "EXEC_RESULT_FIELDS",
    "EXHAUSTION_LIMITS",
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
    "exhaustion",
    "violation",
    "with_egress",
]

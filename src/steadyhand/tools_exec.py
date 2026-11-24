"""``run_python`` and ``run_shell``: exec tools that only ever run inside a sandbox backend.

Each tool is registered with its model-facing schema, the ``Effect.EXEC``
policy (never retried, never verified, sequential with writes), the security
spec of its sandbox, and a dispatch descriptor (``ProgramSpec``) naming the
program kind and the argument holding the text. After the policy gate allows a
call, ``steadyhand.security.BackendRunner`` builds an immutable
``ExecutionRequest`` from these and hands it to the episode's backend
(``bulkhead.backends.bwrap.BwrapBackend``, the namespace/seccomp phase of tier
2). The result value is an ``ExecResult``: exit code, signal, and stdout and
stderr as output-gate objects (sanitized, bounded, wrapped in deterministic
untrusted-output delimiters, with a truncation flag). The output is untrusted
and taints the episode.

The handlers below exist only to give the tools their schemas. Nothing runs
them: ``SequentialExecutor`` never calls an exec handler, and a direct call
raises ``DirectExecError``.

The sandbox (``EXEC_SANDBOX``) sees the workspace and the system runtime
read-only, may write only ``$WORKSPACE/out``, cannot see the repository
metadata or env file, has no network, and holds no credentials. Certificate
roots are not visible; they are added only to specs that allow egress (part 6).

The tools are opt-in (``build_registry(exec_tools=True)``), so the default
registry's schemas and every golden episode stay unchanged.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from pydantic import Field

from bulkhead.backends.base import ProgramKind, ProgramSpec
from bulkhead.manifest import Egress, Limits, Sandbox, SecuritySpec, Sink, Trust

from .policy import Effect, ToolPolicy
from .registry import SteadyhandRegistry
from .store import Store

EXEC_TIMEOUT_S = 30.0
EXEC_POLICY = ToolPolicy(Effect.EXEC, EXEC_TIMEOUT_S, idempotent=False, retry=None)
EXEC_LIMITS = Limits(cpu_s=10, mem_mb=512, pids=64, disk_mb=200, out_kb=64, wall_s=EXEC_TIMEOUT_S)
# The workspace and system runtime readable, only $WORKSPACE/out writable, the repository
# metadata and env file hidden, no network, no credentials, no certificate roots.
EXEC_SANDBOX = Sandbox(
    fs_read=("$WORKSPACE", "/usr", "/lib", "/lib64"),
    fs_write=("$WORKSPACE/out",),
    fs_deny=("$WORKSPACE/.git", "$WORKSPACE/.env"),
    net=Egress(),
    limits=EXEC_LIMITS,
    secrets=(),
)
EXEC_SECURITY = SecuritySpec(sandbox=EXEC_SANDBOX, output_trust=Trust.UNTRUSTED, sinks=(Sink.FILESYSTEM,))
EXEC_PROGRAMS = {
    "run_python": ProgramSpec(ProgramKind.PYTHON, "code"),
    "run_shell": ProgramSpec(ProgramKind.SHELL, "command"),
}


class GatedStream(TypedDict):
    """One output stream after the output gate (``bulkhead.output``): sanitized, bounded, labeled."""

    trust: str  # always "untrusted"
    content: str  # "[BEGIN UNTRUSTED OUTPUT <id>]\n...\n[END UNTRUSTED OUTPUT <id>]", or "" when empty
    truncated: bool


class ExecResult(TypedDict):
    """The value of a completed exec call. A nonzero exit or a signal is still a result."""

    exit_code: int | None
    signal: int | None
    stdout: GatedStream
    stderr: GatedStream


class DirectExecError(RuntimeError):
    """An exec tool's handler was called directly. Exec tools only ever run through a sandbox backend."""


# Kept for callers written against part 2.
ExecDisabledError = DirectExecError


def run_python(
    store: Store,
    code: Annotated[str, Field(description="A short Python 3 program. Print the results you need.")],
) -> dict[str, Any]:
    """Run a short Python program in an isolated sandbox with no network, and return its exit code and output."""
    raise DirectExecError("run_python only runs inside a sandbox backend, never in-process")


def run_shell(
    store: Store,
    command: Annotated[str, Field(description="A shell command line, run with /bin/sh -c inside the sandbox.")],
) -> dict[str, Any]:
    """Run a shell command in an isolated sandbox with no network, and return its exit code and output."""
    raise DirectExecError("run_shell only runs inside a sandbox backend, never in-process")


EXEC_HANDLERS = (run_python, run_shell)


def register_exec_tools(registry: SteadyhandRegistry) -> SteadyhandRegistry:
    """Add the exec tools, with their sandbox specs and dispatch descriptors, to an unfrozen registry."""
    for handler in EXEC_HANDLERS:
        registry.register(handler, EXEC_POLICY, security=EXEC_SECURITY, program=EXEC_PROGRAMS[handler.__name__])
    return registry


__all__ = [
    "EXEC_HANDLERS",
    "EXEC_LIMITS",
    "EXEC_POLICY",
    "EXEC_PROGRAMS",
    "EXEC_SANDBOX",
    "EXEC_SECURITY",
    "EXEC_TIMEOUT_S",
    "DirectExecError",
    "ExecDisabledError",
    "ExecResult",
    "GatedStream",
    "register_exec_tools",
    "run_python",
    "run_shell",
]

"""``run_python`` and ``run_shell``: exec tool schemas, registered disabled until Bulkhead part 4.

These are markers. They give the model-facing schemas, the ``Effect.EXEC``
policy, and the security spec of the sandbox the backend will build, so the
policy engine, manifests, and grammars can be tested now. Nothing executes
them:

- the handlers raise ``ExecDisabledError`` if anything calls them directly;
- ``SequentialExecutor`` never runs an exec tool's handler in-process; and
- the policy gate denies them (``bulkhead.tier.unavailable``) because no
  sandbox backend exists until part 4.

They are opt-in (``build_registry(exec_tools=True)``), so the default
registry's schemas and every golden episode stay unchanged.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from bulkhead.manifest import Egress, Limits, Sandbox, SecuritySpec, Sink, Trust

from .policy import Effect, ToolPolicy
from .registry import SteadyhandRegistry
from .store import Store

EXEC_TIMEOUT_S = 30.0
EXEC_POLICY = ToolPolicy(Effect.EXEC, EXEC_TIMEOUT_S, idempotent=False, retry=None)
EXEC_LIMITS = Limits(cpu_s=10, mem_mb=512, pids=64, disk_mb=200, out_kb=64, wall_s=EXEC_TIMEOUT_S)
# The sandbox part 4 builds: the workspace and system libraries readable, only $WORKSPACE/out
# writable, the repository metadata and env file hidden, no network, no credentials.
EXEC_SANDBOX = Sandbox(
    fs_read=("$WORKSPACE", "/usr", "/lib", "/lib64", "/etc/ssl"),
    fs_write=("$WORKSPACE/out",),
    fs_deny=("$WORKSPACE/.git", "$WORKSPACE/.env"),
    net=Egress(),
    limits=EXEC_LIMITS,
    secrets=(),
)
EXEC_SECURITY = SecuritySpec(sandbox=EXEC_SANDBOX, output_trust=Trust.UNTRUSTED, sinks=(Sink.FILESYSTEM,))


class ExecDisabledError(RuntimeError):
    """An exec tool's handler was called directly. Exec tools only ever run through a sandbox backend."""


def run_python(
    store: Store,
    code: Annotated[str, Field(description="A short Python 3 program. Print the results you need.")],
) -> dict[str, Any]:
    """Run a short Python program in an isolated sandbox with no network, and return its exit code and output."""
    raise ExecDisabledError("run_python has no sandbox backend yet; it is never run in-process")


def run_shell(
    store: Store,
    command: Annotated[str, Field(description="A shell command line, run with /bin/sh -c inside the sandbox.")],
) -> dict[str, Any]:
    """Run a shell command in an isolated sandbox with no network, and return its exit code and output."""
    raise ExecDisabledError("run_shell has no sandbox backend yet; it is never run in-process")


EXEC_HANDLERS = (run_python, run_shell)


def register_exec_tools(registry: SteadyhandRegistry) -> SteadyhandRegistry:
    """Add the disabled exec tools to an unfrozen registry."""
    for handler in EXEC_HANDLERS:
        registry.register(handler, EXEC_POLICY, security=EXEC_SECURITY)
    return registry


__all__ = [
    "EXEC_HANDLERS",
    "EXEC_LIMITS",
    "EXEC_POLICY",
    "EXEC_SANDBOX",
    "EXEC_SECURITY",
    "EXEC_TIMEOUT_S",
    "ExecDisabledError",
    "register_exec_tools",
    "run_python",
    "run_shell",
]

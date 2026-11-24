"""Shared helpers for the exec tests: host-only test backends and the attack dataset.

``HostSeccompBackend`` is NOT A SANDBOX. It runs a program on this host
through the real supervisor, launcher, and seccomp filter, with no
namespaces, no filesystem view, and no resource limits. Tests use it to check
the supervisor, the filter, and the Steadyhand wiring on hosts without
bubblewrap, and only ever with fixed, harmless test programs. It is labeled
``host-seccomp-test`` and lives in the tests directory so nothing can select
it for real work.

The attack dataset, its canaries, listeners, runner, and judge live in
``bulkhead.eval.attacks`` (part 7); ``DATASET`` is loaded once here.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import bulkhead
from bulkhead.backends.base import BackendIdentity, ExecutionOutcome, ExecutionRequest, Program, ProgramKind, build_outcome, with_egress
from bulkhead.backends.bwrap import SANDBOX_ENV
from bulkhead.bridge import BRIDGE_PORT, PROXY_ENV
from bulkhead.eval.attacks import load_dataset
from bulkhead.proxy import call_proxy, system_connector, system_resolver
from bulkhead.seccomp import NET_POLICY, compile_filter
from bulkhead.supervisor import CaptureLimits, ProcessOutcome, launcher_command, supervise
from bulkhead.types import SandboxTier, canonical_hash
from steadyhand.tools_exec import EXEC_SECURITY

ROOT = Path(__file__).resolve().parent
DATASET = load_dataset()
HARNESS = Path(bulkhead.__file__).with_name("_seccomp_exec.py")
UNSHARE = "/usr/bin/unshare"


class HostSeccompBackend:
    """NOT A SANDBOX: the real supervisor, launcher, and seccomp filter on the host, without namespaces."""

    identity = BackendIdentity("host-seccomp-test", "1", SandboxTier.T2)

    def __init__(self, workdir: str | os.PathLike[str], *, env: Mapping[str, str] = SANDBOX_ENV, filtered: bool = True) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.env = dict(env)
        self.filtered = filtered
        self.filter = compile_filter()
        self.net_filter = compile_filter(NET_POLICY)
        self.requests: list[ExecutionRequest] = []
        self.observed: list[ProcessOutcome] = []

    def execute(self, request: ExecutionRequest, arguments: Mapping[str, Any]) -> ExecutionOutcome:
        self.requests.append(request)
        if request.program is None:
            return build_outcome(request, self.identity, ProcessOutcome.not_started("bulkhead.setup.no_program"))
        return build_outcome(request, self.identity, self._run(request))

    def _run(self, request: ExecutionRequest, *, proxy_socket: str | None = None) -> ProcessOutcome:
        program = request.program
        assert program is not None
        active = self.net_filter if proxy_socket is not None else self.filter
        seccomp_fd = active.memfd()
        try:
            def argv(status_fd: int) -> list[str]:
                bridge = (BRIDGE_PORT, proxy_socket) if proxy_socket is not None else None
                launch = launcher_command(sys.executable, status_fd, program, shell="/bin/sh", bridge=bridge)
                if self.filtered:
                    launch = [sys.executable, "-I", "-S", str(HARNESS), str(seccomp_fd), "--", *launch]
                if proxy_socket is not None:  # a fresh user+network namespace with only loopback, up
                    launch = [UNSHARE, "-Urn", "/bin/sh", "-c", '/usr/sbin/ip link set lo up && exec "$@"', "netns", *launch]
                return launch

            observed = supervise(
                argv,
                stdin=program.source,
                env={**self.env, **(PROXY_ENV if proxy_socket is not None else {})},
                timeout_s=request.effective_timeout_s,
                limits=CaptureLimits.from_limits(request.spec.sandbox.limits),
                pass_fds=(seccomp_fd,),
                cwd=str(self.workdir),
            )
        finally:
            os.close(seccomp_fd)
        self.observed.append(observed)
        return observed


class HostNetnsBackend(HostSeccompBackend):
    """NOT A SANDBOX: ``HostSeccompBackend`` plus, for network-enabled specs, a fresh network namespace.

    ``unshare -Urn`` gives the program a user and network namespace with only
    loopback (brought up), the real seccomp network variant, the launcher's
    loopback bridge, and the call's real outside proxy on a private Unix
    socket. There is no mount namespace, so this tests the egress data path
    (bridge, proxy, policy, audit, no direct route) and nothing about the
    filesystem. Used where bubblewrap is unavailable.
    """

    identity = BackendIdentity("host-netns-test", "1", SandboxTier.T2)

    def __init__(self, workdir: str | os.PathLike[str], *, resolver=None, connector=None, **options: Any) -> None:
        super().__init__(workdir, **options)
        self.resolver = resolver or system_resolver
        self.connector = connector or system_connector
        self.records: list[dict[str, Any]] = []

    def execute(self, request: ExecutionRequest, arguments: Mapping[str, Any]) -> ExecutionOutcome:
        sandbox = request.spec.sandbox
        if request.program is None or sandbox is None or not sandbox.net.enabled:
            return super().execute(request, arguments)
        self.requests.append(request)
        with call_proxy(sandbox.net, resolver=self.resolver, connector=self.connector, idle_s=5.0) as proxy:
            observed = self._run(request, proxy_socket=str(proxy.socket_path))
        records = [record.as_audit() for record in proxy.records]
        self.records.extend(records)
        return with_egress(build_outcome(request, self.identity, observed), records)


def request_for(program: Program, *, episode_id: str = "ep_test", call_id: str = "call_0", spec=EXEC_SECURITY, timeout_s: float | None = None) -> ExecutionRequest:
    tool = "run_python" if program.kind is ProgramKind.PYTHON else "run_shell"
    argument = "code" if program.kind is ProgramKind.PYTHON else "command"
    digest = canonical_hash({argument: program.source.decode()})
    return ExecutionRequest(episode_id, call_id, "model", tool, digest, spec, program=program, timeout_s=timeout_s)

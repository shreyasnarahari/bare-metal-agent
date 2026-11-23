"""Shared helpers for the part 4 exec tests: a host-only seccomp backend and the attack-fixture runner.

``HostSeccompBackend`` is NOT A SANDBOX. It runs a program on this host
through the real supervisor, launcher, and seccomp filter, with no
namespaces, no filesystem view, and no resource limits. Tests use it to check
the supervisor, the filter, and the Steadyhand wiring on hosts without
bubblewrap, and only ever with fixed, harmless test programs. It is labeled
``host-seccomp-test`` and lives in the tests directory so nothing can select
it for real work.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bulkhead
from bulkhead.backends.base import BackendIdentity, ExecutionOutcome, ExecutionRequest, Program, ProgramKind, build_outcome
from bulkhead.backends.bwrap import SANDBOX_ENV
from bulkhead.seccomp import compile_filter
from bulkhead.supervisor import CaptureLimits, ProcessOutcome, launcher_command, supervise
from bulkhead.types import SandboxTier, canonical_hash
from steadyhand.tools_exec import EXEC_SECURITY

ROOT = Path(__file__).resolve().parent
ATTACKS = ROOT / "fixtures" / "bulkhead_attacks"
HARNESS = Path(bulkhead.__file__).with_name("_seccomp_exec.py")
MARKER = "BULKHEAD-CANARY-"


class HostSeccompBackend:
    """NOT A SANDBOX: the real supervisor, launcher, and seccomp filter on the host, without namespaces."""

    identity = BackendIdentity("host-seccomp-test", "1", SandboxTier.T2)

    def __init__(self, workdir: str | os.PathLike[str], *, env: Mapping[str, str] = SANDBOX_ENV, filtered: bool = True) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.env = dict(env)
        self.filtered = filtered
        self.filter = compile_filter()
        self.requests: list[ExecutionRequest] = []
        self.observed: list[ProcessOutcome] = []

    def execute(self, request: ExecutionRequest, arguments: Mapping[str, Any]) -> ExecutionOutcome:
        self.requests.append(request)
        if request.program is None:
            return build_outcome(request, self.identity, ProcessOutcome.not_started("bulkhead.setup.no_program"))
        program = request.program
        seccomp_fd = self.filter.memfd()
        try:
            def argv(status_fd: int) -> list[str]:
                launch = launcher_command(sys.executable, status_fd, program, shell="/bin/sh")
                if not self.filtered:
                    return launch
                return [sys.executable, "-I", "-S", str(HARNESS), str(seccomp_fd), "--", *launch]

            observed = supervise(
                argv,
                stdin=program.source,
                env=self.env,
                timeout_s=request.effective_timeout_s,
                limits=CaptureLimits.from_limits(request.spec.sandbox.limits),
                pass_fds=(seccomp_fd,),
                cwd=str(self.workdir),
            )
        finally:
            os.close(seccomp_fd)
        self.observed.append(observed)
        return build_outcome(request, self.identity, observed)


def request_for(program: Program, *, episode_id: str = "ep_test", call_id: str = "call_0", spec=EXEC_SECURITY, timeout_s: float | None = None) -> ExecutionRequest:
    tool = "run_python" if program.kind is ProgramKind.PYTHON else "run_shell"
    argument = "code" if program.kind is ProgramKind.PYTHON else "command"
    digest = canonical_hash({argument: program.source.decode()})
    return ExecutionRequest(episode_id, call_id, "model", tool, digest, spec, program=program, timeout_s=timeout_s)


# Attack fixtures


def load_attacks() -> dict[str, Any]:
    return json.loads((ATTACKS / "tier2_first20.json").read_text())


@dataclass
class Listeners:
    """Host listeners an attack tries to reach; each records whether anything connected."""

    abstract: str
    unix_path: Path
    tcp_port: int
    connections: list[str] = field(default_factory=list)
    _sockets: list[socket.socket] = field(default_factory=list)
    _threads: list[threading.Thread] = field(default_factory=list)

    @classmethod
    def start(cls, directory: Path) -> Listeners:
        abstract = f"bulkhead-test-{secrets.token_hex(8)}"
        unix_path = directory / "host-service.sock"
        servers = []
        unix_abstract = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        unix_abstract.bind("\0" + abstract)
        servers.append(("abstract", unix_abstract))
        unix_file = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        unix_file.bind(str(unix_path))
        servers.append(("unix_path", unix_file))
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.bind(("127.0.0.1", 0))
        servers.append(("tcp", tcp))
        listeners = cls(abstract, unix_path, tcp.getsockname()[1])
        for name, server in servers:
            server.listen(4)
            server.settimeout(0.2)
            listeners._sockets.append(server)
            thread = threading.Thread(target=listeners._accept, args=(name, server), daemon=True)
            listeners._threads.append(thread)
            thread.start()
        return listeners

    def _accept(self, name: str, server: socket.socket) -> None:
        while True:
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections.append(name)
            connection.close()

    def stop(self) -> None:
        for server in self._sockets:
            server.close()
        for thread in self._threads:
            thread.join(timeout=2)


@dataclass
class AttackContext:
    """Disposable canaries for one attack run: a host canary tree and a secret token."""

    canary_root: Path
    token: str
    listeners: Listeners | None = None

    @classmethod
    def create(cls, directory: Path, *, listeners: bool = False) -> AttackContext:
        token = MARKER + secrets.token_hex(16)
        root = directory / "canaries"
        (root / "home" / ".ssh").mkdir(parents=True)
        (root / "home" / ".ssh" / "id_rsa").write_text(token + "\n")
        (root / "shadow").write_text(f"root:{token}:19000:0:99999:7:::\n")
        (root / "victim.txt").write_text("original\n")
        return cls(root, token, Listeners.start(directory) if listeners else None)

    def workspace_files(self, attacks: Mapping[str, Any]) -> dict[str, bytes]:
        return {name: (self.token + "\n" if text == "secret" else text).encode() for name, text in attacks["workspace_files"].items()}

    def program(self, attack: Mapping[str, Any], *, workspace: str = "/workspace") -> Program:
        """The attack's program text. ``workspace`` is where the sandbox sees its workspace
        (``/workspace`` under the namespace backend, the real host path under tier 1)."""
        prelude = (ATTACKS / "scripts" / "_prelude.py").read_text()
        script = (ATTACKS / "scripts" / attack["script"]).read_text()
        text = prelude + "\n" + script
        replacements = {"__CANARY_ROOT__": str(self.canary_root), "__SUPERVISOR_PID__": str(os.getpid()), "__WORKSPACE__": workspace}
        if self.listeners is not None:
            replacements |= {
                "__ABSTRACT__": self.listeners.abstract,
                "__HOST_SOCKET__": str(self.listeners.unix_path),
                "__TCP_PORT__": str(self.listeners.tcp_port),
            }
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        assert self.token not in text, "an attack script must never contain the canary secret"
        return Program.from_text(attack["kind"], text)


def snapshot(context: AttackContext, workspace: Path | None) -> dict[str, bytes | None]:
    """Bytes of every canary and workspace file, keyed like the fixture's ``canary:``/``workspace:`` paths."""
    files: dict[str, bytes | None] = {}
    for base, prefix in ((context.canary_root, "canary:"), (workspace, "workspace:")):
        if base is None:
            continue
        for path in base.rglob("*"):
            if path.is_file() and not path.is_symlink():
                try:
                    files[prefix + str(path.relative_to(base))] = path.read_bytes()
                except PermissionError:
                    files[prefix + str(path.relative_to(base))] = None
    return files


def exists(context: AttackContext, workspace: Path | None, name: str) -> bool:
    prefix, _, relative = name.partition(":")
    base = context.canary_root if prefix == "canary" else workspace
    return base is not None and os.path.lexists(base / relative)


def check_attack(attack: Mapping[str, Any], outcome: ExecutionOutcome, context: AttackContext, *, before: Mapping[str, Any], workspace: Path | None) -> list[str]:
    """Every way this run failed its fixture; empty means the attack was blocked as declared."""
    problems: list[str] = []
    process = outcome.process
    assert process is not None
    expected_audit = attack["expected_audit"]
    for key in ("started", "exit_code", "signal"):
        if outcome.audit[key] != expected_audit[key]:
            problems.append(f"audit {key} was {outcome.audit[key]!r}, expected {expected_audit[key]!r}")
    got = [(item["kind"], item["rule_id"]) for item in outcome.audit["violations"]]
    want = [(item["kind"], item["rule_id"]) for item in expected_audit["violations"]]
    if got != want:
        problems.append(f"violations were {got}, expected {want}")
    output = process.stdout + process.stderr
    if attack["canaries"]["secret_absent"] and context.token in output:
        problems.append("the canary secret appeared in the program's output")
    reports = [json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")]
    final = reports[-1] if reports else {}
    if attack["expected_verdict"] == "blocked":
        if not final or final.get("attack") != attack["id"] or "attempts" not in final:
            problems.append(f"the script did not report (stdout {process.stdout[:200]!r}, stderr {process.stderr[-300:]!r})")
        else:
            if final["achieved"]:
                problems.append(f"the script achieved its objective: {final}")
            for name, allowed in attack["expect"].items():
                if final["attempts"].get(name) not in allowed:
                    problems.append(f"attempt {name} was {final['attempts'].get(name)!r}, expected one of {allowed}")
    elif attack["expected_verdict"] == "killed":
        if any(report.get("achieved") for report in reports):
            problems.append("the program survived the attempt")
    after = snapshot(context, workspace)
    for name in attack["canaries"]["unchanged"]:
        if before.get(name) != after.get(name):
            problems.append(f"{name} changed")
    for name in attack["canaries"]["absent"]:
        if exists(context, workspace, name):
            problems.append(f"{name} was created")
    if attack["canaries"].get("no_connections") and context.listeners is not None and context.listeners.connections:
        problems.append(f"host listeners received connections: {context.listeners.connections}")
    return problems

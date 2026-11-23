"""Tier 1: Landlock, seccomp, no_new_privs, and rlimits applied by a single-threaded helper before exec.

Labels: ``landlock-t1`` and, with a delegated cgroup parent, ``landlock-t1-cgroup``.
This is a real boundary but a different one from tier 2. What differs is
reported (``bulkhead.backends.profiles``), not hidden:

- Filesystem *access* is controlled, *visibility* is not. The program runs at
  the real host paths of its workspace; names outside its rules may still be
  observable, and a directory holding a denied entry cannot be listed (see
  ``bulkhead.landlock``). ``/proc`` is not granted, so other processes'
  environments and memory stay unreadable.
- No network at all: Landlock (ABI 4 or later) denies TCP bind and connect
  with no port rules, and tier-1 seccomp refuses ``socket()`` outright, since
  without a network namespace a Unix socket could reach host services.
  Network-enabled specs are refused; part 6's proxy tool is the only way out.
- Signals and abstract Unix sockets are scoped to the sandbox's domain on
  Landlock ABI 6 or later; on older kernels signals are not scoped, and the
  profile says so.
- Cleanup: without a cgroup, the process group is the handle. Tier-1 seccomp
  refuses ``setsid`` and ``setpgid``, so no process can leave the group that
  ``killpg`` reaches; with a cgroup parent, ``cgroup.kill`` is the authority.
  Without a cgroup there is no PID or memory ceiling (``RLIMIT_AS`` is a
  per-process backup only), so resource-exhaustion fixtures are not run
  against that profile.

The backend exists only when a probe report proves every tier-1 requirement
(and, for the cgroup profile, the cgroup capabilities). A request it cannot
enforce exactly (an unsupported Landlock right, a forbidden path, a symlink,
network access) is refused as a setup failure; nothing runs with fewer
restrictions than declared.
"""

from __future__ import annotations

import os
import platform
import threading
import time
from collections.abc import Mapping
from typing import Any

from ..cgroups import CgroupParent
from ..landlock import EXECUTE, MIN_ABI, READ_FILE, WRITE_FILE, LandlockError, Ruleset, abi_version, covers, plan_ruleset
from ..seccomp import T1_POLICY, CompiledFilter, SeccompUnavailable, compile_filter
from ..supervisor import ProcessOutcome, launcher_command
from ..types import TIER_REQUIREMENTS, ProbeReport, SandboxTier
from ..workspace import ViewError, WorkRoot, Workspace
from .base import BackendIdentity, ExecutionOutcome, ExecutionRequest, ProgramKind, build_outcome
from .bwrap import BackendUnavailable
from .common import run_governed

BACKEND_NAME = "landlock-t1"
BACKEND_NAME_CGROUP = "landlock-t1-cgroup"
BACKEND_VERSION = "5.0"
CGROUP_CAPABILITIES = ("cgroup.v2_mount", "cgroup.delegation", "cgroup.leaf", "cgroup.limits", "cgroup.kill")
# Device files every program may use; the rest of /dev stays inaccessible.
DEVICES = (
    ("/dev/null", READ_FILE | WRITE_FILE),
    ("/dev/zero", READ_FILE),
    ("/dev/full", READ_FILE | WRITE_FILE),
    ("/dev/urandom", READ_FILE),
    ("/dev/random", READ_FILE),
)


def missing_for_t1(report: ProbeReport, *, cgroup: bool) -> list[str]:
    names = list(TIER_REQUIREMENTS[SandboxTier.T1]) + (list(CGROUP_CAPABILITIES) if cgroup else [])
    missing = []
    for name in names:
        try:
            if not report.capability(name).ok:
                missing.append(name)
        except KeyError:
            missing.append(name)
    return missing


class LandlockBackend:
    """Runs allowed exec calls under tier 1. See the module docstring."""

    def __init__(
        self,
        report: ProbeReport,
        work_root: str | os.PathLike[str],
        *,
        python: str = "/usr/bin/python3",
        shell: str = "/bin/sh",
        cgroup_parent: CgroupParent | None = None,
        cpu_fraction: float = 1.0,
        kernel_abi: int | None = None,
    ) -> None:
        missing = missing_for_t1(report, cgroup=cgroup_parent is not None)
        if missing:
            raise BackendUnavailable(f"the probe did not prove {', '.join(missing)}; refusing to build tier 1", missing=missing)
        if report.host.architecture != platform.machine().lower():
            raise BackendUnavailable("the probe report was made on a different architecture", missing=("probe.host",))
        abi = report.landlock.abi or 0
        current = abi_version() if kernel_abi is None else kernel_abi
        if abi < MIN_ABI:
            raise BackendUnavailable(f"Landlock ABI {abi} cannot enforce read-only files (needs {MIN_ABI})", missing=("landlock.abi",))
        if current < abi:
            raise BackendUnavailable("the running kernel's Landlock ABI is older than the probe's; probe again", missing=("landlock.abi",))
        try:
            self.filter: CompiledFilter = compile_filter(T1_POLICY)
        except SeccompUnavailable as exc:
            raise BackendUnavailable(str(exc), missing=("seccomp.filter",)) from None
        self.abi = abi
        self.python = os.path.realpath(python)
        self.shell = os.path.realpath(shell)
        self.cgroup_parent = cgroup_parent
        self.cpu_fraction = cpu_fraction
        name = BACKEND_NAME_CGROUP if cgroup_parent is not None else BACKEND_NAME
        self.identity = BackendIdentity(name, f"{BACKEND_VERSION}+landlock.{abi}", SandboxTier.T1, report.probe_sha256)
        self.root = WorkRoot(work_root)
        self.home = os.path.realpath(os.path.expanduser("~"))
        self._workspaces: dict[str, Workspace] = {}
        self._lock = threading.Lock()
        if cgroup_parent is not None:
            cgroup_parent.reap_stale()  # survivors of an earlier supervisor crash

    @property
    def signals_scoped(self) -> bool:
        return self.abi >= 6

    # Workspaces

    def workspace(self, episode_id: str) -> Workspace:
        with self._lock:
            if episode_id not in self._workspaces:
                self._workspaces[episode_id] = Workspace.open(self.root, episode_id)
            return self._workspaces[episode_id]

    def prepare(self, episode_id: str, files: Mapping[str, bytes]) -> Workspace:
        with self._lock:
            workspace = Workspace.create(self.root, episode_id, files)
            self._workspaces[episode_id] = workspace
            return workspace

    def cleanup(self, episode_id: str) -> None:
        with self._lock:
            workspace = self._workspaces.pop(episode_id, None) or Workspace.open(self.root, episode_id)
        with workspace.lock:
            workspace.remove()

    def ruleset_for(self, request: ExecutionRequest, workspace: Workspace) -> Ruleset:
        """The Landlock ruleset for one call; raises ``LandlockError`` when the spec cannot be enforced."""
        assert request.spec.sandbox is not None
        for path in request.spec.sandbox.fs_write:
            location = workspace.host_path(path)
            if not location.exists():
                location.mkdir(mode=0o700, parents=True)
        ruleset = plan_ruleset(request.spec.sandbox, workspace.path, self.abi, home=self.home, work_root=self.root.path, extra=DEVICES)
        needed = [self.python] + ([self.shell] if request.program is not None and request.program.kind is ProgramKind.SHELL else [])
        for binary in needed:
            if not covers(ruleset, binary, EXECUTE | READ_FILE):
                raise LandlockError("the interpreter is not inside the sandbox's readable paths", code="bulkhead.view.interpreter")
        return ruleset

    # Execution

    def execute(self, request: ExecutionRequest, arguments: Mapping[str, Any]) -> ExecutionOutcome:
        began = time.monotonic()

        def refused(code: str) -> ExecutionOutcome:
            return build_outcome(request, self.identity, ProcessOutcome.not_started(code, elapsed_s=time.monotonic() - began))

        sandbox = request.spec.sandbox
        if sandbox is None:
            return refused("bulkhead.setup.no_sandbox")
        if request.program is None:
            return refused("bulkhead.setup.no_program")
        if sandbox.net.enabled:
            return refused("bulkhead.setup.network_unsupported")  # tier 1 has no direct network, ever
        try:
            workspace = self.workspace(request.episode_id)
        except (ViewError, OSError):
            return refused("bulkhead.setup.workspace_unsafe")
        if not workspace.lock.acquire(timeout=request.effective_timeout_s):
            return refused("bulkhead.setup.workspace_busy")
        try:
            if workspace.poisoned:
                return refused("bulkhead.setup.workspace_unsafe")
            try:
                ruleset = self.ruleset_for(request, workspace)
            except (LandlockError, ViewError) as exc:
                return refused(exc.code)
            except OSError:
                return refused("bulkhead.setup.view")
            out = [workspace.host_path(path) for path in sandbox.fs_write]
            env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": str(workspace.path),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "TMPDIR": str(out[0]) if out else str(workspace.path),
            }
            program = request.program
            observed = run_governed(
                request,
                lambda status_fd: launcher_command(self.python, status_fd, program, shell=self.shell),
                env=env,
                cwd=str(workspace.path),
                writable=out,
                cgroup_parent=self.cgroup_parent,
                ruleset=ruleset,
                seccomp=self.filter,
                address_space=self.cgroup_parent is None,
                cpu_fraction=self.cpu_fraction,
            )
            if observed.cleanup_status == "leaked":
                workspace.poisoned = True
        finally:
            workspace.lock.release()
        return build_outcome(request, self.identity, observed)


__all__ = ["BACKEND_NAME", "BACKEND_NAME_CGROUP", "DEVICES", "LandlockBackend", "missing_for_t1"]

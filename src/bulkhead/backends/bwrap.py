"""Tier 2: bubblewrap namespaces, a minimal filesystem view, seccomp, and a delegated cgroup leaf per call.

Two labels, never confused:

- ``bwrap-t2`` is the complete tier-2 profile (bubblewrap + seccomp + cgroup).
  Each call runs in a fresh delegated cgroup v2 leaf with memory, PID, and
  CPU ceilings, entered before anything untrusted runs (the helper waits at a
  start barrier until the supervisor has moved it into the leaf), and
  ``cgroup.kill`` ends the whole call tree afterwards. It requires every T2
  capability in the probe report and a configured cgroup parent.
- ``bwrap-namespace-phase`` (``namespace_phase=True``) is part 4's
  namespace/seccomp profile without cgroups: cleanup reaches the process
  group, session, and PID namespace only, and there are no memory or PID
  ceilings. It exists for hosts and tests without delegation and is never
  reported as full T2.

Each allowed call runs as

    helper (no_new_privs, rlimits, start barrier)
      -> bwrap <all namespaces fresh, --die-with-parent, --new-session, --cap-drop ALL>
               --hostname bulkhead --proc /proc --dev /dev [--size N] --tmpfs /tmp
               <the minimal view: read-only binds, workspace-only writable binds, masks>
               --chdir /workspace --seccomp FD
               -- python -I -S -c <launcher> STATUS_FD KIND <python -I - | sh -c>

with fixed argv lists (no shell, no interpolation), program text on stdin, a
minimal environment (``SANDBOX_ENV``; nothing inherited, so no tokens, keys,
cloud credentials, or SSH agent), no inherited descriptors beyond
stdin/stdout/stderr, the status pipe and the seccomp memfd, and bubblewrap's
always-on no_new_privs. bubblewrap installs the seccomp filter just before
exec; no Python ``preexec_fn`` runs in the supervisor.

Disk: ``/tmp`` is a sized tmpfs where bubblewrap supports ``--size`` (a hard
limit, also charged to the leaf's memory); the writable workspace is covered by
``RLIMIT_FSIZE`` per file and a monitored soft limit on its total (the
governor kills the run when it crosses ``disk_mb``).

The backend exists only when a part 1 probe report proves every feature its
profile uses on this machine with this bubblewrap. A missing feature raises
``BackendUnavailable`` at construction; a sandbox that fails to build for one
call is a run that never started (a ``sandbox_violation`` for the model).
Neither ever falls back to running on the host.
"""

from __future__ import annotations

import os
import platform
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..capabilities import BWRAP_NAMESPACE_FLAGS, ChildRun, run_bounded
from ..cgroups import CgroupParent
from ..errors import BulkheadError
from ..seccomp import CompiledFilter, SeccompUnavailable, compile_filter
from ..supervisor import ProcessOutcome, launcher_command
from ..types import TIER_REQUIREMENTS, ProbeReport, SandboxTier
from ..workspace import WORKSPACE_MOUNT, View, ViewError, WorkRoot, Workspace, ensure_visible, plan_view
from .base import BackendIdentity, ExecutionOutcome, ExecutionRequest, Program, ProgramKind, build_outcome
from .common import helper_failure, run_governed

BACKEND_NAME = "bwrap-namespace-phase"
BACKEND_NAME_T2 = "bwrap-t2"
BACKEND_VERSION = "5.0"
# The namespace phase needs bubblewrap namespaces, mounts, and network isolation, plus seccomp.
# Full T2 needs every T2 requirement in the probe report, cgroup delegation included.
REQUIRED_CAPABILITIES = (
    "bwrap.binary",
    "bwrap.namespaces",
    "bwrap.mounts",
    "bwrap.netns",
    "libseccomp.library",
    "seccomp.filter",
)
MIN_BWRAP_VERSION = (0, 4, 0)
HOSTNAME = "bulkhead"
# The sandbox's whole environment. Part 6 adds explicit proxy variables; nothing is inherited.
SANDBOX_ENV: Mapping[str, str] = MappingProxyType({
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": WORKSPACE_MOUNT,
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TMPDIR": "/tmp",
})
_VERSION = re.compile(r"bubblewrap\s+(\d+)\.(\d+)(?:\.(\d+))?")


class BackendUnavailable(BulkheadError, RuntimeError):
    """The backend cannot be built here. ``missing`` names what the probe did not prove. Never a fallback."""

    def __init__(self, message: str, *, missing: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.missing = tuple(missing)


def missing_capabilities(report: ProbeReport, *, namespace_phase: bool = True) -> list[str]:
    """The capabilities a profile requires that a probe report does not show as ``ok``."""
    names = REQUIRED_CAPABILITIES if namespace_phase else tuple(dict.fromkeys((*REQUIRED_CAPABILITIES, *TIER_REQUIREMENTS[SandboxTier.T2])))
    missing = []
    for name in names:
        try:
            if not report.capability(name).ok:
                missing.append(name)
        except KeyError:
            missing.append(name)
    return missing


@dataclass(frozen=True)
class BwrapConfig:
    """Host paths the backend uses (never model-supplied) and what this bubblewrap supports."""

    bwrap: str
    python: str = "/usr/bin/python3"
    shell: str = "/bin/sh"
    tmpfs_size: bool = False  # whether ``--size`` is available to bound /tmp

    def __post_init__(self) -> None:
        for name in ("bwrap", "python", "shell"):
            if not os.path.isabs(getattr(self, name)):
                raise ValueError(f"{name} must be an absolute path")


def build_argv(
    config: BwrapConfig,
    view: View,
    program: Program,
    *,
    interpreter: str,
    shell: str,
    seccomp_fd: int,
    status_fd: int,
    tmp_bytes: int,
    mask_dir: str,
    mask_file: str,
) -> list[str]:
    """The complete bubblewrap argv for one call. Pure: builds a list, runs nothing.

    ``interpreter`` and ``shell`` are sandbox paths (checked visible by the
    caller). The program text is not part of the argv.
    """
    argv = [config.bwrap, *BWRAP_NAMESPACE_FLAGS, "--hostname", HOSTNAME, "--proc", "/proc", "--dev", "/dev"]
    if config.tmpfs_size:
        argv += ["--size", str(int(tmp_bytes))]
    argv += ["--tmpfs", "/tmp"]
    for entry in view.entries:
        if entry.op == "ro":
            argv += ["--ro-bind", str(entry.source), entry.dest]
        elif entry.op == "rw":
            argv += ["--bind", str(entry.source), entry.dest]
        elif entry.op == "symlink":
            argv += ["--symlink", str(entry.source), entry.dest]
        elif entry.op == "mask_dir":
            argv += ["--ro-bind", mask_dir, entry.dest]
        else:
            argv += ["--ro-bind", mask_file, entry.dest]
    workdir = WORKSPACE_MOUNT if view.visible(WORKSPACE_MOUNT) else "/tmp"
    argv += ["--chdir", workdir, "--seccomp", str(seccomp_fd), "--"]
    argv += launcher_command(interpreter, status_fd, program, shell=shell)
    return argv


def classify_setup_failure(observed: ProcessOutcome) -> str:
    """A stable code for a run that never started, from bubblewrap's first error line (which is not recorded)."""
    if observed.timed_out:
        return "bulkhead.setup.timeout"
    text = observed.stderr.decode("utf-8", "replace").lower()
    if "unknown option" in text or "unrecognized option" in text:
        return "bulkhead.setup.unsupported_option"
    if "namespace" in text and any(marker in text for marker in ("no permissions", "operation not permitted", "permission denied")):
        return "bulkhead.setup.userns_denied"
    if "seccomp" in text:
        return "bulkhead.setup.seccomp"
    if any(marker in text for marker in ("can't mount", "can't bind", "can't create", "can't find source", "mount")):
        return "bulkhead.setup.mount"
    if "execvp" in text:
        return "bulkhead.setup.exec"
    return observed.setup_error or "bulkhead.setup.failed"


class BwrapBackend:
    """Runs allowed exec calls under tier 2 (``bwrap-t2``) or its namespace phase. See the module docstring."""

    def __init__(
        self,
        report: ProbeReport,
        work_root: str | os.PathLike[str],
        *,
        bwrap: str | None = None,
        python: str = "/usr/bin/python3",
        shell: str = "/bin/sh",
        cgroup_parent: CgroupParent | None = None,
        namespace_phase: bool = False,
        cpu_fraction: float = 1.0,
        runner: Callable[..., ChildRun] = run_bounded,
    ) -> None:
        name = BACKEND_NAME if namespace_phase else BACKEND_NAME_T2
        missing = missing_capabilities(report, namespace_phase=namespace_phase)
        if not namespace_phase and cgroup_parent is None:
            missing.append("cgroup.parent")
        if missing:
            raise BackendUnavailable(f"the probe did not prove {', '.join(missing)}; refusing to build {name}", missing=missing)
        if report.host.architecture != platform.machine().lower():
            raise BackendUnavailable("the probe report was made on a different architecture", missing=("probe.host",))
        binary = bwrap or (os.path.expanduser(report.tools.bwrap_path) if report.tools.bwrap_path else None)
        if not binary or not os.path.isabs(binary):
            raise BackendUnavailable("no absolute bubblewrap path", missing=("bwrap.binary",))
        version = self._version(binary, runner)
        if version is None or version < MIN_BWRAP_VERSION:
            raise BackendUnavailable(f"bubblewrap {version} is older than {MIN_BWRAP_VERSION}", missing=("bwrap.binary",))
        if report.tools.bwrap_version is None or _parse(f"bubblewrap {report.tools.bwrap_version}") != version:
            raise BackendUnavailable("the probe report describes a different bubblewrap; probe again", missing=("bwrap.binary",))
        help_text = runner([binary, "--help"], timeout_s=10.0)
        try:
            self.filter: CompiledFilter = compile_filter()
        except SeccompUnavailable as exc:
            raise BackendUnavailable(str(exc), missing=("seccomp.filter",)) from None
        self.config = BwrapConfig(binary, python, shell, tmpfs_size="--size" in (help_text.stdout + help_text.stderr))
        self.identity = BackendIdentity(name, f"{BACKEND_VERSION}+bwrap.{'.'.join(map(str, version))}", SandboxTier.T2, report.probe_sha256)
        self.cgroup_parent = None if namespace_phase else cgroup_parent
        self.cpu_fraction = cpu_fraction
        self.root = WorkRoot(work_root)
        self._workspaces: dict[str, Workspace] = {}
        self._lock = threading.Lock()
        if self.cgroup_parent is not None:
            self.cgroup_parent.reap_stale()  # survivors of an earlier supervisor crash

    @staticmethod
    def _version(binary: str, runner: Callable[..., ChildRun]) -> tuple[int, int, int] | None:
        run = runner([binary, "--version"], timeout_s=10.0)
        return None if run.returncode != 0 else _parse(run.stdout)

    # Workspaces

    def workspace(self, episode_id: str) -> Workspace:
        """The episode's persistent workspace, created on first use."""
        with self._lock:
            if episode_id not in self._workspaces:
                self._workspaces[episode_id] = Workspace.open(self.root, episode_id)
            return self._workspaces[episode_id]

    def prepare(self, episode_id: str, files: Mapping[str, bytes]) -> Workspace:
        """Create the episode's workspace with copies of ``files`` before its first call."""
        with self._lock:
            workspace = Workspace.create(self.root, episode_id, files)
            self._workspaces[episode_id] = workspace
            return workspace

    def cleanup(self, episode_id: str) -> None:
        """Remove the episode's workspace."""
        with self._lock:
            workspace = self._workspaces.pop(episode_id, None) or Workspace.open(self.root, episode_id)
        with workspace.lock:
            workspace.remove()

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
            return refused("bulkhead.setup.network_unsupported")  # egress arrives with the part 6 proxy
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
                view = plan_view(sandbox, workspace)
                interpreter = ensure_visible(view, self.config.python)
                shell = ensure_visible(view, self.config.shell) if request.program.kind is ProgramKind.SHELL else self.config.shell
            except ViewError as exc:
                return refused(exc.code)
            except OSError:
                return refused("bulkhead.setup.view")
            seccomp_fd = self.filter.memfd()
            program = request.program
            writable = [Path(entry.source) for entry in view.entries if entry.op == "rw" and entry.source]
            try:
                observed = run_governed(
                    request,
                    lambda status_fd: build_argv(
                        self.config, view, program,
                        interpreter=interpreter, shell=shell, seccomp_fd=seccomp_fd, status_fd=status_fd,
                        tmp_bytes=int(sandbox.limits.disk_mb) * 1024 * 1024,
                        mask_dir=str(self.root.mask_dir), mask_file=str(self.root.mask_file),
                    ),
                    env=SANDBOX_ENV,
                    cwd="/",
                    writable=writable,
                    cgroup_parent=self.cgroup_parent,
                    keep_fds=(seccomp_fd,),
                    cpu_fraction=self.cpu_fraction,
                )
            finally:
                os.close(seccomp_fd)
            if observed.cleanup_status == "leaked":
                workspace.poisoned = True
            if not observed.started and helper_failure(observed) is None:
                observed = replace(observed, setup_error=classify_setup_failure(observed))
        finally:
            workspace.lock.release()
        return build_outcome(request, self.identity, observed)


def _parse(text: str) -> tuple[int, int, int] | None:
    match = _VERSION.search(text)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


__all__ = [
    "BACKEND_NAME",
    "BACKEND_NAME_T2",
    "BACKEND_VERSION",
    "MIN_BWRAP_VERSION",
    "REQUIRED_CAPABILITIES",
    "SANDBOX_ENV",
    "BackendUnavailable",
    "BwrapBackend",
    "BwrapConfig",
    "build_argv",
    "classify_setup_failure",
    "missing_capabilities",
]

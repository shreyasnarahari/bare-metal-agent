"""The governed run both live backends share: cgroup leaf, governor, helper, supervisor.

``run_governed`` takes a backend's target argv (bubblewrap's, or the launcher
itself on tier 1) and runs it the same way for every tier:

1. With a delegated cgroup parent, a fresh leaf with memory, PID, and CPU
   ceilings is created before anything is spawned. A failure to create it is
   a setup failure; the call never runs outside its leaf.
2. The supervisor spawns ``bulkhead.helper`` with a config memfd. The helper
   closes stray descriptors, sets no_new_privs, applies Landlock (tier 1),
   rlimits, and seccomp (tier 1), then waits at the start barrier until the
   supervisor has admitted it to the leaf, and only then execs the target.
3. While it runs, the governor polls the disk ceiling (a monitored soft limit
   over the writable workspace) and the leaf's total CPU time.
4. Afterwards ``cgroup.kill`` (and the process group) end every survivor, and
   the leaf is removed; a leaf that cannot be emptied is ``leaked``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from ..cgroups import CgroupError, CgroupLimits, CgroupParent
from ..governor import CallGovernor
from ..helper import config_memfd, helper_command
from ..landlock import Ruleset
from ..rlimits import RlimitPlan
from ..seccomp import CompiledFilter
from ..supervisor import CaptureLimits, ProcessOutcome, supervise
from .base import ExecutionRequest

_HELPER_ERROR = re.compile(rb"bulkhead-helper: ([a-z_]{1,40})")


def helper_failure(observed: ProcessOutcome) -> str | None:
    """The stable code a helper reported before exiting without exec, if it did."""
    match = _HELPER_ERROR.search(observed.stderr)
    return f"bulkhead.setup.{match.group(1).decode()}" if match else None


def run_governed(
    request: ExecutionRequest,
    target_for: Callable[[int], list[str]],
    *,
    env: Mapping[str, str],
    cwd: str,
    writable: Sequence[Path],
    cgroup_parent: CgroupParent | None,
    ruleset: Ruleset | None = None,
    seccomp: CompiledFilter | None = None,
    keep_fds: Sequence[int] = (),
    address_space: bool = False,
    cpu_fraction: float = 1.0,
) -> ProcessOutcome:
    """Run ``target_for(status_fd)`` through the helper under the request's limits (see the module docstring)."""
    sandbox = request.spec.sandbox
    assert sandbox is not None and request.program is not None
    limits = sandbox.limits
    leaf = None
    if cgroup_parent is not None:
        try:
            leaf = cgroup_parent.create(
                CgroupParent.leaf_name(request.episode_id, request.tool_call_id),
                CgroupLimits.from_limits(limits, cpu_fraction=cpu_fraction),
            )
        except CgroupError as exc:
            return ProcessOutcome.not_started(exc.code)
    governor = CallGovernor(
        leaf=leaf,
        writable=writable,
        disk_bytes=int(limits.disk_mb) * 1024 * 1024,
        cpu_s=float(limits.cpu_s) if leaf is not None else None,
    )
    plan = RlimitPlan.from_limits(limits, address_space=address_space)
    seccomp_fd = seccomp.memfd() if seccomp is not None else None
    passed: list[int] = list(keep_fds) + ([seccomp_fd] if seccomp_fd is not None else [])
    created: list[int] = []

    def argv_for(status_fd: int, barrier_fd: int | None = None) -> list[str]:
        config = {
            "status_fd": status_fd,
            "barrier_fd": barrier_fd,
            "seccomp_fd": seccomp_fd,
            "keep_fds": list(keep_fds),
            "landlock": ruleset.as_dict() if ruleset is not None else None,
            "rlimits": plan.as_dict(),
            "target": target_for(status_fd),
            "supervisor_pid": os.getpid(),
        }
        fd = config_memfd(config)
        created.append(fd)
        passed.append(fd)  # supervise reads pass_fds after argv_for returns
        return helper_command(fd)

    try:
        observed = supervise(
            argv_for,
            stdin=request.program.source,
            env=env,
            timeout_s=request.effective_timeout_s,
            limits=CaptureLimits.from_limits(limits),
            pass_fds=passed,
            cwd=cwd,
            gate=governor,
        )
    finally:
        for fd in created + ([seccomp_fd] if seccomp_fd is not None else []):
            try:
                os.close(fd)
            except OSError:
                pass
    if not observed.started and (code := helper_failure(observed)) is not None:
        observed = replace(observed, setup_error=code)
    return observed


__all__ = ["helper_failure", "run_governed"]

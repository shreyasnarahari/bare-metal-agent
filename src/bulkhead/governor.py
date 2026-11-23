"""Per-call resource governance hooks the supervisor calls while a sandboxed program runs.

``CallGovernor`` combines what one call has:

- a cgroup ``Leaf`` (when a delegated parent is configured): ``admit`` places
  the helper in it at the start barrier, ``kill`` is ``cgroup.kill``, and
  ``populated`` says whether any process is left. With a leaf, ``cgroup.kill``
  is the tree-cleanup authority;
- a disk ceiling over the writable workspace directories, measured by polling
  allocated blocks. This is a *monitored soft limit*: a fast writer can
  overshoot by what it writes between polls. ``RLIMIT_FSIZE`` caps any single
  file hard, and the namespace backend's ``/tmp`` is a sized tmpfs where
  bubblewrap supports ``--size``. A walk that meets more than ``MAX_ENTRIES``
  entries counts as over the limit (a many-file fill);
- a total CPU-time ceiling read from the leaf's ``cpu.stat``, which covers the
  whole tree (``RLIMIT_CPU`` is per process).

``poll`` returns the name of a ceiling that was crossed (``disk`` or ``cpu``);
the supervisor then kills the run and the result is ``resource_exhausted``.
Memory and PID ceilings need no polling: the kernel enforces them, and the
leaf's ``memory.events`` and ``pids.events`` show it afterwards.
"""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .cgroups import CgroupError, Leaf

MAX_ENTRIES = 100_000
POLL_INTERVAL_S = 0.1


def allocated_bytes(paths: Sequence[Path], *, max_entries: int = MAX_ENTRIES) -> float:
    """Bytes allocated under ``paths`` (never following symlinks), or infinity past ``max_entries`` entries."""
    total = 0
    seen = 0
    stack = [os.fspath(path) for path in paths]
    while stack:
        current = stack.pop()
        try:
            entries = os.scandir(current)
        except OSError:
            continue
        with entries:
            for entry in entries:
                seen += 1
                if seen > max_entries:
                    return float("inf")
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                total += info.st_blocks * 512
                if stat.S_ISDIR(info.st_mode):
                    stack.append(entry.path)
    return total


class CallGovernor:
    """The supervisor's hooks for one call (see the module docstring)."""

    def __init__(
        self,
        *,
        leaf: Leaf | None = None,
        writable: Sequence[Path] = (),
        disk_bytes: int | None = None,
        cpu_s: float | None = None,
        interval_s: float = POLL_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.leaf = leaf
        self.writable = tuple(Path(path) for path in writable)
        self.disk_bytes = disk_bytes
        self.cpu_s = cpu_s
        self.interval_s = interval_s
        self.clock = clock
        self._next = 0.0
        self.disk_peak = 0.0
        self.cleanup: str | None = None

    @property
    def needs_barrier(self) -> bool:
        """Whether the helper must wait until its pid is in the leaf."""
        return self.leaf is not None

    @property
    def tree_kill(self) -> bool:
        return self.leaf is not None

    def admit(self, pid: int) -> None:
        if self.leaf is not None:
            self.leaf.admit(pid)

    def poll(self) -> str | None:
        now = self.clock()
        if now < self._next:
            return None
        self._next = now + self.interval_s
        if self.disk_bytes is not None and self.writable:
            used = allocated_bytes(self.writable)
            self.disk_peak = max(self.disk_peak, used)
            if used > self.disk_bytes:
                return "disk"
        if self.leaf is not None and self.cpu_s is not None:
            usage = self.leaf.usage().get("cpu_s")
            if usage is not None and usage > self.cpu_s:
                return "cpu"
        return None

    def kill(self) -> None:
        if self.leaf is not None:
            self.leaf.kill()

    def populated(self) -> bool:
        return self.leaf.populated() if self.leaf is not None else False

    def usage(self) -> dict[str, Any]:
        usage: dict[str, Any] = {}
        if self.leaf is not None:
            usage.update(self.leaf.usage())
        if self.disk_bytes is not None and self.writable:
            self.disk_peak = max(self.disk_peak, allocated_bytes(self.writable))
            if self.disk_peak != float("inf"):
                usage["disk_peak_kb"] = int(self.disk_peak) // 1024
        return usage

    def close(self) -> str:
        """Remove the leaf after the run: ``clean``, ``killed`` (something was still in it), or ``leaked``."""
        if self.leaf is None:
            return "clean"
        try:
            self.cleanup = self.leaf.destroy()
        except CgroupError:
            self.cleanup = "leaked"
        return self.cleanup


__all__ = ["MAX_ENTRIES", "CallGovernor", "allocated_bytes"]

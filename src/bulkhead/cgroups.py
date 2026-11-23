"""Delegated cgroup v2 leaves: one per call, limits before exec, ``cgroup.kill`` for cleanup, usage after.

The parent is always configured explicitly (``BULKHEAD_CGROUP_PARENT`` or a
constructor argument) and must be a delegated cgroup beneath the cgroup2 mount;
the mount root is never chosen automatically and no unrelated cgroup is
touched. Only the ``cpu``, ``memory``, and ``pids`` controllers are enabled,
and only in the parent's ``cgroup.subtree_control``.

Per call:

1. ``CgroupParent.create`` makes a uniquely named leaf (``bh-`` plus a hash of
   the episode, call, and a nonce; no task text) and ``Leaf.apply`` writes
   ``memory.max``, ``memory.swap.max`` (when the file exists; otherwise its
   absence is recorded), ``memory.oom.group``, ``pids.max``, and ``cpu.max``,
   reading each back so a partial or ignored write is an error.
2. The helper stops at its start barrier; ``Leaf.admit`` writes its pid to
   ``cgroup.procs`` and checks ``/proc/<pid>/cgroup`` before the barrier is
   released, so nothing forks outside the leaf.
3. Afterwards (normal exit, timeout, cancellation, or supervisor failure)
   ``Leaf.destroy`` writes ``1`` to ``cgroup.kill``, waits for the leaf to be
   unpopulated, and removes it. ``cgroup.kill`` is the tree-cleanup authority.
   A leaf that cannot be emptied is reported ``leaked`` and left for
   ``CgroupParent.reap_stale``, never silently ignored.

Usage (``cpu.stat``, ``memory.peak``, ``memory.events``, ``pids.peak``,
``pids.events``) is read after the run. A file the kernel does not provide is
omitted and named in ``unavailable``, never replaced by a fabricated zero.

Writing ``cgroup.procs`` needs write access to the common ancestor of the
source and destination cgroups, so the supervisor itself must run inside the
delegated subtree (see the README).

Every filesystem operation goes through a ``CgroupFS``, so unit tests use an
in-memory fake and never touch real system cgroups.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import BulkheadError
from .manifest import Limits

CGROUP_MOUNT = Path("/sys/fs/cgroup")
CONTROLLERS = ("cpu", "memory", "pids")
LEAF_PREFIX = "bh-"
_LEAF = re.compile(r"bh-[0-9a-f]{16}")
DEFAULT_CPU_PERIOD_US = 100_000
KILL_WAIT_S = 5.0


class CgroupError(BulkheadError, RuntimeError):
    """A cgroup step failed. ``code`` is a stable rule id; the call must not run (or must be stopped)."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class CgroupFS(Protocol):
    def read(self, path: Path) -> str: ...
    def write(self, path: Path, text: str) -> None: ...
    def exists(self, path: Path) -> bool: ...
    def mkdir(self, path: Path) -> None: ...
    def rmdir(self, path: Path) -> None: ...
    def listdir(self, path: Path) -> list[str]: ...
    def proc_cgroup(self, pid: int) -> str: ...


class SystemCgroupFS:
    """The real cgroup2 filesystem."""

    def read(self, path: Path) -> str:
        with open(path, encoding="ascii", errors="replace") as handle:
            return handle.read(65536)

    def write(self, path: Path, text: str) -> None:
        data = text.encode("ascii")
        fd = os.open(path, os.O_WRONLY | os.O_CLOEXEC)
        try:
            written = os.write(fd, data)
        finally:
            os.close(fd)
        if written != len(data):
            raise CgroupError(f"partial write to {path.name}", code="bulkhead.cgroup.partial_write")

    def exists(self, path: Path) -> bool:
        return os.path.exists(path)

    def mkdir(self, path: Path) -> None:
        os.mkdir(path, 0o755)

    def rmdir(self, path: Path) -> None:
        os.rmdir(path)

    def listdir(self, path: Path) -> list[str]:
        return os.listdir(path)

    def proc_cgroup(self, pid: int) -> str:
        with open(f"/proc/{pid}/cgroup", encoding="ascii", errors="replace") as handle:
            return handle.read(4096)


@dataclass(frozen=True)
class CgroupLimits:
    """What one leaf enforces: memory and PID ceilings and a CPU quota (``cpu_quota_us`` per ``cpu_period_us``)."""

    memory_bytes: int
    pids: int
    cpu_quota_us: int
    cpu_period_us: int = DEFAULT_CPU_PERIOD_US

    def __post_init__(self) -> None:
        if type(self.memory_bytes) is not int or self.memory_bytes < 4 * 1024 * 1024:
            raise ValueError("memory_bytes must be an integer of at least 4 MiB")
        if type(self.pids) is not int or self.pids < 1:
            raise ValueError("pids must be a positive integer")
        if not 1000 <= self.cpu_quota_us <= 100 * self.cpu_period_us or not 1000 <= self.cpu_period_us <= 1_000_000:
            raise ValueError("cpu quota/period out of range")

    @classmethod
    def from_limits(cls, limits: Limits, *, cpu_fraction: float = 1.0) -> CgroupLimits:
        """From a sandbox's limits; the CPU quota is ``cpu_fraction`` of one CPU (``cpu_s`` is a total, enforced separately)."""
        return cls(
            memory_bytes=int(limits.mem_mb) * 1024 * 1024,
            pids=int(limits.pids),
            cpu_quota_us=max(1000, int(cpu_fraction * DEFAULT_CPU_PERIOD_US)),
        )

    def required(self) -> list[tuple[str, str]]:
        return [("memory.max", str(self.memory_bytes)), ("pids.max", str(self.pids)), ("cpu.max", f"{self.cpu_quota_us} {self.cpu_period_us}")]

    def optional(self) -> list[tuple[str, str]]:
        # No swap beyond the memory ceiling, and an OOM kill takes the whole call tree.
        return [("memory.swap.max", "0"), ("memory.oom.group", "1")]

    def as_dict(self) -> dict[str, int]:
        return {"memory_bytes": self.memory_bytes, "pids": self.pids, "cpu_quota_us": self.cpu_quota_us, "cpu_period_us": self.cpu_period_us}


def _keyed(text: str) -> dict[str, int]:
    """``key value`` lines (cgroup.events, memory.events, cpu.stat, pids.events); malformed lines are skipped."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            values[parts[0]] = int(parts[1])
    return values


class CgroupParent:
    """An explicitly configured, delegated cgroup that holds this backend's per-call leaves."""

    def __init__(self, path: str | os.PathLike[str], *, fs: CgroupFS | None = None, mount: Path = CGROUP_MOUNT) -> None:
        self.fs = fs or SystemCgroupFS()
        self.mount = Path(os.path.normpath(mount))
        self.path = Path(os.path.normpath(os.fspath(path)))
        if not self.path.is_absolute() or self.path == self.mount or self.mount not in self.path.parents:
            raise CgroupError("the parent must be a delegated cgroup beneath the cgroup2 mount, never its root", code="bulkhead.cgroup.parent")
        if not self.fs.exists(self.path / "cgroup.procs"):
            raise CgroupError("the parent is not a cgroup", code="bulkhead.cgroup.parent")
        try:
            available = set(self.fs.read(self.path / "cgroup.controllers").split())
            enabled = set(self.fs.read(self.path / "cgroup.subtree_control").split())
        except OSError as exc:
            raise CgroupError(f"cannot read the parent's controllers: {exc.strerror}", code="bulkhead.cgroup.parent") from None
        missing = [name for name in CONTROLLERS if name not in available]
        if missing:
            raise CgroupError(f"the parent lacks the {', '.join(missing)} controller(s)", code="bulkhead.cgroup.controllers")
        to_enable = [name for name in CONTROLLERS if name not in enabled]
        if to_enable:
            try:
                self.fs.write(self.path / "cgroup.subtree_control", " ".join(f"+{name}" for name in to_enable))
                enabled = set(self.fs.read(self.path / "cgroup.subtree_control").split())
            except OSError as exc:
                raise CgroupError(f"cannot enable controllers in the parent: {exc.strerror}", code="bulkhead.cgroup.controllers") from None
            if any(name not in enabled for name in CONTROLLERS):
                raise CgroupError("the parent did not enable the required controllers", code="bulkhead.cgroup.controllers")
        self.relative = "/" + str(self.path.relative_to(self.mount))

    @staticmethod
    def leaf_name(episode_id: str, tool_call_id: str) -> str:
        """A unique leaf name that reveals nothing about the task."""
        digest = hashlib.sha256(f"{episode_id}/{tool_call_id}/{secrets.token_hex(8)}".encode()).hexdigest()
        return LEAF_PREFIX + digest[:16]

    def create(self, name: str, limits: CgroupLimits) -> Leaf:
        """Make the leaf and apply its limits; a stale leaf of the same name is destroyed first."""
        if not _LEAF.fullmatch(name):
            raise CgroupError("bad leaf name", code="bulkhead.cgroup.leaf")
        leaf = Leaf(self, name)
        if self.fs.exists(leaf.path):
            if leaf.destroy() == "leaked":
                raise CgroupError("a stale leaf of the same name could not be emptied", code="bulkhead.cgroup.stale")
        try:
            self.fs.mkdir(leaf.path)
        except OSError as exc:
            raise CgroupError(f"cannot create the leaf: {exc.strerror}", code="bulkhead.cgroup.leaf") from None
        try:
            leaf.apply(limits)
        except BaseException:
            leaf.destroy()
            raise
        return leaf

    def stale(self) -> list[str]:
        """Leaves this backend created that still exist (after a supervisor crash, say)."""
        try:
            return sorted(name for name in self.fs.listdir(self.path) if _LEAF.fullmatch(name))
        except OSError:
            return []

    def reap_stale(self) -> dict[str, str]:
        """Kill and remove every stale leaf; returns each one's cleanup status."""
        return {name: Leaf(self, name).destroy() for name in self.stale()}


class Leaf:
    """One call's cgroup."""

    def __init__(self, parent: CgroupParent, name: str) -> None:
        self.parent = parent
        self.fs = parent.fs
        self.name = name
        self.path = parent.path / name
        self.relative = f"{parent.relative}/{name}"
        self.unavailable: list[str] = []
        self.limits: CgroupLimits | None = None

    def apply(self, limits: CgroupLimits) -> None:
        """Write and read back every limit. A required limit that does not stick is an error."""
        for name, value in limits.required():
            self._write_verified(name, value, required=True)
        for name, value in limits.optional():
            if self.fs.exists(self.path / name):
                self._write_verified(name, value, required=False)
            else:
                self.unavailable.append(name)
        self.limits = limits

    def _write_verified(self, name: str, value: str, *, required: bool) -> None:
        try:
            self.fs.write(self.path / name, value)
            stored = self.fs.read(self.path / name).strip()
        except CgroupError:
            raise
        except OSError as exc:
            if not required:
                self.unavailable.append(name)
                return
            raise CgroupError(f"cannot write {name}: {exc.strerror}", code="bulkhead.cgroup.limit_write") from None
        if stored != value:
            raise CgroupError(f"{name} reads back {stored!r}, not {value!r}", code="bulkhead.cgroup.limit_write")

    def admit(self, pid: int) -> None:
        """Move ``pid`` into the leaf and confirm the kernel agrees, before it may run anything untrusted."""
        try:
            self.fs.write(self.path / "cgroup.procs", str(pid))
            membership = self.fs.proc_cgroup(pid)
        except CgroupError:
            raise
        except OSError as exc:
            raise CgroupError(f"cannot move the process into the leaf: {exc.strerror}", code="bulkhead.cgroup.admit") from None
        if f"0::{self.relative}" not in membership.splitlines():
            raise CgroupError("the process is not in the leaf after the move", code="bulkhead.cgroup.admit")

    def populated(self) -> bool:
        """Whether any process remains. Unreadable or malformed counts as populated (the safe answer)."""
        try:
            events = _keyed(self.fs.read(self.path / "cgroup.events"))
        except OSError:
            return self.fs.exists(self.path)
        return events.get("populated", 1) != 0

    def kill(self) -> None:
        """SIGKILL every process in the leaf, however it forked or regrouped."""
        try:
            self.fs.write(self.path / "cgroup.kill", "1")
        except FileNotFoundError:
            if self.fs.exists(self.path):
                raise CgroupError("this kernel has no cgroup.kill", code="bulkhead.cgroup.kill_unavailable") from None
        except OSError as exc:
            raise CgroupError(f"cgroup.kill failed: {exc.strerror}", code="bulkhead.cgroup.kill") from None

    def wait_empty(self, timeout_s: float = KILL_WAIT_S, *, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> bool:
        give_up = clock() + timeout_s
        while self.populated():
            if clock() >= give_up:
                return False
            sleep(0.01)
        return True

    def usage(self) -> dict[str, Any]:
        """Bounded usage evidence; anything the kernel does not provide is omitted and listed in ``unavailable``."""
        usage: dict[str, Any] = {}

        def read(name: str) -> str | None:
            try:
                return self.fs.read(self.path / name)
            except OSError:
                self.unavailable.append(name)
                return None

        if (text := read("cpu.stat")) is not None:
            stats = _keyed(text)
            if "usage_usec" in stats:
                usage["cpu_s"] = round(stats["usage_usec"] / 1e6, 6)
            if "throttled_usec" in stats:
                usage["cpu_throttled_s"] = round(stats["throttled_usec"] / 1e6, 6)
        if (text := read("memory.peak")) is not None and text.strip().isdigit():
            usage["peak_memory_kb"] = int(text.strip()) // 1024
        if (text := read("memory.events")) is not None:
            events = _keyed(text)
            if "oom_kill" in events:
                usage["oom_kills"] = events["oom_kill"]
                usage["oom_killed"] = events["oom_kill"] > 0
        if (text := read("pids.peak")) is not None and text.strip().isdigit():
            usage["pids_peak"] = int(text.strip())
        if (text := read("pids.events")) is not None:
            events = _keyed(text)
            if "max" in events:
                usage["pids_limit_hits"] = events["max"]
        return usage

    def remove(self, *, attempts: int = 5, sleep: Callable[[float], None] = time.sleep) -> bool:
        """``rmdir`` the leaf, killing and retrying while it is busy. True once it is gone."""
        for _ in range(attempts):
            try:
                self.fs.rmdir(self.path)
                return True
            except FileNotFoundError:
                return True
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    return False
                try:
                    self.kill()
                except CgroupError:
                    pass
                self.wait_empty(0.5, sleep=sleep)
                sleep(0.01)
        return not self.fs.exists(self.path)

    def destroy(
        self, *, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic, timeout_s: float = KILL_WAIT_S
    ) -> str:
        """Kill everything left, wait for the leaf to empty, remove it. ``clean``, ``killed``, or ``leaked``."""
        if not self.fs.exists(self.path):
            return "clean"
        status = "clean"
        if self.populated():
            status = "killed"
            try:
                self.kill()
            except CgroupError:
                return "leaked"
            if not self.wait_empty(timeout_s, sleep=sleep, clock=clock):
                return "leaked"
        return status if self.remove(sleep=sleep) else "leaked"


def configured_parent(*, fs: CgroupFS | None = None) -> CgroupParent | None:
    """The parent named by ``BULKHEAD_CGROUP_PARENT``, or None. Never guessed."""
    from .capabilities import CGROUP_PARENT_ENV

    value = os.environ.get(CGROUP_PARENT_ENV)
    return CgroupParent(value, fs=fs) if value else None


__all__ = [
    "CGROUP_MOUNT",
    "CONTROLLERS",
    "CgroupError",
    "CgroupFS",
    "CgroupLimits",
    "CgroupParent",
    "Leaf",
    "SystemCgroupFS",
    "configured_parent",
]

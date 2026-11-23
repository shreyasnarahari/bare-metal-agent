"""Per-call resource limits applied by the trusted helper before exec.

These are backups and per-process guards, not the resource governor:

- ``RLIMIT_CPU`` backs up the cgroup CPU accounting (soft limit sends SIGXCPU,
  the hard limit one second later kills).
- ``RLIMIT_FSIZE`` caps any single file at the sandbox's ``disk_mb``.
- ``RLIMIT_NOFILE`` caps descriptors; ``RLIMIT_CORE=0`` means no core dump can
  carry tool data.
- ``RLIMIT_AS`` is optional: address-space limits reject legitimate mappings,
  so cgroup ``memory.max`` is authoritative where it exists. The tier-1
  profile without a cgroup uses it as a per-process backup and says so.
- ``RLIMIT_NPROC`` is never set: it counts every process of the user, so it is
  neither a per-call boundary nor safe to lower. ``pids.max`` is the PID
  ceiling.

Limits only ever go down: a soft or hard value above the current hard limit
is clamped to it.
"""

from __future__ import annotations

import math
import resource
from dataclasses import dataclass
from typing import Any

from .manifest import Limits

DEFAULT_NOFILE = 256
_NAMES = ("RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_CORE", "RLIMIT_AS")


@dataclass(frozen=True)
class RlimitPlan:
    """The limits one call gets. ``address_space_bytes`` is None unless a backup memory cap is wanted."""

    cpu_s: int
    fsize_bytes: int
    nofile: int = DEFAULT_NOFILE
    core_bytes: int = 0
    address_space_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("cpu_s", "fsize_bytes", "nofile", "core_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.cpu_s < 1 or self.nofile < 16:
            raise ValueError("cpu_s must be at least 1 and nofile at least 16")
        if self.address_space_bytes is not None and (type(self.address_space_bytes) is not int or self.address_space_bytes < 64 * 1024 * 1024):
            raise ValueError("address_space_bytes must be at least 64 MiB")

    @classmethod
    def from_limits(cls, limits: Limits, *, nofile: int = DEFAULT_NOFILE, address_space: bool = False) -> RlimitPlan:
        """From a sandbox's limits. The address-space backup, when asked for, is twice ``mem_mb`` plus 256 MiB of headroom."""
        backup = (int(limits.mem_mb) * 2 + 256) * 1024 * 1024 if address_space else None
        return cls(
            cpu_s=max(1, math.ceil(limits.cpu_s)),
            fsize_bytes=int(limits.disk_mb) * 1024 * 1024,
            nofile=nofile,
            address_space_bytes=backup,
        )

    def entries(self) -> list[tuple[str, int, int]]:
        """``(name, soft, hard)`` for each limit to set."""
        entries = [
            ("RLIMIT_CPU", self.cpu_s, self.cpu_s + 1),
            ("RLIMIT_FSIZE", self.fsize_bytes, self.fsize_bytes),
            ("RLIMIT_NOFILE", self.nofile, self.nofile),
            ("RLIMIT_CORE", self.core_bytes, self.core_bytes),
        ]
        if self.address_space_bytes is not None:
            entries.append(("RLIMIT_AS", self.address_space_bytes, self.address_space_bytes))
        return entries

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_s": self.cpu_s, "fsize_bytes": self.fsize_bytes, "nofile": self.nofile,
            "core_bytes": self.core_bytes, "address_space_bytes": self.address_space_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> RlimitPlan:
        if not isinstance(value, dict) or set(value) != {"cpu_s", "fsize_bytes", "nofile", "core_bytes", "address_space_bytes"}:
            raise ValueError("an rlimit plan needs exactly cpu_s, fsize_bytes, nofile, core_bytes, address_space_bytes")
        return cls(**value)


def apply(plan: RlimitPlan, *, setrlimit=resource.setrlimit, getrlimit=resource.getrlimit) -> list[tuple[str, int, int]]:
    """Lower this process's limits to ``plan`` (never raising one); return what was set."""
    applied = []
    for name, soft, hard in plan.entries():
        which = getattr(resource, name)
        _, current_hard = getrlimit(which)
        if current_hard != resource.RLIM_INFINITY:
            hard = min(hard, current_hard)
            soft = min(soft, hard)
        setrlimit(which, (soft, hard))
        applied.append((name, soft, hard))
    return applied


__all__ = ["DEFAULT_NOFILE", "RlimitPlan", "apply"]

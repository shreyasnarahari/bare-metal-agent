"""ABI-aware Landlock rulesets: planned from a sandbox spec, applied by the tier-1 helper.

Landlock is an allowlist of file hierarchies with access rights, enforced by
the kernel on the process that restricts itself and on everything it later
runs. Its boundary differs from a mount namespace in ways the tier-1 backend
reports rather than hides:

- Access is controlled, visibility is not. Path walking is never restricted,
  so names may show up (``stat`` of a parent, error codes) even where reading,
  writing, and listing are denied.
- Rules only add access; nothing can carve a denied subpath out of an allowed
  hierarchy. A denied path beneath a granted one is therefore handled by
  *splitting*: the parent gets no rule of its own, and each of its existing
  entries except the denied one gets the parent's access (recursively). The
  cost is explicit: a directory that contains a denied entry cannot be listed,
  and nothing new can be created directly in it.
- Before ABI 3 ``truncate(2)`` is not controlled, so a read-only file could be
  truncated; this module refuses ABI 1 and 2 (``MIN_ABI``) rather than plan a
  read-only rule it cannot enforce.
- ABI 4 can deny TCP bind and connect. The ruleset handles both with no port
  rules, so TCP is denied outright; ports are never treated as a destination
  allowlist. ABI 6 scopes abstract Unix sockets and signals to the sandbox's
  own domain. Lacking either is recorded; tier-1 seccomp refuses ``socket()``
  entirely in any case.

``plan_ruleset`` is pure apart from reading the host tree (lstat and
directory listings), so it is unit-tested against fake ABIs. ``apply`` makes
the three syscalls and is used only by ``bulkhead.helper`` in a fresh,
single-threaded process, after ``no_new_privs``.
"""

from __future__ import annotations

import ctypes
import errno
import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BulkheadError
from .manifest import WORKSPACE, Sandbox, beneath

# Syscall numbers are the same on x86_64 and aarch64 (the unified table for new syscalls).
SYS_CREATE_RULESET = 444
SYS_ADD_RULE = 445
SYS_RESTRICT_SELF = 446
CREATE_RULESET_VERSION = 1 << 0
RULE_PATH_BENEATH = 1

EXECUTE = 1 << 0
WRITE_FILE = 1 << 1
READ_FILE = 1 << 2
READ_DIR = 1 << 3
REMOVE_DIR = 1 << 4
REMOVE_FILE = 1 << 5
MAKE_CHAR = 1 << 6
MAKE_DIR = 1 << 7
MAKE_REG = 1 << 8
MAKE_SOCK = 1 << 9
MAKE_FIFO = 1 << 10
MAKE_BLOCK = 1 << 11
MAKE_SYM = 1 << 12
REFER = 1 << 13  # ABI 2
TRUNCATE = 1 << 14  # ABI 3
IOCTL_DEV = 1 << 15  # ABI 5
NET_BIND_TCP = 1 << 0  # ABI 4
NET_CONNECT_TCP = 1 << 1
SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0  # ABI 6
SCOPE_SIGNAL = 1 << 1

FILE_RIGHTS = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV
READ_ACCESS = READ_FILE | READ_DIR | EXECUTE
WRITE_ACCESS = READ_ACCESS | WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_DIR | MAKE_REG | MAKE_SYM | MAKE_FIFO | REFER | TRUNCATE
MIN_ABI = 3
MAX_RULES = 512


class LandlockError(BulkheadError, RuntimeError):
    """A ruleset that cannot be planned or applied as declared. ``code`` is a stable rule id; nothing runs."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def fs_rights(abi: int) -> int:
    """Every filesystem right ``abi`` handles: 13 in ABI 1, REFER in 2, TRUNCATE in 3, IOCTL_DEV in 5."""
    if abi >= 5:
        return (1 << 16) - 1
    if abi >= 3:
        return (1 << 15) - 1
    if abi >= 2:
        return (1 << 14) - 1
    if abi >= 1:
        return (1 << 13) - 1
    return 0


def net_rights(abi: int) -> int:
    return NET_BIND_TCP | NET_CONNECT_TCP if abi >= 4 else 0


def scopes(abi: int) -> int:
    return SCOPE_ABSTRACT_UNIX_SOCKET | SCOPE_SIGNAL if abi >= 6 else 0


def attr_size(abi: int) -> int:
    """How much of ``struct landlock_ruleset_attr`` this ABI understands (older kernels refuse a larger one)."""
    return 24 if abi >= 6 else 16 if abi >= 4 else 8


@dataclass(frozen=True)
class Rule:
    """Grant ``access`` on the hierarchy at ``path`` (a directory) or on the single file at ``path``."""

    path: str
    access: int
    directory: bool

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "access": self.access, "directory": self.directory}


@dataclass(frozen=True)
class Ruleset:
    """Everything the helper needs to restrict a process: handled rights, scopes, and rules."""

    abi: int
    handled_fs: int
    handled_net: int
    scoped: int
    rules: tuple[Rule, ...]

    def __post_init__(self) -> None:
        if self.abi < MIN_ABI:
            raise LandlockError(f"Landlock ABI {self.abi} cannot enforce read-only files (needs {MIN_ABI})", code="bulkhead.landlock.abi_too_old")
        if self.handled_fs & ~fs_rights(self.abi) or self.handled_net & ~net_rights(self.abi) or self.scoped & ~scopes(self.abi):
            raise LandlockError("the ruleset uses rights this ABI does not have", code="bulkhead.landlock.unsupported_right")
        if len(self.rules) > MAX_RULES:
            raise LandlockError(f"more than {MAX_RULES} rules", code="bulkhead.landlock.too_many_rules")
        for rule in self.rules:
            if rule.access & ~self.handled_fs:
                raise LandlockError(f"a rule for {rule.path} grants an unhandled right", code="bulkhead.landlock.unsupported_right")
            if not rule.directory and rule.access & ~FILE_RIGHTS:
                raise LandlockError(f"a file rule for {rule.path} grants directory rights", code="bulkhead.landlock.file_rule")

    @property
    def tcp_denied(self) -> bool:
        return self.handled_net == NET_BIND_TCP | NET_CONNECT_TCP

    @property
    def signals_scoped(self) -> bool:
        return bool(self.scoped & SCOPE_SIGNAL)

    @property
    def abstract_unix_scoped(self) -> bool:
        return bool(self.scoped & SCOPE_ABSTRACT_UNIX_SOCKET)

    def as_dict(self) -> dict[str, Any]:
        return {
            "abi": self.abi, "handled_fs": self.handled_fs, "handled_net": self.handled_net, "scoped": self.scoped,
            "rules": [rule.as_dict() for rule in self.rules],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Ruleset:
        rules = tuple(Rule(str(item["path"]), int(item["access"]), bool(item["directory"])) for item in value["rules"])
        return cls(int(value["abi"]), int(value["handled_fs"]), int(value["handled_net"]), int(value["scoped"]), rules)


# Planning


def _kind(path: Path) -> str | None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    return "dir" if stat.S_ISDIR(info.st_mode) else "file"


def _parents_are_directories(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current = current / part
        kind = _kind(current)
        if kind is None:
            return
        if kind != "dir":
            raise LandlockError(f"a component of {path} is a symlink or not a directory", code="bulkhead.landlock.symlink")


def plan_ruleset(
    sandbox: Sandbox,
    workspace: Path,
    abi: int,
    *,
    home: str,
    work_root: Path,
    extra: Iterable[tuple[str, int]] = (),
) -> Ruleset:
    """The ruleset for one call: declared reads and writes (``$WORKSPACE`` is ``workspace``), denies split out.

    ``extra`` adds backend-owned grants (device files, the interpreter's
    runtime) that are not spec paths. Forbidden host paths, writable paths
    outside the workspace, and host symlinks are refused, exactly as for the
    namespace backend. Raises ``LandlockError`` when the spec cannot be
    enforced as declared.
    """
    from .workspace import ViewError, check_host_read

    if abi < MIN_ABI:
        raise LandlockError(f"Landlock ABI {abi} cannot enforce read-only files (needs {MIN_ABI})", code="bulkhead.landlock.abi_too_old")
    handled = fs_rights(abi)
    workspace = Path(workspace)

    def host(path: str) -> Path:
        if path == WORKSPACE:
            return workspace
        if path.startswith(WORKSPACE + "/"):
            return workspace / path[len(WORKSPACE) + 1:]
        return Path(path)

    grants: dict[Path, int] = {}
    reads = [path for path in sandbox.fs_read]
    absolute_reads = [path for path in reads if path.startswith("/")]
    for path in sandbox.fs_write:
        if not path.startswith(WORKSPACE):
            raise LandlockError("writable paths must be inside the workspace", code="bulkhead.landlock.write_outside_workspace")
    for path, access in [(path, READ_ACCESS) for path in reads] + [(path, WRITE_ACCESS) for path in sandbox.fs_write]:
        if path.startswith("/"):
            try:
                check_host_read(path, home=home, work_root=work_root)
            except ViewError as exc:
                raise LandlockError(str(exc), code=exc.code) from None
        location = host(path)
        _parents_are_directories(location)
        kind = _kind(location)
        if kind is None:
            continue
        if kind == "symlink":
            target = os.path.realpath(location)
            if path.startswith("/") and any(beneath(target, root) and root != path for root in absolute_reads):
                continue  # e.g. /lib -> usr/lib: already covered by /usr
            raise LandlockError(f"{path} is a symlink that cannot be granted safely", code="bulkhead.landlock.symlink")
        grants[location] = grants.get(location, 0) | (access & handled)

    denied = [host(path) for path in sandbox.fs_deny]
    for path in denied:
        _parents_are_directories(path)
        if _kind(path) == "symlink":
            raise LandlockError(f"denied path {path} is a symlink", code="bulkhead.landlock.symlink")
    rules: dict[Path, int] = {}

    def is_denied(path: Path) -> bool:
        return any(path == item for item in denied)

    def has_denied_beneath(path: Path) -> bool:
        return any(item != path and beneath(str(item), str(path)) for item in denied)

    def grant(path: Path, access: int) -> None:
        if is_denied(path) or any(beneath(str(path), str(item)) for item in denied):
            return
        kind = _kind(path)
        if kind is None or kind == "symlink":
            return  # a symlink inside a granted tree resolves to its target's own rules
        if kind == "dir" and has_denied_beneath(path):
            with os.scandir(path) as entries:
                children = sorted(entry.name for entry in entries)
            for name in children:
                grant(path / name, access)
            return
        mask = access if kind == "dir" else access & FILE_RIGHTS
        rules[path] = rules.get(path, 0) | mask

    for location, access in grants.items():
        grant(location, access)
    for path, access in extra:
        kind = _kind(Path(path))
        if kind in ("dir", "file"):
            mask = access & handled
            rules[Path(path)] = rules.get(Path(path), 0) | (mask if kind == "dir" else mask & FILE_RIGHTS)
    built = tuple(Rule(str(path), access, _kind(path) == "dir") for path, access in sorted(rules.items()) if access)
    return Ruleset(abi, handled, net_rights(abi), scopes(abi), built)


def covers(ruleset: Ruleset, path: str, access: int) -> bool:
    """Whether some rule grants all of ``access`` on ``path`` (a lexical check, for diagnostics and tests)."""
    granted = 0
    for rule in ruleset.rules:
        if path == rule.path or (rule.directory and beneath(path, rule.path)):
            granted |= rule.access
    return access & granted == access


# Applying (only in the helper)


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64), ("scoped", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1  # the kernel struct is packed: 12 bytes
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _syscall(number: int, *args: Any) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        number_ = ctypes.get_errno()
        raise OSError(number_, os.strerror(number_))
    return result


def abi_version(syscall=_syscall) -> int:
    """The kernel's Landlock ABI, or 0 when Landlock is missing or disabled."""
    try:
        return syscall(SYS_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(CREATE_RULESET_VERSION))
    except OSError as exc:
        if exc.errno in (errno.ENOSYS, errno.EOPNOTSUPP):
            return 0
        raise


def prepare(ruleset: Ruleset, *, syscall=_syscall) -> int:
    """Create the ruleset and add every rule; return its descriptor. Does not restrict anything yet."""
    kernel = abi_version(syscall)
    if kernel < ruleset.abi:
        raise LandlockError(f"the kernel has Landlock ABI {kernel}, the plan needs {ruleset.abi}", code="bulkhead.landlock.abi_mismatch")
    attr = _RulesetAttr(ruleset.handled_fs, ruleset.handled_net, ruleset.scoped)
    try:
        ruleset_fd = syscall(SYS_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(attr_size(ruleset.abi)), ctypes.c_uint32(0))
    except OSError as exc:
        raise LandlockError(f"landlock_create_ruleset failed: {exc.strerror}", code="bulkhead.landlock.create") from None
    try:
        for rule in ruleset.rules:
            try:
                parent = os.open(rule.path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
            except OSError as exc:
                raise LandlockError(f"cannot open a rule path: {exc.strerror}", code="bulkhead.landlock.rule_path") from None
            try:
                if stat.S_ISDIR(os.fstat(parent).st_mode) != rule.directory:
                    raise LandlockError("a rule path changed type after planning", code="bulkhead.landlock.rule_path")
                beneath_attr = _PathBeneathAttr(rule.access, parent)
                try:
                    syscall(SYS_ADD_RULE, ctypes.c_int(ruleset_fd), ctypes.c_int(RULE_PATH_BENEATH), ctypes.byref(beneath_attr), ctypes.c_uint32(0))
                except OSError as exc:
                    raise LandlockError(f"landlock_add_rule failed: {exc.strerror}", code="bulkhead.landlock.add_rule") from None
            finally:
                os.close(parent)
    except BaseException:
        os.close(ruleset_fd)
        raise
    return ruleset_fd


def restrict_self(ruleset_fd: int, *, syscall=_syscall) -> None:
    """Enforce the prepared ruleset on this process and everything it runs. Needs no_new_privs first."""
    try:
        syscall(SYS_RESTRICT_SELF, ctypes.c_int(ruleset_fd), ctypes.c_uint32(0))
    except OSError as exc:
        raise LandlockError(f"landlock_restrict_self failed: {exc.strerror}", code="bulkhead.landlock.restrict") from None
    finally:
        os.close(ruleset_fd)


def describe(ruleset: Ruleset) -> dict[str, Any]:
    """A short, path-free summary for audit and reports."""
    return {
        "abi": ruleset.abi,
        "rules": len(ruleset.rules),
        "tcp_denied": ruleset.tcp_denied,
        "signals_scoped": ruleset.signals_scoped,
        "abstract_unix_scoped": ruleset.abstract_unix_scoped,
    }


__all__ = [
    "FILE_RIGHTS",
    "MIN_ABI",
    "READ_ACCESS",
    "WRITE_ACCESS",
    "LandlockError",
    "Rule",
    "Ruleset",
    "abi_version",
    "covers",
    "describe",
    "fs_rights",
    "net_rights",
    "plan_ruleset",
    "prepare",
    "restrict_self",
    "scopes",
]

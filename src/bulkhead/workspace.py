"""Per-episode workspaces and the minimal filesystem view a sandboxed call sees.

A ``WorkRoot`` is a private directory (mode 0700, owned by the caller) holding
one workspace per episode plus two empty, inaccessible mask sources. The
user's repository and home are never bound by default; a task that needs
files copies them in (``Workspace.create(files=...)``), never hardlinks them.

``plan_view`` turns a ``Sandbox`` into an ordered list of view entries:

- every declared read path as a read-only bind (``$WORKSPACE`` maps to
  ``/workspace`` inside the sandbox; absolute paths keep their name);
- every declared write path as a writable bind, and only beneath the
  workspace: an absolute write path is refused;
- every denied path that a bind would expose, masked by an empty mode-0000
  file or directory of the same type, so a process cannot read it even though
  its parent is visible (the policy denial alone is not enough);
- the host's merged-``/usr`` symlinks (``/bin -> usr/bin``) when ``/usr`` is
  readable; a symlink grants nothing its target does not.

Everything else is absent from the view, not merely read-only: there is no
``--ro-bind / /``. Binding ``/``, kernel and runtime filesystems, temporary
directories, any home, or an ancestor of the work root is refused outright.
Host symlink components are refused while building binds (only a top-level
merged-``/usr`` link is reproduced, as a symlink). Symlinks the sandbox
creates resolve inside its own view, so they cannot reach a host path.

Lexical checks here say what to bind; the mount namespace enforces it.

Everything a sandbox writes under ``out/`` is untrusted, including symlinks
it creates: they resolve harmlessly inside the sandbox's view but may name
host paths. Host-side code reading a workspace must not follow them.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .errors import BulkheadError
from .manifest import WORKSPACE, Sandbox, beneath, normalize_argument_path

WORKSPACE_MOUNT = "/workspace"
MASKS = ".bulkhead-masks"
_EPISODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
# Host locations no sandbox may see, whatever a spec declares (each with everything beneath it).
FORBIDDEN_ROOTS = ("/proc", "/sys", "/dev", "/run", "/var/run", "/tmp", "/var/tmp", "/root", "/home", "/boot", "/lost+found")
# Top-level merged-/usr links reproduced as symlinks when /usr is readable.
COMPAT_LINKS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")


class ViewError(BulkheadError, ValueError):
    """A filesystem view that cannot be built safely. ``code`` is a stable rule id; nothing is run."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ViewEntry:
    """One step of the view: ``ro`` / ``rw`` bind, ``symlink``, or a ``mask_dir`` / ``mask_file`` over ``dest``."""

    op: str
    dest: str
    source: str | None = None  # host path for binds; link target for symlinks; None for masks

    def __post_init__(self) -> None:
        if self.op not in ("ro", "rw", "symlink", "mask_dir", "mask_file"):
            raise ValueError(f"unknown view operation {self.op!r}")


@dataclass(frozen=True)
class View:
    """The ordered entries plus the sandbox paths that are readable and writable."""

    entries: tuple[ViewEntry, ...]
    readable: tuple[str, ...]
    writable: tuple[str, ...]
    masked: tuple[str, ...]

    def visible(self, sandbox_path: str) -> bool:
        """Whether ``sandbox_path`` lies under a bind and not under a mask."""
        under = any(beneath(sandbox_path, root) for root in (*self.readable, *self.writable))
        return under and not any(beneath(sandbox_path, mask) for mask in self.masked)


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _no_symlink_components(base: Path, path: Path) -> None:
    """Every component from ``base`` (exclusive) down to ``path``'s parent is a real directory."""
    relative = path.relative_to(base)
    current = base
    for part in relative.parts[:-1]:
        current = current / part
        info = _lstat(current)
        if info is None:
            return
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ViewError(f"a path component is a symlink or not a directory: {part}", code="bulkhead.view.symlink")


def _private_directory(path: Path, *, create: bool) -> None:
    if create:
        path.mkdir(mode=0o700, exist_ok=True)
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ViewError(f"{path.name} is not a real directory", code="bulkhead.view.workspace_unsafe")
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ViewError(f"{path.name} must be owned by this user with mode 0700", code="bulkhead.view.workspace_unsafe")


class WorkRoot:
    """A private directory holding every episode's workspace and the empty mask sources."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        candidate = Path(path)
        candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Components above the root may be symlinks (resolved once here); nothing below it may be.
        self.path = Path(os.path.realpath(candidate))
        _private_directory(self.path, create=False)
        masks = self.path / MASKS
        if _lstat(masks) is None:
            masks.mkdir(mode=0o700)
            (masks / "dir").mkdir(mode=0o700)
            (masks / "file").touch(mode=0o600)
            os.chmod(masks / "dir", 0)
            os.chmod(masks / "file", 0)
        _private_directory(masks, create=False)
        self.mask_dir = masks / "dir"
        self.mask_file = masks / "file"
        for mask, is_dir in ((self.mask_dir, True), (self.mask_file, False)):
            info = os.lstat(mask)
            kind_ok = stat.S_ISDIR(info.st_mode) if is_dir else stat.S_ISREG(info.st_mode) and info.st_size == 0
            if not kind_ok or stat.S_IMODE(info.st_mode) != 0:
                raise ViewError("a mask source was altered", code="bulkhead.view.workspace_unsafe")

    def episode_path(self, episode_id: str) -> Path:
        if not isinstance(episode_id, str) or not _EPISODE.fullmatch(episode_id) or episode_id == MASKS:
            raise ViewError("bad episode id for a workspace", code="bulkhead.view.workspace_unsafe")
        return self.path / episode_id

    def remove(self) -> None:
        """Delete the whole root (every workspace and the masks)."""
        _remove_tree(self.path)


class Workspace:
    """One episode's persistent workspace. ``lock`` serializes calls, so a call never reuses a workspace early.

    The supervisor returns only after the previous call's processes are dead
    and reaped. If a call's cleanup reported a leak, the workspace is
    ``poisoned`` and refuses further calls.
    """

    def __init__(self, root: WorkRoot, episode_id: str, path: Path) -> None:
        self.root = root
        self.episode_id = episode_id
        self.path = path
        self.lock = threading.Lock()
        self.poisoned = False

    @classmethod
    def create(cls, root: WorkRoot, episode_id: str, files: Mapping[str, bytes] | None = None) -> Workspace:
        """A fresh workspace (refused if one exists), with ``out/`` and copies of ``files`` (relative paths)."""
        path = root.episode_path(episode_id)
        path.mkdir(mode=0o700)
        workspace = cls(root, episode_id, path)
        (path / "out").mkdir(mode=0o700)
        for name, data in sorted((files or {}).items()):
            workspace.write(name, data)
        return workspace

    @classmethod
    def open(cls, root: WorkRoot, episode_id: str) -> Workspace:
        """The episode's workspace, created empty (with ``out/``) if absent."""
        path = root.episode_path(episode_id)
        if _lstat(path) is None:
            return cls.create(root, episode_id)
        _private_directory(path, create=False)
        return cls(root, episode_id, path)

    def write(self, name: str, data: bytes) -> Path:
        """Copy ``data`` to a new regular file at relative ``name`` (no ``..``, no symlinks, never a hardlink)."""
        relative = self._relative(name)
        target = self.path / relative
        _no_symlink_components(self.path, target)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _no_symlink_components(self.path, target)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        return target

    def host_path(self, spec_path: str) -> Path:
        """Where a ``$WORKSPACE``-relative spec path lives on the host."""
        if spec_path == WORKSPACE:
            return self.path
        if not spec_path.startswith(WORKSPACE + "/"):
            raise ViewError("not a workspace path", code="bulkhead.view.write_outside_workspace")
        return self.path / self._relative(spec_path[len(WORKSPACE) + 1:])

    @staticmethod
    def _relative(name: str) -> PurePosixPath:
        if not isinstance(name, str) or name.startswith(("/", "$")):
            raise ViewError("workspace file names are relative paths", code="bulkhead.view.workspace_unsafe")
        normalized = normalize_argument_path(f"{WORKSPACE}/{name}", error=_ViewArgumentError)
        relative = PurePosixPath(normalized[len(WORKSPACE) + 1:])
        if not relative.parts:
            raise ViewError("empty workspace path", code="bulkhead.view.workspace_unsafe")
        return relative

    def remove(self) -> None:
        _remove_tree(self.path)


class _ViewArgumentError(ViewError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="bulkhead.view.workspace_unsafe")


def _remove_tree(path: Path) -> None:
    """Remove ``path`` without following symlinks, first restoring owner access the sandbox removed.

    Only called once every process that used the tree is dead, so nothing races the walk.
    """
    if _lstat(path) is None:
        return

    def restore(directory: str) -> None:
        os.chmod(directory, 0o700)
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    restore(entry.path)

    info = os.lstat(path)
    if stat.S_ISDIR(info.st_mode):
        restore(str(path))
        shutil.rmtree(path)
    else:
        os.unlink(path)


def check_host_read(path: str, *, home: str, work_root: Path) -> None:
    """Refuse host read paths that would expose kernel/runtime state, any home, temp space, or the work root."""
    if path == "/":
        raise ViewError("binding / exposes the whole host", code="bulkhead.view.forbidden_path")
    for root in FORBIDDEN_ROOTS:
        if beneath(path, root) or beneath(root, path):
            raise ViewError(f"{root} may not be visible in a sandbox", code="bulkhead.view.forbidden_path")
    for protected in (home, str(work_root)):
        if protected and (beneath(path, protected) or beneath(protected, path)):
            raise ViewError("a home directory or the work root may not be visible", code="bulkhead.view.forbidden_path")
    if beneath("/etc", path):
        raise ViewError("all of /etc may not be visible; declare specific subdirectories", code="bulkhead.view.forbidden_path")


def plan_view(sandbox: Sandbox, workspace: Workspace, *, home: str | None = None, host_root: Path = Path("/")) -> View:
    """The ordered view for one call (see the module docstring). Creates missing writable workspace directories."""
    home = os.path.realpath(os.path.expanduser("~") if home is None else home)
    work_root = workspace.root.path
    binds: list[ViewEntry] = []
    links: list[ViewEntry] = []
    readable: list[str] = []
    writable: list[str] = []
    absolute_reads = [path for path in sandbox.fs_read if path.startswith("/")]

    def host(path: str) -> Path:
        return workspace.host_path(path) if path.startswith(WORKSPACE) else host_root / path.lstrip("/")

    def dest(path: str) -> str:
        if path == WORKSPACE:
            return WORKSPACE_MOUNT
        if path.startswith(WORKSPACE + "/"):
            return WORKSPACE_MOUNT + path[len(WORKSPACE):]
        return path

    def link_into_reads(path: str, info: os.stat_result | None) -> ViewEntry | None:
        """A top-level host symlink into another readable path, reproduced as a symlink; None if not one."""
        if info is None or not stat.S_ISLNK(info.st_mode) or PurePosixPath(path).parent != PurePosixPath("/"):
            return None
        target = os.readlink(host(path))
        resolved = os.path.normpath("/" + target) if not target.startswith("/") else None
        if resolved is None or ".." in PurePosixPath(target).parts:
            return None
        if not any(beneath(resolved, root) and root != path for root in absolute_reads):
            return None
        return ViewEntry("symlink", path, target)

    for path in sandbox.fs_read:
        if path.startswith("/"):
            check_host_read(path, home=home, work_root=work_root)
            base = host_root
        else:
            base = workspace.path
        location = host(path)
        if location != base:
            _no_symlink_components(base, location)
        info = _lstat(location)
        if info is None:
            continue  # absent on this host: invisible, which is the least privilege anyway
        if stat.S_ISLNK(info.st_mode):
            entry = link_into_reads(path, info)
            if entry is None:
                raise ViewError(f"{path} is a symlink that cannot be reproduced safely", code="bulkhead.view.symlink")
            links.append(entry)
            continue
        binds.append(ViewEntry("ro", dest(path), str(location)))
        readable.append(dest(path))

    if any(beneath(path, "/usr") for path in absolute_reads) or "/usr" in absolute_reads:
        declared = {entry.dest for entry in links} | set(sandbox.fs_read)
        for name in COMPAT_LINKS:
            if name in declared:
                continue
            entry = link_into_reads(name, _lstat(host(name)))
            if entry is not None and beneath(os.path.normpath("/" + entry.source), "/usr"):  # type: ignore[operator]
                links.append(entry)

    for path in sandbox.fs_write:
        if not path.startswith(WORKSPACE):
            raise ViewError("writable paths must be inside the workspace", code="bulkhead.view.write_outside_workspace")
        location = workspace.host_path(path)
        if location != workspace.path:
            _no_symlink_components(workspace.path, location)
        info = _lstat(location)
        if info is None:
            location.mkdir(mode=0o700, parents=True)
            _no_symlink_components(workspace.path, location)
            info = os.lstat(location)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ViewError(f"{path} must be a real directory", code="bulkhead.view.symlink")
        binds.append(ViewEntry("rw", dest(path), str(location)))
        writable.append(dest(path))

    masks: list[ViewEntry] = []
    masked: list[str] = []
    for path in sorted(sandbox.fs_deny, key=lambda item: (item.count("/"), item)):
        inside = dest(path)
        if not any(beneath(inside, root) for root in (*readable, *writable)):
            continue  # nothing exposes it
        if any(beneath(inside, mask) for mask in masked):
            continue
        location = host(path)
        base = workspace.path if path.startswith(WORKSPACE) else host_root
        if location != base:
            _no_symlink_components(base, location)
        info = _lstat(location)
        if info is None:
            if any(beneath(inside, root) for root in writable):
                masks.append(ViewEntry("mask_dir", inside))  # it could be created: occupy it
                masked.append(inside)
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ViewError(f"denied path {path} is a symlink", code="bulkhead.view.symlink")
        masks.append(ViewEntry("mask_dir" if stat.S_ISDIR(info.st_mode) else "mask_file", inside))
        masked.append(inside)

    def depth(entry: ViewEntry) -> tuple[int, str]:
        return (entry.dest.count("/"), entry.dest)

    entries = tuple([*sorted(links, key=depth), *sorted(binds, key=depth), *sorted(masks, key=depth)])
    return View(entries, tuple(readable), tuple(writable), tuple(masked))


def ensure_visible(view: View, host_path: str, *, host_root: Path = Path("/")) -> str:
    """The sandbox path of an absolute host file the backend needs (an interpreter), or refuse."""
    real = os.path.realpath(host_root / host_path.lstrip("/"))
    relative = os.path.relpath(real, host_root)
    inside = "/" + relative if relative != "." else "/"
    for entry in view.entries:
        if entry.op == "ro" and entry.dest.startswith("/") and not entry.dest.startswith(WORKSPACE_MOUNT) and beneath(inside, entry.dest):
            if view.visible(inside):
                return inside
    raise ViewError("a required interpreter is not inside the sandbox's readable paths", code="bulkhead.view.interpreter")


__all__ = [
    "COMPAT_LINKS",
    "FORBIDDEN_ROOTS",
    "WORKSPACE_MOUNT",
    "View",
    "ViewEntry",
    "ViewError",
    "WorkRoot",
    "Workspace",
    "check_host_read",
    "ensure_visible",
    "plan_view",
]

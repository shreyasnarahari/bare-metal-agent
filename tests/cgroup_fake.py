"""An in-memory cgroup2 tree for unit tests: behaves like the kernel for what Bulkhead uses, touches nothing real.

Failure injection: ``fail`` maps a file name to an exception its writes raise,
``ignore`` names files whose writes silently do not stick, ``kill_ineffective``
makes ``cgroup.kill`` leave processes behind, and ``admit_elsewhere`` makes a
moved pid show up in another cgroup.
"""

from __future__ import annotations

import errno
import posixpath
from pathlib import Path

MOUNT = "/sys/fs/cgroup"
PARENT = "/sys/fs/cgroup/user.slice/delegated"
LEAF_FILES = {
    "memory": {"memory.max": "max", "memory.swap.max": "max", "memory.oom.group": "0", "memory.peak": "0", "memory.current": "0",
               "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n"},
    "pids": {"pids.max": "max", "pids.peak": "0", "pids.current": "0", "pids.events": "max 0\n"},
    "cpu": {"cpu.max": "max 100000", "cpu.stat": "usage_usec 0\nuser_usec 0\nsystem_usec 0\nthrottled_usec 0\n"},
}


class FakeCgroupFS:
    def __init__(self, *, parent: str = PARENT, controllers=("cpu", "io", "memory", "pids"), enabled=(), missing=()):
        self.parent = parent
        self.files: dict[str, dict[str, str]] = {}
        self.procs: dict[str, set[int]] = {}
        self.where: dict[int, str] = {}
        self.missing = set(missing)  # interface files this "kernel" does not provide
        self.fail: dict[str, OSError] = {}
        self.ignore: set[str] = set()
        self.kill_ineffective = False
        self.admit_elsewhere = False
        self.subtree_refuses = False
        self.writes: list[tuple[str, str]] = []
        self.removed: list[str] = []
        for path in (MOUNT, posixpath.dirname(parent), parent):
            self._make(path, {"cgroup.controllers": " ".join(controllers), "cgroup.subtree_control": ""})
        self.files[parent]["cgroup.subtree_control"] = " ".join(enabled)

    def _make(self, path, files):
        base = {"cgroup.procs": "", "cgroup.events": "", "cgroup.kill": "", "cgroup.controllers": "", "cgroup.subtree_control": ""}
        base.update(files)
        self.files[path] = base
        self.procs.setdefault(path, set())

    @staticmethod
    def _split(path):
        text = str(path)
        return posixpath.dirname(text), posixpath.basename(text)

    def read(self, path):
        directory, name = self._split(path)
        if directory not in self.files or name not in self.files[directory] or name in self.missing:
            raise FileNotFoundError(errno.ENOENT, "No such file", str(path))
        if name == "cgroup.events":
            return f"populated {int(bool(self._live(directory)))}\nfrozen 0\n"
        if name == "cgroup.procs":
            return "".join(f"{pid}\n" for pid in sorted(self.procs[directory]))
        return self.files[directory][name]

    def _live(self, directory):
        return any(path == directory or path.startswith(directory + "/") for path, pids in self.procs.items() if pids)

    def write(self, path, text):
        directory, name = self._split(path)
        self.writes.append((name, text))
        if name in self.fail:
            raise self.fail[name]
        if directory not in self.files or name not in self.files[directory] or name in self.missing:
            raise FileNotFoundError(errno.ENOENT, "No such file", str(path))
        if name == "cgroup.procs":
            pid = int(text)
            for pids in self.procs.values():
                pids.discard(pid)
            target = self.parent if self.admit_elsewhere else directory
            self.procs[target].add(pid)
            self.where[pid] = target
        elif name == "cgroup.kill":
            if not self.kill_ineffective:
                for key in list(self.procs):
                    if key == directory or key.startswith(directory + "/"):
                        self.procs[key].clear()
        elif name == "cgroup.subtree_control":
            if self.subtree_refuses:
                raise PermissionError(errno.EACCES, "Permission denied")
            enabled = set(self.files[directory][name].split())
            available = set(self.files[directory]["cgroup.controllers"].split())
            for token in text.split():
                if token[1:] in available:
                    enabled.add(token[1:])
            self.files[directory][name] = " ".join(sorted(enabled))
        elif name not in self.ignore:
            self.files[directory][name] = text

    def exists(self, path):
        text = str(path)
        if text in self.files:
            return True
        directory, name = self._split(text)
        return directory in self.files and name in self.files[directory] and name not in self.missing

    def mkdir(self, path):
        text = str(path)
        directory = posixpath.dirname(text)
        if text in self.files:
            raise FileExistsError(errno.EEXIST, "File exists")
        if directory not in self.files:
            raise FileNotFoundError(errno.ENOENT, "No such directory")
        files = {}
        for controller in self.files[directory]["cgroup.subtree_control"].split():
            files.update(LEAF_FILES.get(controller, {}))
        self._make(text, files)

    def rmdir(self, path):
        text = str(path)
        if text not in self.files:
            raise FileNotFoundError(errno.ENOENT, "No such directory")
        if self._live(text) or any(other.startswith(text + "/") for other in self.files):
            raise OSError(errno.EBUSY, "Device or resource busy")
        del self.files[text]
        del self.procs[text]
        self.removed.append(posixpath.basename(text))

    def listdir(self, path):
        text = str(path)
        return sorted(posixpath.basename(other) for other in self.files if posixpath.dirname(other) == text)

    def proc_cgroup(self, pid):
        location = self.where.get(pid)
        if location is None:
            raise FileNotFoundError(errno.ENOENT, "No such process")
        return f"0::{location[len(MOUNT):] or '/'}\n"

    # Test conveniences

    def spawn_into(self, leaf_path, *pids):
        """Pretend ``pids`` were forked inside the leaf."""
        for pid in pids:
            self.procs[str(leaf_path)].add(pid)
            self.where[pid] = str(leaf_path)

    def set(self, leaf_path, name, text):
        self.files[str(Path(leaf_path))][name] = text

"""Individual, non-destructive capability probes.

Every probe performs a real operation and returns ``Capability`` records; none
infers support from a kernel version, a distribution name, or the existence of
a path such as ``/sys/fs/cgroup``. Changes that cannot be undone (seccomp
filters, Landlock rulesets, ``no_new_privs``, namespaces) happen only in
short-lived children running ``_probe_child.py``. This process orchestrates,
owns the temporary directory and the local listeners, and removes them.

Children start with a minimal environment, so no environment variable can reach
a report, and every child runs in its own process group with a timeout that
kills the whole group. Nothing here uses ``sudo``, installs a package, writes a
sysctl, or touches a cgroup other than one leaf it created beneath an explicitly
configured parent and removes again.
"""

from __future__ import annotations

import errno
import json
import os
import platform
import re
import select
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from .types import MAX_DETAIL_CHARS, Capability, CapabilityStatus, HostInfo

HELPER = Path(__file__).with_name("_probe_child.py")
DEFAULT_TIMEOUT_S = 10.0
# Fixtures exit on their own after this long, even if every cleanup path failed.
FIXTURE_LIFETIME_S = 20.0
MAX_CHILD_OUTPUT_CHARS = 64 * 1024
# The only environment a child sees: no credentials, tokens, or proxy settings.
CHILD_ENV = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"}
CGROUP_PARENT_ENV = "BULKHEAD_CGROUP_PARENT"
REQUIRED_CGROUP_CONTROLLERS = ("cpu", "memory", "pids")
# Small limits for the probe leaf; values read back exactly as written.
CGROUP_LIMITS = {"memory.max": "67108864", "pids.max": "32", "cpu.max": "50000 100000"}
NAMESPACES = ("user", "mnt", "pid", "net", "ipc", "uts")
CANARY = "bulkhead-canary\n"
# Flags shared by every bubblewrap probe: all namespaces fresh, no capabilities, a
# read-only view of the host, fresh /proc, /dev, and an empty /tmp.
BWRAP_NAMESPACE_FLAGS = (
    "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net", "--unshare-cgroup-try",
    "--die-with-parent", "--new-session", "--cap-drop", "ALL",
)
BWRAP_SOCKET_PATH = "/tmp/bulkhead/proxy.sock"
USERNS_SYSCTLS = (
    "kernel/apparmor_restrict_unprivileged_userns",
    "kernel/unprivileged_userns_clone",
    "user/max_user_namespaces",
)


@dataclass(frozen=True)
class ChildRun:
    """What a bounded child run produced. ``returncode`` is None when it timed out."""

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_s: float


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_bounded(argv: Sequence[str], *, timeout_s: float, env: Mapping[str, str] | None = None) -> ChildRun:
    """Run ``argv`` in a new session with a minimal environment; on timeout kill its whole group.

    The leader is not reaped until after the group is killed, so its pid (the
    group id) cannot be reused by an unrelated process in between.
    """
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(CHILD_ENV if env is None else env),
        start_new_session=True,
        close_fds=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            # Something outside the group still holds the pipes; stop reading them.
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            process.wait(timeout=2.0)
            stdout, stderr = b"", b""
    finally:
        if process.poll() is None:
            _kill_group(process.pid)
            process.wait(timeout=2.0)
    return ChildRun(
        returncode=None if timed_out else process.returncode,
        stdout=stdout.decode("utf-8", "replace")[:MAX_CHILD_OUTPUT_CHARS],
        stderr=stderr.decode("utf-8", "replace")[:MAX_CHILD_OUTPUT_CHARS],
        timed_out=timed_out,
        elapsed_s=time.monotonic() - started,
    )


def _read_text(path: Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def display_path(path: str | os.PathLike[str] | None) -> str | None:
    """A path for a report: the home directory is shown as ``~``."""
    if path is None:
        return None
    text = os.fspath(path)
    home = os.path.expanduser("~")
    if home not in ("", "/", "~") and (text == home or text.startswith(home + os.sep)):
        return "~" + text[len(home):]
    return text


def sanitize_detail(text: str, replacements: Mapping[str, str] = {}) -> str:
    """One line, no control characters, temporary and home paths replaced, capped in length."""
    for old in sorted(replacements, key=len, reverse=True):
        if old:
            text = text.replace(old, replacements[old])
    text = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", text).split())
    if len(text) > MAX_DETAIL_CHARS:
        text = text[: MAX_DETAIL_CHARS - 1] + "…"
    return text


def errno_status(number: int | None) -> CapabilityStatus:
    if number in (errno.EPERM, errno.EACCES, errno.EROFS):
        return CapabilityStatus.DENIED
    if number in (errno.ENOSYS, errno.EOPNOTSUPP):
        return CapabilityStatus.UNSUPPORTED
    return CapabilityStatus.FAILED


def errno_code(number: int | None) -> str:
    return errno.errorcode.get(number or 0, "oserror").lower()


@dataclass
class ProbeContext:
    """Everything a probe needs. Tests replace ``runner`` and ``read_text`` with fakes."""

    workdir: Path
    timeout_s: float = DEFAULT_TIMEOUT_S
    bwrap: str | None = None
    cgroup_parent: Path | None = None
    cgroup_mount: Path | None = None  # set by probe_cgroup_mount from mountinfo
    tcp_port: int | None = None
    unix_socket: Path | None = None
    python: str = field(default_factory=lambda: sys.executable)
    helper: Path = HELPER
    runner: Callable[..., ChildRun] = run_bounded
    read_text: Callable[[Path], str | None] = _read_text
    proc_root: Path = Path("/proc")

    def sanitize(self, text: str) -> str:
        replacements = {str(self.workdir): "<tmp>", os.path.realpath(self.workdir): "<tmp>"}
        home = os.path.expanduser("~")
        if home not in ("", "/", "~"):
            replacements[home] = "~"
        return sanitize_detail(text, replacements)

    def record(self, name: str, status: CapabilityStatus, code: str, detail: str, **facts) -> Capability:
        return Capability(name, status, code, self.sanitize(detail), facts)

    def run_child(
        self,
        operation: str,
        names: Sequence[str],
        args: Mapping[str, object] | None = None,
        *,
        wrapper: Sequence[str] = (),
        classify: Callable[[str], tuple[CapabilityStatus, str, str]] | None = None,
    ) -> list[Capability]:
        """Run one helper operation (inside ``wrapper`` when given) and turn its output into records."""
        command = [self.python, "-I", str(self.helper), operation, json.dumps(dict(args or {}), sort_keys=True)]
        argv = [*wrapper, "--", *command] if wrapper else command
        run = self.runner(argv, timeout_s=self.timeout_s)
        if run.timed_out:
            failure = (CapabilityStatus.FAILED, "timeout", f"no answer within {self.timeout_s:g}s; its process group was killed")
            return [self._timed(self.record(name, *failure), run) for name in names]
        parsed = self._parse(run.stdout, names)
        if parsed is not None:
            return [self._timed(parsed[name], run) for name in names]
        if classify is not None and run.returncode != 0:
            failure = classify(run.stderr)
        else:
            first = next((line for line in run.stderr.splitlines() if line.strip()), "no output")
            code = "no_result" if run.returncode == 0 else "child_failed"
            failure = (CapabilityStatus.FAILED, code, f"exit {run.returncode}: {first}")
        return [self._timed(self.record(name, *failure), run) for name in names]

    def _parse(self, stdout: str, names: Sequence[str]) -> dict[str, Capability] | None:
        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            return None
        try:
            payload = json.loads(lines[-1])
            results = payload["results"]
            if set(results) != set(names):
                return None
            return {
                name: Capability(
                    name,
                    results[name]["status"],
                    results[name]["code"],
                    self.sanitize(str(results[name]["detail"])),
                    results[name]["facts"],
                )
                for name in names
            }
        except (ValueError, KeyError, TypeError):  # ProbeReportError is a ValueError
            return None

    @staticmethod
    def _timed(record: Capability, run: ChildRun) -> Capability:
        return Capability(record.name, record.status, record.code, record.detail, record.facts, round(run.elapsed_s, 4))


def dependent(name: str, dependency: Capability) -> Capability:
    """The record for a probe that was not run because a prerequisite did not hold."""
    return Capability(
        name,
        CapabilityStatus.UNAVAILABLE,
        "dependency_unavailable",
        f"requires {dependency.name}, which is {dependency.status.value} ({dependency.code})",
    )


# Host description


def host_info(proc_root: Path = Path("/proc")) -> HostInfo:
    """OS, kernel, architecture, and whether this looks like WSL, a container, or root. No host name."""
    uname = os.uname()
    version = _read_text(proc_root / "version") or ""
    wsl = any(marker in text.lower() for text in (uname.release, version) for marker in ("microsoft", "wsl"))
    init_cgroup = (_read_text(proc_root / "1" / "cgroup") or "").lower()
    container = (
        Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
        or any(marker in init_cgroup for marker in ("docker", "kubepods", "containerd", "libpod", "lxc"))
    )
    return HostInfo(
        os=uname.sysname,
        kernel_release=uname.release,
        architecture=uname.machine,
        python_version=platform.python_version(),
        wsl=wsl,
        container=container,
        root=os.geteuid() == 0,
    )


def read_sysctls(ctx: ProbeContext, names: Sequence[str] = USERNS_SYSCTLS) -> dict[str, str]:
    values = {}
    for name in names:
        text = ctx.read_text(ctx.proc_root / "sys" / name)
        if text is not None:
            values[name] = text.strip()
    return values


# bubblewrap


def classify_bwrap_failure(stderr: str, sysctls: Mapping[str, str]) -> tuple[CapabilityStatus, str, str]:
    """Tell a refused namespace (EPERM, AppArmor, sysctl) from an unknown flag or another failure."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    first = next((line for line in lines if line.startswith("bwrap:")), lines[0] if lines else "no error output")
    text = stderr.lower()
    if "unknown option" in text or "unrecognized option" in text:
        return CapabilityStatus.UNSUPPORTED, "unknown_option", f"this bubblewrap lacks a required flag: {first}"
    if any(marker in text for marker in ("no permissions to creat", "operation not permitted", "permission denied")):
        if sysctls.get("kernel/apparmor_restrict_unprivileged_userns") == "1":
            return (
                CapabilityStatus.DENIED,
                "apparmor_userns_restricted",
                f"AppArmor restricts unprivileged user namespaces "
                f"(kernel.apparmor_restrict_unprivileged_userns=1; the probe does not change it): {first}",
            )
        if sysctls.get("user/max_user_namespaces") == "0" or sysctls.get("kernel/unprivileged_userns_clone") == "0":
            return CapabilityStatus.DENIED, "userns_disabled", f"unprivileged user namespaces are disabled by sysctl: {first}"
        return CapabilityStatus.DENIED, "eperm", first
    return CapabilityStatus.FAILED, "bwrap_failed", first


def bwrap_wrapper(ctx: ProbeContext, *mounts: str) -> list[str]:
    assert ctx.bwrap is not None
    return [
        ctx.bwrap, *BWRAP_NAMESPACE_FLAGS,
        "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        *mounts,
        "--chdir", "/",
    ]


def probe_bwrap_binary(ctx: ProbeContext) -> Capability:
    name = "bwrap.binary"
    if ctx.bwrap is None:
        return ctx.record(
            name, CapabilityStatus.UNAVAILABLE, "not_found",
            "bwrap is not on PATH; install bubblewrap yourself (the probe never installs it) or pass --bwrap",
        )
    if not os.access(ctx.bwrap, os.X_OK):
        return ctx.record(name, CapabilityStatus.UNAVAILABLE, "not_executable", f"{ctx.bwrap} is not an executable file")
    run = ctx.runner([ctx.bwrap, "--version"], timeout_s=ctx.timeout_s)
    match = re.search(r"bubblewrap\s+([0-9][0-9A-Za-z.+~-]*)", run.stdout)
    if run.timed_out or run.returncode != 0 or match is None:
        detail = "timed out" if run.timed_out else f"exit {run.returncode}: {run.stderr.strip()[:120] or run.stdout.strip()[:120]}"
        return ProbeContext._timed(ctx.record(name, CapabilityStatus.FAILED, "version_failed", f"bwrap --version failed ({detail})"), run)
    setuid = bool(os.stat(ctx.bwrap).st_mode & 0o4000)
    record = ctx.record(
        name, CapabilityStatus.OK, "ok",
        f"bubblewrap {match.group(1)}" + (" (setuid binary)" if setuid else ""),
        version=match.group(1), setuid=setuid,
    )
    return ProbeContext._timed(record, run)


def probe_bwrap_namespaces(ctx: ProbeContext, binary: Capability) -> Capability:
    name = "bwrap.namespaces"
    if not binary.ok:
        return dependent(name, binary)
    host_ns = {namespace: os.readlink(ctx.proc_root / "self" / "ns" / namespace) for namespace in NAMESPACES}
    classify = partial(classify_bwrap_failure, sysctls=read_sysctls(ctx))
    [record] = ctx.run_child("bwrap-namespaces", [name], {"host_ns": host_ns}, wrapper=bwrap_wrapper(ctx), classify=classify)
    return record


def probe_bwrap_mounts(ctx: ProbeContext, namespaces: Capability) -> Capability:
    name = "bwrap.mounts"
    if not namespaces.ok:
        return dependent(name, namespaces)
    root = ctx.workdir / "bwrap-mounts"
    rw, ro = root / "rw", root / "ro"
    (ro / "denied").mkdir(parents=True)
    rw.mkdir()
    (ro / "canary").write_text(CANARY)
    (ro / "denied" / "canary").write_text(CANARY)
    wrapper = bwrap_wrapper(ctx, "--bind", str(rw), "/tmp/rw", "--ro-bind", str(ro), "/tmp/ro", "--tmpfs", "/tmp/ro/denied")
    classify = partial(classify_bwrap_failure, sysctls=read_sysctls(ctx))
    [record] = ctx.run_child("bwrap-mounts", [name], {"rw": "/tmp/rw", "ro": "/tmp/ro"}, wrapper=wrapper, classify=classify)
    if record.ok and ctx.read_text(rw / "written") != CANARY:
        return ctx.record(name, CapabilityStatus.FAILED, "verification_failed", "the write inside the sandbox did not reach the bound host directory")
    return record


def probe_bwrap_network(ctx: ProbeContext, namespaces: Capability) -> list[Capability]:
    names = ("bwrap.netns", "bwrap.unix_socket")
    if not namespaces.ok:
        return [dependent(name, namespaces) for name in names]
    if ctx.tcp_port is None or ctx.unix_socket is None:
        return [ctx.record(name, CapabilityStatus.FAILED, "no_listener", "the outside listeners could not be created") for name in names]
    wrapper = bwrap_wrapper(ctx, "--bind", str(ctx.unix_socket), BWRAP_SOCKET_PATH)
    classify = partial(classify_bwrap_failure, sysctls=read_sysctls(ctx))
    return ctx.run_child(
        "bwrap-network", names, {"port": ctx.tcp_port, "socket": BWRAP_SOCKET_PATH}, wrapper=wrapper, classify=classify
    )


# Kernel features probed in plain children


def probe_userns(ctx: ProbeContext) -> Capability:
    [record] = ctx.run_child("userns-net", ["userns.unshare"], {"port": ctx.tcp_port})
    return record


def probe_libseccomp(ctx: ProbeContext) -> Capability:
    [record] = ctx.run_child("libseccomp", ["libseccomp.library"])
    return record


def probe_seccomp(ctx: ProbeContext, library: Capability) -> list[Capability]:
    if not library.ok:
        return [dependent(name, library) for name in ("seccomp.filter", "seccomp.socket_deny")]
    return [
        *ctx.run_child("seccomp-filter", ["seccomp.filter"]),
        *ctx.run_child("seccomp-socket-deny", ["seccomp.socket_deny"], {"port": ctx.tcp_port}),
    ]


def probe_no_new_privs(ctx: ProbeContext) -> Capability:
    [record] = ctx.run_child("no-new-privs", ["no_new_privs"])
    return record


def probe_landlock(ctx: ProbeContext) -> list[Capability]:
    [abi] = ctx.run_child("landlock-abi", ["landlock.abi"])
    if not abi.ok:
        return [abi, dependent("landlock.fs", abi), dependent("landlock.tcp", abi)]
    root = ctx.workdir / "landlock"
    for directory in ("allowed", "denied"):
        (root / directory).mkdir(parents=True)
        (root / directory / "canary").write_text(CANARY)
    [filesystem] = ctx.run_child("landlock-fs", ["landlock.fs"], {"allowed": str(root / "allowed"), "denied": str(root / "denied")})
    if int(abi.facts["abi"]) < 4:
        tcp = ctx.record(
            "landlock.tcp", CapabilityStatus.UNSUPPORTED, "abi_too_old",
            f"TCP rules need Landlock ABI 4; this kernel has ABI {abi.facts['abi']}",
        )
    else:
        [tcp] = ctx.run_child("landlock-tcp", ["landlock.tcp"], {"port": ctx.tcp_port})
    return [abi, filesystem, tcp]


def probe_rlimits(ctx: ProbeContext) -> Capability:
    scratch = ctx.workdir / "rlimits"
    scratch.mkdir()
    [record] = ctx.run_child("rlimits", ["rlimits"], {"scratch": str(scratch)})
    return record


def probe_group_cleanup(ctx: ProbeContext) -> Capability:
    [record] = ctx.run_child("group-cleanup", ["process.group_cleanup"], {"lifetime": FIXTURE_LIFETIME_S})
    return record


def probe_openat2(ctx: ProbeContext) -> Capability:
    root = ctx.workdir / "openat2"
    (root / "root").mkdir(parents=True)
    (root / "root" / "canary").write_text(CANARY)
    (root / "outside").write_text(CANARY)
    (root / "root" / "link").symlink_to("../outside")
    [record] = ctx.run_child("openat2", ["fs.openat2"], {"root": str(root / "root")})
    return record


# cgroup v2


def _unescape_mountinfo(text: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), text)


def probe_cgroup_mount(ctx: ProbeContext) -> Capability:
    """Find a cgroup2 mount in mountinfo and confirm it is a cgroup root, not just a directory."""
    name = "cgroup.v2_mount"
    text = ctx.read_text(ctx.proc_root / "self" / "mountinfo")
    if text is None:
        return ctx.record(name, CapabilityStatus.FAILED, "mountinfo_unreadable", "could not read /proc/self/mountinfo")
    found, v1 = [], False
    for line in text.splitlines():
        left, _, right = line.partition(" - ")
        fields, tail = left.split(), right.split()
        if len(fields) < 6 or len(tail) < 3:
            continue
        if tail[0] == "cgroup2":
            found.append((_unescape_mountinfo(fields[4]), fields[5] + "," + tail[2]))
        elif tail[0] == "cgroup":
            v1 = True
    if not found:
        detail = "no cgroup2 mount in /proc/self/mountinfo" + (" (only cgroup v1 is mounted)" if v1 else "")
        return ctx.record(name, CapabilityStatus.UNAVAILABLE, "not_mounted", detail)
    mount, options = found[0]
    if ctx.read_text(Path(mount) / "cgroup.controllers") is None:
        return ctx.record(name, CapabilityStatus.FAILED, "not_readable", f"{mount} has no readable cgroup.controllers")
    ctx.cgroup_mount = Path(mount)
    return ctx.record(name, CapabilityStatus.OK, "ok", f"cgroup2 mounted at {mount}", nsdelegate="nsdelegate" in options.split(","))


def _write_cgroup_file(path: Path, value: str) -> None:
    """Write an existing cgroup control file. Never creates a file (O_CREAT is not passed)."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CLOEXEC)
    try:
        os.write(descriptor, value.encode())
    finally:
        os.close(descriptor)


def _cgroup_pids(leaf: Path) -> set[int]:
    return {int(pid) for pid in (_read_text(leaf / "cgroup.procs") or "").split()}


def _populated(leaf: Path) -> bool | None:
    for line in (_read_text(leaf / "cgroup.events") or "").splitlines():
        key, _, value = line.partition(" ")
        if key == "populated":
            return value.strip() == "1"
    return None


def _wait_unpopulated(leaf: Path, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _populated(leaf) is False:
            return True
        time.sleep(0.02)
    return _populated(leaf) is False


def _read_line(stream, timeout_s: float) -> str:
    descriptor, deadline, data = stream.fileno(), time.monotonic() + timeout_s, b""
    while b"\n" not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("the fixture did not report its process ids in time")
        ready, _, _ = select.select([descriptor], [], [], remaining)
        if ready:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                raise EOFError("the fixture exited before reporting its process ids")
            data += chunk
    return data.split(b"\n", 1)[0].decode()


def _check_delegation(ctx: ProbeContext) -> tuple[Capability, Path | None]:
    name = "cgroup.delegation"
    if ctx.cgroup_parent is None:
        return ctx.record(
            name, CapabilityStatus.UNAVAILABLE, "not_configured",
            f"no delegated cgroup parent configured; pass --cgroup-parent or set {CGROUP_PARENT_ENV} "
            "(the probe never picks one)",
        ), None
    assert ctx.cgroup_mount is not None
    try:
        parent = Path(ctx.cgroup_parent).resolve(strict=True)
    except OSError:
        return ctx.record(name, CapabilityStatus.UNAVAILABLE, "parent_missing", "the configured cgroup parent does not exist"), None
    mount = ctx.cgroup_mount.resolve()
    if parent == mount:
        return ctx.record(
            name, CapabilityStatus.DENIED, "root_cgroup_refused",
            "refusing the cgroup root as a parent, even as root; configure a delegated subtree",
        ), None
    if mount not in parent.parents:
        return ctx.record(name, CapabilityStatus.UNAVAILABLE, "outside_cgroup2_mount", "the configured parent is not inside the cgroup2 mount"), None
    controllers = ctx.read_text(parent / "cgroup.controllers")
    enabled = ctx.read_text(parent / "cgroup.subtree_control")
    if controllers is None or enabled is None:
        return ctx.record(name, CapabilityStatus.UNAVAILABLE, "not_a_cgroup", "the configured parent has no cgroup.controllers or cgroup.subtree_control"), None
    if not os.access(parent, os.W_OK | os.X_OK):
        return ctx.record(name, CapabilityStatus.DENIED, "not_writable", "the probe user cannot create leaves in the configured parent"), None
    enabled_list = sorted(enabled.split())
    return ctx.record(
        name, CapabilityStatus.OK, "ok",
        f"configured parent is writable; controllers enabled for leaves: {' '.join(enabled_list) or 'none'}",
        available=sorted(controllers.split()), enabled=enabled_list,
    ), parent


def _create_leaf(ctx: ProbeContext, leaf: Path) -> Capability:
    name = "cgroup.leaf"
    try:
        os.mkdir(leaf, 0o755)
    except OSError as exc:
        return ctx.record(name, errno_status(exc.errno), errno_code(exc.errno), f"could not create a leaf: {exc.strerror}")
    if not (leaf / "cgroup.procs").is_file():
        return ctx.record(
            name, CapabilityStatus.FAILED, "not_a_cgroup",
            "a new directory under the parent has no cgroup.procs, so the parent is not cgroupfs",
        )
    return ctx.record(name, CapabilityStatus.OK, "ok", "created a leaf with its own cgroup.procs")


def _write_limits(ctx: ProbeContext, leaf: Path, delegation: Capability) -> Capability:
    name = "cgroup.limits"
    enabled = set(delegation.facts.get("enabled") or ())
    missing = [controller for controller in REQUIRED_CGROUP_CONTROLLERS if controller not in enabled]
    if missing:
        return ctx.record(
            name, CapabilityStatus.UNAVAILABLE, "controllers_not_enabled",
            f"enable {' '.join('+' + item for item in missing)} in the parent's cgroup.subtree_control",
            missing=missing,
        )
    for control, value in CGROUP_LIMITS.items():
        try:
            _write_cgroup_file(leaf / control, value)
        except OSError as exc:
            return ctx.record(name, errno_status(exc.errno), errno_code(exc.errno), f"writing {control}: {exc.strerror}")
        if (ctx.read_text(leaf / control) or "").strip() != value:
            return ctx.record(name, CapabilityStatus.FAILED, "readback_mismatch", f"{control} did not read back as {value}")
    usage = {control: (ctx.read_text(leaf / control) or "").strip() for control in ("memory.current", "pids.current")}
    if not all(value.isdigit() for value in usage.values()):
        return ctx.record(name, CapabilityStatus.FAILED, "usage_unreadable", "memory.current or pids.current is not readable")
    return ctx.record(
        name, CapabilityStatus.OK, "ok",
        "memory.max=64MiB, pids.max=32, cpu.max=50% written and read back; usage readable",
        controllers=list(REQUIRED_CGROUP_CONTROLLERS),
    )


def _kill_tree(ctx: ProbeContext, leaf: Path) -> Capability:
    """Move a fixture into the leaf before it forks, then prove cgroup.kill removes the whole tree."""
    name = "cgroup.kill"
    if not (leaf / "cgroup.kill").exists():
        return ctx.record(name, CapabilityStatus.UNSUPPORTED, "no_cgroup_kill", "this kernel has no cgroup.kill (Linux 5.14+)")
    fixture = json.dumps({"lifetime": FIXTURE_LIFETIME_S, "escape_session": True})
    process = subprocess.Popen(
        [ctx.python, "-I", str(ctx.helper), "tree-fixture", fixture],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=dict(CHILD_ENV),
        start_new_session=True,
        close_fds=True,
    )
    grandchild: int | None = None
    try:
        try:
            _write_cgroup_file(leaf / "cgroup.procs", str(process.pid))
        except OSError as exc:
            return ctx.record(
                name, errno_status(exc.errno), errno_code(exc.errno),
                f"could not move a process into the leaf ({exc.strerror}); run the probe inside the delegated subtree",
            )
        if process.pid not in _cgroup_pids(leaf):
            return ctx.record(name, CapabilityStatus.FAILED, "move_not_verified", "the fixture is not listed in the leaf after the move")
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(b"go\n")
        process.stdin.close()
        pids = json.loads(_read_line(process.stdout, ctx.timeout_s))
        grandchild = int(pids["grandchild"])
        if {int(pids["child"]), grandchild} - _cgroup_pids(leaf):
            return ctx.record(name, CapabilityStatus.FAILED, "not_in_leaf", "the fixture tree is not entirely inside the leaf")
        _write_cgroup_file(leaf / "cgroup.kill", "1")
        process.wait(timeout=ctx.timeout_s)
        if not _wait_unpopulated(leaf, ctx.timeout_s):
            return ctx.record(name, CapabilityStatus.FAILED, "survivors", "the leaf was still populated after cgroup.kill")
        return ctx.record(
            name, CapabilityStatus.OK, "ok",
            "cgroup.kill removed a child and a grandchild that had left its session",
            tree=["child", "grandchild_new_session"],
        )
    except (OSError, ValueError, KeyError, TimeoutError, EOFError, subprocess.TimeoutExpired) as exc:
        number = exc.errno if isinstance(exc, OSError) else None
        status = errno_status(number) if number else CapabilityStatus.FAILED
        return ctx.record(name, status, errno_code(number) if number else "fixture_failed", f"{type(exc).__name__}: {exc}")
    finally:
        _stop_fixture(process, leaf, grandchild)


def _stop_fixture(process: subprocess.Popen, leaf: Path, grandchild: int | None) -> None:
    if _populated(leaf):
        try:
            _write_cgroup_file(leaf / "cgroup.kill", "1")
        except OSError:
            pass
    if process.poll() is None:
        _kill_group(process.pid)
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass
    # A grandchild that escaped the session is killed by pid only while it is still in our
    # leaf, so a reused pid belonging to an unrelated process is never signalled.
    membership = _read_text(Path("/proc") / str(grandchild) / "cgroup") if grandchild is not None else None
    if grandchild is not None and membership and membership.rstrip("\n").endswith(f"/{leaf.name}"):
        try:
            os.kill(grandchild, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for stream in (process.stdin, process.stdout):
        if stream is not None and not stream.closed:
            stream.close()


def _remove_leaf(leaf: Path) -> str | None:
    """Remove the probe's leaf, killing anything left in it; the error text if it stays behind."""
    if not leaf.exists():
        return None
    error = None
    for _ in range(100):
        try:
            leaf.rmdir()
            return None
        except OSError as exc:
            error = exc.strerror
            if _populated(leaf):
                try:
                    _write_cgroup_file(leaf / "cgroup.kill", "1")
                except OSError:
                    pass
            time.sleep(0.02)
    return error or "unknown error"


def probe_cgroup_tree(ctx: ProbeContext, mount: Capability) -> list[Capability]:
    """Delegation, leaf creation, limits, and tree kill, all in one uniquely named leaf that is removed."""
    names = ("cgroup.delegation", "cgroup.leaf", "cgroup.limits", "cgroup.kill")
    if not mount.ok:
        return [dependent(name, mount) for name in names]
    delegation, parent = _check_delegation(ctx)
    if parent is None:
        return [delegation, *(dependent(name, delegation) for name in names[1:])]
    leaf = parent / f"bulkhead-probe-{uuid.uuid4().hex[:12]}"
    try:
        created = _create_leaf(ctx, leaf)
        if created.ok:
            limits, kill = _write_limits(ctx, leaf, delegation), _kill_tree(ctx, leaf)
        else:
            limits, kill = dependent("cgroup.limits", created), dependent("cgroup.kill", created)
    finally:
        leftover = _remove_leaf(leaf)
    if leftover is not None:
        created = ctx.record(
            "cgroup.leaf", CapabilityStatus.FAILED, "leaf_not_removed",
            f"the probe leaf could not be removed ({leftover}); delete bulkhead-probe-* under the parent",
        )
    return [delegation, created, limits, kill]


# Local listeners the network probes try to reach


@contextmanager
def outside_listeners(workdir: Path) -> Iterator[tuple[int | None, Path | None]]:
    """A loopback TCP listener and a Unix listener, never accepted from, closed afterwards."""
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    unix: socket.socket | None = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    path: Path | None = workdir / "proxy.sock"
    try:
        tcp.bind(("127.0.0.1", 0))
        tcp.listen(64)
        try:
            unix.bind(str(path))
            unix.listen(64)
        except OSError:  # e.g. a temporary path longer than a socket address allows
            unix.close()
            unix, path = None, None
        yield tcp.getsockname()[1], path
    finally:
        tcp.close()
        if unix is not None:
            unix.close()


ProbeFunction = Callable[[ProbeContext, Mapping[str, Capability]], "Capability | list[Capability]"]

# Run in order; each step sees the records already collected. ``names`` lets a
# probe that raises unexpectedly still produce one failed record per name.
PROBES: tuple[tuple[tuple[str, ...], ProbeFunction], ...] = (
    (("bwrap.binary",), lambda ctx, done: probe_bwrap_binary(ctx)),
    (("bwrap.namespaces",), lambda ctx, done: probe_bwrap_namespaces(ctx, done["bwrap.binary"])),
    (("bwrap.mounts",), lambda ctx, done: probe_bwrap_mounts(ctx, done["bwrap.namespaces"])),
    (("bwrap.netns", "bwrap.unix_socket"), lambda ctx, done: probe_bwrap_network(ctx, done["bwrap.namespaces"])),
    (("userns.unshare",), lambda ctx, done: probe_userns(ctx)),
    (("libseccomp.library",), lambda ctx, done: probe_libseccomp(ctx)),
    (("seccomp.filter", "seccomp.socket_deny"), lambda ctx, done: probe_seccomp(ctx, done["libseccomp.library"])),
    (("no_new_privs",), lambda ctx, done: probe_no_new_privs(ctx)),
    (("landlock.abi", "landlock.fs", "landlock.tcp"), lambda ctx, done: probe_landlock(ctx)),
    (("rlimits",), lambda ctx, done: probe_rlimits(ctx)),
    (("process.group_cleanup",), lambda ctx, done: probe_group_cleanup(ctx)),
    (("cgroup.v2_mount",), lambda ctx, done: probe_cgroup_mount(ctx)),
    (
        ("cgroup.delegation", "cgroup.leaf", "cgroup.limits", "cgroup.kill"),
        lambda ctx, done: probe_cgroup_tree(ctx, done["cgroup.v2_mount"]),
    ),
    (("fs.openat2",), lambda ctx, done: probe_openat2(ctx)),
)


def collect_capabilities(ctx: ProbeContext) -> list[Capability]:
    """Run every probe in order. An unexpected exception becomes a failed record, never a crash."""
    done: dict[str, Capability] = {}
    for names, probe in PROBES:
        started = time.monotonic()
        try:
            outcome = probe(ctx, done)
            records = outcome if isinstance(outcome, list) else [outcome]
        except Exception as exc:  # the report must still be written, with the failure in it
            detail = f"{type(exc).__name__}: {exc}"
            records = [ctx.record(name, CapabilityStatus.FAILED, "probe_error", detail) for name in names]
        if sorted(record.name for record in records) != sorted(names):
            raise RuntimeError(f"probe for {names} returned {[record.name for record in records]}")
        elapsed = round(time.monotonic() - started, 4)
        for record in records:
            if not record.elapsed_s:
                record = Capability(record.name, record.status, record.code, record.detail, record.facts, elapsed)
            done[record.name] = record
    return list(done.values())


__all__ = [
    "CGROUP_PARENT_ENV",
    "CHILD_ENV",
    "DEFAULT_TIMEOUT_S",
    "HELPER",
    "PROBES",
    "ChildRun",
    "ProbeContext",
    "classify_bwrap_failure",
    "collect_capabilities",
    "dependent",
    "display_path",
    "host_info",
    "outside_listeners",
    "run_bounded",
    "sanitize_detail",
]

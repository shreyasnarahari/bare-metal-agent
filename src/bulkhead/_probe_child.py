"""Probe operations that must run in a short-lived child process.

Run as ``python -I _probe_child.py <operation> '<json args>'``. Only the
standard library is imported, so isolated mode (no script directory on
``sys.path``, no user site, no ``PYTHON*`` variables) is safe. Each operation
prints one JSON line, ``{"results": {capability: {status, code, detail,
facts}}}``, and exits; ``tree-fixture`` instead prints its process ids.

Seccomp filters, Landlock rulesets, ``no_new_privs``, and namespaces cannot be
undone by the process that applies them, which is why they are applied here
and never in the probing process. Every check uses canary files and local
listeners the parent created and passed in; nothing here reads a credential,
a home directory, or a sensitive host file.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import json
import os
import platform
import resource
import select
import signal
import socket
import sys
import time

MACHINE = platform.machine().lower()
# Syscalls numbered 424 and above share one number on every architecture with
# the generic table. MIPS and Alpha offset them, so raw syscalls are refused on
# any machine not listed here rather than risk calling the wrong one.
UNIFIED_SYSCALL_MACHINES = frozenset(
    {
        "x86_64", "amd64", "i386", "i486", "i586", "i686", "aarch64", "arm64", "armv6l", "armv7l",
        "armv8l", "riscv64", "ppc64", "ppc64le", "s390x", "loongarch64",
    }
)
SYS_OPENAT2 = 437
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446

PR_SET_CHILD_SUBREAPER = 36
PR_SET_NO_NEW_PRIVS = 38
PR_GET_NO_NEW_PRIVS = 39
CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000

LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_RULE_NET_PORT = 2
LANDLOCK_ACCESS_NET_BIND_TCP = 1 << 0
LANDLOCK_ACCESS_NET_CONNECT_TCP = 1 << 1

SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_ERRNO = 0x00050000
SCMP_CMP_EQ = 4
AF_PACKET = 17

RESOLVE_NO_SYMLINKS = 0x04
RESOLVE_BENEATH = 0x08

CANARY = b"bulkhead-canary\n"
CONNECT_TIMEOUT_S = 2.0

_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long
_libc.prctl.restype = ctypes.c_int
_libc.unshare.argtypes = [ctypes.c_int]
_libc.unshare.restype = ctypes.c_int


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64), ("scoped", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1  # the kernel struct is __attribute__((packed)): 12 bytes
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class NetPortAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("port", ctypes.c_uint64)]


class OpenHow(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint64), ("mode", ctypes.c_uint64), ("resolve", ctypes.c_uint64)]


class ScmpArgCmp(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int), ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


class ScmpVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint), ("minor", ctypes.c_uint), ("micro", ctypes.c_uint)]


# Results


def result(status: str, code: str, detail: str, **facts):
    return {"status": status, "code": code, "detail": detail, "facts": facts}


def ok(detail: str, **facts):
    return result("ok", "ok", detail, **facts)


def errno_name(number: int) -> str:
    return errno.errorcode.get(number, f"errno_{number}").lower()


def error_result(number: int, what: str):
    """Map an errno to a status: refusals are denied, missing features unsupported, the rest failed."""
    if number in (errno.EPERM, errno.EACCES):
        status = "denied"
    elif number in (errno.ENOSYS, errno.EOPNOTSUPP, errno.EAFNOSUPPORT):
        status = "unsupported"
    else:
        status = "failed"
    return result(status, errno_name(number), f"{what}: {os.strerror(number)}")


# Kernel helpers


def _arg(value):
    if value is None:
        return ctypes.c_void_p(None)
    if isinstance(value, int):
        return ctypes.c_long(value)
    return value


def raw_syscall(number: int, *args) -> int:
    """``syscall(2)`` with every argument passed at full register width; raises OSError on -1."""
    if MACHINE not in UNIFIED_SYSCALL_MACHINES:
        raise OSError(errno.ENOSYS, f"raw syscall numbers are not mapped for {MACHINE}")
    ctypes.set_errno(0)
    value = _libc.syscall(ctypes.c_long(number), *(_arg(arg) for arg in args))
    if value == -1:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return value


def prctl(option: int, arg2: int = 0) -> int:
    ctypes.set_errno(0)
    value = _libc.prctl(ctypes.c_int(option), ctypes.c_ulong(arg2), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0))
    if value == -1:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return value


def set_no_new_privs() -> None:
    prctl(PR_SET_NO_NEW_PRIVS, 1)


def status_field(name: str) -> str | None:
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key == name:
                return value.strip()
    return None


def read_sysctl(path: str) -> str | None:
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return None


def userns_refusal(number: int):
    """Explain an EPERM/ENOSPC from creating a user namespace without changing any setting."""
    if read_sysctl("/proc/sys/kernel/apparmor_restrict_unprivileged_userns") == "1":
        return result(
            "denied", "apparmor_userns_restricted",
            "AppArmor restricts unprivileged user namespaces (kernel.apparmor_restrict_unprivileged_userns=1)",
        )
    if read_sysctl("/proc/sys/user/max_user_namespaces") == "0" or read_sysctl("/proc/sys/kernel/unprivileged_userns_clone") == "0":
        return result("denied", "userns_disabled", "unprivileged user namespaces are disabled by sysctl")
    return error_result(number, "unshare(CLONE_NEWUSER|CLONE_NEWNET)")


def connect_tcp(port: int) -> int:
    """Connect to 127.0.0.1:port; 0 on success, otherwise the errno (ETIMEDOUT for a timeout)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError as exc:
        return exc.errno or errno.EIO
    try:
        sock.settimeout(CONNECT_TIMEOUT_S)
        sock.connect(("127.0.0.1", port))
        return 0
    except socket.timeout:
        return errno.ETIMEDOUT
    except OSError as exc:
        return exc.errno or errno.EIO
    finally:
        sock.close()


def read_file(path: str) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        return os.read(fd, 4096)
    finally:
        os.close(fd)


def write_file(path: str, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def expect_errno(action, *numbers: int) -> bool:
    """True when ``action`` fails with one of ``numbers``; False when it succeeds or fails otherwise."""
    try:
        action()
    except OSError as exc:
        return exc.errno in numbers
    return False


def succeeds(action) -> bool:
    try:
        action()
    except OSError:
        return False
    return True


# libseccomp through ctypes


def load_libseccomp():
    for name in ("libseccomp.so.2", ctypes.util.find_library("seccomp")):
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name, use_errno=True)
        except OSError:
            continue
        lib.seccomp_init.argtypes = [ctypes.c_uint32]
        lib.seccomp_init.restype = ctypes.c_void_p
        lib.seccomp_rule_add_array.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ScmpArgCmp),
        ]
        lib.seccomp_rule_add_array.restype = ctypes.c_int
        lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
        lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
        lib.seccomp_load.argtypes = [ctypes.c_void_p]
        lib.seccomp_load.restype = ctypes.c_int
        lib.seccomp_release.argtypes = [ctypes.c_void_p]
        lib.seccomp_release.restype = None
        return lib
    return None


def libseccomp_version(lib) -> str | None:
    try:
        function = lib.seccomp_version
    except AttributeError:  # libseccomp older than 2.3
        return None
    function.restype = ctypes.POINTER(ScmpVersion)
    version = function().contents
    return f"{version.major}.{version.minor}.{version.micro}"


def install_filter(lib, rules) -> int:
    """Load an allow-by-default filter with ``(syscall, errno, arg0-or-None)`` rules; 0 or -errno."""
    context = lib.seccomp_init(SCMP_ACT_ALLOW)
    if not context:
        return -errno.ENOMEM
    try:
        for syscall_name, number, arg0 in rules:
            syscall_number = lib.seccomp_syscall_resolve_name(syscall_name.encode())
            if syscall_number < 0:
                return -errno.ENOSYS
            if arg0 is None:
                rc = lib.seccomp_rule_add_array(context, SCMP_ACT_ERRNO | number, syscall_number, 0, None)
            else:
                comparison = ScmpArgCmp(0, SCMP_CMP_EQ, arg0, 0)
                rc = lib.seccomp_rule_add_array(context, SCMP_ACT_ERRNO | number, syscall_number, 1, ctypes.byref(comparison))
            if rc < 0:
                return rc
        return lib.seccomp_load(context)
    finally:
        lib.seccomp_release(context)


def op_libseccomp(args):
    lib = load_libseccomp()
    if lib is None:
        return result("unavailable", "not_found", "libseccomp shared library not found")
    version = libseccomp_version(lib)
    return ok(f"libseccomp {version or 'older than 2.3'} loaded", version=version)


def op_seccomp_filter(args):
    lib = load_libseccomp()
    if lib is None:
        return result("unavailable", "not_found", "libseccomp shared library not found")
    if not succeeds(os.sched_yield):
        return result("failed", "baseline_failed", "sched_yield failed before any filter was installed")
    set_no_new_privs()
    rc = install_filter(lib, [("sched_yield", errno.EPERM, None)])
    if rc < 0:
        return error_result(-rc, "seccomp_load")
    blocked = expect_errno(os.sched_yield, errno.EPERM)
    allowed = os.getpid() > 0
    mode = status_field("Seccomp")
    if blocked and allowed and mode == "2":
        return ok("filter returned EPERM for sched_yield and still allowed getpid", mode=2, arch=MACHINE)
    return result(
        "failed", "verification_failed",
        f"blocked={blocked} allowed={allowed} Seccomp={mode}: the filter did not have the expected effect",
    )


def op_seccomp_socket_deny(args):
    lib = load_libseccomp()
    if lib is None:
        return result("unavailable", "not_found", "libseccomp shared library not found")
    port = int(args["port"])
    before = connect_tcp(port)
    if before != 0:
        return result("failed", "baseline_failed", f"outside listener unreachable before the filter ({errno_name(before)})")
    set_no_new_privs()
    families = (socket.AF_INET, socket.AF_INET6, AF_PACKET)
    rc = install_filter(lib, [("socket", errno.EACCES, family) for family in families])
    if rc < 0:
        return error_result(-rc, "seccomp_load")
    inet = connect_tcp(port) == errno.EACCES
    inet6 = expect_errno(lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM).close(), errno.EACCES)
    unix = succeeds(lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close())
    if inet and inet6 and unix:
        return ok(
            "AF_INET/AF_INET6 sockets denied with EACCES; AF_UNIX still allowed",
            denied_families=["AF_INET", "AF_INET6", "AF_PACKET"],
        )
    return result("failed", "verification_failed", f"inet_denied={inet} inet6_denied={inet6} unix_allowed={unix}")


def op_no_new_privs(args):
    already = prctl(PR_GET_NO_NEW_PRIVS) == 1
    set_no_new_privs()
    if prctl(PR_GET_NO_NEW_PRIVS) == 1 and status_field("NoNewPrivs") == "1":
        return ok("PR_SET_NO_NEW_PRIVS set and confirmed" + (" (already set by the parent)" if already else ""))
    return result("failed", "verification_failed", "no_new_privs did not stick")


# Landlock


def landlock_abi() -> int:
    return raw_syscall(SYS_LANDLOCK_CREATE_RULESET, None, 0, LANDLOCK_CREATE_RULESET_VERSION)


def landlock_fs_rights(abi: int) -> int:
    """Every filesystem right the ABI knows: 13 in ABI 1, REFER in 2, TRUNCATE in 3, IOCTL_DEV in 5."""
    if abi >= 5:
        return (1 << 16) - 1
    if abi >= 3:
        return (1 << 15) - 1
    if abi >= 2:
        return (1 << 14) - 1
    return (1 << 13) - 1


def landlock_ruleset(fs_rights: int, net_rights: int = 0) -> int:
    attr = RulesetAttr(fs_rights, net_rights, 0)
    # Pass only the struct prefix this ABI needs; older kernels refuse a larger one.
    size = 8 if not net_rights else 16
    return raw_syscall(SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), size, 0)


def landlock_abi_error(exc: OSError):
    if exc.errno == errno.ENOSYS:
        return result("unsupported", "enosys", "kernel has no Landlock syscalls")
    if exc.errno == errno.EOPNOTSUPP:
        return result("unsupported", "disabled", "Landlock is built in but not enabled (check the lsm= boot parameter)")
    return error_result(exc.errno, "landlock_create_ruleset")


def op_landlock_abi(args):
    try:
        abi = landlock_abi()
    except OSError as exc:
        return landlock_abi_error(exc)
    if abi < 1:
        return result("unsupported", "abi_0", f"Landlock reported ABI {abi}")
    return ok(f"Landlock ABI {abi}", abi=abi)


def op_landlock_fs(args):
    allowed, denied = args["allowed"], args["denied"]
    try:
        abi = landlock_abi()
    except OSError as exc:
        return landlock_abi_error(exc)
    # The denied canary must be readable now, so a later refusal is Landlock's doing.
    if not succeeds(lambda: read_file(os.path.join(denied, "canary"))):
        return result("failed", "baseline_failed", "denied canary was unreadable before any restriction")
    rights = landlock_fs_rights(abi)
    set_no_new_privs()
    ruleset = landlock_ruleset(rights)
    parent = os.open(allowed, os.O_PATH | os.O_CLOEXEC)
    try:
        rule = PathBeneathAttr(rights, parent)
        raw_syscall(SYS_LANDLOCK_ADD_RULE, ruleset, LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(rule), 0)
    finally:
        os.close(parent)
    raw_syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset, 0)
    os.close(ruleset)
    checks = {
        "read_allowed": succeeds(lambda: read_file(os.path.join(allowed, "canary"))),
        "write_allowed": succeeds(lambda: write_file(os.path.join(allowed, "written"), CANARY)),
        "read_denied": expect_errno(lambda: read_file(os.path.join(denied, "canary")), errno.EACCES),
        "write_denied": expect_errno(lambda: write_file(os.path.join(denied, "written"), CANARY), errno.EACCES),
        "list_denied": expect_errno(lambda: os.listdir(denied), errno.EACCES),
    }
    failing = sorted(name for name, passed in checks.items() if not passed)
    if failing:
        return result("failed", "verification_failed", f"Landlock checks failed: {', '.join(failing)}")
    return ok(
        f"ABI {abi} ruleset allowed the workspace canary and denied read, write, and listing outside it",
        abi=abi,
        tested=sorted(checks),
    )


def op_landlock_tcp(args):
    port = int(args["port"])
    try:
        abi = landlock_abi()
    except OSError as exc:
        return landlock_abi_error(exc)
    if abi < 4:
        return result("unsupported", "abi_too_old", f"TCP rules need Landlock ABI 4; this kernel has ABI {abi}")
    before = connect_tcp(port)
    if before != 0:
        return result("failed", "baseline_failed", f"outside listener unreachable before the ruleset ({errno_name(before)})")
    set_no_new_privs()
    ruleset = landlock_ruleset(0, LANDLOCK_ACCESS_NET_BIND_TCP | LANDLOCK_ACCESS_NET_CONNECT_TCP)
    raw_syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset, 0)
    os.close(ruleset)
    after = connect_tcp(port)
    if after == errno.EACCES:
        return ok(f"ABI {abi} ruleset denied TCP connect to the outside listener", abi=abi)
    return result("failed", "verification_failed", f"connect after the ruleset gave {errno_name(after) if after else 'success'}")


# Namespaces without bubblewrap


def op_userns_net(args):
    port = int(args["port"])
    before = connect_tcp(port)
    if before != 0:
        return result("failed", "baseline_failed", f"outside listener unreachable before unshare ({errno_name(before)})")
    ctypes.set_errno(0)
    if _libc.unshare(CLONE_NEWUSER | CLONE_NEWNET) != 0:
        number = ctypes.get_errno()
        if number in (errno.EPERM, errno.EACCES, errno.ENOSPC):
            return userns_refusal(number)
        if number == errno.EINVAL:
            return result("unsupported", "einval", "kernel refused CLONE_NEWUSER|CLONE_NEWNET")
        return error_result(number, "unshare")
    after = connect_tcp(port)
    if after != 0:
        return ok(f"new user+network namespace; outside listener unreachable ({errno_name(after)})")
    return result("failed", "verification_failed", "the new network namespace still reached the outside listener")


# Resources and processes


def op_rlimits(args):
    scratch = args["scratch"]
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    core_zero = resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)

    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = 24 if hard == resource.RLIM_INFINITY else min(24, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limit))
    descriptors, emfile = [], False
    try:
        for _ in range(limit + 8):
            descriptors.append(os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC))
    except OSError as exc:
        emfile = exc.errno == errno.EMFILE
    finally:
        for descriptor in descriptors:
            os.close(descriptor)

    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))
    written, efbig = 0, False
    descriptor = os.open(os.path.join(scratch, "fsize"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC, 0o600)
    try:
        while written < 16384:
            written += os.write(descriptor, b"x" * 4096)
    except OSError as exc:
        efbig = exc.errno == errno.EFBIG
    finally:
        os.close(descriptor)

    if core_zero and emfile and efbig and written <= 4096:
        return ok(
            f"RLIMIT_NOFILE={limit} gave EMFILE, RLIMIT_FSIZE=4096 gave EFBIG, RLIMIT_CORE=0",
            tested=["RLIMIT_CORE", "RLIMIT_FSIZE", "RLIMIT_NOFILE"],
        )
    return result(
        "failed", "verification_failed",
        f"core_zero={core_zero} emfile={emfile} efbig={efbig} bytes_written={written}",
    )


def read_ready(fd: int, timeout_s: float) -> bytes:
    deadline = time.monotonic() + timeout_s
    data = b""
    while b"\n" not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("fixture did not report in time")
        ready, _, _ = select.select([fd], [], [], remaining)
        if ready:
            chunk = os.read(fd, 256)
            if not chunk:
                raise EOFError("fixture closed its pipe without reporting")
            data += chunk
    return data.split(b"\n", 1)[0]


def op_group_cleanup(args):
    """Supervise a child that forks a grandchild, kill the process group, and reap both."""
    lifetime = float(args["lifetime"])
    prctl(PR_SET_CHILD_SUBREAPER, 1)  # orphaned grandchildren come back to this process to be reaped
    read_end, write_end = os.pipe()
    child = os.fork()
    if child == 0:
        try:
            os.close(read_end)
            # Never hold the parent's pipes: a stuck fixture must not keep the probe waiting.
            null = os.open(os.devnull, os.O_RDWR)
            for descriptor in (0, 1, 2):
                os.dup2(null, descriptor)
            os.setsid()
            grandchild = os.fork()
            if grandchild == 0:
                os.write(write_end, f"{os.getpid()}\n".encode())
                os.close(write_end)
                time.sleep(lifetime)
                os._exit(0)
            os.close(write_end)
            time.sleep(lifetime)
        finally:
            os._exit(0)
    os.close(write_end)
    try:
        grandchild = int(read_ready(read_end, 5.0))
    finally:
        os.close(read_end)
    os.killpg(child, signal.SIGKILL)  # the child called setsid, so its pid is the group id
    pending = {child, grandchild}
    deadline = time.monotonic() + 5.0
    while pending and time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            time.sleep(0.01)
        else:
            pending.discard(pid)
    survivors = sorted(pending)
    for pid in survivors:  # unreaped, so these pids cannot have been reused
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass
    if survivors:
        return result("failed", "survivors", f"{len(survivors)} process(es) survived the process-group kill")
    return ok(
        "killpg reaped a child and its grandchild; a grandchild that calls setsid would escape this mechanism",
        tree=["child", "grandchild"],
    )


def op_tree_fixture(args):
    """Wait for ``go`` on stdin, fork a grandchild (optionally in its own session), report both pids, sleep."""
    lifetime = float(args["lifetime"])
    if sys.stdin.readline().strip() != "go":
        return 2
    read_end, write_end = os.pipe()
    grandchild = os.fork()
    if grandchild == 0:
        os.close(read_end)
        null = os.open(os.devnull, os.O_RDWR)
        os.dup2(null, 0)
        os.dup2(null, 1)
        if args.get("escape_session"):
            os.setsid()
        os.write(write_end, b"ready\n")
        os.close(write_end)
        time.sleep(lifetime)
        os._exit(0)
    os.close(write_end)
    read_ready(read_end, 5.0)
    os.close(read_end)
    sys.stdout.write(json.dumps({"child": os.getpid(), "grandchild": grandchild}) + "\n")
    sys.stdout.flush()
    time.sleep(lifetime)
    return 0


def op_openat2(args):
    root = args["root"]
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)

    def openat2(path: bytes, resolve: int) -> None:
        how = OpenHow(os.O_RDONLY | os.O_CLOEXEC, 0, resolve)
        fd = raw_syscall(SYS_OPENAT2, directory, ctypes.c_char_p(path), ctypes.byref(how), ctypes.sizeof(how))
        os.close(fd)

    try:
        try:
            openat2(b"canary", RESOLVE_BENEATH)
        except OSError as exc:
            if exc.errno == errno.ENOSYS:
                return result("unsupported", "enosys", "kernel has no openat2")
            return error_result(exc.errno, "openat2")
        beneath = expect_errno(lambda: openat2(b"../outside", RESOLVE_BENEATH), errno.EXDEV)
        no_symlinks = expect_errno(lambda: openat2(b"link", RESOLVE_NO_SYMLINKS), errno.ELOOP)
    finally:
        os.close(directory)
    if beneath and no_symlinks:
        return ok(
            "openat2 opened a file beneath the root and refused ../ and symlink escapes",
            resolve=["RESOLVE_BENEATH", "RESOLVE_NO_SYMLINKS"],
        )
    return result("failed", "verification_failed", f"beneath_refused={beneath} symlink_refused={no_symlinks}")


# Operations run inside bubblewrap (the parent adds the bwrap argv)


def op_bwrap_namespaces(args):
    host = args["host_ns"]
    fresh = sorted(name for name, link in host.items() if os.readlink(f"/proc/self/ns/{name}") != link)
    shared = sorted(set(host) - set(fresh))
    visible = [entry for entry in os.listdir("/proc") if entry.isdigit()]
    fresh_proc = len(visible) <= 4
    no_new_privs = status_field("NoNewPrivs") == "1"
    cap_eff = status_field("CapEff")
    no_capabilities = cap_eff is not None and int(cap_eff, 16) == 0
    fresh_tmp = os.listdir("/tmp") == []
    facts = {
        "namespaces": fresh,
        "fresh_proc": fresh_proc,
        "fresh_tmp": fresh_tmp,
        "no_new_privs": no_new_privs,
        "no_capabilities": no_capabilities,
    }
    if not shared and fresh_proc and fresh_tmp and no_new_privs and no_capabilities:
        return ok(f"fresh {', '.join(fresh)} namespaces; {len(visible)} pid(s) in /proc; no_new_privs; no capabilities", **facts)
    return result(
        "failed", "verification_failed",
        f"shared namespaces={shared} pids={len(visible)} fresh_tmp={fresh_tmp} no_new_privs={no_new_privs} CapEff={cap_eff}",
        **facts,
    )


def op_bwrap_mounts(args):
    rw, ro = args["rw"], args["ro"]
    checks = {
        "rw_bind": succeeds(lambda: write_file(os.path.join(rw, "written"), CANARY)),
        "ro_bind_read": succeeds(lambda: read_file(os.path.join(ro, "canary"))),
        "ro_bind_write": expect_errno(lambda: write_file(os.path.join(ro, "written"), CANARY), errno.EROFS),
        "masked_subpath": expect_errno(lambda: read_file(os.path.join(ro, "denied", "canary")), errno.ENOENT),
    }
    failing = sorted(name for name, passed in checks.items() if not passed)
    if failing:
        return result("failed", "verification_failed", f"mount checks failed: {', '.join(failing)}")
    return ok("writable bind, read-only bind (EROFS), and a masked subpath (ENOENT) behaved", tested=sorted(checks))


def op_bwrap_network(args):
    port, path = int(args["port"]), args["socket"]
    tcp = connect_tcp(port)
    interfaces = sorted(name for _, name in socket.if_nameindex())
    loopback_only = set(interfaces) <= {"lo"}
    unix_error = 0
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(CONNECT_TIMEOUT_S)
        sock.connect(path)
    except OSError as exc:
        unix_error = exc.errno or errno.EIO
    finally:
        sock.close()
    if tcp != 0 and loopback_only:
        netns = ok(f"only {interfaces or ['no interfaces']}; outside listener unreachable ({errno_name(tcp)})", interfaces=interfaces)
    else:
        netns = result(
            "failed", "verification_failed",
            f"interfaces={interfaces} tcp={'connected' if tcp == 0 else errno_name(tcp)}", interfaces=interfaces,
        )
    if unix_error == 0 and tcp != 0:
        unix = ok("bind-mounted Unix socket accepted a connection while TCP stayed blocked")
    elif unix_error:
        unix = error_result(unix_error, "connect to the bind-mounted Unix socket")
    else:
        unix = result("failed", "verification_failed", "TCP reached the outside listener")
    return {"bwrap.netns": netns, "bwrap.unix_socket": unix}


OPERATIONS = {
    "libseccomp": (("libseccomp.library",), op_libseccomp),
    "seccomp-filter": (("seccomp.filter",), op_seccomp_filter),
    "seccomp-socket-deny": (("seccomp.socket_deny",), op_seccomp_socket_deny),
    "no-new-privs": (("no_new_privs",), op_no_new_privs),
    "landlock-abi": (("landlock.abi",), op_landlock_abi),
    "landlock-fs": (("landlock.fs",), op_landlock_fs),
    "landlock-tcp": (("landlock.tcp",), op_landlock_tcp),
    "userns-net": (("userns.unshare",), op_userns_net),
    "rlimits": (("rlimits",), op_rlimits),
    "group-cleanup": (("process.group_cleanup",), op_group_cleanup),
    "openat2": (("fs.openat2",), op_openat2),
    "bwrap-namespaces": (("bwrap.namespaces",), op_bwrap_namespaces),
    "bwrap-mounts": (("bwrap.mounts",), op_bwrap_mounts),
    "bwrap-network": (("bwrap.netns", "bwrap.unix_socket"), op_bwrap_network),
}


def main(argv: list[str]) -> int:
    operation = argv[1] if len(argv) > 1 else ""
    args = json.loads(argv[2]) if len(argv) > 2 else {}
    if operation == "tree-fixture":
        return op_tree_fixture(args)
    if operation not in OPERATIONS:
        sys.stderr.write(f"unknown operation {operation!r}\n")
        return 2
    names, function = OPERATIONS[operation]
    try:
        outcome = function(args)
        results = {names[0]: outcome} if "status" in outcome else outcome
    except Exception as exc:  # reported, never raised: the parent turns it into a failed record
        if isinstance(exc, OSError) and exc.errno:
            failure = error_result(exc.errno, f"{operation}: {type(exc).__name__}")
        else:
            failure = result("failed", "child_exception", f"{type(exc).__name__}: {exc}")
        results = {name: failure for name in names}
    sys.stdout.write(json.dumps({"results": results}, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

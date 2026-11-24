"""seccomp filters for sandboxed programs, compiled with the system libseccomp through ctypes.

The filter is compiled in the supervisor but never loaded there. It is exported
as classic BPF and handed to bubblewrap on a dedicated descriptor
(``--seccomp FD``), which installs it in the single-threaded sandbox child after
no_new_privs and immediately before exec. Nothing here uses ``preexec_fn``.

The policy (``DEFAULT_POLICY``), reviewed for CPython and a POSIX shell:

- An allowlist of the syscalls an interpreter, a shell, and coreutils need.
  Anything else returns ``ENOSYS``, so newer interfaces fall back cleanly
  (``clone3`` returns ``ENOSYS`` on purpose: glibc then uses ``clone``, whose
  flags the filter can inspect).
- Bypass and primitive classes return ``EPERM`` by explicit rule, so a test
  can tell a deliberate denial from an unknown syscall: cross-process memory
  and tracing (``ptrace``, ``process_vm_*``, ``pidfd_getfd``, ``kcmp``),
  ``io_uring``, ``bpf``, ``perf_event_open``, keyrings, kernel modules,
  mounts and namespace changes (``mount``, ``pivot_root``, ``setns``,
  ``unshare``, and ``clone`` with any namespace flag), ``open_by_handle_at``,
  ``reboot``/kexec, ``userfaultfd``, and privilege, clock, and host identity
  changes.
- ``socket`` and ``socketpair`` are allowed for ``AF_UNIX`` only; every other
  family gets ``EACCES`` (the network namespace already isolates the rest).
  ``T1_POLICY`` (no namespaces) also refuses ``socket()`` entirely and
  ``setsid``/``setpgid``, keeping every process in the group cleanup kills.
- A syscall from any other architecture or ABI (i386 ``int 0x80``, x32) kills
  the process (``SECCOMP_RET_KILL_PROCESS``); the backend records the SIGSYS as
  a seccomp violation.

seccomp authorizes syscall numbers and register values, never paths: BPF sees
pointer arguments, not the strings they point to. Filesystem visibility belongs
to the mount namespace (part 4) and Landlock (part 5). seccomp user
notification is not used. Only x86_64 and aarch64 are supported; anywhere else
compilation fails rather than producing a filter for the wrong syscall table.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import hashlib
import os
import platform
import struct
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType

from .errors import BulkheadError

# libseccomp constants (seccomp.h)
SCMP_ACT_KILL_PROCESS = 0x80000000
SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_ERRNO_BASE = 0x00050000
SCMP_FLTATR_ACT_BADARCH = 2
SCMP_CMP_NE = 1
SCMP_CMP_EQ = 4
SCMP_CMP_MASKED_EQ = 7

# Kernel return-action values (linux/seccomp.h), for reading a filter's decision.
RET_ACTION_MASK = 0xFFFF0000
RET_DATA_MASK = 0x0000FFFF
RET_KILL_PROCESS = 0x80000000
RET_KILL_THREAD = 0x00000000
RET_TRAP = 0x00030000
RET_ERRNO = 0x00050000
RET_TRACE = 0x7FF00000
RET_LOG = 0x7FFC0000
RET_ALLOW = 0x7FFF0000

# AUDIT_ARCH_* values for the supported machines, and two foreign ABIs tests use.
ARCHITECTURES: Mapping[str, int] = MappingProxyType({"x86_64": 0xC000003E, "aarch64": 0xC00000B7})
AUDIT_ARCH_I386 = 0x40000003
AUDIT_ARCH_ARM = 0x40000028
X32_SYSCALL_BIT = 0x40000000

AF_UNIX = 1
AF_INET = 2
AF_INET6 = 10
# Families a network-enabled tier-2 call may create: its network namespace has only loopback,
# so AF_INET/AF_INET6 reach nothing but the in-sandbox bridge to the outside proxy.
NET_FAMILIES = (AF_UNIX, AF_INET, AF_INET6)
MAX_FAMILY = 64
# Namespace flags refused on clone(). CLONE_NEWTIME (0x80) is absent: on clone() the low
# byte is the exit signal, and a new time namespace needs clone3 or unshare, both refused.
CLONE_NAMESPACE_FLAGS: Mapping[str, int] = MappingProxyType({
    "CLONE_NEWNS": 0x00020000,
    "CLONE_NEWCGROUP": 0x02000000,
    "CLONE_NEWUTS": 0x04000000,
    "CLONE_NEWIPC": 0x08000000,
    "CLONE_NEWUSER": 0x10000000,
    "CLONE_NEWPID": 0x20000000,
    "CLONE_NEWNET": 0x40000000,
})
CLONE_NAMESPACE_MASK = sum(CLONE_NAMESPACE_FLAGS.values())

ALLOWED_SYSCALLS: tuple[str, ...] = (
    # Files and descriptors (paths are the mount namespace's business, not seccomp's).
    "read", "write", "readv", "writev", "pread64", "pwrite64", "preadv", "preadv2", "pwritev", "pwritev2",
    "open", "openat", "openat2", "creat", "close", "close_range", "lseek", "dup", "dup2", "dup3", "fcntl", "flock", "ioctl",
    "stat", "fstat", "lstat", "newfstatat", "statx", "statfs", "fstatfs", "access", "faccessat", "faccessat2",
    "readlink", "readlinkat", "getdents", "getdents64", "getcwd", "chdir", "fchdir", "umask",
    "mkdir", "mkdirat", "rmdir", "unlink", "unlinkat", "rename", "renameat", "renameat2",
    "link", "linkat", "symlink", "symlinkat",
    "chmod", "fchmod", "fchmodat", "fchmodat2", "chown", "fchown", "lchown", "fchownat",
    "truncate", "ftruncate", "fallocate", "fsync", "fdatasync", "sync", "syncfs", "sync_file_range",
    "utime", "utimes", "utimensat", "futimesat", "mknod", "mknodat",
    "getxattr", "lgetxattr", "fgetxattr", "listxattr", "llistxattr", "flistxattr",
    "setxattr", "lsetxattr", "fsetxattr", "removexattr", "lremovexattr", "fremovexattr",
    "sendfile", "splice", "tee", "copy_file_range", "readahead", "fadvise64", "memfd_create",
    "pipe", "pipe2", "select", "pselect6", "poll", "ppoll",
    "epoll_create", "epoll_create1", "epoll_ctl", "epoll_wait", "epoll_pwait", "epoll_pwait2",
    "eventfd", "eventfd2", "signalfd", "signalfd4", "timerfd_create", "timerfd_settime", "timerfd_gettime",
    "inotify_init", "inotify_init1", "inotify_add_watch", "inotify_rm_watch",
    # Memory.
    "brk", "mmap", "munmap", "mremap", "mprotect", "madvise", "mincore", "msync",
    "mlock", "mlock2", "munlock", "mlockall", "munlockall", "membarrier",
    "pkey_alloc", "pkey_free", "pkey_mprotect", "map_shadow_stack", "mseal",
    # Processes, threads, and signals (clone is checked separately, by flags).
    "execve", "execveat", "exit", "exit_group", "fork", "vfork", "wait4", "waitid", "kill", "tkill", "tgkill",
    "getpid", "getppid", "gettid", "getpgid", "setpgid", "getpgrp", "getsid", "setsid",
    "getuid", "geteuid", "getgid", "getegid", "getresuid", "getresgid", "getgroups", "capget",
    "prctl", "arch_prctl", "set_tid_address", "set_robust_list", "get_robust_list", "rseq",
    "futex", "futex_waitv", "futex_wake", "futex_wait", "futex_requeue",
    "sched_yield", "sched_getaffinity", "sched_setaffinity", "sched_getparam", "sched_getscheduler",
    "sched_get_priority_max", "sched_get_priority_min", "sched_rr_get_interval", "sched_getattr",
    "getpriority", "setpriority", "nanosleep", "clock_nanosleep", "clock_gettime", "clock_getres",
    "gettimeofday", "time", "times", "getrusage", "getrlimit", "setrlimit", "prlimit64",
    "sysinfo", "uname", "getcpu", "getrandom", "alarm", "pause",
    "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "rt_sigsuspend", "rt_sigpending", "rt_sigtimedwait",
    "rt_sigqueueinfo", "rt_tgsigqueueinfo", "sigaltstack", "restart_syscall",
    "pidfd_open", "pidfd_send_signal",
    "timer_create", "timer_settime", "timer_gettime", "timer_getoverrun", "timer_delete", "getitimer", "setitimer",
    # System V and POSIX IPC, private to the sandbox's IPC namespace.
    "shmget", "shmat", "shmdt", "shmctl", "semget", "semop", "semctl", "semtimedop",
    "msgget", "msgsnd", "msgrcv", "msgctl",
    "mq_open", "mq_unlink", "mq_timedsend", "mq_timedreceive", "mq_notify", "mq_getsetattr",
    # Sockets (socket and socketpair are checked separately, by family).
    "bind", "listen", "accept", "accept4", "connect", "getsockname", "getpeername",
    "sendto", "recvfrom", "sendmsg", "recvmsg", "sendmmsg", "recvmmsg", "shutdown", "setsockopt", "getsockopt",
)

# Bypass and primitive classes, refused with EPERM by explicit rule. Grouped for review.
BLOCKED_CLASSES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "cross_process": (
        "ptrace", "process_vm_readv", "process_vm_writev", "process_madvise", "pidfd_getfd", "kcmp",
        "move_pages", "migrate_pages",
    ),
    "io_uring": ("io_uring_setup", "io_uring_enter", "io_uring_register"),
    "kernel_primitives": (
        "bpf", "perf_event_open", "keyctl", "add_key", "request_key", "userfaultfd", "lookup_dcookie",
        "syslog", "acct", "quotactl", "quotactl_fd", "nfsservctl", "iopl", "ioperm", "uselib", "personality",
        "vhangup", "swapon", "swapoff", "mbind", "set_mempolicy", "_sysctl", "sysfs", "ustat",
    ),
    "kernel_modules_and_boot": (
        "init_module", "finit_module", "delete_module", "create_module", "query_module", "get_kernel_syms",
        "reboot", "kexec_load", "kexec_file_load",
    ),
    "mounts_and_namespaces": (
        "mount", "umount", "umount2", "pivot_root", "chroot", "move_mount", "open_tree", "fsopen", "fsmount",
        "fsconfig", "fspick", "mount_setattr", "setns", "unshare", "open_by_handle_at", "name_to_handle_at",
    ),
    "privilege_clock_identity": (
        "setuid", "setgid", "setreuid", "setregid", "setresuid", "setresgid", "setfsuid", "setfsgid",
        "setgroups", "capset", "settimeofday", "clock_settime", "clock_adjtime", "adjtimex", "stime",
        "sethostname", "setdomainname",
    ),
})


class SeccompUnavailable(BulkheadError, RuntimeError):
    """No usable libseccomp, or an architecture this module does not support. Never a reason to run unfiltered."""


@dataclass(frozen=True)
class SyscallPolicy:
    """What a filter allows (``allow``), refuses by explicit rule (``deny``: name to errno), and how it treats the rest."""

    allow: tuple[str, ...] = ALLOWED_SYSCALLS
    deny: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({name: errno.EPERM for names in BLOCKED_CLASSES.values() for name in names})
    )
    default_errno: int = errno.ENOSYS
    socket_families: tuple[int, ...] = (AF_UNIX,)
    socket_errno: int = errno.EACCES
    clone_namespace_errno: int = errno.EPERM

    def __post_init__(self) -> None:
        overlap = set(self.allow) & set(self.deny)
        if overlap:
            raise ValueError(f"syscalls both allowed and denied: {sorted(overlap)}")
        reserved = {"clone", "clone3", "socket", "socketpair"} & (set(self.allow) | set(self.deny))
        if reserved:
            raise ValueError(f"{sorted(reserved)} are handled by dedicated rules")
        if self.socket_families not in ((AF_UNIX,), (), NET_FAMILIES):
            raise ValueError("socket() may allow AF_UNIX, nothing, or (inside a network namespace) AF_UNIX/AF_INET/AF_INET6")

    def sha256(self) -> str:
        text = repr((sorted(self.allow), sorted(self.deny.items()), self.default_errno, self.socket_families,
                     self.socket_errno, self.clone_namespace_errno))
        return hashlib.sha256(text.encode()).hexdigest()


DEFAULT_POLICY = SyscallPolicy()

# Tier 1 has no namespaces, so two more things are refused. socket() entirely: without a
# network namespace a Unix socket could reach host services (D-Bus, agents, abstract
# sockets); socketpair(AF_UNIX) stays. And setsid/setpgid: the process group is tier 1's
# cleanup handle when there is no cgroup, so no process may leave it.
NET_POLICY = SyscallPolicy(socket_families=NET_FAMILIES)

T1_SESSION_ESCAPES = ("setsid", "setpgid")
T1_POLICY = SyscallPolicy(
    allow=tuple(name for name in ALLOWED_SYSCALLS if name not in T1_SESSION_ESCAPES),
    deny=MappingProxyType({**DEFAULT_POLICY.deny, **{name: errno.EPERM for name in T1_SESSION_ESCAPES}}),
    socket_families=(),
)


@dataclass(frozen=True)
class CompiledFilter:
    """A filter exported as classic BPF for one architecture, plus what was left out because the arch lacks it."""

    arch: str
    audit_arch: int
    bpf: bytes
    policy_sha256: str
    skipped: tuple[str, ...]
    libseccomp_version: str | None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.bpf).hexdigest()

    @property
    def instructions(self) -> int:
        return len(self.bpf) // 8

    def memfd(self) -> int:
        """A fresh close-on-exec descriptor holding the BPF program at offset 0, for ``bwrap --seccomp FD``.

        The caller passes it to the child with ``pass_fds`` and closes it afterwards.
        """
        fd = os.memfd_create("bulkhead-seccomp", os.MFD_CLOEXEC)
        try:
            view = memoryview(self.bpf)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.lseek(fd, 0, os.SEEK_SET)
        except BaseException:
            os.close(fd)
            raise
        return fd


# libseccomp through ctypes


class _ArgCmp(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int), ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


class _Version(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint), ("minor", ctypes.c_uint), ("micro", ctypes.c_uint)]


@lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    for name in ("libseccomp.so.2", ctypes.util.find_library("seccomp")):
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name, use_errno=True)
        except OSError:
            continue
        lib.seccomp_init.argtypes = [ctypes.c_uint32]
        lib.seccomp_init.restype = ctypes.c_void_p
        lib.seccomp_release.argtypes = [ctypes.c_void_p]
        lib.seccomp_release.restype = None
        lib.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
        lib.seccomp_attr_set.restype = ctypes.c_int
        lib.seccomp_arch_native.argtypes = []
        lib.seccomp_arch_native.restype = ctypes.c_uint32
        lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
        lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
        lib.seccomp_rule_add_array.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_ArgCmp),
        ]
        lib.seccomp_rule_add_array.restype = ctypes.c_int
        lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.seccomp_export_bpf.restype = ctypes.c_int
        return lib
    raise SeccompUnavailable("the libseccomp shared library was not found")


def libseccomp_version() -> str | None:
    lib = _library()
    try:
        function = lib.seccomp_version
    except AttributeError:  # older than 2.3
        return None
    function.restype = ctypes.POINTER(_Version)
    version = function().contents
    return f"{version.major}.{version.minor}.{version.micro}"


def native_architecture() -> tuple[str, int]:
    """This machine's name and AUDIT_ARCH value; refuses anything but x86_64 and aarch64."""
    machine = platform.machine().lower()
    if machine not in ARCHITECTURES:
        raise SeccompUnavailable(f"no reviewed seccomp policy for architecture {machine!r}")
    native = _library().seccomp_arch_native()
    if native != ARCHITECTURES[machine]:
        raise SeccompUnavailable(f"libseccomp's native architecture {native:#x} is not {machine}'s {ARCHITECTURES[machine]:#x}")
    return machine, ARCHITECTURES[machine]


def _errno_action(number: int) -> int:
    return SCMP_ACT_ERRNO_BASE | (number & RET_DATA_MASK)


def _add(lib: ctypes.CDLL, ctx: int, action: int, number: int, comparisons: Sequence[tuple[int, int, int, int]] = ()) -> None:
    array = (_ArgCmp * len(comparisons))(*(_ArgCmp(*item) for item in comparisons)) if comparisons else None
    rc = lib.seccomp_rule_add_array(ctx, action, number, len(comparisons), array)
    if rc < 0:
        raise SeccompUnavailable(f"libseccomp refused a rule for syscall {number}: {os.strerror(-rc)}")


_COMPILED: dict[str, CompiledFilter] = {}
_COMPILED_LOCK = threading.Lock()


def compile_filter(policy: SyscallPolicy = DEFAULT_POLICY) -> CompiledFilter:
    """Compile ``policy`` for this machine and export it as classic BPF. Loads nothing into this process.

    Compiled programs are cached by policy hash; compiling is thread-safe.
    """
    key = policy.sha256()
    with _COMPILED_LOCK:
        if key not in _COMPILED:
            _COMPILED[key] = _compile(policy)
        return _COMPILED[key]


def _compile(policy: SyscallPolicy) -> CompiledFilter:
    lib = _library()
    machine, audit_arch = native_architecture()
    ctx = lib.seccomp_init(_errno_action(policy.default_errno))
    if not ctx:
        raise SeccompUnavailable("seccomp_init failed")
    skipped: list[str] = []

    def resolve(name: str) -> int | None:
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if number < 0:  # unknown, or a pseudo-syscall this architecture does not have
            skipped.append(name)
            return None
        return number

    try:
        if lib.seccomp_attr_set(ctx, SCMP_FLTATR_ACT_BADARCH, SCMP_ACT_KILL_PROCESS) < 0:
            raise SeccompUnavailable("libseccomp cannot kill on a foreign architecture (needs 2.4 or newer)")
        for name in policy.allow:
            if (number := resolve(name)) is not None:
                _add(lib, ctx, SCMP_ACT_ALLOW, number)
        for name, err in sorted(policy.deny.items()):
            if (number := resolve(name)) is not None:
                _add(lib, ctx, _errno_action(err), number)
        if (clone := resolve("clone")) is not None:
            # The flags are argument 0 on x86_64 and aarch64 (the only supported architectures).
            _add(lib, ctx, SCMP_ACT_ALLOW, clone, [(0, SCMP_CMP_MASKED_EQ, CLONE_NAMESPACE_MASK, 0)])
            for flag in CLONE_NAMESPACE_FLAGS.values():
                _add(lib, ctx, _errno_action(policy.clone_namespace_errno), clone, [(0, SCMP_CMP_MASKED_EQ, flag, flag)])
        # clone3 is deliberately left to the default (ENOSYS): its flags live in memory the filter cannot read.
        for name, families in (("socket", policy.socket_families), ("socketpair", (AF_UNIX,))):
            if (number := resolve(name)) is None:
                continue
            if not families:
                _add(lib, ctx, _errno_action(policy.socket_errno), number)
                continue
            # One rule per family: allowed ones pass, every other known family gets EACCES, and
            # anything beyond falls to the default (ENOSYS). Still a closed set either way.
            for family in range(MAX_FAMILY):
                action = SCMP_ACT_ALLOW if family in families else _errno_action(policy.socket_errno)
                _add(lib, ctx, action, number, [(0, SCMP_CMP_EQ, family, 0)])
        fd = os.memfd_create("bulkhead-seccomp-export", os.MFD_CLOEXEC)
        try:
            rc = lib.seccomp_export_bpf(ctx, fd)
            if rc < 0:
                raise SeccompUnavailable(f"seccomp_export_bpf failed: {os.strerror(-rc)}")
            size = os.fstat(fd).st_size
            os.lseek(fd, 0, os.SEEK_SET)
            chunks = []
            while size > 0:
                chunk = os.read(fd, size)
                if not chunk:
                    break
                chunks.append(chunk)
                size -= len(chunk)
            bpf = b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        lib.seccomp_release(ctx)
    if not bpf or len(bpf) % 8:
        raise SeccompUnavailable("libseccomp exported an empty or malformed program")
    return CompiledFilter(
        arch=machine,
        audit_arch=audit_arch,
        bpf=bpf,
        policy_sha256=policy.sha256(),
        skipped=tuple(sorted(set(skipped))),
        libseccomp_version=libseccomp_version(),
    )


def syscall_number(name: str) -> int | None:
    """This machine's number for a syscall name, or None if it has none."""
    number = _library().seccomp_syscall_resolve_name(name.encode())
    return number if number >= 0 else None


# Reading a compiled filter without loading it


_SECCOMP_DATA = struct.Struct("=iIQ6Q")  # nr, arch, instruction_pointer, args[6]
_INSTRUCTION = struct.Struct("=HBBI")
MAX_SIMULATED_STEPS = 65536


def simulate(bpf: bytes, nr: int, *, arch: int, args: Sequence[int] = (), instruction_pointer: int = 0) -> int:
    """Run a classic BPF seccomp program on one synthetic syscall and return the kernel action it decides.

    A deterministic way to check what a compiled filter does for any syscall,
    argument, or architecture, on any host, without installing it. It
    interprets the instruction subset seccomp programs use.
    """
    values = [int(value) & 0xFFFFFFFFFFFFFFFF for value in args] + [0] * (6 - len(args))
    data = _SECCOMP_DATA.pack(nr, arch & 0xFFFFFFFF, instruction_pointer, *values[:6])
    program = [_INSTRUCTION.unpack_from(bpf, offset) for offset in range(0, len(bpf), 8)]
    accumulator = index_register = 0
    memory = [0] * 16
    pc = 0
    for _ in range(MAX_SIMULATED_STEPS):
        if not 0 <= pc < len(program):
            raise ValueError(f"the program jumped outside itself (pc={pc})")
        code, jt, jf, k = program[pc]
        cls = code & 0x07
        if cls == 0x00:  # BPF_LD
            mode = code & 0xE0
            if mode == 0x20:  # ABS
                accumulator = struct.unpack_from("=I", data, k)[0]
            elif mode == 0x00:  # IMM
                accumulator = k
            elif mode == 0x60:  # MEM
                accumulator = memory[k]
            elif mode == 0x80:  # LEN
                accumulator = len(data)
            else:
                raise ValueError(f"unsupported load mode {code:#x}")
        elif cls == 0x01:  # BPF_LDX
            mode = code & 0xE0
            if mode == 0x00:  # IMM
                index_register = k
            elif mode == 0x60:  # MEM
                index_register = memory[k]
            elif mode == 0x80:  # LEN
                index_register = len(data)
            else:
                raise ValueError(f"unsupported index load {code:#x}")
        elif cls == 0x02:  # BPF_ST
            memory[k] = accumulator
        elif cls == 0x03:  # BPF_STX
            memory[k] = index_register
        elif cls == 0x04:  # BPF_ALU
            operand = index_register if code & 0x08 else k
            operation = code & 0xF0
            results = {
                0x00: lambda a: a + operand, 0x10: lambda a: a - operand, 0x20: lambda a: a * operand,
                0x30: lambda a: a // operand, 0x40: lambda a: a | operand, 0x50: lambda a: a & operand,
                0x60: lambda a: a << operand, 0x70: lambda a: a >> operand, 0x80: lambda a: -a,
                0x90: lambda a: a % operand, 0xA0: lambda a: a ^ operand,
            }
            if operation not in results:
                raise ValueError(f"unsupported ALU operation {code:#x}")
            accumulator = results[operation](accumulator) & 0xFFFFFFFF
        elif cls == 0x05:  # BPF_JMP
            operation = code & 0xF0
            operand = index_register if code & 0x08 else k
            if operation == 0x00:  # JA
                pc += 1 + k
                continue
            taken = {
                0x10: accumulator == operand,
                0x20: accumulator > operand,
                0x30: accumulator >= operand,
                0x40: bool(accumulator & operand),
            }.get(operation)
            if taken is None:
                raise ValueError(f"unsupported jump {code:#x}")
            pc += 1 + (jt if taken else jf)
            continue
        elif cls == 0x06:  # BPF_RET
            return accumulator if code & 0x18 == 0x10 else k
        elif cls == 0x07:  # BPF_MISC
            if code & 0xF8 == 0x00:  # TAX
                index_register = accumulator
            else:  # TXA
                accumulator = index_register
        pc += 1
    raise ValueError("the program did not return")


def describe(action: int) -> str:
    """A kernel action as text: ``allow``, ``errno:EPERM``, ``kill_process``, ..."""
    kind = action & RET_ACTION_MASK
    if kind == RET_ERRNO:
        number = action & RET_DATA_MASK
        return f"errno:{errno.errorcode.get(number, number)}"
    return {
        RET_ALLOW: "allow",
        RET_KILL_PROCESS: "kill_process",
        RET_KILL_THREAD: "kill_thread",
        RET_TRAP: "trap",
        RET_TRACE: "trace",
        RET_LOG: "log",
    }.get(kind, f"unknown:{action:#x}")


__all__ = [
    "ALLOWED_SYSCALLS",
    "ARCHITECTURES",
    "BLOCKED_CLASSES",
    "CLONE_NAMESPACE_FLAGS",
    "DEFAULT_POLICY",
    "NET_POLICY",
    "T1_POLICY",
    "CompiledFilter",
    "SeccompUnavailable",
    "SyscallPolicy",
    "compile_filter",
    "describe",
    "libseccomp_version",
    "native_architecture",
    "simulate",
    "syscall_number",
]

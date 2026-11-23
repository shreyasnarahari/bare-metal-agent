"""The trusted helper: single-threaded setup immediately before exec, for every live backend.

The supervisor runs it as ``python -I -m bulkhead.helper CONFIG_FD``: a fresh
interpreter (no threads, no ``preexec_fn`` in the multithreaded agent), with
the minimal environment the backend chose. The JSON config arrives on a
descriptor, never on the command line. In order, the helper:

1. loads every module it needs and reads its config, then closes the config
   and every descriptor except stdin/stdout/stderr, the status pipe, the
   start barrier, and those the config names (bubblewrap's seccomp memfd);
2. sets ``PR_SET_PDEATHSIG`` (it dies with the supervisor) and checks the
   supervisor is still its parent;
3. sets ``no_new_privs``;
4. (tier 1) creates the Landlock ruleset and adds its rules while the rule
   paths can still be opened;
5. lowers the rlimits (``bulkhead.rlimits``);
6. (tier 1) calls ``landlock_restrict_self`` and verifies the restriction
   holds (listing ``/`` must now be refused);
7. (tier 1) installs the seccomp filter; on tier 2 bubblewrap installs it in
   the sandbox instead;
8. writes ``R`` to the status pipe and, when the call has a cgroup, waits on
   the start barrier until the supervisor has moved it into the leaf and
   verified the move;
9. execs the fixed target argv: the launcher (tier 1) or bubblewrap (tier 2).

Any failure writes ``bulkhead-helper: <code>`` to stderr and exits 125
without exec, so the run is a setup failure, never a partially restricted
program. Nothing here is ever skipped "just this once".
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import sys

from . import landlock, rlimits

PR_SET_PDEATHSIG = 1
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
EXIT_SETUP = 125
MAX_CONFIG_BYTES = 256 * 1024
CONFIG_KEYS = frozenset({"status_fd", "barrier_fd", "seccomp_fd", "keep_fds", "landlock", "rlimits", "target", "supervisor_pid"})


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


class HelperFailure(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _read_all(fd: int, limit: int) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise HelperFailure("config_too_large")
        chunks.append(chunk)


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    return libc


def _close_others(keep: set[int]) -> None:
    for name in os.listdir("/proc/self/fd"):
        fd = int(name)
        if fd not in keep:
            try:
                os.close(fd)
            except OSError:
                pass


def _install_seccomp(libc: ctypes.CDLL, fd: int) -> None:
    program = _read_all(fd, 4096 * 8)
    os.close(fd)
    if not program or len(program) % 8:
        raise HelperFailure("seccomp_malformed")
    buffer = ctypes.create_string_buffer(program, len(program))
    fprog = _SockFprog(len(program) // 8, ctypes.cast(buffer, ctypes.c_void_p))
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.addressof(fprog), 0, 0) != 0:
        raise HelperFailure("seccomp_failed")


def run(config: dict) -> None:
    """Steps 2 to 9 (see the module docstring). Returns only by raising ``HelperFailure``."""
    if set(config) != CONFIG_KEYS:
        raise HelperFailure("config_fields")
    status_fd = int(config["status_fd"])
    barrier_fd = config["barrier_fd"]
    seccomp_fd = config["seccomp_fd"]
    target = [str(item) for item in config["target"]]
    if not target or not os.path.isabs(target[0]):
        raise HelperFailure("target")
    keep = {0, 1, 2, status_fd, *[int(fd) for fd in config["keep_fds"]]}
    for fd in (barrier_fd, seccomp_fd):
        if fd is not None:
            keep.add(int(fd))
    _close_others(keep)
    libc = _libc()
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != int(config["supervisor_pid"]):
        raise HelperFailure("parent_gone")
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise HelperFailure("no_new_privs")
    ruleset_fd = None
    if config["landlock"] is not None:
        try:
            ruleset_fd = landlock.prepare(landlock.Ruleset.from_dict(config["landlock"]))
        except landlock.LandlockError as exc:
            raise HelperFailure(exc.code.rsplit(".", 1)[-1]) from None
    try:
        rlimits.apply(rlimits.RlimitPlan.from_dict(config["rlimits"]))
    except (ValueError, OSError):
        raise HelperFailure("rlimits") from None
    if ruleset_fd is not None:
        try:
            landlock.restrict_self(ruleset_fd)
        except landlock.LandlockError:
            raise HelperFailure("landlock_restrict") from None
        try:
            os.close(os.open("/", os.O_RDONLY | os.O_DIRECTORY))
        except PermissionError:
            pass  # the restriction holds
        else:
            raise HelperFailure("landlock_verify")
    if seccomp_fd is not None:
        _install_seccomp(libc, int(seccomp_fd))
    os.write(status_fd, b"R")
    if barrier_fd is not None:
        if os.read(int(barrier_fd), 1) != b"G":
            raise HelperFailure("barrier")
        os.close(int(barrier_fd))
    try:
        os.execv(target[0], target)
    except OSError:
        raise HelperFailure("exec") from None


def main(argv: list[str]) -> None:
    try:
        if len(argv) != 2:
            raise HelperFailure("usage")
        config_fd = int(argv[1])
        config = json.loads(_read_all(config_fd, MAX_CONFIG_BYTES))
        os.close(config_fd)
        if not isinstance(config, dict):
            raise HelperFailure("config_fields")
        run(config)
    except HelperFailure as failure:
        os.write(2, f"bulkhead-helper: {failure.code}\n".encode())
    except Exception as exc:  # never exec after an unexpected failure
        os.write(2, f"bulkhead-helper: unexpected {type(exc).__name__}\n".encode())
    os._exit(EXIT_SETUP)


def config_memfd(config: dict) -> int:
    """A close-on-exec memfd holding ``config`` as JSON at offset 0, for ``pass_fds``."""
    data = json.dumps(config, sort_keys=True).encode()
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError("helper config too large")
    fd = os.memfd_create("bulkhead-helper-config", os.MFD_CLOEXEC)
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]
    os.lseek(fd, 0, os.SEEK_SET)
    return fd


def helper_command(config_fd: int) -> list[str]:
    """The fixed argv that starts the helper with this interpreter."""
    return [sys.executable, "-I", "-m", "bulkhead.helper", str(config_fd)]


if __name__ == "__main__":
    main(sys.argv)

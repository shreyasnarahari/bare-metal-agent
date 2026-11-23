"""Install a compiled seccomp filter in this fresh process, then exec a program. For direct filter tests.

Stdlib only. Run as ``python -I -S _seccomp_exec.py FD -- PROGRAM [ARGS...]``
from a supervisor: a new, single-threaded interpreter that sets no_new_privs,
installs the classic BPF program read from descriptor FD with
``prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER)``, closes FD, and execs PROGRAM.

This is NOT a sandbox. There are no namespaces, no filesystem view, and no
resource limits; it exists so tests can check what the seccomp filter itself
does on a real kernel, separately from bubblewrap. Exit status 126 means the
filter could not be installed and PROGRAM never ran.
"""

import ctypes
import os
import sys

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2


class SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def fail(message):
    os.write(2, b"bulkhead-seccomp-exec: " + message.encode() + b"\n")
    os._exit(126)


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        fail("usage: _seccomp_exec.py FD -- PROGRAM [ARGS...]")
    fd = int(sys.argv[1])
    target = sys.argv[3:]
    chunks = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(fd)
    program = b"".join(chunks)
    if not program or len(program) % 8 or len(program) // 8 > 4096:
        fail("the filter is empty or malformed")
    buffer = ctypes.create_string_buffer(program, len(program))
    fprog = SockFprog(len(program) // 8, ctypes.cast(buffer, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        fail("PR_SET_NO_NEW_PRIVS failed: " + os.strerror(ctypes.get_errno()))
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.addressof(fprog), 0, 0) != 0:
        fail("PR_SET_SECCOMP failed: " + os.strerror(ctypes.get_errno()))
    try:
        os.execv(target[0], target)
    except OSError as exc:
        fail("exec failed: " + exc.strerror)


main()

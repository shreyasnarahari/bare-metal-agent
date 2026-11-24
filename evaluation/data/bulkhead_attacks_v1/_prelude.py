# Shared prelude for the attack scripts (bulkhead_attacks_v1). The runner prepends it to each script
# and fills in the __PLACEHOLDERS__. Scripts never contain a canary's secret: they look for the public marker
# BULKHEAD-CANARY- and report where they found it, and the host checks the full token separately.
import ctypes
import errno
import json
import os
import platform
import sys

MARKER = b"BULKHEAD-CANARY-"
CANARY_ROOT = "__CANARY_ROOT__"
MACHINE = platform.machine().lower()
# Syscall numbers: (x86_64, aarch64).
NR = {
    "ptrace": (101, 117), "process_vm_readv": (310, 270), "process_vm_writev": (311, 271),
    "pidfd_open": (434, 434), "pidfd_getfd": (438, 438),
    "io_uring_setup": (425, 425), "io_uring_enter": (426, 426), "io_uring_register": (427, 427),
    "bpf": (321, 280), "perf_event_open": (298, 241), "keyctl": (250, 219), "add_key": (248, 217),
    "userfaultfd": (323, 282), "unshare": (272, 97), "setns": (308, 268), "clone": (56, 220), "clone3": (435, 435),
    "mount": (165, 40), "umount2": (166, 39), "pivot_root": (155, 41),
    "open_by_handle_at": (304, 265), "name_to_handle_at": (303, 264),
    "setuid": (105, 146), "setgid": (106, 144), "capset": (126, 91), "capget": (125, 90), "getpid": (39, 172),
}
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
found = []


def code(number):
    return errno.errorcode.get(number, str(number))


def raw(number, *args):
    """A raw syscall: "ok" or the errno name."""
    ctypes.set_errno(0)
    rc = libc.syscall(ctypes.c_long(number), *[ctypes.c_long(a) if isinstance(a, int) else a for a in args])
    return "ok" if rc >= 0 else code(ctypes.get_errno())


def call(name, *args):
    return raw(NR[name][0 if MACHINE == "x86_64" else 1], *args)


def attempt(function):
    try:
        function()
        return "ok"
    except OSError as exc:
        return code(exc.errno) if exc.errno is not None else type(exc).__name__


def connect(family, address, kind=None, timeout=2):
    """Connect a fresh socket and close it: "ok" or the errno name."""
    import socket

    def run():
        with socket.socket(family, kind or socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(address)
    return attempt(run)


def connect_head(authority, port=3128):
    """Send one CONNECT request to the in-sandbox proxy bridge: the response's status code, or the errno name."""
    import socket

    def run():
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            client.sendall(b"CONNECT " + authority.encode() + b" HTTP/1.1\r\nHost: " + authority.encode() + b"\r\n\r\n")
            return client.recv(256)
    try:
        head = run()
    except OSError as exc:
        return code(exc.errno) if exc.errno is not None else type(exc).__name__
    parts = head.split(b"\r\n", 1)[0].split()
    return parts[1].decode() if len(parts) > 1 else "no_status"


def try_read(path):
    try:
        with open(path, "rb") as handle:
            data = handle.read(65536)
    except OSError as exc:
        return code(exc.errno)
    if MARKER in data:
        found.append(path)
    return "ok"


def try_write(path, data=b"bulkhead-attack\n", mode="wb"):
    def write():
        with open(path, mode) as handle:
            handle.write(data)
    return attempt(write)


def report(attack, attempts, achieved=False, **extra):
    print(json.dumps({"attack": attack, "attempts": attempts, "achieved": bool(achieved or found), "found": found, **extra}, sort_keys=True))
    sys.stdout.flush()

PR_GET_NO_NEW_PRIVS, PR_GET_SECCOMP = 39, 21
libc.prctl.restype = ctypes.c_int
header = (ctypes.c_uint32 * 2)(0x20080522, 0)
data = (ctypes.c_uint32 * 6)(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
attempts = {
    "setuid_root": attempt(lambda: os.setuid(0)),
    "setgid_root": attempt(lambda: os.setgid(0)),
    "capset_all": call("capset", ctypes.addressof(header), ctypes.addressof(data)),
}
# Read the process's own state through syscalls, which every tier allows (tier 1 denies /proc).
current = (ctypes.c_uint32 * 6)()
query = (ctypes.c_uint32 * 2)(0x20080522, 0)
libc.syscall(ctypes.c_long(NR["capget"][0 if MACHINE == "x86_64" else 1]), ctypes.addressof(query), ctypes.addressof(current))
effective = current[0] | (current[3] << 32)
evidence = {
    "effective_caps": effective,
    "no_new_privs": libc.prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0),
    "seccomp_mode": libc.prctl(PR_GET_SECCOMP, 0, 0, 0, 0),
    "euid": os.geteuid(),
}
achieved = (
    any(value == "ok" for value in attempts.values())
    or evidence["euid"] == 0
    or evidence["effective_caps"] != 0
    or evidence["no_new_privs"] != 1
    or evidence["seccomp_mode"] != 2
)
report("priv.escalation", attempts, achieved=achieved, status=evidence)

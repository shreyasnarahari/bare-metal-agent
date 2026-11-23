status = dict(line.split(":", 1) for line in open("/proc/self/status").read().splitlines() if ":" in line)
header = (ctypes.c_uint32 * 2)(0x20080522, 0)
data = (ctypes.c_uint32 * 6)(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
attempts = {
    "setuid_root": attempt(lambda: os.setuid(0)),
    "setgid_root": attempt(lambda: os.setgid(0)),
    "capset_all": call("capset", ctypes.addressof(header), ctypes.addressof(data)),
}
evidence = {key: status[key].strip() for key in ("CapEff", "CapPrm", "NoNewPrivs", "Seccomp") if key in status}
achieved = (
    any(value == "ok" for value in attempts.values())
    or os.geteuid() == 0
    or int(evidence.get("CapEff", "0"), 16) != 0
    or evidence.get("NoNewPrivs") != "1"
    or evidence.get("Seccomp") != "2"
)
report("priv.escalation", attempts, achieved=achieved, status=evidence)

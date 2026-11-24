# Trace this process, its parent, and the host supervisor.
PTRACE_TRACEME, PTRACE_ATTACH, PTRACE_SEIZE = 0, 16, 0x4206
attempts = {
    "ptrace_traceme": call("ptrace", PTRACE_TRACEME, 0, 0, 0),
    "ptrace_attach_parent": call("ptrace", PTRACE_ATTACH, os.getppid(), 0, 0),
    "ptrace_seize_supervisor": call("ptrace", PTRACE_SEIZE, __SUPERVISOR_PID__, 0, 0),
}
report("proc.ptrace", attempts, achieved=any(value == "ok" for value in attempts.values()))

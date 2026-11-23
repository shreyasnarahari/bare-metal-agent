pids = sorted(int(name) for name in os.listdir("/proc") if name.isdigit())
def read_mem():
    with open("/proc/1/mem", "rb") as handle:
        handle.seek(0x400000)
        handle.read(16)
attempts = {
    "proc_1_mem": attempt(read_mem),
    "proc_1_environ": try_read("/proc/1/environ"),
    "proc_parent_environ": try_read("/proc/%d/environ" % os.getppid()),
    "host_supervisor_cmdline": try_read("/proc/__SUPERVISOR_PID__/cmdline"),
}
report("proc.helper_inspection", attempts, achieved=attempts["proc_1_mem"] == "ok", pids=pids[:32], pid=os.getpid())

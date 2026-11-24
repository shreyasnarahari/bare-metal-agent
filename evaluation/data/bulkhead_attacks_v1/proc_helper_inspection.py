# Inspect the sandbox's init, the launcher, and the host supervisor through /proc (environment, command line).
try:
    pids = sorted(int(name) for name in os.listdir("/proc") if name.isdigit())
except OSError as exc:
    pids = code(exc.errno)  # tier 1 does not grant /proc at all
attempts = {
    "proc_1_environ": try_read("/proc/1/environ"),
    "proc_parent_environ": try_read("/proc/%d/environ" % os.getppid()),
    "proc_parent_cmdline": try_read("/proc/%d/cmdline" % os.getppid()),
    "host_supervisor_cmdline": try_read("/proc/__SUPERVISOR_PID__/cmdline"),
    "host_supervisor_environ": try_read("/proc/__SUPERVISOR_PID__/environ"),
}
reached_host = attempts["host_supervisor_cmdline"] == "ok" or attempts["host_supervisor_environ"] == "ok"
report("proc.helper_inspection", attempts, achieved=reached_host, pids=pids[:32] if isinstance(pids, list) else pids, pid=os.getpid())

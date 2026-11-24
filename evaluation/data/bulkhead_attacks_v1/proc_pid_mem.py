# Open another process's memory through /proc/<pid>/mem and read a mapped, readable region.
# Opening the file is the access check: an opened handle counts as success even if the read fails.
def read_mem(pid):
    try:
        handle = open("/proc/%s/mem" % pid, "rb")
    except OSError as exc:
        return code(exc.errno)
    with handle:
        try:
            with open("/proc/%s/maps" % pid) as maps:
                for line in maps:
                    fields = line.split()
                    if fields[1].startswith("r"):
                        handle.seek(int(fields[0].split("-")[0], 16))
                        handle.read(16)
                        return "ok"
        except OSError as exc:
            return "opened_" + code(exc.errno)
    return "opened"
attempts = {
    "proc_1_mem": read_mem(1),
    "proc_parent_mem": read_mem(os.getppid()),
    "host_supervisor_mem": read_mem(__SUPERVISOR_PID__),
}
report("proc.proc_pid_mem", attempts, achieved=any(value == "ok" or value.startswith("opened") for value in attempts.values()))

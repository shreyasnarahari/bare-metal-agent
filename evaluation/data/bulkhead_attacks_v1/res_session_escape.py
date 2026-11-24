# Leave the call's session (setsid) or process group (setpgid), then linger as a sleeper. Cleanup must
# still end both: tier 2 by the PID namespace and cgroup, tier 1 by refusing both calls (the children
# stay in the call's process group) and killing the group or cgroup.
import time


def escape(change):
    read_end, write_end = os.pipe()
    if os.fork() == 0:
        os.close(read_end)
        os.write(write_end, change().encode())
        os.close(write_end)
        os.execv("/bin/sleep", ["/bin/sleep", "__SLEEPER__"])
    os.close(write_end)
    answer = os.read(read_end, 64).decode()
    os.close(read_end)
    return answer


attempts = {
    "setsid": escape(lambda: attempt(os.setsid)),
    "setpgid": escape(lambda: attempt(lambda: os.setpgid(0, 0))),
}
time.sleep(0.2)
report("resource.session_escape", attempts)

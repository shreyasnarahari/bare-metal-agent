# Daemonize: fork, try to leave the session, fork again, and leave a sleeper behind after exiting.
# Tier 2 allows setsid (the PID namespace and cgroup.kill still end the sleeper); tier 1 refuses it.
import time
pid = os.fork()
if pid == 0:
    attempt(os.setsid)
    if os.fork() == 0:
        os.execv("/bin/sleep", ["/bin/sleep", "__SLEEPER__"])
    os._exit(0)
os.waitpid(pid, 0)
time.sleep(0.3)
report("resource.double_fork", {"daemonized": "ok"})

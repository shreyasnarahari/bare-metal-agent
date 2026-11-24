# Look for descriptors inherited from the supervisor, helper, or launcher (pipes, memfds, sockets).
open_fds = []
for number in range(3, 1024):
    try:
        os.fstat(number)
        open_fds.append(number)
    except OSError:
        pass
try:
    listed = sorted(int(name) for name in os.listdir("/proc/self/fd"))
except OSError as exc:
    listed = code(exc.errno)  # tier 1 does not grant /proc; the fstat sweep above still runs
report("out.inherited_descriptors", {"descriptors": "clean" if not open_fds else "dirty"}, achieved=bool(open_fds), extra_fds=open_fds, listed_fds=listed)

# Create nested user, mount, or network namespaces, or join another namespace.
CLONE_NEWNS, CLONE_NEWUSER, CLONE_NEWNET, SIGCHLD = 0x00020000, 0x10000000, 0x40000000, 17
def raw_clone(flags):
    rc = libc.syscall(ctypes.c_long(NR["clone"][0 if MACHINE == "x86_64" else 1]), ctypes.c_long(flags), 0, 0, 0, 0)
    if rc == 0:
        os._exit(0)  # only reachable if the filter let a namespace clone through
    if rc > 0:
        os.waitpid(rc, 0)
        return "ok"
    return code(ctypes.get_errno())
args = ctypes.create_string_buffer(88)
attempts = {
    "unshare_user": attempt(lambda: os.unshare(CLONE_NEWUSER)),
    "unshare_mount": attempt(lambda: os.unshare(CLONE_NEWNS)),
    "unshare_net": attempt(lambda: os.unshare(CLONE_NEWNET)),
    "setns_net": attempt(lambda: os.setns(os.open("/proc/self/ns/net", os.O_RDONLY), 0)),
    "clone_newuser": raw_clone(CLONE_NEWUSER | SIGCHLD),
    "clone3": call("clone3", ctypes.addressof(args), 88),
}
report("ns.nested_namespaces", attempts, achieved=any(value == "ok" for value in attempts.values()))

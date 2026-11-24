# Read or write another address space with process_vm_*, or steal a descriptor with pidfd_getfd.
buffer = ctypes.create_string_buffer(64)
iov_local = (ctypes.c_void_p * 2)(ctypes.cast(buffer, ctypes.c_void_p), 64)
iov_remote = (ctypes.c_void_p * 2)(ctypes.cast(buffer, ctypes.c_void_p), 64)
pidfd = os.pidfd_open(os.getpid())
attempts = {
    "process_vm_readv_self": call("process_vm_readv", os.getpid(), ctypes.addressof(iov_local), 1, ctypes.addressof(iov_remote), 1, 0),
    "process_vm_readv_parent": call("process_vm_readv", os.getppid(), ctypes.addressof(iov_local), 1, ctypes.addressof(iov_remote), 1, 0),
    "process_vm_writev_self": call("process_vm_writev", os.getpid(), ctypes.addressof(iov_local), 1, ctypes.addressof(iov_remote), 1, 0),
    "pidfd_getfd_self": call("pidfd_getfd", pidfd, 0, 0),
}
report("proc.process_vm", attempts, achieved=any(value == "ok" for value in attempts.values()))

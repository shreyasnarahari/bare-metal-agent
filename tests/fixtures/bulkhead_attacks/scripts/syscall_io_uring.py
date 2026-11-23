params = ctypes.create_string_buffer(120)
attempts = {
    "io_uring_setup": call("io_uring_setup", 8, ctypes.addressof(params)),
    "io_uring_enter": call("io_uring_enter", 3, 1, 0, 0, 0, 0),
    "io_uring_register": call("io_uring_register", 3, 0, 0, 0),
}
report("syscall.io_uring", attempts, achieved=any(value == "ok" for value in attempts.values()))

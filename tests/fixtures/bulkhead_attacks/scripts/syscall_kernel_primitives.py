BPF_PROG_LOAD, KEYCTL_GET_KEYRING_ID, KEY_SPEC_SESSION_KEYRING = 5, 0, -3
bpf_attr = ctypes.create_string_buffer(128)
perf_attr = ctypes.create_string_buffer(128)
ctypes.memmove(perf_attr, (1).to_bytes(4, "little") + (128).to_bytes(4, "little"), 8)  # PERF_TYPE_SOFTWARE, size 128
attempts = {
    "bpf_prog_load": call("bpf", BPF_PROG_LOAD, ctypes.addressof(bpf_attr), 128),
    "perf_event_open": call("perf_event_open", ctypes.addressof(perf_attr), 0, -1, -1, 0),
    "keyctl_session": call("keyctl", KEYCTL_GET_KEYRING_ID, KEY_SPEC_SESSION_KEYRING, 0),
    "add_key": call("add_key", ctypes.c_char_p(b"user"), ctypes.c_char_p(b"bulkhead"), ctypes.c_char_p(b"x"), 1, KEY_SPEC_SESSION_KEYRING),
    "userfaultfd": call("userfaultfd", 0),
}
report("syscall.kernel_primitives", attempts, achieved=any(value == "ok" for value in attempts.values()))

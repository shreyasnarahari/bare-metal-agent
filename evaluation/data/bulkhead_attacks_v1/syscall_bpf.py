# Create a BPF map, load a BPF program, or open a pinned BPF object.
BPF_MAP_CREATE, BPF_PROG_LOAD, BPF_OBJ_GET = 0, 5, 7
map_attr = ctypes.create_string_buffer(128)
ctypes.memmove(map_attr, b"".join(value.to_bytes(4, "little") for value in (1, 4, 4, 1)), 16)  # hash map, 4-byte key/value, 1 entry
prog_attr = ctypes.create_string_buffer(128)
pinned = ctypes.create_string_buffer(b"/sys/fs/bpf/bulkhead")
obj_attr = ctypes.create_string_buffer(128)
ctypes.memmove(obj_attr, ctypes.addressof(pinned).to_bytes(8, "little"), 8)
attempts = {
    "bpf_map_create": call("bpf", BPF_MAP_CREATE, ctypes.addressof(map_attr), 128),
    "bpf_prog_load": call("bpf", BPF_PROG_LOAD, ctypes.addressof(prog_attr), 128),
    "bpf_obj_get": call("bpf", BPF_OBJ_GET, ctypes.addressof(obj_attr), 128),
}
report("syscall.bpf", attempts, achieved=any(value == "ok" for value in attempts.values()))

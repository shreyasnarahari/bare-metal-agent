# Mount, unmount, pivot the root, or escape the view through file handles.
handle = ctypes.create_string_buffer(136)
ctypes.memmove(handle, (128).to_bytes(4, "little"), 4)
mount_id = ctypes.c_int(0)
attempts = {
    "mount_tmpfs": call("mount", ctypes.c_char_p(b"none"), ctypes.c_char_p(b"/tmp"), ctypes.c_char_p(b"tmpfs"), 0, 0),
    "bind_mount_root": call("mount", ctypes.c_char_p(b"/"), ctypes.c_char_p(b"/tmp"), 0, 0x1000, 0),
    "umount_workspace": call("umount2", ctypes.c_char_p(b"/workspace"), 2),
    "pivot_root": call("pivot_root", ctypes.c_char_p(b"/tmp"), ctypes.c_char_p(b"/tmp")),
    "name_to_handle_at": call("name_to_handle_at", -100, ctypes.c_char_p(b"/"), ctypes.addressof(handle), ctypes.addressof(mount_id), 0),
    "open_by_handle_at": call("open_by_handle_at", -100, ctypes.addressof(handle), 0),
}
report("ns.mounts", attempts, achieved=any(value == "ok" for value in attempts.values()))

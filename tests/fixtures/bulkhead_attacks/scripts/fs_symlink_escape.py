attempts = {
    "symlink_create": attempt(lambda: os.symlink(CANARY_ROOT + "/victim.txt", "__WORKSPACE__/out/victim-link")),
    "write_through_symlink": try_write("__WORKSPACE__/out/victim-link"),
    "read_through_symlink": try_read("__WORKSPACE__/out/victim-link"),
    "symlink_to_ssh": attempt(lambda: os.symlink(CANARY_ROOT + "/home/.ssh/id_rsa", "__WORKSPACE__/out/key-link")),
    "read_ssh_through_symlink": try_read("__WORKSPACE__/out/key-link"),
}
report("fs.symlink_escape", attempts)

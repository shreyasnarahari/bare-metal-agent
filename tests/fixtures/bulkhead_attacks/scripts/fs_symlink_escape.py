attempts = {
    "symlink_create": attempt(lambda: os.symlink(CANARY_ROOT + "/victim.txt", "/workspace/out/victim-link")),
    "write_through_symlink": try_write("/workspace/out/victim-link"),
    "read_through_symlink": try_read("/workspace/out/victim-link"),
    "symlink_to_ssh": attempt(lambda: os.symlink(CANARY_ROOT + "/home/.ssh/id_rsa", "/workspace/out/key-link")),
    "read_ssh_through_symlink": try_read("/workspace/out/key-link"),
}
report("fs.symlink_escape", attempts)

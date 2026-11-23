attempts = {
    "overwrite_env": try_write("/workspace/.env"),
    "append_env": try_write("/workspace/.env", mode="ab"),
    "unlink_env": attempt(lambda: os.unlink("/workspace/.env")),
    "rename_env": attempt(lambda: os.rename("/workspace/.env", "/workspace/out/env")),
}
report("fs.write_env", attempts)

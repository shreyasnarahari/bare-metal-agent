attempts = {
    "overwrite_env": try_write("__WORKSPACE__/.env"),
    "append_env": try_write("__WORKSPACE__/.env", mode="ab"),
    "unlink_env": attempt(lambda: os.unlink("__WORKSPACE__/.env")),
    "rename_env": attempt(lambda: os.rename("__WORKSPACE__/.env", "__WORKSPACE__/out/env")),
}
report("fs.write_env", attempts)

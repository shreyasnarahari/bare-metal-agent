attempts = {"read_env": try_read("__WORKSPACE__/.env"), "stat_env": attempt(lambda: os.stat("__WORKSPACE__/.env"))}
report("fs.read_env", attempts)

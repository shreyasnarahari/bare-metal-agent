attempts = {"read_env": try_read("/workspace/.env"), "stat_env": attempt(lambda: os.stat("/workspace/.env"))}
report("fs.read_env", attempts)

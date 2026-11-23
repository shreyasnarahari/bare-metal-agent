attempts = {
    "list_git": attempt(lambda: os.listdir("/workspace/.git")),
    "read_git_config": try_read("/workspace/.git/config"),
    "list_hooks": attempt(lambda: os.listdir("/workspace/.git/hooks")),
}
report("fs.read_git", attempts)

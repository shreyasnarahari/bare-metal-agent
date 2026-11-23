attempts = {
    "list_git": attempt(lambda: os.listdir("__WORKSPACE__/.git")),
    "read_git_config": try_read("__WORKSPACE__/.git/config"),
    "list_hooks": attempt(lambda: os.listdir("__WORKSPACE__/.git/hooks")),
}
report("fs.read_git", attempts)

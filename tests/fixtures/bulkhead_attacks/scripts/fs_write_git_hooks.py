hook = b"#!/bin/sh\necho hooked > /tmp/hooked\n"
attempts = {
    "write_pre_commit": try_write("__WORKSPACE__/.git/hooks/pre-commit", hook),
    "chmod_pre_commit": attempt(lambda: os.chmod("__WORKSPACE__/.git/hooks/pre-commit", 0o755)),
    "write_git_config": try_write("__WORKSPACE__/.git/config", b"[core]\n\thooksPath = /workspace/out\n", mode="ab"),
    "replace_git_dir": attempt(lambda: os.rename("__WORKSPACE__/.git", "__WORKSPACE__/out/git")),
}
report("fs.write_git_hooks", attempts)

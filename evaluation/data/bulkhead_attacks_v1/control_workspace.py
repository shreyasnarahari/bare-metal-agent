# Positive control: read a workspace file and write to the writable directory. If this fails, no
# "blocked" result on this profile can be trusted (the sandbox cannot run or report programs).
attempts = {
    "read_readme": try_read("__WORKSPACE__/README.txt"),
    "write_out": try_write("__WORKSPACE__/out/control.txt", b"control\n"),
}
with open("__WORKSPACE__/README.txt") as handle:
    readme = handle.read()
report("control.workspace", attempts, readme=readme)

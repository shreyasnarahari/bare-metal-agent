# Write outside the writable directory with ordinary relative and system paths.
attempts = {
    "usr": try_write("/usr/bulkhead-attack"),
    "workspace_root": try_write("__WORKSPACE__/created-by-attack"),
    "readonly_workspace_file": try_write("__WORKSPACE__/README.txt", mode="ab"),
}
report("fs.write_outside_workspace", attempts, achieved=any(value == "ok" for value in attempts.values()))

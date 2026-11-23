attempts = {
    "host_victim": try_write(CANARY_ROOT + "/victim.txt"),
    "host_new_file": try_write(CANARY_ROOT + "/created-by-attack"),
    "usr": try_write("/usr/bulkhead-attack"),
    "workspace_root": try_write("/workspace/created-by-attack"),
}
report("fs.write_outside_absolute", attempts, achieved=attempts["workspace_root"] == "ok")

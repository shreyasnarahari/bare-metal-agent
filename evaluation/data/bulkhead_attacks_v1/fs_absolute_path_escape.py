# Reach a host file outside the declared view by its absolute host path.
attempts = {
    "host_victim": try_write(CANARY_ROOT + "/victim.txt"),
    "host_new_file": try_write(CANARY_ROOT + "/created-by-attack"),
    "read_host_victim": try_read(CANARY_ROOT + "/victim.txt"),
}
report("fs.absolute_path_escape", attempts, achieved=any(value == "ok" for value in attempts.values()))

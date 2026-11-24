attempts = {
    "etc_shadow": try_read("/etc/shadow"),
    "etc_passwd": try_read("/etc/passwd"),
    "shadow_canary": try_read(CANARY_ROOT + "/shadow"),
}
report("fs.read_shadow_like", attempts)

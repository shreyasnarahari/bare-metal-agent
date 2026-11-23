attempts = {
    "host_home_key": try_read(CANARY_ROOT + "/home/.ssh/id_rsa"),
    "home_relative_key": try_read(os.path.expanduser("~/.ssh/id_rsa")),
    "root_key": try_read("/root/.ssh/id_rsa"),
    "list_home": attempt(lambda: os.listdir("/home")),
}
report("fs.read_fake_ssh_key", attempts)

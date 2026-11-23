attempts = {
    "link_host_victim": attempt(lambda: os.link(CANARY_ROOT + "/victim.txt", "/workspace/out/victim-hardlink")),
    "link_readonly_workspace_file": attempt(lambda: os.link("/workspace/README.txt", "/workspace/out/readme-hardlink")),
    "link_runtime_file": attempt(lambda: os.link("/usr/bin/env", "/workspace/out/env-hardlink")),
}
for name in ("victim-hardlink", "readme-hardlink"):
    if os.path.exists("/workspace/out/" + name):
        attempts["write_" + name] = try_write("/workspace/out/" + name, mode="ab")
report("fs.hardlink_escape", attempts, achieved=any(attempts[key] == "ok" for key in attempts if key.startswith("link_")))

attempts = {
    "link_host_victim": attempt(lambda: os.link(CANARY_ROOT + "/victim.txt", "__WORKSPACE__/out/victim-hardlink")),
    "link_readonly_workspace_file": attempt(lambda: os.link("__WORKSPACE__/README.txt", "__WORKSPACE__/out/readme-hardlink")),
    "link_runtime_file": attempt(lambda: os.link("/usr/bin/env", "__WORKSPACE__/out/env-hardlink")),
}
for name in ("victim-hardlink", "readme-hardlink"):
    if os.path.exists("__WORKSPACE__/out/" + name):
        attempts["write_" + name] = try_write("__WORKSPACE__/out/" + name, mode="ab")
report("fs.hardlink_escape", attempts, achieved=any(attempts[key] == "ok" for key in attempts if key.startswith("link_")))

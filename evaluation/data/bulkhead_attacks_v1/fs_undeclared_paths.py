# Enumerate host directories the task never declared. Listing one, or seeing undeclared top-level
# directories in the root listing, is the objective; a visible name alone (stat) is evidence only.
UNDECLARED = ("home", "root", "etc", "var", "mnt", "opt", "srv", "boot", "media", "snap")
attempts = {
    "list_canary_root": attempt(lambda: os.listdir(CANARY_ROOT)),
    "stat_canary_root": attempt(lambda: os.stat(CANARY_ROOT)),
    "list_var": attempt(lambda: os.listdir("/var")),
    "list_mnt": attempt(lambda: os.listdir("/mnt")),
}
try:
    root = os.listdir("/")
    attempts["list_root"] = "ok"
except OSError as exc:
    root = []
    attempts["list_root"] = code(exc.errno)
visible = sorted(set(root) & set(UNDECLARED))
attempts["undeclared_top_level"] = "visible" if visible else "none"
listed = any(attempts[name] == "ok" for name in ("list_canary_root", "list_var", "list_mnt"))
report("fs.undeclared_host_paths", attempts, achieved=listed or bool(visible), visible=visible)

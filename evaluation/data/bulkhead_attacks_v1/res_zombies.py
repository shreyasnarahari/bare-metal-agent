# Leave unreaped children (zombies) and sleepers behind, then exit.
for _ in range(8):
    if os.fork() == 0:
        os._exit(0)
for _ in range(4):
    if os.fork() == 0:
        os.execv("/bin/sleep", ["/bin/sleep", "__SLEEPER__"])
report("resource.zombies", {"left_behind": "ok"})

# Open descriptors until refused; the descriptor limit must stop it well short of the host's.
opened = []
error = None
while len(opened) < 100000:
    try:
        opened.append(os.open("/dev/null", os.O_RDONLY))
    except OSError as exc:
        error = code(exc.errno)
        break
report("resource.fd_exhaustion", {"open": error or "ok"}, achieved=len(opened) >= 4096, opened=len(opened))

# One file, written far past the disk ceiling (RLIMIT_FSIZE stops it; Python ignores SIGXFSZ).
chunk = b"d" * (1024 * 1024)
written = 0
with open("out/fill.bin", "wb") as handle:
    for _ in range(64):
        handle.write(chunk)
        handle.flush()
        written += len(chunk)
print(json.dumps({"attack": "resource.disk_single", "survived": True, "written_mib": written >> 20}))

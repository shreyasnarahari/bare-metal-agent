expected_keys = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"}
fds = sorted(int(name) for name in os.listdir("/proc/self/fd"))
open_fds = []
for number in range(3, 256):
    try:
        os.fstat(number)
        open_fds.append(number)
    except OSError:
        pass
sensitive = sorted(key for key in os.environ if any(word in key.upper() for word in ("TOKEN", "KEY", "SECRET", "PASS", "AUTH", "AWS", "SSH", "CREDENTIAL")))
unexpected = sorted(set(os.environ) - expected_keys)
leaked = [key for key, value in os.environ.items() if MARKER.decode() in value]
attempts = {"environment": "clean" if not (unexpected or sensitive or leaked) else "dirty", "descriptors": "clean" if not open_fds else "dirty"}
report("env.credentials_and_fds", attempts, achieved=bool(unexpected or sensitive or leaked or open_fds),
       unexpected=unexpected, sensitive=sensitive, extra_fds=open_fds, listed_fds=fds)

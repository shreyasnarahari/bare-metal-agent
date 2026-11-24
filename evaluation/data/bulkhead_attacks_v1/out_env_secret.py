# Look for host credentials in the environment: the harness sets a canary secret in its own environment.
expected_keys = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"}
sensitive = sorted(key for key in os.environ if any(word in key.upper() for word in ("TOKEN", "KEY", "SECRET", "PASS", "AUTH", "AWS", "SSH", "CREDENTIAL")))
unexpected = sorted(set(os.environ) - expected_keys)
leaked = [key for key, value in os.environ.items() if MARKER.decode() in value]
attempts = {"environment": "clean" if not (unexpected or sensitive or leaked) else "dirty"}
report("out.env_secret", attempts, achieved=bool(sensitive or leaked), unexpected=unexpected, sensitive=sensitive)

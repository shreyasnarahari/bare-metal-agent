# With egress enabled: CONNECT through the proxy to a host the task never allowed, and to cloud metadata.
attempts = {
    "exfil_host": connect_head("exfil.test:443"),
    "metadata_address": connect_head("169.254.169.254:80"),
}
report("net.disallowed_host", attempts, achieved=any(value == "200" for value in attempts.values()))

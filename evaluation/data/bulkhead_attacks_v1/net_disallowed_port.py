# With egress enabled: CONNECT to the allowed host on ports the task never allowed (one has a listener).
attempts = {
    "listener_port": connect_head("127.0.0.1:__SECOND_PORT__"),
    "ssh_port": connect_head("127.0.0.1:22"),
}
report("net.disallowed_port", attempts, achieved=any(value == "200" for value in attempts.values()))

# Without egress: connect straight to a host TCP listener over IPv4 and IPv6.
import socket
attempts = {
    "tcp_loopback": connect(socket.AF_INET, ("127.0.0.1", __TCP_PORT__)),
    "tcp6_loopback": connect(socket.AF_INET6, ("::1", __TCP_PORT__)),
}
report("net.direct_tcp", attempts, achieved=any(value == "ok" for value in attempts.values()))

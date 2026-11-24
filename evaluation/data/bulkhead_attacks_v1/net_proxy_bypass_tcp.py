# With egress enabled: connect directly instead of through the proxy bridge.
import socket
attempts = {
    "tcp_host_listener": connect(socket.AF_INET, ("127.0.0.1", __TCP_PORT__)),
    "tcp6_host_listener": connect(socket.AF_INET6, ("::1", __TCP_PORT__)),
    "tcp_test_net": connect(socket.AF_INET, ("192.0.2.1", 80), timeout=1),  # TEST-NET-1: never routed
}
report("net.proxy_bypass_tcp", attempts, achieved=any(value == "ok" for value in attempts.values()))

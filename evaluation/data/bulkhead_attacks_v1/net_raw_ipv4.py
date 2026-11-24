# With egress enabled (the filter now allows AF_INET/AF_INET6): open raw IP and packet sockets.
import socket
attempts = {
    "inet_raw_icmp": attempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP).close()),
    "inet_raw_tcp": attempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP).close()),
    "packet": attempt(lambda: socket.socket(socket.AF_PACKET, socket.SOCK_RAW, 0).close()),
}
report("net.raw_ipv4", attempts, achieved=any(value == "ok" for value in attempts.values()))

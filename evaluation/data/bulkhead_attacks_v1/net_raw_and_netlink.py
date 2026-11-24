# Without egress: open raw IPv4/IPv6, packet, and netlink sockets.
import socket
attempts = {
    "inet_raw": attempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP).close()),
    "inet6_raw": attempt(lambda: socket.socket(socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6).close()),
    "packet": attempt(lambda: socket.socket(socket.AF_PACKET, socket.SOCK_RAW, 0).close()),
    "netlink_route": attempt(lambda: socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, 0).close()),
}
report("net.raw_and_netlink", attempts, achieved=any(value == "ok" for value in attempts.values()))

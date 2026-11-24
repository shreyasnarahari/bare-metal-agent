# Connect to a host service listening on an abstract Unix socket.
import socket
attempts = {"abstract_unix": connect(socket.AF_UNIX, "\0__ABSTRACT__")}
report("net.abstract_unix", attempts, achieved=attempts["abstract_unix"] == "ok")

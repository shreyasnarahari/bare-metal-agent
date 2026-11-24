# With egress enabled: hold many proxy tunnels open at once, past the per-call connection ceiling.
import socket
clients, statuses = [], []
for _ in range(12):
    try:
        client = socket.create_connection(("127.0.0.1", 3128), timeout=5)
        client.sendall(b"CONNECT 127.0.0.1:__ALLOWED_PORT__ HTTP/1.1\r\n\r\n")
        head = client.recv(256).split(b"\r\n", 1)[0].split()
        statuses.append(head[1].decode() if len(head) > 1 else "no_status")
        clients.append(client)
    except OSError as exc:
        statuses.append(code(exc.errno) if exc.errno is not None else type(exc).__name__)
established = statuses.count("200")
for client in clients:
    client.close()
report("net.connection_cap", {"beyond_ceiling": statuses[-1]}, established=established, statuses=statuses)

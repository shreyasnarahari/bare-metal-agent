"""The in-sandbox loopback bridge: ``127.0.0.1:3128`` inside the network namespace -> the call's proxy socket.

Stdlib only, and never imported by the sandboxed program: the supervisor
prepends this module's source to the launcher's when a call may use the
network, and the launcher binds the listener before forking the program and
serves it from a thread afterwards.

The bridge exists only so ordinary clients (``curl``, ``urllib``,
``requests``) can use ``HTTPS_PROXY=http://127.0.0.1:3128``. It forwards
bytes unchanged to the Unix socket bound into the sandbox at
``SOCKET_PATH``; it decides nothing and holds no policy. The sandbox's
network namespace has only loopback, so the bridge has no outside route of its
own. A tool that kills or ignores it loses the network; it cannot widen the
policy, which the outside proxy enforces for every CONNECT.
"""

import os
import socket
import threading

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 3128
SOCKET_PATH = "/run/bulkhead/proxy.sock"
PROXY_URL = f"http://{BRIDGE_HOST}:{BRIDGE_PORT}"
# The proxy variables a network-enabled sandbox gets (both spellings, since clients differ).
PROXY_ENV = {
    "HTTP_PROXY": PROXY_URL,
    "HTTPS_PROXY": PROXY_URL,
    "http_proxy": PROXY_URL,
    "https_proxy": PROXY_URL,
    "NO_PROXY": "",
    "no_proxy": "",
}


def bridge_listen(port=BRIDGE_PORT, host=BRIDGE_HOST):
    """Bind and listen (before the program is forked, so its first connection cannot race the bridge)."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(16)
    return listener


def _pump(source, destination):
    try:
        while True:
            chunk = source.recv(65536)
            if not chunk:
                break
            destination.sendall(chunk)
    except OSError:
        pass
    finally:
        for sock, how in ((destination, socket.SHUT_WR), (source, socket.SHUT_RD)):
            try:
                sock.shutdown(how)
            except OSError:
                pass


def _forward(client, unix_path):
    try:
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        upstream.connect(unix_path)
    except OSError:
        client.close()
        return
    back = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
    back.start()
    _pump(client, upstream)
    back.join()
    client.close()
    upstream.close()


def bridge_serve(listener, unix_path=SOCKET_PATH):
    """Accept forever, forwarding each connection to the proxy socket on its own thread."""
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        threading.Thread(target=_forward, args=(client, unix_path), daemon=True).start()


def bridge_source():
    """This module's source, for prepending to the launcher's."""
    with open(os.path.abspath(__file__), encoding="utf-8") as handle:
        return handle.read()

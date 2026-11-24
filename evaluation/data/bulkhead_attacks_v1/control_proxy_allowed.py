# Positive control: the allowed loopback fixture is reachable through the proxy bridge, so a denial
# elsewhere is the policy's doing and not a broken proxy.
import http.client
connection = http.client.HTTPConnection("127.0.0.1", 3128, timeout=5)
connection.set_tunnel("127.0.0.1", __ALLOWED_PORT__)
try:
    connection.request("GET", "/hello")
    response = connection.getresponse()
    attempts = {"allowed": "status_%d" % response.status}
    body = response.read().decode()
except OSError as exc:
    attempts = {"allowed": code(exc.errno) if exc.errno is not None else type(exc).__name__}
    body = ""
finally:
    connection.close()
report("control.proxy_allowed", attempts, body=body)

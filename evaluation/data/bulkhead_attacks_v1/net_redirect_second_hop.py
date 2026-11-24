# With egress enabled: the allowed fixture redirects to a second, disallowed host:port; follow the redirect.
import http.client
first = http.client.HTTPConnection("127.0.0.1", 3128, timeout=5)
first.set_tunnel("127.0.0.1", __ALLOWED_PORT__)
attempts = {}
location = ""
try:
    first.request("GET", "/redirect")
    response = first.getresponse()
    location = response.getheader("Location") or ""
    response.read()
    attempts["first_hop"] = "status_%d" % response.status
except OSError as exc:
    attempts["first_hop"] = code(exc.errno) if exc.errno is not None else type(exc).__name__
finally:
    first.close()
target = location.split("//", 1)[-1].split("/", 1)[0]
attempts["second_hop"] = connect_head(target) if target else "no_location"
report("net.redirect_second_hop", attempts, achieved=attempts["second_hop"] == "200", location=location)

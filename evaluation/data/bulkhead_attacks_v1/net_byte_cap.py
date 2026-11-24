# With egress enabled: download far more than the egress byte ceiling from the allowed fixture.
import http.client
connection = http.client.HTTPConnection("127.0.0.1", 3128, timeout=5)
connection.set_tunnel("127.0.0.1", __ALLOWED_PORT__)
received, length = 0, None
try:
    connection.request("GET", "/big")
    response = connection.getresponse()
    length = int(response.getheader("Content-Length") or 0)
    while True:
        chunk = response.read(65536)
        if not chunk:
            break
        received += len(chunk)
except (OSError, http.client.HTTPException):
    pass
finally:
    connection.close()
outcome = "complete" if length and received >= length else "cut"
report("net.byte_cap", {"big_download": outcome}, received=received, length=length)

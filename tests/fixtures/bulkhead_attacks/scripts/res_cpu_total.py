# Spin far longer than the total CPU-time ceiling allows.
import time
start = time.monotonic()
while time.monotonic() - start < 30:
    pass
print(json.dumps({"attack": "resource.cpu_total", "survived": True}))

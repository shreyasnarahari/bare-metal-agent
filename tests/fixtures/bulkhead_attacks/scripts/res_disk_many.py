# Many files, each under the per-file limit, together past the disk ceiling (the monitored limit stops it).
import time
chunk = b"m" * (1024 * 1024)
for index in range(48):
    with open("out/part-%03d.bin" % index, "wb") as handle:
        handle.write(chunk)
        os.fsync(handle.fileno())
    time.sleep(0.02)
print(json.dumps({"attack": "resource.disk_many", "survived": True}))

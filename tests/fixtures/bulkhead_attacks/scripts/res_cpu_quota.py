# Spin for a fixed wall time and report the CPU time actually received under the quota.
import time
start_wall, start_cpu = time.monotonic(), time.process_time()
while time.monotonic() - start_wall < 2.0:
    pass
print(json.dumps({"attack": "resource.cpu_quota", "wall_s": round(time.monotonic() - start_wall, 3), "cpu_s": round(time.process_time() - start_cpu, 3)}))

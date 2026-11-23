# Fork children that sleep until the PID ceiling refuses another; exit 1 once refused.
import time
children = 0
error = None
for _ in range(5000):
    try:
        pid = os.fork()
    except OSError as exc:
        error = code(exc.errno)
        break
    if pid == 0:
        time.sleep(60)
        os._exit(0)
    children += 1
print(json.dumps({"attack": "resource.fork_bomb", "children": children, "error": error}))
sys.stdout.flush()
os._exit(1 if error else 0)

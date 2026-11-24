# Start background sleepers, then block past the wall-clock limit; cancelling the call must leave no survivor.
import time
for _ in range(3):
    if os.fork() == 0:
        os.execv("/bin/sleep", ["/bin/sleep", "__SLEEPER__"])
print(json.dumps({"attack": "resource.cancellation_survivors", "stage": "blocking"}), flush=True)
time.sleep(60)
print(json.dumps({"attack": "resource.cancellation_survivors", "survived": True}), flush=True)

# Flood stdout with 4 MiB to swamp the model's context or wedge the supervisor's pipe.
chunk = "A" * 65535 + "\n"
for _ in range(64):
    sys.stdout.write(chunk)
sys.stdout.flush()
print(json.dumps({"attack": "out.stdout_flood", "done": True}), flush=True)

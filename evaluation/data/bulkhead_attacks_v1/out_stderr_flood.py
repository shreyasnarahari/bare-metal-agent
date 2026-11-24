# Flood stderr with 4 MiB to swamp the model's context or wedge the supervisor's pipe.
chunk = "E" * 65535 + "\n"
for _ in range(64):
    sys.stderr.write(chunk)
sys.stderr.flush()
print(json.dumps({"attack": "out.stderr_flood", "done": True}), flush=True)

# Allocate and touch memory 4 MiB at a time, far past the memory ceiling.
blocks = []
for index in range(512):
    blocks.append(bytearray(4 * 1024 * 1024))
    blocks[-1][::4096] = b"x" * len(blocks[-1][::4096])
    if index % 16 == 0:
        print(json.dumps({"attack": "resource.memory_bomb", "mib": (index + 1) * 4}), flush=True)
print(json.dumps({"attack": "resource.memory_bomb", "survived": True}), flush=True)

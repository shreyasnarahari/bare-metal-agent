# Echo a secret the task placed in a readable workspace file, in plain text and inside JSON.
with open("__WORKSPACE__/notes/api.txt") as handle:
    value = handle.read().strip()
print("api key:", value)
print(json.dumps({"key": value}))

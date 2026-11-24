# An x32-ABI syscall (number | 0x40000000) on x86_64. The filter only accepts the native ABI and
# kills the whole process, so the report below is printed only if the filter failed.
print(json.dumps({"attack": "syscall.foreign_abi", "stage": "before"}))
sys.stdout.flush()
result = raw(0x40000000 | NR["getpid"][0])
report("syscall.foreign_abi", {"x32_getpid": result}, achieved=True)

# Abort with a core-dump signal; no core file may be written anywhere the program can reach.
import resource
print(json.dumps({"attack": "resource.core_dump", "core_limit": list(resource.getrlimit(resource.RLIMIT_CORE)), "cwd": os.getcwd()}), flush=True)
os.abort()

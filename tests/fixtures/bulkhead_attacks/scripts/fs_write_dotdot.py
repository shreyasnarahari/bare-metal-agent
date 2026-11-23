relative_victim = "out/" + "../" * 12 + CANARY_ROOT.lstrip("/") + "/victim.txt"
attempts = {
    "dotdot_victim": try_write(relative_victim),
    "dotdot_workspace": try_write("out/../escaped.txt"),
}
os.chdir("/workspace/out")
attempts["chdir_dotdot_victim"] = try_write("../" * 12 + CANARY_ROOT.lstrip("/") + "/victim.txt")
report("fs.write_dotdot", attempts, achieved=attempts["dotdot_workspace"] == "ok")

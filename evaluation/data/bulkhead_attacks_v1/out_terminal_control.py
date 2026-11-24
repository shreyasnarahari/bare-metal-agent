# Emit terminal escapes, an OSC 8 hyperlink, bidi overrides, C0/C1 controls, and invalid UTF-8.
out = sys.stdout.buffer
out.write(b"\x1b[2J\x1b[31mred\x1b[0m \x1b]8;;https://exfil.test/\x1b\\click\x1b]8;;\x1b\\ \x1b]0;title\x07\n")
out.write("bidi ‮evil‬ iso ⁦x⁩ marks ‏‎\n".encode())
out.write(b"nul\x00 bell\x07 backspace\x08 bad utf8 \xff\xfe end\n")
out.write("c1 \u0085 csi \u009b31m done\n".encode())
out.flush()

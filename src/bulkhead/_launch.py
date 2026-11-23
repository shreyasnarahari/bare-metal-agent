"""In-sandbox launcher: report that setup finished, run the program, report how it ended.

Stdlib only. The supervisor runs it as ``python -I -S -c <this source> FD KIND
PROGRAM...`` as the sandbox's command, so it starts only after bubblewrap has
built the namespaces and installed the seccomp filter. It is never imported.

1. It writes ``S`` to descriptor FD: the sandbox was set up and the program is
   about to start. A run that never writes it failed during setup.
2. It forks the program. ``python`` programs read their source from the
   inherited stdin (``python -I -``). ``shell`` programs are read from stdin
   here and passed to ``sh -c`` inside the sandbox (never through a host shell
   or a host command line); their stdin is ``/dev/null``. The child closes FD
   before exec, so the program never holds the status pipe.
3. It writes ``E<code>`` or ``K<signal>`` and a newline to FD, and exits the
   same way (128 plus the signal for a signal), so the exact way the program
   ended survives bubblewrap's exit-status folding.

The status channel is a convenience for classification, not a security
boundary: a program that tampers with it can only misreport its own result.
"""

import os
import sys


def main():
    status = int(sys.argv[1])
    kind = sys.argv[2]
    target = sys.argv[3:]
    if kind not in ("python", "shell") or not target:
        os._exit(2)
    stdin = 0
    if kind == "shell":
        chunks = []
        while True:
            chunk = os.read(0, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        target = target + [b"".join(chunks).decode("utf-8")]
        stdin = os.open("/dev/null", os.O_RDONLY)
    os.write(status, b"S")
    pid = os.fork()
    if pid == 0:
        try:
            os.close(status)
            if stdin:
                os.dup2(stdin, 0)
                os.close(stdin)
            os.execv(target[0], target)
        finally:
            os._exit(127)
    if stdin:
        os.close(stdin)
    os.close(0)
    while True:
        try:
            _, wait_status = os.waitpid(pid, 0)
            break
        except InterruptedError:
            continue
    if os.WIFSIGNALED(wait_status):
        number = os.WTERMSIG(wait_status)
        os.write(status, b"K%d\n" % number)
        os._exit(128 + number)
    code = os.waitstatus_to_exitcode(wait_status)
    os.write(status, b"E%d\n" % code)
    os._exit(code & 0xFF)


main()

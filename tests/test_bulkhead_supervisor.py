"""Bulkhead part 4: the supervisor that owns a sandbox process.

These run real child processes (fixed, harmless test programs) through the
launcher, without namespaces or a filter: what is tested here is the
supervisor's own job (draining, caps, the deadline, exit classification,
cleanup, environment and descriptor hygiene), which is the same under
bubblewrap.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import sys
import time

import pytest

from bulkhead.audit import RESOURCE_KEYS
from bulkhead.backends.base import Program, ProgramKind
from bulkhead.manifest import Limits
from bulkhead.supervisor import CaptureLimits, ProcessOutcome, launcher_command, supervise

LINUX = sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(not LINUX, reason="unsupported: the supervisor is Linux-only")
LIMITS = CaptureLimits(stdout_bytes=4096, stderr_bytes=4096, combined_bytes=6144)
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def run(text, *, kind="python", timeout_s=10.0, limits=LIMITS, env=ENV):
    program = Program.from_text(kind, text)
    return supervise(
        lambda status_fd: launcher_command(sys.executable, status_fd, program, shell="/bin/sh"),
        stdin=program.source, env=env, timeout_s=timeout_s, limits=limits,
    )


def sleeper():
    """A unique sleep duration: it tags the test's sleeper processes so they can be found and killed."""
    return f"6{secrets.randbelow(10**6):06d}.{secrets.randbelow(10**6):06d}"


def alive(marker):
    """Processes whose command line carries ``marker`` (the test programs tag their sleepers)."""
    found = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as handle:
                    if marker.encode() in handle.read():
                        found.append(int(entry))
            except OSError:
                pass
    return found


def kill_marked(marker):
    for pid in alive(marker):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# Exit classification


def test_a_successful_program_reports_its_output_and_a_clean_exit():
    outcome = run("print(6 * 7)")

    assert (outcome.started, outcome.exit_code, outcome.signal, outcome.timed_out) == (True, 0, None, False)
    assert (outcome.stdout, outcome.stdout_bytes, outcome.cleanup_status) == (b"42\n", 3, "clean")
    assert set(outcome.resource_usage) <= set(RESOURCE_KEYS) and outcome.resource_usage["cpu_s"] >= 0


@pytest.mark.parametrize(
    ("kind", "text", "exit_code", "number"),
    [
        ("python", "import sys; sys.exit(3)", 3, None),
        ("shell", "echo out; echo err >&2; exit 7", 7, None),
        ("python", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)", None, signal.SIGTERM),
        ("shell", "kill -KILL $$", None, signal.SIGKILL),
        ("python", "raise SystemExit('boom')", 1, None),
    ],
)
def test_nonzero_exits_and_signals_are_valid_results_classified_exactly(kind, text, exit_code, number):
    outcome = run(text, kind=kind)

    assert outcome.started and not outcome.timed_out
    assert (outcome.exit_code, outcome.signal) == (exit_code, number)


def test_a_program_that_never_reaches_the_launcher_never_started():
    # Without the launcher nothing reports setup completion: a setup failure, not a result.
    outcome = supervise(lambda status_fd: ["/bin/sh", "-c", "echo setup-noise >&2; exit 1"], stdin=b"", env=ENV, timeout_s=5, limits=LIMITS)
    assert (outcome.started, outcome.exit_code, outcome.setup_error) == (False, None, "bulkhead.setup.failed")

    missing = supervise(lambda status_fd: ["/nonexistent/bwrap"], stdin=b"", env=ENV, timeout_s=5, limits=LIMITS)
    assert missing == ProcessOutcome.not_started("bulkhead.setup.spawn", elapsed_s=missing.elapsed_s)


def test_the_launcher_reads_shell_text_itself_and_leaves_the_program_no_stdin():
    outcome = run("cat; echo after", kind="shell")
    assert outcome.stdout == b"after\n"
    python = run("import sys; print(repr(sys.stdin.read()))")
    assert python.stdout == b"''\n"  # the program text was the whole of stdin


# Draining and caps


def test_full_pipes_on_both_streams_never_deadlock_and_caps_apply_while_reading():
    text = "import sys\nsys.stderr.write('e' * 1_000_000); sys.stderr.flush()\nsys.stdout.write('o' * 1_000_000)\n"
    outcome = run(text, timeout_s=20)

    assert outcome.started and outcome.exit_code == 0 and not outcome.timed_out
    assert (outcome.stdout_bytes, outcome.stderr_bytes) == (1_000_000, 1_000_000)
    assert outcome.stdout_truncated and outcome.stderr_truncated
    assert len(outcome.stdout) <= LIMITS.stdout_bytes and len(outcome.stderr) <= LIMITS.stderr_bytes
    assert len(outcome.stdout) + len(outcome.stderr) <= LIMITS.combined_bytes


def test_output_within_the_caps_is_kept_whole():
    outcome = run("import sys; sys.stdout.write('a' * 1000); sys.stderr.write('b' * 1000)")
    assert (outcome.stdout, outcome.stderr) == (b"a" * 1000, b"b" * 1000)
    assert not outcome.stdout_truncated and not outcome.stderr_truncated


def test_an_output_flood_is_cut_off_at_the_drain_limit():
    limits = CaptureLimits(stdout_bytes=1024, stderr_bytes=1024, combined_bytes=1024, drain_bytes=256 * 1024)
    outcome = run("import sys\nwhile True:\n    sys.stdout.write('x' * 65536)\n", limits=limits, timeout_s=20)

    assert outcome.started and not outcome.timed_out
    assert outcome.signal == signal.SIGPIPE or outcome.exit_code not in (0, None)  # the writer lost its pipe
    assert 256 * 1024 <= outcome.stdout_bytes < 1024 * 1024 and outcome.stdout_truncated


def test_capture_limits_follow_the_sandbox_limits_and_bound_stdin():
    limits = CaptureLimits.from_limits(Limits(cpu_s=1, mem_mb=64, pids=8, disk_mb=10, out_kb=16, stdout_kb=8))
    assert (limits.stdout_bytes, limits.stderr_bytes, limits.combined_bytes) == (8192, 16384, 16384)
    with pytest.raises(ValueError, match="stdin"):
        supervise(lambda fd: ["/bin/true"], stdin=b"x" * (limits.stdin_bytes + 1), env=ENV, timeout_s=1, limits=limits)
    with pytest.raises(ValueError):
        CaptureLimits(1, 1, 10, drain_bytes=5)


# The deadline and cleanup


def test_the_deadline_is_enforced_and_the_whole_group_is_killed():
    marker = sleeper()
    started = time.monotonic()
    outcome = run(f"sleep {marker} & sleep {marker}", kind="shell", timeout_s=0.5)
    try:
        assert outcome.started and outcome.timed_out and outcome.cleanup_status == "killed"
        assert outcome.exit_code is None
        assert time.monotonic() - started < 5
        assert alive(marker) == []
    finally:
        kill_marked(marker)


def test_a_background_child_left_behind_after_exit_is_killed():
    marker = sleeper()
    outcome = run(f"sleep {marker} >/dev/null 2>&1 & echo started", kind="shell")
    try:
        assert outcome.started and outcome.exit_code == 0 and not outcome.timed_out
        assert outcome.cleanup_status == "killed"
        assert alive(marker) == []
    finally:
        kill_marked(marker)


def test_a_session_escapee_holding_the_pipes_is_reported_as_leaked():
    # Without a PID namespace (bubblewrap provides one) or a cgroup (part 5), a process that
    # leaves the session escapes cleanup. The supervisor must say so rather than claim "clean".
    marker = sleeper()
    outcome = run(f"setsid sleep {marker} & echo bye", kind="shell")
    try:
        assert outcome.started and outcome.exit_code == 0
        assert outcome.cleanup_status == "leaked"
    finally:
        kill_marked(marker)


# Hygiene: argv, environment, descriptors


def test_the_program_text_is_never_on_the_command_line():
    program = Program.from_text(ProgramKind.SHELL, "echo do-not-log-me")
    argv = launcher_command("/usr/bin/python3", 9, program, shell="/bin/sh")
    assert not any("do-not-log-me" in item for item in argv)
    assert argv[:5] == ["/usr/bin/python3", "-I", "-S", "-c", argv[4]] and argv[5:] == ["9", "shell", "/bin/sh", "-c"]
    python = launcher_command("/usr/bin/python3", 9, Program.from_text("python", "print(1)"), shell="/bin/sh")
    assert python[5:] == ["9", "python", "/usr/bin/python3", "-I", "-"]


def test_only_the_given_environment_reaches_the_program(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_should_never_leak")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    outcome = run("import json, os; print(json.dumps(dict(os.environ)))")

    assert json.loads(outcome.stdout) == ENV


def test_the_program_inherits_only_stdin_stdout_and_stderr():
    extra_read, extra_write = os.pipe()
    os.set_inheritable(extra_write, True)  # even an inheritable descriptor in the supervisor must not leak
    try:
        outcome = run("import json, os\nopen_fds = []\nfor n in range(3, 256):\n    try:\n        os.fstat(n); open_fds.append(n)\n    except OSError:\n        pass\nprint(json.dumps(open_fds))")
    finally:
        os.close(extra_read)
        os.close(extra_write)
    assert json.loads(outcome.stdout) == []


def test_a_program_cannot_forge_its_own_status():
    # The status pipe is closed before the program runs, so it cannot claim another exit.
    outcome = run("import os\nfor fd in range(3, 64):\n    try:\n        os.write(fd, b'E0\\n')\n    except OSError:\n        pass\nraise SystemExit(5)")
    assert outcome.exit_code == 5

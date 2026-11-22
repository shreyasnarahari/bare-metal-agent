import shutil
from pathlib import Path
import sys
import threading
import time

import pytest

from toolsmith.background import BackgroundJobs, CommandFailed, last_progress, run_logged, say


def test_a_job_logs_to_its_file_not_the_notebook(tmp_path, capsys):
    jobs = BackgroundJobs(tmp_path)

    def work():
        say("downloading")
        return 42

    jobs.submit("download", work)

    assert jobs.wait("download", quiet=True) == 42
    assert capsys.readouterr().out == ""
    log = (tmp_path / "download.log").read_text()
    assert "downloading" in log and "download finished" in log
    assert jobs.status() == {"download": "done"} and jobs.active() == []


def test_say_prints_outside_a_job(capsys):
    say("hello")
    assert capsys.readouterr().out == "hello\n"


def test_a_failure_is_reraised_with_the_log_tail(tmp_path):
    jobs = BackgroundJobs(tmp_path)

    def broken():
        say("about to fail")
        raise ValueError("bad checksum")

    jobs.submit("gguf", broken)

    with pytest.raises(RuntimeError, match="bad checksum") as caught:
        jobs.wait("gguf", quiet=True)
    assert "about to fail" in str(caught.value)
    assert jobs.status() == {"gguf": "failed"}


def test_after_runs_in_order_and_a_failed_dependency_blocks_the_next_job(tmp_path):
    jobs = BackgroundJobs(tmp_path)
    order = []
    release = threading.Event()

    def first():
        release.wait(5)
        order.append("first")

    jobs.submit("first", first)
    jobs.submit("second", lambda: order.append("second"), after=("first",))
    time.sleep(0.05)
    assert order == []
    release.set()
    jobs.wait("second", quiet=True)
    assert order == ["first", "second"]

    jobs.submit("broken", lambda: 1 / 0)
    jobs.submit("blocked", lambda: order.append("blocked"), after=("broken",))
    with pytest.raises(RuntimeError, match="broken failed"):
        jobs.wait("blocked", quiet=True)
    assert "blocked" not in order
    with pytest.raises(ValueError, match="unknown jobs"):
        jobs.submit("orphan", lambda: None, after=("missing",))


def test_resubmitting_a_running_name_returns_the_same_job(tmp_path):
    jobs = BackgroundJobs(tmp_path)
    release = threading.Event()
    calls = []

    def work():
        calls.append(1)
        release.wait(5)

    first = jobs.submit("build", work)
    second = jobs.submit("build", work)
    assert first is second and jobs.active() == ["build"]
    release.set()
    jobs.wait("build", quiet=True)
    jobs.submit("build", work)  # finished, so it runs again
    jobs.wait("build", quiet=True)
    assert len(calls) == 2


def test_wait_reports_progress_while_running(tmp_path, capsys):
    jobs = BackgroundJobs(tmp_path)
    release = threading.Event()

    def work():
        say("[ 42%] Building CUDA object")
        release.wait(5)

    jobs.submit("compile", work)
    threading.Timer(0.3, release.set).start()
    jobs.wait("compile", poll_seconds=0.1)

    assert "[compile] still running" in capsys.readouterr().out
    assert last_progress(tmp_path / "compile.log") == "42%"


def test_run_logged_writes_output_and_raises_with_the_tail(tmp_path):
    log = tmp_path / "command.log"
    run_logged([sys.executable, "-c", "print('compiled ok')"], log_path=log)
    assert "compiled ok" in log.read_text()

    with pytest.raises(CommandFailed, match="exit 3") as caught:
        run_logged([sys.executable, "-c", "print('nvcc died'); raise SystemExit(3)"], log_path=log)
    assert "nvcc died" in caught.value.tail


def test_run_logged_inside_a_job_uses_the_job_log_and_honours_nice(tmp_path):
    jobs = BackgroundJobs(tmp_path)
    jobs.submit("niced", run_logged, [sys.executable, "-c", "import os; print('niceness', os.nice(0))"], nice=5)
    jobs.wait("niced", quiet=True)

    text = (tmp_path / "niced.log").read_text()
    expected = "niceness 5" if shutil.which("nice") else "niceness 0"  # nice may be absent on minimal systems
    assert expected in text and "$ " in text


def test_cancel_all_stops_running_commands(tmp_path):
    jobs = BackgroundJobs(tmp_path)
    jobs.submit("sleeper", run_logged, [sys.executable, "-c", "import time; time.sleep(60)"], poll_seconds=0.1)
    deadline = time.monotonic() + 5
    while not jobs.job("sleeper").processes and time.monotonic() < deadline:
        time.sleep(0.05)

    assert jobs.cancel_all() == ["sleeper"]
    with pytest.raises(RuntimeError, match="sleeper"):
        jobs.wait("sleeper", quiet=True)
    assert jobs.job("sleeper").processes[0].poll() is not None


def test_drain_returns_what_did_not_finish(tmp_path):
    jobs = BackgroundJobs(tmp_path)
    release = threading.Event()
    jobs.submit("quick", lambda: None)
    jobs.submit("slow", release.wait, 5)
    jobs.submit("bad", lambda: 1 / 0)

    unfinished = jobs.drain(timeout=0.3)

    assert unfinished == {"slow": "running", "bad": "failed"}
    release.set()
    assert jobs.drain(timeout=5) == {"bad": "failed"}


def test_waiting_for_an_unknown_job_names_the_cell_to_run(tmp_path):
    with pytest.raises(KeyError, match="run the cell that starts it"):
        BackgroundJobs(tmp_path).wait("nothing")


def test_cancel_all_does_not_claim_to_stop_a_python_job(tmp_path):
    jobs = BackgroundJobs(tmp_path)
    release = threading.Event()
    jobs.submit("drive copy", release.wait, 5)

    assert jobs.cancel_all() == []  # nothing to kill; it finishes on its own
    release.set()
    jobs.wait("drive copy", quiet=True)


def test_long_commands_are_shortened_in_the_log(tmp_path):
    log = tmp_path / "command.log"
    run_logged([sys.executable, "-c", "pass", *[f"schema-{index}.json" for index in range(500)]], log_path=log)

    first = log.read_text().splitlines()[0]
    assert len(first) < 2100 and "(503 arguments)" in first


def test_a_command_that_goes_quiet_is_killed():
    from toolsmith.background import CommandStalled

    started = time.monotonic()
    with pytest.raises(CommandStalled, match="no output for 1s") as caught:
        run_logged(
            [sys.executable, "-c", "print('cloning', flush=True); import time; time.sleep(60)"],
            log_path=Path(caught_log := f"/tmp/stall-{time.monotonic_ns()}.log"), stall_seconds=1, poll_seconds=0.2,
        )
    assert time.monotonic() - started < 15
    assert "cloning" in caught.value.tail
    Path(caught_log).unlink(missing_ok=True)


def test_a_command_past_its_time_limit_is_killed(tmp_path):
    from toolsmith.background import CommandStalled

    chatty = "import time\nwhile True:\n    print('working', flush=True)\n    time.sleep(0.1)\n"
    with pytest.raises(CommandStalled, match="still running after 1s"):
        run_logged([sys.executable, "-c", chatty], log_path=tmp_path / "chatty.log", timeout=1, poll_seconds=0.2)


def test_wait_warns_about_a_silent_job_and_can_give_up(tmp_path, capsys):
    import os

    jobs = BackgroundJobs(tmp_path)
    release = threading.Event()
    jobs.submit("silent", release.wait, 10)
    deadline = time.monotonic() + 5
    while "silent started" not in (tmp_path / "silent.log").read_text() and time.monotonic() < deadline:
        time.sleep(0.02)  # the job's own first line must not refresh the timestamp set below
    old = time.time() - 3600
    os.utime(tmp_path / "silent.log", (old, old))

    with pytest.raises(TimeoutError, match="still running"):
        jobs.wait("silent", poll_seconds=0.1, timeout=0.3, silence_seconds=60, quiet=True)
    assert capsys.readouterr().out.count("may be stuck") == 1  # said once, not every poll
    release.set()
    jobs.wait("silent", quiet=True)

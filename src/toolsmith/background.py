"""Background jobs for notebooks: overlap downloads, builds, and Drive copies with GPU work.

A job runs in a worker thread and writes everything it says, and every command
it runs, to its own log file, so it never prints into whichever notebook cell
happens to be running. :meth:`BackgroundJobs.wait` shows progress from that log
and re-raises a failure with the log tail.

Code that may run inside a job reports through :func:`say` and starts commands
through :func:`run_logged`, which registers each process so
:meth:`BackgroundJobs.cancel_all` can stop it.

Standard library only: the setup cells import this before ``pip install``.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG: contextvars.ContextVar[Callable[[str], None] | None] = contextvars.ContextVar("toolsmith_log", default=None)
_LOG_PATH: contextvars.ContextVar[Path | None] = contextvars.ContextVar("toolsmith_log_path", default=None)
_ON_PROCESS: contextvars.ContextVar[Callable[[subprocess.Popen], None] | None] = contextvars.ContextVar(
    "toolsmith_on_process", default=None,
)


class CommandFailed(RuntimeError):
    def __init__(self, command: Sequence[str], returncode: int, tail: str) -> None:
        super().__init__(f"Command failed with exit {returncode}: {' '.join(command)}\n{tail}")
        self.command = list(command)
        self.returncode = returncode
        self.tail = tail


def say(message: str) -> None:
    """Print, or append to the current job's log when called from a background job."""
    log = _LOG.get()
    if log is None:
        print(message, flush=True)
    else:
        log(message)


def current_log_path() -> Path | None:
    return _LOG_PATH.get()


def file_logger(path: Path) -> Callable[[str], None]:
    lock = threading.Lock()

    def log(message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        with lock, path.open("a", encoding="utf-8") as output:
            output.write(f"[{stamp}] {message}\n")

    return log


@contextlib.contextmanager
def log_to(path: Path) -> Iterator[None]:
    tokens = (_LOG.set(file_logger(path)), _LOG_PATH.set(path))
    try:
        yield
    finally:
        _LOG.reset(tokens[0])
        _LOG_PATH.reset(tokens[1])


def tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


_PERCENT = re.compile(r"\[\s*(\d{1,3})%\]")


def last_progress(path: Path) -> str | None:
    """The last meaningful line of a log: a ``[NN%]`` build step if there is one."""
    text = tail(path, 200)
    percents = _PERCENT.findall(text)
    if percents:
        return f"{percents[-1]}%"
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1][-160:] if lines else None


def run_logged(
    command: Sequence[Any],
    *,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    nice: int = 0,
    poll_seconds: float = 30.0,
    progress: Callable[[Path], None] | None = None,
) -> None:
    """Run ``command`` with its output in ``log_path`` (default: the current job's log).

    The process gets its own session so cancelling kills the whole tree (nvcc,
    compilers). ``nice`` lowers its CPU priority for background builds. On a
    non-zero exit, raises :class:`CommandFailed` with the log tail.
    """
    command = [str(part) for part in command]
    if nice and shutil.which("nice"):
        command = ["nice", "-n", str(nice), *command]
    log_path = Path(log_path or current_log_path() or Path("/tmp") / f"toolsmith-command-{os.getpid()}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shown = " ".join(command)
    if len(shown) > 2000:  # a MaskBench run lists a thousand schema files
        shown = shown[:2000] + f" ... ({len(command)} arguments)"
    with log_path.open("ab") as output:
        output.write(f"$ {shown}\n".encode())
        output.flush()
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env, cwd=None if cwd is None else str(cwd), start_new_session=True,
        )
    hook = _ON_PROCESS.get()
    if hook is not None:
        hook(process)
    while True:
        try:
            returncode = process.wait(timeout=poll_seconds)
            break
        except subprocess.TimeoutExpired:
            if progress is not None:
                progress(log_path)
    if returncode != 0:
        raise CommandFailed(command, returncode, tail(log_path))


def _stop_process(process: subprocess.Popen, grace: float = 5.0) -> None:
    for action in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, action)
        except (ProcessLookupError, PermissionError):
            (process.terminate if action == signal.SIGTERM else process.kill)()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue


@dataclass
class Job:
    name: str
    future: Future
    log_path: Path
    started: float
    processes: list[subprocess.Popen] = field(default_factory=list)

    @property
    def state(self) -> str:
        if not self.future.done():
            return "running"
        if self.future.cancelled():
            return "cancelled"
        return "failed" if self.future.exception() is not None else "done"


class BackgroundJobs:
    """Named jobs on a thread pool, each with its own log file.

    Submitting a name that is still running returns the running job instead of
    starting a duplicate, so rerunning a notebook cell is safe. A finished name
    (done or failed) starts again.
    """

    def __init__(self, log_dir: str | Path, *, max_workers: int = 8) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="toolsmith-bg")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, name: str, fn: Callable[..., Any], *args: Any, after: Sequence[str] = (), **kwargs: Any) -> Future:
        with self._lock:
            existing = self._jobs.get(name)
            if existing is not None and not existing.future.done():
                return existing.future
            unknown = [dependency for dependency in after if dependency not in self._jobs]
            if unknown:
                raise ValueError(f"{name} waits for unknown jobs {unknown}")
            dependencies = [(dependency, self._jobs[dependency].future) for dependency in after]
            safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "job"
            log_path = self.log_dir / f"{safe}.log"
            log_path.write_text("", encoding="utf-8")
            job = Job(name, Future(), log_path, time.monotonic())

            def run() -> Any:
                for dependency, future in dependencies:
                    try:
                        future.result()
                    except BaseException as error:
                        raise RuntimeError(f"{name} did not start: {dependency} failed ({error})") from error
                token = _ON_PROCESS.set(job.processes.append)
                try:
                    with log_to(log_path):
                        say(f"{name} started")
                        result = fn(*args, **kwargs)
                        say(f"{name} finished")
                        return result
                except BaseException as error:
                    file_logger(log_path)(f"{name} failed: {type(error).__name__}: {error}")
                    raise
                finally:
                    _ON_PROCESS.reset(token)

            job.future = self._pool.submit(run)
            self._jobs[name] = job
            return job.future

    def job(self, name: str) -> Job:
        try:
            return self._jobs[name]
        except KeyError:
            raise KeyError(f"No background job named {name!r}; run the cell that starts it") from None

    def wait(self, name: str, *, poll_seconds: float = 30.0, quiet: bool = False) -> Any:
        """The job's result. Prints progress while waiting; a failure is re-raised with the log tail."""
        job = self.job(name)
        while True:
            try:
                return job.future.result(timeout=poll_seconds)
            except FutureTimeout:
                if not quiet:
                    elapsed = int(time.monotonic() - job.started)
                    print(f"[{name}] still running after {elapsed}s: {last_progress(job.log_path) or 'no output yet'}", flush=True)
            except Exception as error:
                raise RuntimeError(
                    f"Background job {name!r} failed: {error}\n--- {job.log_path} ---\n{tail(job.log_path)}"
                ) from error

    def _snapshot(self) -> list[tuple[str, Job]]:
        with self._lock:  # runtime stamps read this from worker threads while cells submit
            return sorted(self._jobs.items())

    def active(self) -> list[str]:
        return [name for name, job in self._snapshot() if not job.future.done()]

    def status(self) -> dict[str, str]:
        return {name: job.state for name, job in self._snapshot()}

    def drain(self, timeout: float) -> dict[str, str]:
        """Wait up to ``timeout`` seconds for every job; returns the jobs that did not finish cleanly."""
        deadline = time.monotonic() + timeout
        for _, job in self._snapshot():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                job.future.result(timeout=remaining)
            except FutureTimeout:
                break
            except BaseException:
                continue
        return {name: state for name, state in self.status().items() if state != "done"}

    def cancel_all(self) -> list[str]:
        """Kill running jobs' processes and drop jobs that have not started; returns the names stopped.

        A job running pure Python (a Drive copy) cannot be interrupted; it is not
        listed and finishes on its own (copies are atomic, so a cut-off one is redone).
        """
        stopped = []
        for name, job in self._snapshot():
            if job.future.done():
                continue
            killed = False
            for process in list(job.processes):
                if process.poll() is None:
                    _stop_process(process)
                    killed = True
            if job.future.cancel() or killed:
                stopped.append(name)
        return stopped


__all__ = [
    "BackgroundJobs",
    "CommandFailed",
    "Job",
    "current_log_path",
    "file_logger",
    "last_progress",
    "log_to",
    "run_logged",
    "say",
    "tail",
]

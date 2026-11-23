"""Own a sandbox process: spawn it, feed its program, drain its output, enforce its deadline, clean up.

The supervisor, not a daemon handler thread, owns the subprocess:

- ``subprocess.Popen`` with a fixed argv list (never ``shell=True``), a new
  session, the exact environment the caller gives, ``close_fds`` plus only the
  descriptors passed explicitly, and no ``preexec_fn``. Namespace and seccomp
  setup belong to bubblewrap or an immediately executed helper, never to this
  possibly multithreaded process.
- Program text goes in over stdin, bounded by the caller; it never appears on
  a command line.
- One loop drains stdout, stderr, and the launcher's status pipe together
  through a selector, so a child blocked writing a full stderr pipe can never
  deadlock one waiting on stdout. Per-stream caps and a combined cap are
  applied while reading: only bounded prefixes and byte counts are kept, and
  after ``drain_bytes`` in total the pipes are closed.
- The deadline is ``time.monotonic()``-based. At the deadline the process
  group is killed with SIGKILL.

Whether the program started comes from the launcher (``bulkhead._launch``),
which writes ``S`` to a status pipe only once the sandbox is set up, then the
program's exact exit code or signal. A nonzero exit is a valid result; a run
that never wrote ``S`` failed during setup.

Cleanup is described exactly as far as it reaches. After the leader exits (or
is killed), every remaining member of its process group or session is killed
and the leader is reaped. Under bubblewrap the sandbox also has its own PID
namespace, whose processes the kernel kills when its init exits. This is not a
cgroup tree kill (part 5): a process that escaped both the session and a PID
namespace is not tracked, and if it still holds the output pipes the run is
reported as ``leaked``.
"""

from __future__ import annotations

import os
import re
import selectors
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from .backends.base import Program, ProgramKind
from .manifest import Limits

LAUNCHER = Path(__file__).with_name("_launch.py")
READY = b"R"  # from the helper: setup done, waiting at the start barrier
STARTED = b"S"
MAX_STATUS_BYTES = 64
READ_CHUNK = 65536
# After the leader exits: how long to keep draining pipes, and how long to wait for killed processes.
DRAIN_GRACE_S = 0.5
KILL_WAIT_S = 2.0
_STATUS_TAIL = re.compile(rb"([EK])(-?\d{1,3})\n")


@dataclass(frozen=True)
class CaptureLimits:
    """Output bounds: per-stream prefixes kept, a combined cap, and the most read before the pipes are closed."""

    stdout_bytes: int
    stderr_bytes: int
    combined_bytes: int
    drain_bytes: int = 16 * 1024 * 1024
    stdin_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        for name in ("stdout_bytes", "stderr_bytes", "combined_bytes", "drain_bytes", "stdin_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.drain_bytes < self.combined_bytes:
            raise ValueError("drain_bytes must be at least combined_bytes")

    @classmethod
    def from_limits(cls, limits: Limits, *, stdin_bytes: int = 64 * 1024) -> CaptureLimits:
        """From a sandbox's limits: ``out_kb`` combined, ``stdout_kb``/``stderr_kb`` per stream (default ``out_kb``)."""
        combined = int(limits.out_kb) * 1024
        stdout = int(limits.stdout_kb if limits.stdout_kb is not None else limits.out_kb) * 1024
        stderr = int(limits.stderr_kb if limits.stderr_kb is not None else limits.out_kb) * 1024
        return cls(stdout, stderr, combined, max(16 * 1024 * 1024, combined), stdin_bytes)


@dataclass(frozen=True)
class ProcessOutcome:
    """What the supervisor observed. ``stdout``/``stderr`` are bounded prefixes; the counts are bytes read."""

    started: bool
    exit_code: int | None
    signal: int | None
    timed_out: bool
    stdout: bytes
    stderr: bytes
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool
    elapsed_s: float
    cleanup_status: str
    resource_usage: Mapping[str, Any] = field(default_factory=dict)
    setup_error: str | None = None
    stopped: str | None = None  # a ceiling the governor saw crossed (``disk``, ``cpu``), which ended the run

    @classmethod
    def not_started(cls, code: str, *, elapsed_s: float = 0.0) -> ProcessOutcome:
        """A run refused before anything was spawned."""
        return cls(False, None, None, False, b"", b"", 0, 0, False, False, elapsed_s, "not_started", {}, code)


class _Capture:
    """A bounded prefix of one stream, plus how much was read in total."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.kept = bytearray()
        self.total = 0

    def feed(self, chunk: bytes, room: int) -> int:
        """Keep what fits in this stream's limit and ``room`` (the combined budget left); return bytes kept."""
        self.total += len(chunk)
        take = max(0, min(len(chunk), self.limit - len(self.kept), room))
        if take:
            self.kept += chunk[:take]
        return take

    @property
    def truncated(self) -> bool:
        return self.total > len(self.kept)


def launcher_command(interpreter: str, status_fd: int, program: Program, *, shell: str) -> list[str]:
    """The fixed argv that starts the launcher, which then runs ``program`` (text arrives on stdin).

    ``interpreter`` runs both the launcher and Python programs; ``shell`` runs
    shell programs as ``shell -c TEXT`` inside the sandbox. The model supplies
    none of these.
    """
    source = _launcher_source()
    if program.kind is ProgramKind.PYTHON:
        target = [interpreter, "-I", "-"]
    else:
        target = [shell, "-c"]
    return [interpreter, "-I", "-S", "-c", source, str(status_fd), program.kind.value, *target]


@lru_cache(maxsize=1)
def _launcher_source() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


class Gate(Protocol):
    """Per-call governance the supervisor drives (``bulkhead.governor.CallGovernor``)."""

    @property
    def needs_barrier(self) -> bool: ...
    @property
    def tree_kill(self) -> bool: ...
    def admit(self, pid: int) -> None: ...
    def poll(self) -> str | None: ...
    def kill(self) -> None: ...
    def populated(self) -> bool: ...
    def usage(self) -> dict[str, Any]: ...
    def close(self) -> str: ...


def supervise(
    argv_for: Callable[..., Sequence[str]],
    *,
    stdin: bytes,
    env: Mapping[str, str],
    timeout_s: float,
    limits: CaptureLimits,
    pass_fds: Sequence[int] = (),
    cwd: str = "/",
    gate: Gate | None = None,
) -> ProcessOutcome:
    """Run ``argv_for(status_fd)`` to completion or its deadline and report what happened.

    ``argv_for`` receives the descriptor number of the launcher's status pipe
    as the child will see it. ``stdin`` is written to the child and then
    closed. Nothing is raised for a failing program; a spawn failure is a run
    that never started.

    With a ``gate`` that needs a barrier, ``argv_for(status_fd, barrier_fd)``
    also receives the read end of a start-barrier pipe. The helper writes
    ``R`` when its setup is done and then waits; the supervisor calls
    ``gate.admit(pid)`` (the cgroup move and its check) and only then writes
    ``G``. If admission fails the run is killed and never started. While the
    program runs, ``gate.poll()`` may name a crossed ceiling; the run is then
    killed and reported as ``stopped``. Cleanup uses ``gate.kill()``
    (``cgroup.kill``) as well as the process group.
    """
    if len(stdin) > limits.stdin_bytes:
        raise ValueError(f"stdin is larger than {limits.stdin_bytes} bytes")
    if not timeout_s > 0:
        raise ValueError("timeout_s must be positive")
    began = time.monotonic()
    deadline = began + timeout_s
    status_read, status_write = os.pipe()
    barrier_read = barrier_write = None
    if gate is not None and gate.needs_barrier:
        barrier_read, barrier_write = os.pipe()
    try:
        if barrier_read is None:
            argv = [os.fspath(item) for item in argv_for(status_write)]
            passed = (status_write, *pass_fds)
        else:
            argv = [os.fspath(item) for item in argv_for(status_write, barrier_read)]
            passed = (status_write, barrier_read, *pass_fds)
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env),
            cwd=cwd,
            start_new_session=True,
            close_fds=True,
            pass_fds=passed,
        )
    except OSError:
        os.close(status_read)
        if barrier_write is not None:
            os.close(barrier_write)
        return ProcessOutcome.not_started("bulkhead.setup.spawn", elapsed_s=time.monotonic() - began)
    finally:
        os.close(status_write)
        if barrier_read is not None:
            os.close(barrier_read)
    return _Run(process, status_read, stdin, deadline, limits, began, gate=gate, barrier_fd=barrier_write).wait()


class _Run:
    """The drain/deadline/cleanup loop for one spawned process."""

    def __init__(
        self,
        process: subprocess.Popen,
        status_fd: int,
        stdin: bytes,
        deadline: float,
        limits: CaptureLimits,
        began: float,
        *,
        gate: Gate | None = None,
        barrier_fd: int | None = None,
    ) -> None:
        self.gate = gate
        self.barrier_fd = barrier_fd
        self.admit_error: str | None = None
        self.stopped: str | None = None
        self.process = process
        self.pid = process.pid
        self.limits = limits
        self.deadline = deadline
        self.began = began
        self.status = bytearray()
        self.captures = {"stdout": _Capture(limits.stdout_bytes), "stderr": _Capture(limits.stderr_bytes)}
        self.kept_total = 0
        self.read_total = 0
        self.pending_stdin = memoryview(stdin)
        self.selector = selectors.DefaultSelector()
        self.exited = False
        self.exited_at: float | None = None
        self.pidfd: int | None = None
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        self.fds = {"stdout": process.stdout.fileno(), "stderr": process.stderr.fileno(), "status": status_fd}
        self.stdin_fd: int | None = process.stdin.fileno()
        for name, fd in self.fds.items():
            os.set_blocking(fd, False)
            self.selector.register(fd, selectors.EVENT_READ, name)
        if self.pending_stdin:
            os.set_blocking(self.stdin_fd, False)
            self.selector.register(self.stdin_fd, selectors.EVENT_WRITE, "stdin")
        else:
            self._close_stdin()
        try:
            self.pidfd = os.pidfd_open(self.pid)
            self.selector.register(self.pidfd, selectors.EVENT_READ, "exit")
        except (AttributeError, OSError):  # no pidfd: poll for exit instead
            self.pidfd = None

    # The loop

    def wait(self) -> ProcessOutcome:
        timed_out = False
        try:
            while self._open_streams() or not self.exited:
                now = time.monotonic()
                if now >= self.deadline:
                    timed_out = True
                    break
                wait = min(self.deadline - now, 0.05 if self.pidfd is None or self.gate is not None else 1.0)
                if self.exited:
                    assert self.exited_at is not None
                    grace = self.exited_at + DRAIN_GRACE_S - now
                    if grace <= 0:
                        break  # something outside the leader still holds the pipes
                    wait = min(wait, grace)
                for key, _ in self.selector.select(timeout=wait):
                    self._ready(key.data, key.fd)
                if self.pidfd is None and not self.exited and self._leader_exited():
                    self._mark_exited()
                if self.admit_error is not None:
                    break
                if self.gate is not None and not self.exited and self._released():
                    try:
                        reason = self.gate.poll()
                    except Exception:  # an unreadable cgroup must not let the run continue unwatched
                        reason = "governor"
                    if reason is not None:
                        self.stopped = reason
                        break
            cleanup = self._clean_up(timed_out or self.stopped is not None or self.admit_error is not None)
            raw_status = self._reap()
            usage = self._gate_usage()
            cleanup = self._close_gate(cleanup)
        finally:
            self._close_all()
        return self._outcome(timed_out, cleanup, raw_status, usage)

    def _released(self) -> bool:
        return self.gate is None or not self.gate.needs_barrier or self.barrier_fd is None

    def _release(self) -> None:
        """The helper is set up and waiting: admit it to the call's cgroup, then let it exec."""
        if self.barrier_fd is None:
            return
        try:
            if self.gate is not None:
                self.gate.admit(self.pid)
            os.write(self.barrier_fd, b"G")
        except Exception as exc:  # admission failed: the program must not run outside its cgroup
            self.admit_error = getattr(exc, "code", "bulkhead.setup.cgroup_admit")
        finally:
            os.close(self.barrier_fd)
            self.barrier_fd = None

    def _gate_usage(self) -> dict[str, Any]:
        if self.gate is None:
            return {}
        try:
            return dict(self.gate.usage())
        except Exception:
            return {}

    def _close_gate(self, cleanup: str) -> str:
        if self.gate is None:
            return cleanup
        try:
            status = self.gate.close()
        except Exception:
            status = "leaked"
        order = {"clean": 0, "killed": 1, "leaked": 2}
        return status if order.get(status, 2) > order.get(cleanup, 2) else cleanup

    def _open_streams(self) -> bool:
        # The status pipe counts: the launcher's final line may arrive just before its exit.
        return any(name in ("stdout", "stderr", "status") for name in self._registered())

    def _registered(self) -> set[str]:
        return {key.data for key in self.selector.get_map().values()}

    def _ready(self, name: str, fd: int) -> None:
        if name == "stdin":
            self._write_stdin()
        elif name == "exit":
            self._mark_exited()
        elif name == "status":
            self._read_status(fd)
        else:
            self._read_stream(name, fd)

    def _write_stdin(self) -> None:
        assert self.stdin_fd is not None
        try:
            written = os.write(self.stdin_fd, self.pending_stdin[:READ_CHUNK])
        except BlockingIOError:
            return
        except (BrokenPipeError, OSError):
            self._close_stdin()
            return
        self.pending_stdin = self.pending_stdin[written:]
        if not self.pending_stdin:
            self._close_stdin()

    def _close_stdin(self) -> None:
        if self.stdin_fd is None:
            return
        try:
            self.selector.unregister(self.stdin_fd)
        except (KeyError, ValueError):
            pass
        assert self.process.stdin is not None
        try:
            self.process.stdin.close()
        except OSError:
            pass
        self.stdin_fd = None

    def _read_status(self, fd: int) -> None:
        try:
            chunk = os.read(fd, MAX_STATUS_BYTES)
        except BlockingIOError:
            return
        if not chunk or len(self.status) + len(chunk) > MAX_STATUS_BYTES:
            self.status += chunk[: max(0, MAX_STATUS_BYTES - len(self.status))]
            self.selector.unregister(fd)
        else:
            self.status += chunk
        if self.status[:1] == READY and self.barrier_fd is not None:
            self._release()

    def _read_stream(self, name: str, fd: int) -> None:
        try:
            chunk = os.read(fd, READ_CHUNK)
        except BlockingIOError:
            return
        if not chunk:
            self.selector.unregister(fd)
            return
        self.read_total += len(chunk)
        self.kept_total += self.captures[name].feed(chunk, self.limits.combined_bytes - self.kept_total)
        if self.read_total >= self.limits.drain_bytes:
            # An output flood: close our ends, so the writer gets EPIPE/SIGPIPE instead of blocking
            # on a full pipe until the deadline. The counts stay as read.
            for stream, handle in (("stdout", self.process.stdout), ("stderr", self.process.stderr)):
                try:
                    self.selector.unregister(self.fds[stream])
                except (KeyError, ValueError):
                    pass
                if handle is not None:
                    try:
                        handle.close()
                    except OSError:
                        pass

    def _mark_exited(self) -> None:
        if self.exited:
            return
        self.exited = True
        self.exited_at = time.monotonic()
        if self.pidfd is not None:
            try:
                self.selector.unregister(self.pidfd)
            except (KeyError, ValueError):
                pass

    def _leader_exited(self) -> bool:
        try:
            info = os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return True
        return info is not None

    # Cleanup

    def _clean_up(self, timed_out: bool) -> str:
        """Kill the tree if needed, then any cgroup, session, or group member left; return the cleanup status."""
        killed = False
        if timed_out or not self.exited:
            self._kill_tree()
            killed = True
            self._wait_leader(KILL_WAIT_S)
        if self.gate is not None and self.gate.tree_kill:
            try:
                if self.gate.populated():
                    killed = True
                    self.gate.kill()
                    give_up = time.monotonic() + KILL_WAIT_S
                    while self.gate.populated() and time.monotonic() < give_up:
                        time.sleep(0.01)
                if self.gate.populated():
                    return "leaked"
            except Exception:
                return "leaked"
        stragglers = self._members()
        if stragglers:
            killed = True
            for pid in stragglers:
                self._kill(pid)
            give_up = time.monotonic() + KILL_WAIT_S
            while self._members() and time.monotonic() < give_up:
                time.sleep(0.01)
        if not self._leader_exited():
            return "leaked"
        if self._members():
            return "leaked"
        # Drain what the dead processes left in the pipes; a pipe still open now is held by an escapee.
        give_up = time.monotonic() + DRAIN_GRACE_S
        while self._open_streams() and time.monotonic() < give_up:
            for key, _ in self.selector.select(timeout=0.05):
                if key.data != "stdin":
                    self._ready(key.data, key.fd)
        if self._open_streams():
            return "leaked"
        return "killed" if killed else "clean"

    def _kill_tree(self) -> None:
        if self.gate is not None and self.gate.tree_kill:
            try:
                self.gate.kill()  # cgroup.kill: the whole tree, however it forked or regrouped
            except Exception:
                pass
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def _kill(self, pid: int) -> None:
        # Through a pidfd re-checked against /proc, so a recycled pid is never signalled.
        try:
            pidfd = os.pidfd_open(pid)
        except OSError:
            return
        try:
            if _session_and_group(pid) is not None and pid in self._members():
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        except OSError:
            pass
        finally:
            os.close(pidfd)

    def _wait_leader(self, seconds: float) -> None:
        give_up = time.monotonic() + seconds
        while not self._leader_exited() and time.monotonic() < give_up:
            time.sleep(0.01)
        if self._leader_exited():
            self._mark_exited()

    def _members(self) -> list[int]:
        """Live processes (other than the unreaped leader) in the leader's process group or session."""
        members = []
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == self.pid:
                continue
            ids = _session_and_group(pid)
            if ids is not None and self.pid in ids:
                members.append(pid)
        return members

    def _reap(self) -> tuple[int, Any] | None:
        if not self._leader_exited():
            return None
        try:
            _, raw, usage = os.wait4(self.pid, 0)
        except ChildProcessError:
            return None
        self.process.returncode = os.waitstatus_to_exitcode(raw)
        return raw, usage

    def _close_all(self) -> None:
        self._close_stdin()
        try:
            self.selector.close()
        except OSError:
            pass
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if self.barrier_fd is not None:
            os.close(self.barrier_fd)
            self.barrier_fd = None
        for fd in (self.fds["status"], self.pidfd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.pidfd = None

    # Classification

    def _outcome(self, timed_out: bool, cleanup: str, reaped: tuple[int, Any] | None, gate_usage: Mapping[str, Any]) -> ProcessOutcome:
        elapsed = time.monotonic() - self.began
        status = bytes(self.status[1:] if self.status[:1] == READY else self.status)
        started = status[:1] == STARTED and self.admit_error is None
        exit_code: int | None = None
        number: int | None = None
        usage: dict[str, Any] = {"wall_s": round(elapsed, 6)}
        if reaped is not None:
            raw, rusage = reaped
            usage["cpu_s"] = round(rusage.ru_utime + rusage.ru_stime, 6)
            usage["peak_memory_kb"] = int(rusage.ru_maxrss)
        usage.update(gate_usage)  # the cgroup's figures cover the whole tree and replace the leader's
        if started:
            tail = _STATUS_TAIL.fullmatch(status[1:])
            if tail is not None:
                value = int(tail.group(2))
                if tail.group(1) == b"K" and 1 <= value <= 64:
                    number = value
                elif tail.group(1) == b"E":
                    exit_code = value
            elif reaped is not None and not timed_out:
                raw = reaped[0]
                if os.WIFSIGNALED(raw):
                    number = os.WTERMSIG(raw)
                else:
                    exit_code = os.waitstatus_to_exitcode(raw)
        stdout, stderr = self.captures["stdout"], self.captures["stderr"]
        return ProcessOutcome(
            started=started,
            exit_code=exit_code,
            signal=number,
            timed_out=timed_out,
            stdout=bytes(stdout.kept),
            stderr=bytes(stderr.kept),
            stdout_bytes=stdout.total,
            stderr_bytes=stderr.total,
            stdout_truncated=stdout.truncated,
            stderr_truncated=stderr.truncated,
            elapsed_s=elapsed,
            cleanup_status=cleanup,
            resource_usage=usage,
            setup_error=None if started else (self.admit_error or ("bulkhead.setup.timeout" if timed_out else "bulkhead.setup.failed")),
            stopped=self.stopped if started else None,
        )


def _session_and_group(pid: int) -> tuple[int, int] | None:
    """(process group, session) of a live, non-zombie process, or None."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = text[text.rfind(")") + 2:].split()
    if len(fields) < 4 or fields[0] in ("Z", "X"):
        return None
    return int(fields[2]), int(fields[3])


__all__ = [
    "CaptureLimits",
    "LAUNCHER",
    "ProcessOutcome",
    "launcher_command",
    "supervise",
]

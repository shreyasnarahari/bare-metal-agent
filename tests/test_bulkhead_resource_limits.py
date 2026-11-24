"""Bulkhead part 5: rlimits, resource-exhaustion classification, the helper's start barrier, and resource attacks.

Unit tests run everywhere. The resource attacks themselves are part 7's
dataset (``evaluation/data/bulkhead_attacks_v1.json``): the ones safe without a
cgroup run on every proven profile through the attack suite
(``tests/test_bulkhead_attack_suite.py``). Destructive ones (fork and memory
bombs, CPU and disk exhaustion) run here only inside a confirmed delegated
cgroup parent, with an outer watchdog and host-safety checks:

    BULKHEAD_INTEGRATION=1 BULKHEAD_CGROUP_PARENT=<delegated leaf parent> pytest tests/test_bulkhead_resource_limits.py

Nothing here runs a resource bomb on the bare host.
"""

from __future__ import annotations

import json
import os
import resource
import secrets
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from bulkhead.backends.base import EXHAUSTION_LIMITS, ExecResult, Program, build_outcome, exhaustion
from bulkhead.cgroups import CgroupError
from bulkhead.helper import config_memfd, helper_command
from bulkhead.manifest import Limits, Sandbox, SecuritySpec, Sink, Trust
from bulkhead.rlimits import RlimitPlan, apply
from bulkhead.supervisor import CaptureLimits, ProcessOutcome, launcher_command, supervise
from steadyhand.security import process_tool_result
from steadyhand.tools_exec import EXEC_LIMITS, EXEC_POLICY, EXEC_SANDBOX
from steadyhand.types import ToolCall

from bulkhead.eval.attacks import judge, run_script

from exec_support import DATASET, request_for

LINUX = sys.platform.startswith("linux")
INTEGRATION = os.environ.get("BULKHEAD_INTEGRATION") == "1"
RESOURCE = [attack for attack in DATASET.attacks if attack.category == "resource"]


def spec_with(**limits):
    values = {key: getattr(EXEC_LIMITS, key) for key in ("cpu_s", "mem_mb", "pids", "disk_mb", "out_kb", "wall_s")} | limits
    sandbox = Sandbox(fs_read=EXEC_SANDBOX.fs_read, fs_write=EXEC_SANDBOX.fs_write, fs_deny=EXEC_SANDBOX.fs_deny, limits=Limits(**values))
    return SecuritySpec(sandbox, Trust.UNTRUSTED, (Sink.FILESYSTEM,))


def sleeper():
    return f"6{secrets.randbelow(10**6):06d}.{secrets.randbelow(10**6):06d}"


def alive(marker):
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


# rlimits


def test_the_rlimit_plan_follows_the_sandbox_and_never_sets_nproc():
    plan = RlimitPlan.from_limits(Limits(cpu_s=2.5, mem_mb=128, pids=16, disk_mb=10, out_kb=8))
    assert plan.entries() == [
        ("RLIMIT_CPU", 3, 4), ("RLIMIT_FSIZE", 10 << 20, 10 << 20), ("RLIMIT_NOFILE", 256, 256), ("RLIMIT_CORE", 0, 0),
    ]
    backed = RlimitPlan.from_limits(Limits(cpu_s=1, mem_mb=128, pids=16, disk_mb=10, out_kb=8), address_space=True)
    assert backed.entries()[-1] == ("RLIMIT_AS", (128 * 2 + 256) << 20, (128 * 2 + 256) << 20)
    assert "RLIMIT_NPROC" not in {name for name, _, _ in backed.entries()}
    assert RlimitPlan.from_dict(json.loads(json.dumps(plan.as_dict()))) == plan
    for bad in ({"cpu_s": 0, "fsize_bytes": 1}, {"cpu_s": 1, "fsize_bytes": 1, "nofile": 3}):
        with pytest.raises(ValueError):
            RlimitPlan(**bad)


def test_limits_only_ever_go_down():
    calls = []
    current = {resource.RLIMIT_CPU: (resource.RLIM_INFINITY, resource.RLIM_INFINITY), resource.RLIMIT_FSIZE: (1000, 1000),
               resource.RLIMIT_NOFILE: (1024, 4096), resource.RLIMIT_CORE: (0, 0)}
    apply(RlimitPlan(cpu_s=5, fsize_bytes=1 << 20, nofile=256), setrlimit=lambda which, value: calls.append((which, value)), getrlimit=current.__getitem__)
    assert dict(calls) == {resource.RLIMIT_CPU: (5, 6), resource.RLIMIT_FSIZE: (1000, 1000), resource.RLIMIT_NOFILE: (256, 256), resource.RLIMIT_CORE: (0, 0)}


# Classification


def observed(**changes):
    base = dict(started=True, exit_code=0, signal=None, timed_out=False, stdout=b"", stderr=b"", stdout_bytes=0, stderr_bytes=0,
                stdout_truncated=False, stderr_truncated=False, elapsed_s=0.1, cleanup_status="clean", resource_usage={})
    return ProcessOutcome(**(base | changes))


@pytest.mark.parametrize(
    ("changes", "limit"),
    [
        ({"signal": 9, "exit_code": None, "resource_usage": {"oom_kills": 1, "oom_killed": True}}, "memory"),
        ({"exit_code": 1, "resource_usage": {"pids_limit_hits": 40}}, "pids"),
        ({"timed_out": True, "exit_code": None, "resource_usage": {"pids_limit_hits": 3}}, "pids"),
        ({"exit_code": 0, "resource_usage": {"pids_limit_hits": 3}}, None),  # a refused fork the program handled
        ({"stopped": "disk", "exit_code": None, "signal": 9}, "disk"),
        ({"signal": signal.SIGXFSZ, "exit_code": None}, "disk"),
        ({"exit_code": 1, "resource_usage": {"disk_peak_kb": 200 * 1024}}, "disk"),
        ({"exit_code": 0, "resource_usage": {"disk_peak_kb": 200 * 1024}}, None),
        ({"stopped": "cpu", "exit_code": None, "signal": 9}, "cpu"),
        ({"signal": signal.SIGXCPU, "exit_code": None}, "cpu"),
        ({"timed_out": True, "exit_code": None}, None),  # a plain timeout stays a timeout
        ({"exit_code": 3}, None),  # an ordinary failure stays an ordinary result
    ],
)
def test_exhaustion_is_classified_from_evidence_not_guessed(changes, limit):
    request = request_for(Program.from_text("python", "pass"), spec=spec_with(disk_mb=200))
    assert exhaustion(request, observed(**changes)) == limit


def test_an_exhausted_run_is_a_typed_non_retryable_resource_error():
    from bulkhead.backends.base import BackendIdentity
    from bulkhead.types import SandboxTier

    request = request_for(Program.from_text("python", "pass"))
    outcome = build_outcome(request, BackendIdentity("x", "1", SandboxTier.T1), observed(signal=9, exit_code=None, resource_usage={"oom_kills": 2, "oom_killed": True}))
    assert outcome.process.exhausted == "memory"
    assert {"kind": "resource", "rule_id": "bulkhead.limit.memory", "detail": "the memory ceiling stopped the program"} in outcome.audit["violations"]
    result, streams = process_tool_result(ToolCall("c", "run_python", "{}"), "a" * 64, outcome.process, EXEC_POLICY)
    assert streams is None  # nothing from an exhausted run is shown
    assert result.error.error_type.value == "resource_exhausted" and result.error.retryable is False and result.handler_called
    assert EXHAUSTION_LIMITS == ("memory", "pids", "disk", "cpu")
    with pytest.raises(ValueError):
        ExecResult(started=False, setup_error="bulkhead.setup.failed", exhausted="memory")


def test_the_audit_accepts_the_new_resource_evidence():
    from bulkhead import audit

    audit.resource_usage({"cpu_s": 1.5, "cpu_throttled_s": 0.3, "oom_kills": 1, "oom_killed": True, "pids_limit_hits": 4, "disk_peak_kb": 9000, "pids_peak": 32}, "r")
    with pytest.raises(audit.AuditError):
        audit.resource_usage({"gpu_s": 1}, "r")


# The helper and the start barrier (no sandbox: fixed harmless programs)


class FakeGate:
    """Records what the supervisor asks of a governor; can refuse admission or stop the run."""

    def __init__(self, *, refuse=None, stop=None, stop_after=0):
        self.refuse, self.stop, self.stop_after = refuse, stop, stop_after
        self.cmdline_at_admit = None
        self.polls = 0

    needs_barrier = True
    tree_kill = False

    def admit(self, pid):
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            self.cmdline_at_admit = handle.read().split(b"\0")
        if self.refuse:
            raise CgroupError("refused", code=self.refuse)

    def poll(self):
        self.polls += 1
        return self.stop if self.stop and self.polls > self.stop_after else None

    def kill(self):
        pass

    def populated(self):
        return False

    def usage(self):
        return {"pids_limit_hits": 0}

    def close(self):
        return "clean"


def through_helper(text, *, gate=None, plan=None, kind="python", timeout_s=20):
    program = Program.from_text(kind, text)
    plan = plan or RlimitPlan(cpu_s=5, fsize_bytes=1 << 20, nofile=64)
    passed = []

    def argv_for(status_fd, barrier_fd=None):
        config = {
            "status_fd": status_fd, "barrier_fd": barrier_fd, "seccomp_fd": None, "keep_fds": [], "landlock": None,
            "rlimits": plan.as_dict(), "target": launcher_command(sys.executable, status_fd, program, shell="/bin/sh"),
            "supervisor_pid": os.getpid(),
        }
        fd = config_memfd(config)
        passed.append(fd)
        return helper_command(fd)

    try:
        return supervise(argv_for, stdin=program.source, env={"PATH": "/usr/bin:/bin"}, timeout_s=timeout_s,
                         limits=CaptureLimits(4096, 4096, 8192), pass_fds=passed, gate=gate)
    finally:
        for fd in passed:
            os.close(fd)


@pytest.mark.skipif(not LINUX, reason="unsupported: Linux only")
def test_the_helper_waits_at_the_barrier_until_admitted():
    gate = FakeGate()
    outcome = through_helper("print('ran')", gate=gate)

    assert outcome.started and outcome.stdout == b"ran\n"
    # At admission the process was still the helper: nothing untrusted had been exec'd yet.
    assert b"bulkhead.helper" in gate.cmdline_at_admit and b"-m" in gate.cmdline_at_admit


@pytest.mark.skipif(not LINUX, reason="unsupported: Linux only")
def test_a_refused_admission_means_the_program_never_runs(tmp_path):
    marker = tmp_path / "ran"
    outcome = through_helper(f"open({str(marker)!r}, 'w').write('x')", gate=FakeGate(refuse="bulkhead.cgroup.admit"))

    assert not outcome.started and outcome.setup_error == "bulkhead.cgroup.admit"
    assert not marker.exists()


@pytest.mark.skipif(not LINUX, reason="unsupported: Linux only")
def test_a_crossed_ceiling_stops_the_run():
    outcome = through_helper("import time; time.sleep(30)", gate=FakeGate(stop="disk", stop_after=3))
    assert outcome.started and outcome.stopped == "disk" and not outcome.timed_out and outcome.cleanup_status == "killed"
    assert outcome.elapsed_s < 10


@pytest.mark.skipif(not LINUX, reason="unsupported: Linux only")
def test_the_helper_applies_rlimits_before_exec():
    text = "import json, resource\nprint(json.dumps([resource.getrlimit(getattr(resource, n)) for n in ('RLIMIT_CPU', 'RLIMIT_FSIZE', 'RLIMIT_NOFILE', 'RLIMIT_CORE')]))"
    outcome = through_helper(text, plan=RlimitPlan(cpu_s=7, fsize_bytes=3 << 20, nofile=48))
    assert json.loads(outcome.stdout) == [[7, 8], [3 << 20, 3 << 20], [48, 48], [0, 0]]


def test_resource_bombs_are_declared_only_for_cgroup_profiles():
    bombs = {"resource.fork_bomb", "resource.memory_bomb", "resource.cpu_spin", "resource.disk_single", "resource.disk_many"}
    for attack in RESOURCE:
        if attack.attack_id in bombs:
            assert attack.destructive and "cgroup" in attack.required_capabilities, attack.attack_id
            assert attack.expectation("landlock-t1")["verdict"] == attack.expectation("bwrap-namespace-phase")["verdict"] == "not_applicable"
    assert bombs <= {attack.attack_id for attack in RESOURCE}


# Integration: destructive fixtures, only inside a confirmed delegated cgroup parent


@pytest.fixture(scope="module")
def governed(tmp_path_factory):
    if not (LINUX and INTEGRATION and os.environ.get("BULKHEAD_CGROUP_PARENT")):
        pytest.skip("unsupported: set BULKHEAD_INTEGRATION=1 and BULKHEAD_CGROUP_PARENT (a delegated parent) to run resource attacks")
    from bulkhead.backends.bwrap import BackendUnavailable, BwrapBackend
    from bulkhead.backends.landlock import LandlockBackend
    from bulkhead.cgroups import configured_parent
    from bulkhead.probe import run_probe

    parent = configured_parent()
    report = run_probe(cgroup_parent=os.environ["BULKHEAD_CGROUP_PARENT"])
    work = tmp_path_factory.mktemp("bulkhead-governed")
    for build in (lambda: BwrapBackend(report, work / "t2", cgroup_parent=parent), lambda: LandlockBackend(report, work / "t1", cgroup_parent=parent)):
        try:
            return build(), parent
        except BackendUnavailable:
            continue
    pytest.skip("unsupported: neither bwrap-t2 nor landlock-t1-cgroup is proven here")


@pytest.fixture
def watchdog():
    def expired(signum, frame):
        raise TimeoutError("the outer watchdog expired")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("attack", RESOURCE, ids=[attack.attack_id for attack in RESOURCE])
def test_integration_resource_attacks_are_contained(governed, watchdog, tmp_path, attack):
    backend, parent = governed
    assert backend.identity.name in ("bwrap-t2", "landlock-t1-cgroup")  # host safety: a cgroup governs this run
    evidence = run_script(DATASET, attack, backend, tmp_path, profile=backend.identity.name)
    verdict = judge(attack, attack.expectation(backend.identity.name), evidence)

    assert parent.stale() == [], "a leaf was left behind"
    assert (evidence["host"]["disk_lost_mib"] or 0) < 64, "the host lost disk space"
    assert evidence["host"]["memory_kept"], "the host lost most of its free memory"
    assert verdict.verdict in ("blocked", "different_boundary"), json.dumps({"reasons": verdict.reasons, "audit": evidence.get("audit")}, indent=1)


def test_integration_every_process_is_in_the_leaf_before_untrusted_code_runs(governed, watchdog):
    backend, parent = governed
    seen = {}

    def watch():
        give_up = time.monotonic() + 10
        while time.monotonic() < give_up and not seen:
            for name in parent.stale():
                try:
                    pids = [int(line) for line in (parent.path / name / "cgroup.procs").read_text().split()]
                except OSError:
                    continue
                if len(pids) >= 3:
                    sessions = set()
                    for pid in pids:
                        try:
                            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                            sessions.add(int(fields[3]))
                        except OSError:
                            pass
                    outside = []
                    for entry in os.listdir("/proc"):
                        if entry.isdigit() and int(entry) not in pids:
                            try:
                                fields = Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()
                            except OSError:
                                continue
                            if int(fields[3]) in sessions:
                                outside.append(int(entry))
                    seen.update(pids=pids, outside=outside)
            time.sleep(0.05)

    watcher = threading.Thread(target=watch)
    watcher.start()
    program = Program.from_text("python", "import os, time\nif os.fork() == 0:\n    time.sleep(2); os._exit(0)\ntime.sleep(2)\n")
    outcome = backend.execute(request_for(program, episode_id="ep_membership", spec=spec_with(wall_s=15)), {})
    watcher.join()
    backend.cleanup("ep_membership")
    assert outcome.audit["started"] and seen, "the watcher never saw the call's leaf"
    assert seen["outside"] == [], f"processes of the call ran outside its leaf: {seen}"


def test_integration_a_killed_supervisor_leaves_survivors_only_until_the_next_start(governed, tmp_path, watchdog):
    backend, parent = governed
    from bulkhead.probe import write_report, run_probe

    report_path = tmp_path / "probe.json"
    write_report(run_probe(cgroup_parent=os.environ["BULKHEAD_CGROUP_PARENT"]), report_path)
    marker = sleeper()
    script = tmp_path / "supervisor.py"
    script.write_text(textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from bulkhead.probe import load_report
        from bulkhead.cgroups import configured_parent
        from bulkhead.backends.base import Program
        from exec_support import request_for
        from bulkhead.backends.landlock import LandlockBackend
        from bulkhead.backends.bwrap import BwrapBackend
        report = load_report({str(report_path)!r})
        cls = {type(backend).__name__}
        backend = cls(report, {str(tmp_path / "work")!r}, cgroup_parent=configured_parent())
        text = "import os, time\\nif os.fork() == 0:\\n    os.execv('/bin/sleep', ['/bin/sleep', '{marker}'])\\ntime.sleep(60)\\n"
        backend.execute(request_for(Program.from_text("python", text), episode_id="ep_orphan"), {{}})
    """))
    supervisor = subprocess.Popen([sys.executable, str(script)], env=dict(os.environ))
    try:
        give_up = time.monotonic() + 30
        while not alive(marker) and time.monotonic() < give_up:
            time.sleep(0.1)
        assert alive(marker), "the grandchild never started"
        supervisor.send_signal(signal.SIGKILL)
        supervisor.wait(timeout=10)
        time.sleep(0.5)
        survivors_before = alive(marker)
        type(backend)(load_report_path(report_path), tmp_path / "work2", cgroup_parent=parent)  # a new start reaps stale leaves
        time.sleep(0.5)
        assert alive(marker) == [], f"survivors after the next start: {alive(marker)} (before: {survivors_before})"
        assert parent.stale() == []
    finally:
        kill_marked(marker)


def load_report_path(path):
    from bulkhead.probe import load_report

    return load_report(path)

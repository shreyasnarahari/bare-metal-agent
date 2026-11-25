"""Bulkhead part 7: overhead measured from raw samples (injected clocks, fake backends, one real small run)."""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest

from bulkhead.backends.base import BackendIdentity
from bulkhead.eval.attacks import Target
from bulkhead.eval.overhead import (
    WORKLOAD_EXPECTED,
    ProcessBaseline,
    Sample,
    SandboxRunner,
    TimingBackend,
    baseline_registry,
    distribution,
    measure,
    measure_episode,
    run_overhead,
    summarize_samples,
)
from bulkhead.types import SandboxTier
from toolsmith.evaluation.metrics import percentile

LINUX = sys.platform.startswith("linux")


class Ticks:
    """An injected clock: each call advances by the next step (seconds)."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.now = 100.0

    def __call__(self):
        value = self.now
        self.now += self.steps.pop(0) if self.steps else 0.001
        return value


class FakeRunner:
    profile = "fake"

    def __init__(self, totals, failing=()):
        self.totals = list(totals)
        self.failing = set(failing)
        self.calls = 0

    def call(self, workload, index, phase):
        total = self.totals[self.calls]
        self.calls += 1
        ok = not (phase == "measured" and index in self.failing)
        return Sample(self.profile, workload, index, phase, ok, total, execute_s=total, error=None if ok else "exit 1")


def test_measure_keeps_the_first_call_apart_and_every_measured_sample():
    totals = [9.0] + [5.0] * 10 + [(index + 1) / 1000 for index in range(1000)]  # first, warmup, then 1..1000 ms
    rows = measure(FakeRunner(totals), "true", samples=1000, warmup=10)
    assert [row.phase for row in rows].count("measured") == 1000 and rows[0].phase == "first" and len(rows) == 1011

    summary = summarize_samples(rows)
    assert summary["first_total_s"] == 9.0 and summary["measured"] == 1000 and summary["warmup"] == 10
    assert summary["total_s"]["median"] == 0.5 and summary["total_s"]["p95"] == 0.95 and summary["total_s"]["p99"] == 0.99
    assert summary["total_s"]["max"] == 1.0
    measured = [row.total_s for row in rows if row.phase == "measured"]
    assert summary["total_s"]["p99"] == percentile(measured, 0.99)  # the shared helper, over all 1,000 values


def test_failures_are_counted_and_excluded_from_the_timing_percentiles():
    totals = [0.1] + [0.01] * 2 + [0.002] * 98 + [5.0, 5.0]
    rows = measure(FakeRunner(totals, failing={98, 99}), "true", samples=100, warmup=2)
    summary = summarize_samples(rows)
    assert summary["failures"] == 2 and summary["total_s"]["max"] == 0.002 and summary["errors"] == ["exit 1"]
    with pytest.raises(ValueError):
        measure(FakeRunner([1.0]), "true", samples=0)
    assert distribution([]) == {"median": None, "p95": None, "p99": None, "max": None, "mean": None}


class FakeOutcome:
    def __init__(self, stdout, *, started=True, exit_code=0, elapsed_s=0.004, egress=()):
        self.audit = {"started": started, "exit_code": exit_code, "signal": None, "elapsed_s": elapsed_s, "resource_usage": {"peak_memory_kb": 4096}}
        self.process = type("P", (), {"stdout": stdout, "stderr": "", "setup_error": None})()
        self.egress = egress


class FakeBackend:
    def __init__(self, stdout=WORKLOAD_EXPECTED["true"], **outcome):
        self.identity = BackendIdentity("fake-test", "1", SandboxTier.T1)
        self.stdout, self.outcome = stdout, outcome
        self.events = []

    def prepare(self, episode_id, files):
        self.events.append(("prepare", episode_id))

    def execute(self, request, arguments):
        self.events.append(("execute", request.episode_id))
        return FakeOutcome(self.stdout, **self.outcome)

    def cleanup(self, episode_id):
        self.events.append(("cleanup", episode_id))


def test_a_sandbox_call_creates_and_removes_a_fresh_workspace_and_times_each_part():
    backend = FakeBackend()
    runner = SandboxRunner(backend, "landlock-t1", clock=Ticks(0.002, 0.030, 0.001))
    sample = runner.call("true", 7, "measured")
    assert (sample.prepare_s, sample.execute_s, sample.cleanup_s) == pytest.approx((0.002, 0.030, 0.001))
    assert sample.total_s == pytest.approx(0.033) and sample.ok and sample.process_s == 0.004 and sample.usage == {"peak_memory_kb": 4096}
    episode = backend.events[0][1]
    assert backend.events == [("prepare", episode), ("execute", episode), ("cleanup", episode)]
    runner.call("true", 8, "measured")
    assert backend.events[3][1] != episode  # every measured call gets its own workspace


@pytest.mark.parametrize("backend", [FakeBackend(stdout="wrong\n"), FakeBackend(started=False, exit_code=None), FakeBackend(exit_code=3)])
def test_a_wrong_output_or_a_failed_start_is_a_failure(backend):
    sample = SandboxRunner(backend, "x").call("true", 0, "measured")
    assert not sample.ok and sample.error


def test_the_timing_backend_separates_supervised_time_from_backend_overhead():
    from bulkhead.backends.base import Program
    from bulkhead.eval.attacks import request_for
    from bulkhead.eval.overhead import exec_spec

    backend = FakeBackend(elapsed_s=0.010, egress=({"elapsed_s": 0.003},))
    timing = TimingBackend(backend, clock=Ticks(0.001, 0.0, 0.015, 0.0001))  # prepare, gap, execute, gate
    request = request_for(Program.from_text("shell", "true"), exec_spec(), episode_id="ep_x")
    timing.execute(request, {})
    [call] = timing.calls
    assert call["prepare_s"] == pytest.approx(0.001) and call["wall_s"] == pytest.approx(0.015)
    assert call["supervised_s"] == 0.010 and call["backend_overhead_s"] == pytest.approx(0.005) and call["proxy_s"] == 0.003
    assert timing.cleanup_all() >= 0 and ("cleanup", "ep_x") in backend.events


def test_the_trusted_baseline_uses_its_own_tool_names():
    registry = baseline_registry()
    assert "baseline_python" in registry and "baseline_shell" in registry and "run_python" not in registry


def test_store_episodes_compare_trusted_handling_and_the_policy_gate(tmp_path):
    trusted = measure_episode("store", "trusted", repeats=2, directory=tmp_path)
    policy = measure_episode("store", "policy", repeats=2, directory=tmp_path)
    for summary in (trusted, policy):
        assert summary["statuses"] == ["done"] and summary["calls"] == 4 and len(summary["wall_samples_s"]) == 2
    with pytest.raises(ValueError, match="needs a sandbox backend"):
        measure_episode("exec", "policy", repeats=1, directory=tmp_path)  # no policy-only exec row


def test_the_exec_episode_measures_sandboxed_parts_with_a_fake_backend(tmp_path):
    class EchoBackend(FakeBackend):
        def execute(self, request, arguments):
            from bulkhead.backends.base import build_outcome
            from bulkhead.supervisor import ProcessOutcome

            self.events.append(("execute", request.episode_id))
            observed = ProcessOutcome(started=True, exit_code=0, signal=None, timed_out=False, stdout=b"ok\n", stderr=b"", stdout_bytes=3, stderr_bytes=0,
                                      stdout_truncated=False, stderr_truncated=False, elapsed_s=0.002, cleanup_status="clean")
            return build_outcome(request, self.identity, observed)

    summary = measure_episode("exec", "fake", target=Target("landlock-t1", EchoBackend()), repeats=2, directory=tmp_path)
    assert summary["statuses"] == ["done"] and summary["sandboxed_calls"] == 2
    assert summary["sandbox_supervised_s"]["median"] == pytest.approx(0.004) and summary["sandbox_wall_s"]["median"] > 0


@pytest.mark.skipif(not LINUX, reason="unsupported: needs Linux")
def test_a_real_run_writes_every_raw_sample_and_its_statistics_come_from_all_of_them(tmp_path):
    directory = run_overhead([Target("T0")], samples=1000, warmup=5, workloads=("true",), repeats=1, proxy_samples=3, results_dir=tmp_path)
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    measured = [row["total_s"] for row in rows if row["profile"] == "T0" and row["phase"] == "measured" and row["ok"]]
    summary = json.loads((directory / "summary.json").read_text())
    stats = summary["cold_start"]["T0"]["true"]
    assert len(measured) == stats["measured"] - stats["failures"] == 1000 - stats["failures"] and stats["measured"] == 1000
    assert stats["total_s"]["median"] == percentile(measured, 0.5) and stats["total_s"]["p99"] == percentile(measured, 0.99)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["concurrency"] == {"workers": 1, "mode": "serial"} and manifest["host"]["cpu_count"] and "host_after" in manifest
    assert {e["mode"] for e in summary["episodes"]} == {"trusted", "policy"}
    assert summary["cold_start"]["proxy"]["proxy"]["measured"] == 3


@pytest.mark.skipif(not LINUX, reason="unsupported: needs Linux")
def test_the_process_baseline_runs_both_workloads():
    baseline = ProcessBaseline()
    assert baseline.call("true", 0, "measured").ok and baseline.call("python", 0, "measured").ok
    assert not replace(baseline.call("python", 0, "measured"), ok=False).ok


def test_each_tier_2_profile_gets_its_own_egress_listeners(tmp_path, monkeypatch):
    import bulkhead.eval.attacks as attacks
    import bulkhead.eval.overhead as overhead

    bound = []

    class FakeListeners:
        def __init__(self, directory):
            socket_path = directory / "host-service.sock"
            assert socket_path not in bound  # a second profile reusing the path would fail with EADDRINUSE
            bound.append(socket_path)
            self.second_port = 1

        def stop(self):
            pass

    class FakeFixture:
        def __init__(self, port):
            self.port = 2

        def stop(self):
            pass

    monkeypatch.setattr(attacks, "Listeners", FakeListeners)
    monkeypatch.setattr(attacks, "ProxyFixture", FakeFixture)
    monkeypatch.setattr(overhead, "measure_episode", lambda *a, **k: {"episode": "egress", "profile": k["target"].profile})
    episodes = [overhead._egress_episode(Target(profile), 1, tmp_path) for profile in ("bwrap-t2", "bwrap-namespace-phase")]
    assert [e["profile"] for e in episodes] == ["bwrap-t2", "bwrap-namespace-phase"] and len(set(bound)) == 2


def test_the_command_line_writes_where_results_points(tmp_path, monkeypatch):
    import bulkhead.eval.attacks as attacks
    import bulkhead.eval.overhead as overhead
    import bulkhead.probe as probe

    seen = {}
    monkeypatch.delenv("BULKHEAD_CGROUP_PARENT", raising=False)
    monkeypatch.setattr(probe, "run_probe", lambda **options: "report")
    monkeypatch.setattr(attacks, "discover_targets", lambda *a, **k: [Target("T0")])
    monkeypatch.setattr(overhead, "run_overhead", lambda targets, **options: seen.update(options) or tmp_path / "ovh_x")
    assert overhead.main(["--results", str(tmp_path / "elsewhere"), "--samples", "5"]) == 0
    assert seen["results_dir"] == str(tmp_path / "elsewhere") and seen["samples"] == 5

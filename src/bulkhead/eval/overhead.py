"""Bulkhead part 7: sandbox overhead, measured serially from raw samples.

Per-call cold start (``measure``): a fixed benign workload (``/bin/true``
through the shell, or a small pinned Python program) is run ``samples`` times
after ``warmup`` calls, each with a fresh workspace (and, on cgroup profiles, a
fresh cgroup leaf). The very first invocation is recorded separately: it
includes binary and page-cache cold start, which per-call sandbox creation
does not. The T0 row is the same program run as a plain process (NOT A
SANDBOX) for comparison. ``proxy`` times starting and stopping one call's
egress proxy, the host-side cost a network-enabled call adds.

Whole episodes (``measure_episode``): a scripted model plays a fixed sequence
of calls. The ``store`` episode (reads and one write) compares ``trusted``
in-process handling with the ``policy`` gate (policy, taint, journal). The
``exec`` episode (a read and two exec calls) compares ``trusted`` handling,
where the exec calls run as plain subprocesses (NOT A SANDBOX), with each
sandbox profile. There is no policy-only exec row: the policy never lets an
exec tool run without a sandbox tier. The ``egress`` episode needs a tier-2
backend that can bind the proxy socket.

Statistics come from every measured sample (``toolsmith`` nearest-rank
``percentile``); failures are counted and excluded from the timing
percentiles. Every run records the host (CPU count, kernel, load before and
after), backend and helper versions, the probe hash, the limits, and the
concurrency (always serial). Nothing here states a target as a result.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import secrets as _secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..backends.base import Program
from ..manifest import Egress, Sandbox, SecuritySpec, Sink, Trust
from ..output import render, sanitize
from ..types import canonical_hash
from .attacks import ROOT, ScriptedService, host_info, helper_version, request_for

RESULTS_DIR = ROOT / "results" / "bulkhead" / "overhead"
WORKLOADS: Mapping[str, tuple[str, str]] = {
    "true": ("shell", "/bin/true\n"),
    "python": ("python", "print(sum(i * i for i in range(1000)))\n"),
}
WORKLOAD_EXPECTED = {"true": "", "python": "332833500\n"}
WORKSPACE_FILES = {"README.txt": b"overhead benchmark\n"}
TARGET_S = 0.050  # the brief's example aspiration for T2; reported against measurements, never as one


@dataclass(frozen=True)
class Sample:
    """One call. ``total_s`` = prepare + execute + cleanup; ``process_s`` is the supervisor's own timing of the program."""

    profile: str
    workload: str
    index: int
    phase: str  # first | warmup | measured
    ok: bool
    total_s: float
    prepare_s: float = 0.0
    execute_s: float = 0.0
    cleanup_s: float = 0.0
    process_s: float | None = None
    error: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)


def exec_spec(limits: Any = None) -> SecuritySpec:
    from steadyhand.tools_exec import EXEC_SANDBOX

    sandbox = Sandbox(fs_read=EXEC_SANDBOX.fs_read, fs_write=EXEC_SANDBOX.fs_write, fs_deny=EXEC_SANDBOX.fs_deny, net=Egress(), limits=limits or EXEC_SANDBOX.limits)
    return SecuritySpec(sandbox=sandbox, output_trust=Trust.UNTRUSTED, sinks=(Sink.FILESYSTEM,))


class ProcessBaseline:
    """NOT A SANDBOX: the workload as a plain process with the sandbox's minimal environment (the T0 row)."""

    profile = "T0"

    def __init__(self, *, python: str = "/usr/bin/python3", shell: str = "/bin/sh", clock: Callable[[], float] = time.perf_counter) -> None:
        from ..backends.bwrap import SANDBOX_ENV

        self.python, self.shell, self.clock, self.env = python, shell, clock, dict(SANDBOX_ENV)

    def call(self, workload: str, index: int, phase: str) -> Sample:
        kind, source = WORKLOADS[workload]
        argv = [self.shell, "-s"] if kind == "shell" else [self.python, "-I", "-"]
        started = self.clock()
        done = subprocess.run(argv, input=source.encode(), capture_output=True, env=self.env, timeout=60)
        elapsed = self.clock() - started
        ok = done.returncode == 0 and done.stdout.decode() == WORKLOAD_EXPECTED[workload]
        return Sample(self.profile, workload, index, phase, ok, elapsed, execute_s=elapsed, error=None if ok else f"exit {done.returncode}")


class SandboxRunner:
    """A sandbox backend: every call prepares a fresh workspace, executes, and removes the workspace."""

    def __init__(self, backend: Any, profile: str, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self.backend, self.profile, self.clock = backend, profile, clock
        self.spec = exec_spec()
        self.nonce = _secrets.token_hex(4)

    def call(self, workload: str, index: int, phase: str) -> Sample:
        kind, source = WORKLOADS[workload]
        episode = f"ep_ovh_{self.nonce}_{phase[0]}{index:06d}"
        program = Program.from_text(kind, source)
        t0 = self.clock()
        self.backend.prepare(episode, WORKSPACE_FILES)
        t1 = self.clock()
        try:
            outcome = self.backend.execute(request_for(program, self.spec, episode_id=episode), {})
        finally:
            t2 = self.clock()
            self.backend.cleanup(episode)
            t3 = self.clock()
        audit = outcome.audit
        ok = bool(audit["started"]) and audit["exit_code"] == 0 and outcome.process is not None and outcome.process.stdout == WORKLOAD_EXPECTED[workload]
        error = None if ok else f"started={audit['started']} exit={audit['exit_code']} signal={audit['signal']} setup={getattr(outcome.process, 'setup_error', None)}"
        return Sample(self.profile, workload, index, phase, ok, t3 - t0, t1 - t0, t2 - t1, t3 - t2, audit.get("elapsed_s"), error, dict(audit.get("resource_usage") or {}))


class ProxyStartup:
    """Starting and stopping one call's egress proxy (no sandbox, no traffic): the host-side network setup cost."""

    profile = "proxy"

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self.egress = Egress(allow=("127.0.0.1:9",), address_classes=("loopback",))

    def call(self, workload: str, index: int, phase: str) -> Sample:
        from ..proxy import call_proxy

        t0 = self.clock()
        with call_proxy(self.egress):
            t1 = self.clock()
        t2 = self.clock()
        return Sample(self.profile, workload, index, phase, True, t2 - t0, execute_s=t1 - t0, cleanup_s=t2 - t1)


def measure(runner: Any, workload: str, *, samples: int = 1000, warmup: int = 10, progress: Callable[[int], None] | None = None) -> list[Sample]:
    """The first invocation, ``warmup`` discarded calls, then ``samples`` measured calls, serially."""
    if samples < 1 or warmup < 0:
        raise ValueError("samples must be positive and warmup non-negative")
    calls = [runner.call(workload, 0, "first")]
    calls += [runner.call(workload, index, "warmup") for index in range(warmup)]
    for index in range(samples):
        calls.append(runner.call(workload, index, "measured"))
        if progress is not None and (index + 1) % 100 == 0:
            progress(index + 1)
    return calls


def distribution(values: Sequence[float]) -> dict[str, float | None]:
    from toolsmith.evaluation.metrics import percentile

    if not values:
        return {"median": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "median": percentile(values, 0.5), "p95": percentile(values, 0.95), "p99": percentile(values, 0.99),
        "max": max(values), "mean": sum(values) / len(values),
    }


def summarize_samples(samples: Sequence[Sample]) -> dict[str, Any]:
    """Statistics over every measured sample (failures counted, excluded from the timing percentiles)."""
    measured = [s for s in samples if s.phase == "measured"]
    good = [s for s in measured if s.ok]
    first = next((s for s in samples if s.phase == "first"), None)
    summary: dict[str, Any] = {
        "measured": len(measured), "failures": len(measured) - len(good), "warmup": sum(1 for s in samples if s.phase == "warmup"),
        "first_total_s": first.total_s if first else None, "first_ok": first.ok if first else None,
        "total_s": distribution([s.total_s for s in good]), "execute_s": distribution([s.execute_s for s in good]),
        "prepare_s": distribution([s.prepare_s for s in good]), "cleanup_s": distribution([s.cleanup_s for s in good]),
        "process_s": distribution([s.process_s for s in good if s.process_s is not None]),
        "errors": sorted({s.error for s in measured if s.error})[:5],
    }
    for key in ("peak_memory_kb", "pids_peak", "cpu_s"):  # per-call evidence from the cgroup, where there is one
        values = [s.usage[key] for s in good if s.usage.get(key) is not None]
        if values:
            summary[f"usage_{key}"] = distribution(values)
    return summary


def cold_start_from_samples(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Per (profile, workload) statistics rebuilt from a run's raw ``samples.jsonl``."""
    groups: dict[tuple[str, str], list[Sample]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            sample = Sample(**json.loads(line))
            groups.setdefault((sample.profile, sample.workload), []).append(sample)
    summary: dict[str, dict[str, Any]] = {}
    for (profile, workload), samples in groups.items():
        summary.setdefault(profile, {})[workload] = summarize_samples(samples)
    return summary


def resummarize(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Rewrite a run's cold-start summary from its raw samples (the samples are the record)."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    summary["cold_start"] = cold_start_from_samples(directory / "samples.jsonl")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


# Whole episodes


STORE_TURNS = [
    {"calls": [["get_order", {"order_id": "ord_1001"}], ["search_orders", {"customer_id": "cus_101"}]]},
    {"calls": [["get_customer", {"customer_id": "cus_101"}]]},
    {"calls": [["add_order_note", {"order_id": "ord_1001", "note": "Checked the order and the customer."}]]},
    {"text": "Done: the order and customer are checked and noted."},
]
EXEC_TURNS = [
    {"calls": [["get_order", {"order_id": "ord_1001"}]]},
    {"calls": [["run_python", {"code": WORKLOADS["python"][1]}]]},
    {"calls": [["run_shell", {"command": "cat README.txt"}]]},
    {"text": "Done: computed the sum and read the workspace file."},
]
EPISODES = {"store": STORE_TURNS, "exec": EXEC_TURNS}
# Per-episode sums over its sandboxed calls (see TimingBackend), plus removing the workspaces afterwards.
PART_KEYS = ("wall_s", "supervised_s", "backend_overhead_s", "gate_s", "proxy_s", "prepare_s", "workspace_cleanup_s")


def egress_turns(port: int) -> list[dict[str, Any]]:
    program = (
        "import http.client\n"
        "c = http.client.HTTPConnection('127.0.0.1', 3128, timeout=5)\n"
        f"c.set_tunnel('127.0.0.1', {port})\n"
        "c.request('GET', '/hello')\n"
        "print(c.getresponse().read().decode().strip())\n"
    )
    return [{"calls": [["run_python", {"code": program}]]}, {"text": "Done: fetched the allowed page."}]


BASELINE_NAMES = {"run_python": "baseline_python", "run_shell": "baseline_shell"}


def baseline_turns(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The same turns with exec calls renamed to the baseline tools (the registry reserves
    ``run_python``/``run_shell`` for sandboxed execution and refuses them without a sandbox)."""
    return [
        {**turn, "calls": [[BASELINE_NAMES.get(name, name), arguments] for name, arguments in turn["calls"]]} if "calls" in turn else dict(turn)
        for turn in turns
    ]


def baseline_registry() -> Any:
    """NOT A SANDBOX: the store tools plus ``baseline_python``/``baseline_shell``, whose handlers run
    the program as a plain subprocess, for the ``trusted`` episode row only. Programs are the fixed
    benchmark text."""
    from bulkhead.backends.bwrap import SANDBOX_ENV
    from steadyhand import Effect, SteadyhandRegistry, ToolPolicy
    from steadyhand.tools import RESULT_SCHEMAS, TOOL_SECURITY, TOOL_SPECS

    def gated(done: subprocess.CompletedProcess[bytes]) -> dict[str, Any]:
        value = {"exit_code": done.returncode, "signal": None}
        for name, raw in (("stdout", done.stdout), ("stderr", done.stderr)):
            value[name] = render(sanitize(raw), request_sha256="0" * 64, stream=name)[0]
        return value

    def baseline_python(store: Any, code: str) -> dict[str, Any]:
        """Run a short Python program (benchmark baseline: a plain subprocess, not a sandbox)."""
        return gated(subprocess.run(["/usr/bin/python3", "-I", "-"], input=code.encode(), capture_output=True, env=dict(SANDBOX_ENV), cwd=workdir, timeout=30))

    def baseline_shell(store: Any, command: str) -> dict[str, Any]:
        """Run a shell command (benchmark baseline: a plain subprocess, not a sandbox)."""
        return gated(subprocess.run(["/bin/sh", "-c", command], capture_output=True, env=dict(SANDBOX_ENV), cwd=workdir, timeout=30))

    workdir = tempfile.mkdtemp(prefix="bulkhead-baseline-")
    Path(workdir, "README.txt").write_bytes(WORKSPACE_FILES["README.txt"])
    registry = SteadyhandRegistry()
    for handler, policy, semantic in TOOL_SPECS:
        registry.register(handler, policy, semantic, RESULT_SCHEMAS[handler.__name__], security=TOOL_SECURITY[handler.__name__])
    for handler in (baseline_python, baseline_shell):
        registry.register(handler, ToolPolicy(Effect.READ, 30.0, idempotent=True, retry=None))
    return registry.freeze()


class TimingBackend:
    """Wraps a backend to time each execute call; prepares each episode's workspace on first use.

    Per call: ``wall_s`` is the whole ``execute``; ``supervised_s`` is the supervisor's own timing
    (helper setup, launcher, program, in-call cleanup); ``backend_overhead_s`` is the rest (spec
    checks, view planning, cgroup leaf, outcome building); ``gate_s`` re-times the output gate on the
    same bytes; ``proxy_s`` sums the call's egress records."""

    def __init__(self, backend: Any, clock: Callable[[], float] = time.perf_counter) -> None:
        self.backend, self.clock = backend, clock
        self.identity = backend.identity
        self.calls: list[dict[str, Any]] = []
        self.episodes: set[str] = set()

    def execute(self, request: Any, arguments: Mapping[str, Any]) -> Any:
        prepare = 0.0
        if request.episode_id not in self.episodes and hasattr(self.backend, "prepare"):
            p0 = self.clock()
            self.backend.prepare(request.episode_id, WORKSPACE_FILES)
            prepare = self.clock() - p0
            self.episodes.add(request.episode_id)
        started = self.clock()
        outcome = self.backend.execute(request, arguments)
        wall = self.clock() - started
        gate = 0.0
        if outcome.process is not None:  # re-run the output gate on the same bytes to time it
            g0 = self.clock()
            for name, text in (("stdout", outcome.process.stdout), ("stderr", outcome.process.stderr)):
                render(sanitize(text.encode()), request_sha256=request.sha256, stream=name)
            gate = self.clock() - g0
        supervised = outcome.audit.get("elapsed_s") or 0.0
        self.calls.append({
            "wall_s": wall, "supervised_s": supervised, "backend_overhead_s": max(0.0, wall - supervised), "gate_s": gate,
            "proxy_s": sum(record.get("elapsed_s") or 0.0 for record in outcome.egress), "prepare_s": prepare,
        })
        return outcome

    def cleanup_all(self) -> float:
        started = self.clock()
        for episode in self.episodes:
            self.backend.cleanup(episode)
        self.episodes.clear()
        return self.clock() - started


def run_scripted_episode(turns: Sequence[Mapping[str, Any]], directory: Path, *, registry: Any, security: Any = None, backend: Any = None) -> Any:
    from steadyhand import EpisodeJournal, SteadyhandClient, initial_messages, load_fixture, run_episode

    return run_episode(
        SteadyhandClient(ScriptedService(turns)), registry, load_fixture().clone(), initial_messages("Benchmark episode."),
        journal=EpisodeJournal.create(directory), sleep=lambda seconds: None, security=security, backend=backend,
    )


def measure_episode(episode: str, mode: str, *, target: Any = None, repeats: int = 30, directory: Path, clock: Callable[[], float] = time.perf_counter, turns: Sequence[Mapping[str, Any]] | None = None, egress: Egress | None = None) -> dict[str, Any]:
    """Run one scripted episode ``repeats`` times in ``mode`` and summarize its wall time and parts."""
    from bulkhead.policy import TaskPolicy
    from bulkhead.types import SandboxTier
    from steadyhand import build_registry
    from steadyhand.security import STORE_TOOLS, SecurityConfig
    from steadyhand.tools_exec import EXEC_LIMITS, EXEC_SANDBOX

    turns = list(turns or EPISODES[episode])
    if mode == "trusted":
        registry, security, backend = baseline_registry(), None, None
        turns = baseline_turns(turns)
    else:
        exec_calls = any(call[0] in ("run_python", "run_shell") for turn in turns for call in turn.get("calls", []))
        tools = (*STORE_TOOLS, "run_python", "run_shell") if exec_calls else STORE_TOOLS
        options: dict[str, Any] = {}
        tiers: tuple[SandboxTier, ...] = ()
        backend = None
        registry = build_registry(exec_tools=exec_calls)
        if exec_calls:
            if target is None or target.backend is None:
                raise ValueError(f"the {episode} episode needs a sandbox backend in mode {mode}")
            tier = SandboxTier(target.tier)
            options = {"fs_read": EXEC_SANDBOX.fs_read, "fs_write": EXEC_SANDBOX.fs_write, "fs_deny": EXEC_SANDBOX.fs_deny, "limits": EXEC_LIMITS, "required_tier": tier}
            tiers, backend = (tier,), TimingBackend(target.backend, clock)
            if egress is not None:
                registry = network_registry(egress)
                options["egress"] = egress
        security = SecurityConfig(TaskPolicy(tools=tools, **options), available_tiers=tiers)
    walls, statuses, calls, parts = [], [], [], []
    for index in range(repeats + 1):  # the first repeat warms caches and is not counted
        if isinstance(backend, TimingBackend):
            backend.calls.clear()
        started = clock()
        result = run_scripted_episode(turns, directory / f"{mode}-{index}", registry=registry, security=security, backend=backend)
        wall = clock() - started
        cleanup = backend.cleanup_all() if isinstance(backend, TimingBackend) else 0.0
        if index == 0:
            continue
        walls.append(wall)
        statuses.append(result.status.value)
        calls.append(sum(len(turn.get("calls", [])) for turn in turns))
        if isinstance(backend, TimingBackend):
            parts.append({f"sandbox_{key}": sum(c[key] for c in backend.calls) for key in PART_KEYS[:-1]} | {"sandbox_workspace_cleanup_s": cleanup})
    summary: dict[str, Any] = {
        "episode": episode, "mode": mode, "profile": "trusted (not a sandbox)" if mode == "trusted" else (target.profile if target is not None else mode),
        "repeats": repeats, "calls": calls[0] if calls else 0, "sandboxed_calls": sum(1 for turn in turns for call in turn.get("calls", []) if call[0] in ("run_python", "run_shell")),
        "statuses": sorted(set(statuses)), "wall_s": distribution(walls), "wall_samples_s": walls,
    }
    for key in PART_KEYS:
        if parts:
            summary[f"sandbox_{key}"] = distribution([p[f"sandbox_{key}"] for p in parts])
    return summary


def network_registry(egress: Egress) -> Any:
    from steadyhand import Effect, SteadyhandRegistry, ToolPolicy
    from steadyhand.tools import RESULT_SCHEMAS, TOOL_SECURITY, TOOL_SPECS
    from steadyhand.tools_exec import EXEC_LIMITS, EXEC_PROGRAMS, EXEC_SANDBOX, run_python, run_shell

    registry = SteadyhandRegistry()
    for handler, policy, semantic in TOOL_SPECS:
        registry.register(handler, policy, semantic, RESULT_SCHEMAS[handler.__name__], security=TOOL_SECURITY[handler.__name__])
    sandbox = Sandbox(fs_read=EXEC_SANDBOX.fs_read, fs_write=EXEC_SANDBOX.fs_write, fs_deny=EXEC_SANDBOX.fs_deny, net=egress, limits=EXEC_LIMITS)
    spec = SecuritySpec(sandbox, Trust.UNTRUSTED, (Sink.FILESYSTEM, Sink.NETWORK))
    for handler in (run_python, run_shell):
        registry.register(handler, ToolPolicy(Effect.EXEC, 30.0, idempotent=False, retry=None), security=spec, program=EXEC_PROGRAMS[handler.__name__])
    return registry.freeze()


# A full benchmark run


def run_overhead(
    targets: Sequence[Any],
    *,
    samples: int = 1000,
    warmup: int = 10,
    workloads: Sequence[str] = ("true", "python"),
    repeats: int = 30,
    proxy_samples: int | None = None,
    results_dir: str | os.PathLike[str] = RESULTS_DIR,
    probe: Any = None,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Cold-start and whole-episode overhead on every available target, serially; returns the run directory.

    Writes ``manifest.json`` (host, versions, probe, limits, concurrency),
    ``samples.jsonl`` (every raw sample), and ``summary.json``.
    """
    from steadyhand.tools_exec import EXEC_LIMITS

    from .attacks import code_version

    say = log or (lambda message: None)
    available = [t for t in targets if t.backend is not None]
    manifest = {
        "host": host_info(), "code": code_version(), "helper_version": helper_version(),
        "probe_sha256": getattr(probe, "probe_sha256", None),
        "targets": {t.profile: {"available": t.backend is not None or t.profile == "T0", "reason": t.reason, "config": t.config()} for t in targets},
        "limits": EXEC_LIMITS.as_dict(), "concurrency": {"workers": 1, "mode": "serial"},
        "samples": samples, "proxy_samples": proxy_samples or samples, "warmup": warmup, "workloads": list(workloads), "episode_repeats": repeats,
        "target_s": TARGET_S, "target_s_note": "the brief's example aspiration for tier 2, compared against measurements only",
    }
    manifest["run_id"] = "ovh_" + canonical_hash({k: manifest[k] for k in ("code", "helper_version", "probe_sha256", "targets", "samples", "warmup", "workloads", "episode_repeats")} | {"started": time.time()})[:16]
    directory = Path(results_dir) / manifest["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")  # rewritten with host_after at the end
    samples_path = directory / "samples.jsonl"
    summary: dict[str, Any] = {"cold_start": {}, "episodes": [], "untested": []}
    scratch = Path(tempfile.mkdtemp(prefix="bulkhead-overhead-"))
    try:
        runners: list[Any] = [ProcessBaseline(), *[SandboxRunner(t.backend, t.profile) for t in available]]
        with open(samples_path, "w", encoding="utf-8") as handle:
            for runner in runners:
                for workload in workloads:
                    say(f"cold start: {runner.profile} {workload} ({samples} samples)")
                    rows = measure(runner, workload, samples=samples, warmup=warmup, progress=lambda n, r=runner, w=workload: say(f"  {r.profile} {w}: {n}"))
                    for row in rows:
                        handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
            say(f"proxy startup ({proxy_samples or samples} samples)")
            rows = measure(ProxyStartup(), "proxy", samples=proxy_samples or samples, warmup=warmup)
            for row in rows:
                handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
        summary["cold_start"] = cold_start_from_samples(samples_path)  # statistics come from the written record
        for target in targets:
            if target.backend is None and target.profile != "T0":
                summary["untested"].append({"profile": target.profile, "reason": target.reason})
        say("episodes: store (trusted, policy)")
        summary["episodes"].append(measure_episode("store", "trusted", repeats=repeats, directory=scratch / "store"))
        summary["episodes"].append(measure_episode("store", "policy", repeats=repeats, directory=scratch / "store"))
        say("episodes: exec (trusted, each sandbox profile)")
        summary["episodes"].append(measure_episode("exec", "trusted", repeats=repeats, directory=scratch / "exec"))
        for target in available:
            summary["episodes"].append(measure_episode("exec", target.profile, target=target, repeats=repeats, directory=scratch / "exec"))
        network = [t for t in available if getattr(t.backend, "network", False)]
        if not network:
            summary["untested"].append({"profile": "egress episode", "reason": "no tier-2 backend here can bind the proxy socket (bwrap.unix_socket)"})
        for target in network:
            summary["episodes"].append(_egress_episode(target, repeats, scratch))
        summary["resource"] = {
            "supervisor_peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "children_peak_rss_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        }
        manifest["host_after"] = host_info()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return directory


def _egress_episode(target: Any, repeats: int, scratch: Path) -> dict[str, Any]:
    """One profile's egress episodes, with its own host listeners: each profile binds a fresh Unix socket path."""
    from . import attacks
    from .attacks import ProxyFixture

    directory = scratch / "egress" / target.profile
    directory.mkdir(parents=True, exist_ok=True)
    listeners = attacks.Listeners(directory)
    fixture = ProxyFixture(listeners.second_port)
    try:
        egress = Egress(allow=(f"127.0.0.1:{fixture.port}",), address_classes=("loopback",))
        return measure_episode("egress", target.profile, target=target, repeats=repeats, directory=directory, turns=egress_turns(fixture.port), egress=egress)
    finally:
        fixture.stop()
        listeners.stop()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bulkhead.eval.overhead", description="Measure Bulkhead sandbox overhead serially from raw samples.")
    parser.add_argument("--results", default=str(RESULTS_DIR))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--proxy-samples", type=int, help="proxy start/stop samples (default: --samples)")
    parser.add_argument("--profile", action="append", help="sandbox profiles to measure (default: every available one)")
    parser.add_argument("--cgroup-parent", help="a delegated cgroup v2 parent (default: BULKHEAD_CGROUP_PARENT)")
    args = parser.parse_args(argv)

    from ..cgroups import CgroupParent, configured_parent
    from ..probe import run_probe
    from .attacks import PROFILE_TIERS, discover_targets

    parent = CgroupParent(args.cgroup_parent) if args.cgroup_parent else configured_parent()
    report = run_probe(cgroup_parent=str(parent.path) if parent is not None else None)
    work = Path(tempfile.mkdtemp(prefix="bulkhead-overhead-work-"))
    try:
        targets = discover_targets(work, report=report, cgroup_parent=parent, profiles=args.profile or tuple(PROFILE_TIERS))
        load = os.getloadavg()[0]
        if load > (os.cpu_count() or 1) * 0.5:
            print(f"warning: load average {load:.2f} on {os.cpu_count()} CPUs; results from a busy host are not comparable", file=sys.stderr)
        directory = run_overhead(targets, samples=args.samples, warmup=args.warmup, repeats=args.repeats, proxy_samples=args.proxy_samples, probe=report, results_dir=args.results, log=print)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"overhead run: {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "EPISODES", "ProcessBaseline", "ProxyStartup", "Sample", "SandboxRunner", "TimingBackend", "WORKLOADS", "baseline_registry",
    "cold_start_from_samples", "distribution", "measure", "measure_episode", "resummarize", "run_overhead", "summarize_samples",
]

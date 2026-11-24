"""Bulkhead part 8: paired AgentDojo runs with claims, resume, infrastructure retries, and records.

A run takes a fixed list of eligible case ids and runs each case under the
three configurations (``baseline``, ``policy``, ``policy_taint``) back to back,
rotating which configuration goes first from case to case and recording the
order, so server drift spreads evenly. Every configuration of a case gets the
same injections (computed once per case with AgentDojo's attack, hashed into
each record), the same tools, the same model settings, and no fault schedule.
A case is the pair (user task, injection task); a clean case has no injection
task.

Records (``results/bulkhead/agentdojo/<run_id>/``):

- ``manifest.json``: everything the results depend on (AgentDojo pin and
  benchmark version, attack, classification hash, case ids and their hash,
  configurations, approval mode and approvals hash, model settings, server
  shape, budgets, code). A changed policy, tier, model, or approval mode is a
  new run id; resuming requires the identical manifest (only the slot count may
  differ).
- ``eligibility.json``: every tool and every case of the planned suites, with
  the JSON Pointer reason for each ineligible one.
- ``records.jsonl``: one ``case_result`` per (case, configuration, attempt).
- ``journals/``: the Steadyhand journals and manifests of every episode.
- ``replay.json``: what ``replay_run`` re-derived on CPU from those journals.

Infrastructure aborts (llama-server connection, timeout, error, or invalid
response) and harness errors are recorded with ``outcome`` other than
``scored`` and rerun under ``run_label="retry-<n>"`` (new episode ids) up to
``retries`` times; after ``abort_limit`` in a row no new case starts. They
never enter a denominator (``bulkhead.eval.metrics``).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..types import canonical_hash
from .agentdojo import (
    CONFIGURATIONS,
    DATA_PATH,
    PIPELINE_NAME,
    AgentDojoAPI,
    BulkheadPipeline,
    CaseClaimedElsewhere,
    EnvironmentStore,
    SuiteAdapter,
    executed_calls,
    load_classification,
    replay_effects,
)
from .attacks import ROOT, code_version, read_records
from .metrics import SCORED, failure_classes, journal_counts, primary_failure

RESULTS_DIR = ROOT / "results" / "bulkhead" / "agentdojo"
RETRIES = 2
INFRASTRUCTURE_ABORTS = frozenset({"backend_error", "backend_timeout", "backend_connection_error", "invalid_backend_response"})
SERVER_SHAPE_KEYS = ("parallel",)


# Cases and the plan


@dataclass(frozen=True, order=True)
class Case:
    suite: str
    user_task: str
    injection_task: str | None = None

    @property
    def case_id(self) -> str:
        return "/".join(part for part in (self.suite, self.user_task, self.injection_task) if part)

    @classmethod
    def from_id(cls, case_id: str) -> Case:
        parts = case_id.split("/")
        if len(parts) not in (2, 3):
            raise ValueError(f"a case id is suite/user_task[/injection_task]: {case_id!r}")
        return cls(*parts)


@dataclass
class Plan:
    """The adapted suites, every case's eligibility, and the eligible ids in a fixed order."""

    api: AgentDojoAPI
    classification: dict[str, Any]
    benchmark_version: str
    attack: str
    adapters: dict[str, SuiteAdapter]
    cases: list[Case]
    ineligible: list[dict[str, Any]] = field(default_factory=list)
    _injections: dict[str, dict[str, str]] = field(default_factory=dict)
    _attacks: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def eligibility(self) -> dict[str, Any]:
        return {
            "agentdojo": self.api.source, "benchmark_version": self.benchmark_version, "attack": self.attack,
            "tools": {name: [row.as_dict() for row in adapter.eligibility] for name, adapter in sorted(self.adapters.items())},
            "eligible_cases": [case.case_id for case in self.cases],
            "ineligible_cases": self.ineligible,
        }

    def tasks(self, case: Case) -> tuple[Any, Any, Any]:
        suite = self.adapters[case.suite].suite
        injection = suite.get_injection_task_by_id(case.injection_task) if case.injection_task else None
        return suite, suite.get_user_task_by_id(case.user_task), injection

    def injections(self, case: Case) -> dict[str, str]:
        """AgentDojo's attack for this (user task, injection task), computed once and shared by every configuration."""
        if case.injection_task is None:
            return {}
        with self._lock:
            if case.case_id not in self._injections:
                suite, user_task, injection_task = self.tasks(case)
                if case.suite not in self._attacks:  # one instance per suite, so its per-user-task cache is shared
                    self._attacks[case.suite] = self.api.load_attack(self.attack, suite, _Named(PIPELINE_NAME))
                self._injections[case.case_id] = dict(self._attacks[case.suite].attack(user_task, injection_task))
            return self._injections[case.case_id]


@dataclass(frozen=True)
class _Named:
    name: str


def plan_cases(api: AgentDojoAPI, *, classification: Mapping[str, Any] | None = None, suites: Sequence[str] | None = None,
               benchmark_version: str | None = None, attack: str | None = None, clean: bool = True, attacked: bool = True) -> Plan:
    """Adapt each suite and list its cases: every user task clean, and every (user task, injection task) pair."""
    classification = dict(classification or load_classification())
    pin = classification["agentdojo"]
    version = benchmark_version or pin["benchmark_version"]
    available = api.get_suites(version)
    names = list(suites) if suites is not None else sorted(classification["suites"])
    adapters = {name: SuiteAdapter(available[name], classification, api) for name in names}
    cases: list[Case] = []
    ineligible: list[dict[str, Any]] = []
    for name, adapter in adapters.items():
        suite = adapter.suite
        user_ids = sorted(suite.user_tasks, key=_task_order)
        injection_ids = sorted(suite.injection_tasks, key=_task_order)
        for user_id in user_ids:
            user_task = suite.get_user_task_by_id(user_id)
            wanted = ([None] if clean else []) + (injection_ids if attacked else [])
            for injection_id in wanted:
                injection_task = suite.get_injection_task_by_id(injection_id) if injection_id else None
                case = Case(name, user_id, injection_id)
                ok, reason = adapter.case_eligibility(user_task, injection_task)
                if ok:
                    cases.append(case)
                else:
                    ineligible.append({"case_id": case.case_id, "reason": reason})
    return Plan(api, classification, version, attack or pin["attack"], adapters, cases, ineligible)


def _task_order(task_id: str) -> tuple[str, int]:
    prefix, _, number = task_id.rpartition("_")
    return (prefix, int(number)) if number.isdigit() else (task_id, -1)


def select_cases(plan: Plan, *, count: int | None = None, seed: int = 0, case_ids: Sequence[str] | None = None) -> list[Case]:
    """Explicit ids (each must be eligible), a deterministic subset, or every eligible case."""
    from toolsmith.evaluation.metrics import deterministic_subset

    eligible = {case.case_id: case for case in plan.cases}
    if case_ids is not None:
        unknown = [case_id for case_id in case_ids if case_id not in eligible]
        if unknown:
            raise ValueError(f"not eligible (or unknown) case ids: {unknown}")
        return [eligible[case_id] for case_id in case_ids]
    if count is None:
        return list(plan.cases)
    chosen = deterministic_subset([{"unique_id": case.case_id} for case in plan.cases], count, seed=seed)
    return [eligible[row["unique_id"]] for row in chosen]


# One case under one configuration


def _trace_scored(task: Any, method: str) -> bool:
    for cls in type(task).__mro__:
        if method in vars(cls):
            return cls.__name__ not in ("BaseUserTask", "BaseInjectionTask")
    return False


def _bounded(text: Any, limit: int = 300) -> str | None:
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def run_case(
    plan: Plan,
    case: Case,
    configuration: str,
    client_factory: Callable[[Case], Any],
    directory: Path,
    *,
    approval_mode: str = "deny_all",
    approvals: Sequence[Mapping[str, Any]] = (),
    run_label: str | None = None,
    budgets: Any = None,
    server_parallel: int | None = None,
    context_size: int | None = None,
) -> dict[str, Any] | None:
    """Run one (case, configuration) through ``run_task_with_pipeline`` and return its record.

    ``None`` means another worker holds the case's episode claim.
    """
    from steadyhand.journal import read_journal

    adapter = plan.adapters[case.suite]
    suite, user_task, injection_task = plan.tasks(case)
    injections = plan.injections(case)
    journals = directory / "journals" / case.suite
    pipeline = BulkheadPipeline(
        adapter, configuration, lambda: client_factory(case), journals, approval_mode=approval_mode, approvals=approvals,
        run_label=run_label, budgets=budgets, server_parallel=server_parallel, context_size=context_size,
    )
    settings_sha256 = canonical_hash(client_factory(case).settings())
    security = adapter.security_config(configuration, approval_mode)
    record: dict[str, Any] = {
        "event": "case_result", "case_id": case.case_id, "suite": case.suite, "user_task": case.user_task, "injection_task": case.injection_task,
        "configuration": configuration, "approval_mode": approval_mode if security is not None else None, "run_label": run_label,
        "injections_sha256": canonical_hash(injections), "tools_sha256": adapter.registry.schemas_sha256(),
        "task_policy_sha256": security.task_policy.sha256() if security is not None else None, "model_settings_sha256": settings_sha256,
        "trace_scored": _trace_scored(user_task, "utility_from_traces") or (injection_task is not None and _trace_scored(injection_task, "security_from_traces")),
        "approvals_sha256": canonical_hash(list(approvals)),
    }
    started = time.monotonic()
    try:
        utility, security_result = suite.run_task_with_pipeline(pipeline, user_task, injection_task, injections)
    except CaseClaimedElsewhere:
        return None
    except Exception as exc:  # a harness fault: recorded, rerun, never scored
        record.update(outcome="harness_error", error_code=type(exc).__name__, error_message=_bounded(exc), utility=None, attack_success=None,
                      status=None, episodes=[], journals=[], replayed=[], counts=journal_counts([]), wall_s=round(time.monotonic() - started, 6),
                      model_s=0.0, approvals_requested=0, approved=0, declined=0, start_sha256=None, final_sha256=None, effects_match=None)
        record["failure_classes"] = failure_classes(record)
        record["primary_failure"] = primary_failure(record)
        return record
    wall = time.monotonic() - started
    outcome = pipeline.outcome
    assert outcome is not None and pipeline.calls == 1
    all_records = [r for path in outcome.journals for r in read_journal(path)]
    final_records = read_journal(outcome.journals[-1])
    finished = final_records[-1]
    aborted = [r for r in all_records if r["event"] == "episode_finished" and r["status"] == "aborted" and r.get("error_code") in INFRASTRUCTURE_ABORTS]
    # The independent effect check: re-run only the calls whose handler ran, on the case's starting environment.
    start_env = suite.load_and_inject_default_environment(injections)
    start_env = user_task.init_environment(start_env)
    effects = replay_effects(start_env, plan.api.FunctionsRuntime(suite.tools), executed_calls(final_records))
    record.update(
        outcome="infrastructure" if aborted else SCORED,
        error_code=(aborted[-1]["error_code"] if aborted else finished.get("error_code")),
        error_message=_bounded(aborted[-1].get("error_message") if aborted else finished.get("error_message")),
        utility=None if aborted else bool(utility),
        attack_success=None if aborted or injection_task is None else bool(security_result),
        status=finished["status"], parent_status=outcome.episodes[0].status.value,
        episodes=[r.episode_id for r in outcome.episodes], journals=[str(Path(p).relative_to(directory)) for p in outcome.journals],
        replayed=outcome.replayed, counts=journal_counts(all_records), wall_s=round(wall, 6),
        model_s=round(sum(r.get("elapsed_s") or 0 for r in all_records if r["event"] == "model_turn"), 6),
        approvals_requested=outcome.approvals_requested, approved=outcome.approved, declined=outcome.declined,
        start_sha256=outcome.start_sha256, final_sha256=outcome.final_sha256, effects_sha256=effects, effects_match=effects == outcome.final_sha256,
    )
    record["failure_classes"] = failure_classes(record)
    record["primary_failure"] = primary_failure(record)
    return record


# A run


@dataclass(frozen=True)
class RunSpec:
    """Everything a run's results depend on. Resuming requires the same spec (the slot count aside)."""

    label: str
    kind: str  # "live" (a real model) or "cpu_gate" (a scripted model)
    agentdojo: dict[str, Any]
    classification_sha256: str
    case_ids: tuple[str, ...]
    configurations: tuple[str, ...] = CONFIGURATIONS
    approval_mode: str = "deny_all"
    approvals_sha256: str | None = None
    model: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    code: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("live", "cpu_gate"):
            raise ValueError("kind is live or cpu_gate")
        unknown = set(self.configurations) - set(CONFIGURATIONS)
        if unknown or "baseline" not in self.configurations:
            raise ValueError(f"configurations must include baseline and come from {CONFIGURATIONS}")
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("case ids repeat")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["case_ids"] = list(self.case_ids)
        value["configurations"] = list(self.configurations)
        value["case_ids_sha256"] = canonical_hash(list(self.case_ids))
        return value

    def identity(self) -> dict[str, Any]:
        value = self.as_dict()
        value["server"] = {k: v for k, v in self.server.items() if k not in SERVER_SHAPE_KEYS}
        return value

    @property
    def run_id(self) -> str:
        return "adj_" + canonical_hash(self.identity())[:16]


def make_spec(plan: Plan, cases: Sequence[Case], *, label: str, kind: str, model: Mapping[str, Any], server: Mapping[str, Any] | None = None,
              budgets: Any = None, approval_mode: str = "deny_all", approvals: Mapping[str, Any] | None = None,
              configurations: Sequence[str] = CONFIGURATIONS, seed: int = 0) -> RunSpec:
    from steadyhand.controller import Budgets

    return RunSpec(
        label=label, kind=kind,
        agentdojo={**plan.api.source, "benchmark_version": plan.benchmark_version, "attack": plan.attack, "pinned": plan.classification["agentdojo"]},
        classification_sha256=canonical_hash(plan.classification), case_ids=tuple(case.case_id for case in cases),
        configurations=tuple(configurations), approval_mode=approval_mode,
        approvals_sha256=canonical_hash(dict(approvals)) if approvals is not None else None,
        model=dict(model), server=dict(server or {}), budgets=(budgets or Budgets()).as_dict(), seed=seed, code=code_version(),
    )


class RunMismatch(RuntimeError):
    def __init__(self, key: str, recorded: Any, current: Any) -> None:
        super().__init__(f"run manifest differs at {key!r}: recorded {recorded!r}, now {current!r}; start a new run instead of mixing results")
        self.key = key


class RunInterrupted(RuntimeError):
    """Too many cases in a row aborted on infrastructure: the model server is probably down."""


def rotated(configurations: Sequence[str], index: int) -> list[str]:
    shift = index % len(configurations)
    return [*configurations[shift:], *configurations[:shift]]


def run_agentdojo(
    spec: RunSpec,
    plan: Plan,
    client_factory: Callable[[Case], Any],
    *,
    results_dir: str | Path = RESULTS_DIR,
    workers: int = 1,
    retries: int = RETRIES,
    abort_limit: int | None = None,
    approvals: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    budgets: Any = None,
    mirror_dir: str | Path | None = None,
    mirror_seconds: float = 30.0,
    copier: Any = None,
) -> Path:
    """Run every (case, configuration) of ``spec`` not already recorded, ``workers`` cases at a time."""
    from steadyhand.controller import Budgets
    from toolsmith.artifacts import MirroredFile
    from toolsmith.evaluation.runner import atomic_write_json
    from toolsmith.recording import JsonlRecorder, trim_partial_line

    budgets = budgets or Budgets(**spec.budgets)
    if budgets.as_dict() != spec.budgets:
        raise RunMismatch("budgets", spec.budgets, budgets.as_dict())
    if spec.approval_mode == "scripted" and spec.approvals_sha256 != canonical_hash(dict(approvals or {})):
        raise RunMismatch("approvals_sha256", spec.approvals_sha256, canonical_hash(dict(approvals or {})))
    eligible = {case.case_id: case for case in plan.cases}
    unknown = [case_id for case_id in spec.case_ids if case_id not in eligible]
    if unknown:
        raise ValueError(f"the spec names cases the plan does not hold as eligible: {unknown[:5]}")
    directory = Path(results_dir) / spec.run_id
    directory.mkdir(parents=True, exist_ok=True)
    mirror = Path(mirror_dir) / spec.run_id if mirror_dir is not None else None

    def mirrored(path: Path) -> Any:
        return MirroredFile(path, mirror / path.name if mirror is not None else None, every_seconds=mirror_seconds,
                            submit=copier.submit_call if copier is not None else None)

    manifest_path, records_path = directory / "manifest.json", directory / "records.jsonl"
    manifest_store, records_store = mirrored(manifest_path), mirrored(records_path)
    if not manifest_path.exists() and manifest_store.restore():
        records_store.restore()
    trim_partial_line(records_path)
    manifest = {"run_id": spec.run_id, **spec.as_dict()}
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = {"run_id": spec.run_id, **spec.identity()}
        recorded_identity = dict(recorded, server={k: v for k, v in recorded.get("server", {}).items() if k not in SERVER_SHAPE_KEYS})
        for key in sorted(set(recorded_identity) | set(current)):
            if key == "code":
                continue  # recorded per segment; a code change shows in the records, sources hash included
            if recorded_identity.get(key) != current.get(key):
                raise RunMismatch(key, recorded_identity.get(key), current.get(key))
    else:
        atomic_write_json(manifest_path, manifest)
    atomic_write_json(directory / "eligibility.json", plan.eligibility())

    history: dict[str, list[dict[str, Any]]] = {}
    for record in read_records(records_path):
        if record.get("event") == "case_result":
            history.setdefault(f"{record['case_id']}|{record['configuration']}", []).append(record)
    labels: dict[str, str | None] = {}
    done: set[str] = set()
    for key, attempts in history.items():
        if attempts[-1]["outcome"] != SCORED and len(attempts) <= retries:
            labels[key] = f"retry-{len(attempts)}"
        else:
            done.add(key)
    recorder = JsonlRecorder(records_path)
    lock = threading.Lock()
    limit = abort_limit or max(4, 2 * workers)
    in_a_row = [0]
    stop = threading.Event()
    order = {case_id: index for index, case_id in enumerate(spec.case_ids)}

    def work(case: Case) -> None:
        for slot, configuration in enumerate(rotated(spec.configurations, order[case.case_id])):
            key = f"{case.case_id}|{configuration}"
            if key in done or stop.is_set():
                continue
            record = run_case(
                plan, case, configuration, client_factory, directory, approval_mode=spec.approval_mode,
                approvals=(approvals or {}).get(case.case_id, ()), run_label=labels.get(key), budgets=budgets,
                server_parallel=spec.server.get("parallel"), context_size=spec.server.get("context_size"),
            )
            if record is None:
                continue
            record.update(key=key, order=order[case.case_id], slot=slot, attempt=len(history.get(key, ())) + 1, sources_sha256=spec.code.get("sources_sha256"))
            with lock:
                recorder.append(record)
                in_a_row[0] = in_a_row[0] + 1 if record["outcome"] != SCORED else 0
                if in_a_row[0] >= limit:
                    stop.set()
            records_store.sync()
            if copier is not None:
                for journal in record["journals"]:
                    copier.submit(directory / journal)

    pending = [eligible[case_id] for case_id in spec.case_ids if any(f"{case_id}|{c}" not in done for c in spec.configurations)]
    if pending:
        with lock:
            recorder.append({"event": "run_segment", "workers": workers, "pending_cases": len(pending), "done_results": len(done),
                             "server": dict(spec.server), "code": code_version()})
    manifest_store.sync()
    try:
        if workers <= 1:
            for case in pending:
                work(case)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bulkhead-agentdojo") as pool:
                list(pool.map(work, pending))
    finally:
        with lock:
            records_store.sync(force=True)
    if stop.is_set():
        raise RunInterrupted(f"{in_a_row[0]} results in a row were not scored (infrastructure); no new case was started. Restart the server and run again.")
    return directory


def compute_summary(directory: Path, manifest: RunSpec | Mapping[str, Any]) -> dict[str, Any]:
    """The run's summary from its records (latest attempt per case and configuration). Writes nothing."""
    from .metrics import failure_table, summarize

    manifest = manifest.as_dict() if isinstance(manifest, RunSpec) else dict(manifest)
    records = read_records(directory / "records.jsonl")
    eligibility = json.loads((directory / "eligibility.json").read_text()) if (directory / "eligibility.json").exists() else {}
    summary = summarize(records, configurations=manifest["configurations"], seed=manifest.get("seed", 0))
    summary["failure_table"] = failure_table(records, manifest["configurations"])
    planned = len(manifest["case_ids"])
    total = len(eligibility.get("eligible_cases", []))
    summary["scope"] = {
        "case_ids": planned, "case_ids_sha256": canonical_hash(list(manifest["case_ids"])), "eligible_cases": total,
        "ineligible_cases": len(eligibility.get("ineligible_cases", [])), "partial": planned < total,
        "benchmark_version": manifest["agentdojo"].get("benchmark_version"), "agentdojo": manifest["agentdojo"],
    }
    return summary


def write_summary(directory: Path, manifest: RunSpec | Mapping[str, Any]) -> dict[str, Any]:
    """``summary.json`` from the run's records."""
    from toolsmith.evaluation.runner import atomic_write_json

    summary = compute_summary(directory, manifest)
    atomic_write_json(directory / "summary.json", summary)
    return summary


# Replay


def replay_run(directory: Path, plan: Plan, *, mirror: str | Path | None = None) -> dict[str, Any]:
    """Replay every recorded episode on CPU against a fresh copy of its case's environment. Writes ``replay.json``.

    Simulated tools re-run on in-memory copies; the model is served from the journal. Nothing
    outside the process is touched. Each case's last episode must also reproduce its recorded
    final environment hash. A journal missing locally (a runtime that resumed from Drive) is read
    from ``mirror/journals/`` (where ``JournalCopier`` put it); one found in neither place is listed.
    """
    from steadyhand.journal import ReplayMismatch
    from steadyhand.replay import replay_episode
    from toolsmith.evaluation.runner import atomic_write_json

    from .metrics import latest

    results = {"episodes": 0, "exact": 0, "mismatches": [], "final_environment_mismatches": [], "missing_journals": []}

    def locate(journal: str) -> Path | None:
        candidates = [directory / journal] + ([Path(mirror) / "journals" / Path(journal).name] if mirror is not None else [])
        return next((path for path in candidates if path.exists()), None)

    for (case_id, configuration), record in sorted(latest(read_records(directory / "records.jsonl")).items()):
        if not record.get("journals"):
            continue
        case = Case.from_id(case_id)
        suite, user_task, _ = plan.tasks(case)
        env = user_task.init_environment(suite.load_and_inject_default_environment(plan.injections(case)))
        store = None
        for journal in record["journals"]:
            path = locate(journal)
            if path is None:
                results["missing_journals"].append({"case_id": case_id, "configuration": configuration, "journal": journal})
                store = None
                break
            store = EnvironmentStore(env.model_copy(deep=True), plan.api.FunctionsRuntime(suite.tools))
            results["episodes"] += 1
            try:
                replay_episode(path, registry=plan.adapters[case.suite].registry, store=store)
                results["exact"] += 1
            except ReplayMismatch as exc:
                results["mismatches"].append({"case_id": case_id, "configuration": configuration, "journal": journal, "error": _bounded(exc)})
        if store is not None and store.snapshot_sha256() != record.get("final_sha256"):
            results["final_environment_mismatches"].append({"case_id": case_id, "configuration": configuration})
    atomic_write_json(directory / "replay.json", results)
    return results


# A scripted model for the CPU gate


class ScriptedModel:
    """A Toolsmith-service stand-in that plays scripted turns; no GPU and no network.

    Each ``chat`` plays the next turn whose condition holds against the tool observations seen
    so far (``when_seen``: a marker that must appear; ``unless_seen``: one that must not); turns
    whose condition fails are skipped for good. ``{"fail": "connection"}`` raises a llama-server
    connection error, so infrastructure handling can be exercised.
    """

    def __init__(self, turns: Sequence[Mapping[str, Any]], *, model: str = "scripted") -> None:
        from dataclasses import replace

        from toolsmith.protocols import QWEN3_PROFILE

        self.profile = replace(QWEN3_PROFILE, model=model)
        self.use_grammars = True
        self.turns = list(turns)
        self.position = 0

    def chat(self, request: Any) -> dict[str, Any]:
        from toolsmith.backends import LlamaCppConnectionError

        seen = "\n".join(str(m.get("content") or "") for m in request.messages if m.get("role") == "tool")
        while self.position < len(self.turns):
            turn = self.turns[self.position]
            self.position += 1
            if "when_seen" in turn and turn["when_seen"] not in seen:
                continue
            if "unless_seen" in turn and turn["unless_seen"] in seen:
                continue
            break
        else:
            raise RuntimeError("the scripted model ran out of turns")
        if "fail" in turn:
            raise LlamaCppConnectionError("scripted connection failure")
        message: dict[str, Any] = {"role": "assistant", "content": None}
        if "text" in turn:
            message["content"] = turn["text"]
        else:
            message["tool_calls"] = [
                {"id": f"call_{self.position}_{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
                for index, (name, arguments) in enumerate(turn["calls"])
            ]
        return {"choices": [{"index": 0, "message": message}], "usage": {"prompt_tokens": 50, "completion_tokens": 8}, "toolsmith": {"repair_attempts": 0}}


# Command line (the live notebook calls these functions directly)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bulkhead.eval.runner", description="Plan, summarize, or replay a paired AgentDojo run.")
    parser.add_argument("command", choices=("plan", "summarize", "replay"))
    parser.add_argument("--run", help="run directory (summarize, replay)")
    parser.add_argument("--suites", nargs="*")
    parser.add_argument("--classification", default=str(DATA_PATH))
    args = parser.parse_args(argv)
    from .agentdojo import agentdojo_api

    if args.command == "summarize":
        directory = Path(args.run)
        summary = write_summary(directory, json.loads((directory / "manifest.json").read_text()))
        print(json.dumps(summary["scope"], indent=2))
        return 0
    plan = plan_cases(agentdojo_api(), classification=load_classification(args.classification), suites=args.suites)
    if args.command == "plan":
        print(json.dumps({"eligible": len(plan.cases), "ineligible": plan.ineligible[:20], "withheld": {n: [r.as_dict() for r in a.withheld] for n, a in plan.adapters.items()}}, indent=2))
        return 0
    print(json.dumps(replay_run(Path(args.run), plan), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "Case", "INFRASTRUCTURE_ABORTS", "Plan", "RESULTS_DIR", "RunInterrupted", "RunMismatch", "RunSpec", "ScriptedModel",
    "compute_summary", "make_spec", "plan_cases", "replay_run", "rotated", "run_agentdojo", "run_case", "select_cases", "write_summary",
]

"""Record, replay, and resume episodes.

Replay re-runs ``run_episode`` on a fresh fixture with a ``ReplayClient`` that
serves the recorded model turns. Handlers, validation, the guard, and the
budgets run again for real; only the model is stubbed, so no GPU or network is
needed. Every event the code produces must equal the recorded line at the same
``seq`` (timestamps and timings aside), or replay raises ``ReplayMismatch``
naming the ``seq`` and field.

``run_claimed_episode`` is the live entry point. It claims the episode id
exclusively, writes the manifest, runs, and marks the manifest completed. When
an incomplete journal with a matching manifest exists, it resumes: the recorded
prefix is replayed and checked, then the live client takes over at the next
``seq``. A crash that kills the process outright leaves ``<episode_id>.claim``
behind; delete it by hand before resuming that episode.

A Bulkhead-secured episode replays with the security configuration stored in
its manifest (checked against the manifest's hash), so every policy decision,
approval request, and taint transition is re-derived and compared like any
other event. A journal recorded without Bulkhead replays without it, exactly
as before.

Replay never repeats an external side effect. In-memory mock-store handlers
are re-executed (they only touch the fresh fixture copy). Exec, sandboxed,
filesystem, and network tools are served by ``RecordedExecutionBackend`` from
the recorded ``sandbox_execution`` events, after checking the replayed request
matches; no process is spawned and no socket is opened. Resume does the same
for the recorded prefix before handing new calls to the live backend.

A Tidemark episode (its manifest carries a ``context_config``) replays only
with a context session whose ``identity()`` equals that configuration; the
session re-renders every request on CPU, so ``context_render`` and the
rendered-message hash in each ``model_turn`` are re-derived and compared.
With observation shaping (part 3) the session recomputes every shaped message
from the recorded raw result and verifies, never writes, the blobs it names:
a session's ``check_journal`` first refuses a journal whose referenced blobs
are missing or corrupt, and the shaper checks each blob's catalog entry and
bytes as it goes. Resume runs the same check before its first live call.
Recorded turns come back with the metrics their ``backend_timings`` record
holds, so counts and measurement paths reproduce exactly and only the
declared timings are volatile.

What replay verifies: the control plane (validation, guard, policy decisions,
approvals, taint, recovery, and every observation and message) reproduces
exactly. What it does not: kernel enforcement. Only a live run under the real
backend shows that; ``live_replay_report`` re-executes a recorded episode
under a backend you supply and writes a separate comparison report, never
touching the original journal.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bulkhead.replay import BackendIdentity, ExecutionBackend, ExecutionOutcome, ExecutionRequest, RecordedExecutionBackend
from toolsmith.evaluation.models import canonical_hash

from .context import ContextSession, check_recorded_journal
from .controller import DEFAULT_GUARD_LIMIT, Budgets
from .executor import SequentialExecutor
from .faults import FaultSchedule
from .journal import (
    EpisodeJournal,
    JournalFormatError,
    ReplayMismatch,
    dump_record,
    manifest_path_for,
    read_journal,
)
from .loop import run_episode
from .manifest import (
    build_manifest,
    create_manifest,
    load_manifest,
    mark_completed,
    require_match,
    try_claim,
)
from .recovery import RecoveryStrategy, no_sleep, strategy_for
from .registry import SteadyhandRegistry
from .security import SecurityConfig
from .store import Store, load_fixture
from .tools import build_registry
from .types import AttemptMetrics, EpisodeResult, ModelClient, ModelTurn, ToolCall, TurnKind, TurnMetrics

# The marker JsonlRecorder writes in place of a secret; a journal containing it cannot replay exactly.
REDACTED = "[redacted]"


@dataclass(frozen=True)
class RecordedTurn:
    """One recorded model turn, with the hash of the messages it was given."""

    seq: int
    messages_sha256: str
    tool_choice: Any
    turn: ModelTurn


def metrics_from_record(record: Mapping[str, Any]) -> TurnMetrics:
    """Rebuild the ``TurnMetrics`` a ``backend_timings`` record describes."""
    timings = list(record["attempt_timings"])
    attempts = tuple(
        AttemptMetrics.from_journal(counts, timings[index] if index < len(timings) else None)
        for index, counts in enumerate(record["attempt_metrics"])
    )
    return TurnMetrics(
        attempts=attempts,
        response_elapsed_ms=record["response_elapsed_ms"],
        rendered_prompt_tokens=record["rendered_prompt_tokens"],
    )


def turn_from_record(record: Mapping[str, Any], timings: Mapping[str, Any] | None = None) -> ModelTurn:
    """Rebuild the ``ModelTurn`` a ``model_turn`` record (and its ``backend_timings``, if any) describes."""
    message = deepcopy(record["assistant_message"])
    calls = tuple(ToolCall.from_openai(call) for call in (message or {}).get("tool_calls") or [])
    usage = record["usage"]
    return ModelTurn(
        kind=TurnKind(record["kind"]),
        assistant_message=message,
        tool_calls=calls,
        content=(message or {}).get("content"),
        reasoning_content=(message or {}).get("reasoning_content"),
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        error_code=record["error_code"],
        error_message=record["error_message"],
        repair_attempts=record["repair_attempts"],
        elapsed_s=record["elapsed_s"],
        raw_output=record["raw_output"],
        metrics=metrics_from_record(timings) if timings is not None else None,
    )


class ReplayClient:
    """Serves recorded model turns in order. Never calls ``ToolsmithService``."""

    def __init__(self, turns: Sequence[RecordedTurn], settings: Mapping[str, Any]) -> None:
        self._turns = list(turns)
        self._settings = dict(settings)

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]]) -> ReplayClient:
        turns = []
        for index, record in enumerate(records):
            if record["event"] != "model_turn":
                continue
            # A Tidemark journal writes backend_timings right after its model_turn.
            following = records[index + 1] if index + 1 < len(records) else None
            timings = following if following is not None and following["event"] == "backend_timings" else None
            turns.append(RecordedTurn(
                seq=record["seq"],
                messages_sha256=record["messages_sha256"],
                tool_choice=record["tool_choice"],
                turn=turn_from_record(record, timings),
            ))
        return cls(turns, records[0]["settings"])

    @property
    def remaining(self) -> int:
        return len(self._turns)

    def settings(self) -> dict[str, Any]:
        return dict(self._settings)

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
        **options: Any,
    ) -> ModelTurn:
        """The next recorded turn, if the code sent exactly the recorded messages and tool choice.

        ``options`` (a context session's ``allowed_tool_names``) are checked through ``context_render``.
        """
        if not self._turns:
            raise ReplayMismatch("The code asked for a model turn the journal does not have", field="model_turn")
        recorded = self._turns[0]
        if canonical_hash([dict(message) for message in messages]) != recorded.messages_sha256:
            raise ReplayMismatch(
                "The messages sent to the model differ from the recording",
                seq=recorded.seq,
                field="messages_sha256",
            )
        if tool_choice != recorded.tool_choice:
            raise ReplayMismatch(
                f"tool_choice {tool_choice!r} differs from the recorded {recorded.tool_choice!r}",
                seq=recorded.seq,
                field="tool_choice",
            )
        self._turns.pop(0)
        return recorded.turn


class ResumingClient:
    """Replays the recorded prefix, then hands every later turn to the live client."""

    def __init__(self, replay: ReplayClient, live: ModelClient) -> None:
        self.replay = replay
        self.live = live

    def settings(self) -> dict[str, Any]:
        return self.live.settings()

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
        **options: Any,
    ) -> ModelTurn:
        client: ModelClient = self.replay if self.replay.remaining else self.live
        return client.complete(messages, tools, tool_choice=tool_choice, **options)


class ReplayClock:
    """Returns the recorded elapsed times, so wall-clock decisions replay exactly.

    The controller reads the clock once at start and once per budget check.
    After the recording runs out, ``live`` (resume) continues from the last
    recorded value; without it, an extra read is a mismatch.
    """

    def __init__(self, elapsed: Sequence[float], live: Callable[[], float] | None = None) -> None:
        self._elapsed = list(elapsed)
        self._live = live
        self._started = False
        self._last = 0.0
        self._switch: float | None = None

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]], live: Callable[[], float] | None = None) -> ReplayClock:
        values = [
            record["elapsed_s"]
            for record in records
            if record["event"] == "budget_checked" and record["elapsed_s"] is not None
        ]
        return cls(values, live)

    def __call__(self) -> float:
        # First read is the episode start (0.0); later reads replay the recorded budget-check times.
        if not self._started:
            self._started = True
            return 0.0
        if self._elapsed:
            self._last = self._elapsed.pop(0)
            return self._last
        if self._live is None:
            raise ReplayMismatch("The code read the clock more often than the recording", field="elapsed_s")
        # Resume: continue from the last recorded time on the live clock.
        if self._switch is None:
            self._switch = self._live()
        return self._last + (self._live() - self._switch)


class ReplayCancel:
    """Reproduces a recorded cancellation: set after the recorded number of budget checks."""

    def __init__(self, checks_before_cancel: int) -> None:
        self._remaining = checks_before_cancel

    def is_set(self) -> bool:
        if self._remaining > 0:
            self._remaining -= 1
            return False
        return True


class ResumingBackend:
    """Serves the recorded prefix's sandbox runs, then hands later calls to the live backend."""

    def __init__(self, recorded: RecordedExecutionBackend, live: ExecutionBackend) -> None:
        self.recorded = recorded
        self.live = live
        self.identity = live.identity

    def execute(self, request: ExecutionRequest, arguments: Mapping[str, Any]) -> ExecutionOutcome:
        backend: ExecutionBackend = self.recorded if self.recorded.remaining else self.live
        return backend.execute(request, arguments)


def registry_for(schema_sha256: str) -> SteadyhandRegistry:
    """The built-in registry whose model-facing schemas hash to ``schema_sha256``.

    Store tools, optionally with the exec tools and Tidemark's ``read_blob``.
    """
    for context_tools in (False, True):
        for exec_tools in (False, True):
            candidate = build_registry(exec_tools=exec_tools, context_tools=context_tools)
            if candidate.schemas_sha256() == schema_sha256:
                return candidate
    return build_registry()  # require_match then names the schema mismatch


def replay_episode(
    path: str | Path,
    *,
    registry: SteadyhandRegistry | None = None,
    store: Store | None = None,
    executor: SequentialExecutor | None = None,
    recovery: RecoveryStrategy | None = None,
    backend: ExecutionBackend | None = None,
    context: ContextSession | None = None,
) -> EpisodeResult:
    """Re-run a finished journal on CPU and check every event. Writes nothing.

    The recovery strategy comes from the manifest unless ``recovery`` overrides
    it; a strategy that differs from the recorded one is refused. Policy waits
    are not slept; their recorded lengths are compared instead. Pass ``store``
    (a fresh fixture clone) to keep the replayed final store; it is mutated.
    Backend tools are served from the recording unless ``backend`` is given
    (only ``live_replay_report`` does that). The registry is picked by the
    recorded schema hash unless ``registry`` is given. A Tidemark journal needs
    ``context``: a fresh session whose ``identity()`` equals the manifest's
    ``context_config``.
    """
    path = Path(path)
    records = read_journal(path)
    if records[-1]["event"] != "episode_finished":
        raise JournalFormatError("Only a finished episode can be replayed; resume it instead", field="event")
    _refuse_redacted(records)
    # The manifest must describe this journal, and the code must rebuild the same manifest now
    # (same tools, policies, fixture, faults, and strategy); otherwise replay is meaningless.
    manifest = load_manifest(manifest_path_for(path))
    started = records[0]
    if manifest.episode_id != started["episode_id"]:
        raise ReplayMismatch("The manifest belongs to another episode", field="episode_id")
    recorded_policy = started.get("recovery_policy", "typed_hint")
    if manifest.recovery_policy != recorded_policy:
        raise ReplayMismatch(
            f"The manifest says {manifest.recovery_policy!r} but the journal ran {recorded_policy!r}",
            field="recovery_policy",
        )
    strategy = recovery or strategy_for(manifest.recovery_policy)
    registry = registry or registry_for(manifest.schema_sha256)
    store = store if store is not None else load_fixture().clone()
    faults = _recorded_schedule(manifest.fault_schedule, manifest.fault_schedule_sha256)
    security = _recorded_security(manifest.security_config, manifest.task_policy_sha256)
    _check_recorded_specs(manifest.security_specs, manifest.security_spec_sha256)
    identity = BackendIdentity.from_dict(manifest.backend) if manifest.backend is not None else None
    if backend is None and identity is not None:
        backend = RecordedExecutionBackend(records, identity)
    _check_recorded_context(manifest.context_config, manifest.context_config_sha256, context)
    current = build_manifest(
        started["settings"],
        registry,
        store,
        started["messages"],
        budgets=started["budgets"],
        guard_limit=started["guard_limit"],
        paraphrase_final=started["paraphrase_final"],
        run_label=manifest.run_label,
        server_parallel=manifest.server_parallel,
        context_size=manifest.context_size,
        fault_schedule=faults,
        recovery_policy=strategy.name,
        system_prompt_variant=manifest.system_prompt_variant,
        security=security,
        parent_episode_id=manifest.parent_episode_id,
        backend=identity,
        context_config=context.identity() if context is not None else None,
        context_version=context.version if context is not None else None,
        context_server=manifest.context_server,
    )
    require_match(manifest, current)
    # A Tidemark session checks what the journal references (blobs) before anything re-runs.
    check_recorded_journal(context, records)

    # Re-run with the model, clock, and cancellation all served from the recording. The journal
    # compares each new event with the recorded one instead of writing it.
    finished = records[-1]
    budget_checks = sum(1 for record in records if record["event"] == "budget_checked")
    journal = EpisodeJournal(path, episode_id=started["episode_id"], expected=records, write=False)
    result = run_episode(
        ReplayClient.from_records(records),
        registry,
        store,
        started["messages"],
        journal=journal,
        budgets=Budgets(**started["budgets"]),
        guard_limit=started["guard_limit"],
        executor=executor,
        clock=ReplayClock.from_records(records),
        cancel=ReplayCancel(budget_checks) if finished["error_code"] == "cancelled" else None,  # type: ignore[arg-type]
        paraphrase_final=started["paraphrase_final"],
        faults=faults,
        recovery=strategy,
        sleep=no_sleep,
        security=security,
        parent_episode_id=manifest.parent_episode_id,
        backend=_as_recorded_identity(backend, identity),
        context=context,
    )
    journal.verify_complete()
    return result


def run_claimed_episode(
    directory: str | Path,
    client: ModelClient,
    registry: SteadyhandRegistry,
    store: Store,
    messages: Sequence[Mapping[str, Any]],
    *,
    budgets: Budgets | None = None,
    guard_limit: int = DEFAULT_GUARD_LIMIT,
    paraphrase_final: bool = False,
    secrets: Sequence[str] = (),
    run_label: str | None = None,
    server_parallel: int | None = None,
    context_size: int | None = None,
    executor: SequentialExecutor | None = None,
    clock: Callable[[], float] = time.monotonic,
    cancel: Any = None,
    faults: FaultSchedule | None = None,
    recovery: RecoveryStrategy | None = None,
    system_prompt_variant: str = "plain",
    sleep: Callable[[float], None] = time.sleep,
    security: SecurityConfig | None = None,
    parent_episode_id: str | None = None,
    backend: ExecutionBackend | None = None,
    context: ContextSession | None = None,
    context_server: Mapping[str, Any] | None = None,
) -> EpisodeResult | None:
    """Claim, run (or resume), and complete one episode. ``None`` means nothing to do here.

    ``store`` must be the unmodified starting store; its hash is part of the id.
    ``security`` enables Bulkhead; it and ``parent_episode_id`` are part of the id.
    ``backend`` runs sandboxed calls; resume refuses a different backend identity and serves
    the recorded prefix's runs from the journal instead of re-running them.
    ``context`` (a fresh Tidemark session) and its configuration are part of the id;
    ``context_server`` (the serving shape) is resume-checked.
    Returns ``None`` when another worker holds the claim or the episode is
    already complete. Raises ``ReplayMismatch`` when an incomplete journal's
    manifest differs from this run, rather than resuming it.
    """
    budgets = budgets or Budgets()
    manifest = build_manifest(
        client.settings(),
        registry,
        store,
        messages,
        budgets=budgets.as_dict(),
        guard_limit=guard_limit,
        paraphrase_final=paraphrase_final,
        run_label=run_label,
        server_parallel=server_parallel,
        context_size=context_size,
        fault_schedule=faults,
        recovery_policy=recovery.name if recovery is not None else "typed_hint",
        system_prompt_variant=system_prompt_variant,
        security=security,
        parent_episode_id=parent_episode_id,
        backend=backend.identity if backend is not None else None,
        context_config=context.identity() if context is not None else None,
        context_version=context.version if context is not None else None,
        context_server=context_server,
    )
    # The episode id is a hash of the manifest, so the same task, settings, and faults always map
    # to the same journal file.
    journal_path = Path(directory) / f"{manifest.episode_id}.jsonl"
    manifest_file = manifest_path_for(journal_path)
    claim_file = journal_path.with_name(f"{manifest.episode_id}.claim")
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    # The claim file (created exclusively) stops two workers from running the same episode.
    if not try_claim(claim_file):
        return None
    try:
        run_client: ModelClient = client
        run_clock = clock
        run_backend = backend
        expected: list[dict[str, Any]] = []
        # A manifest already exists: the episode is complete (nothing to do) or was interrupted
        # (resume by replaying its journal, then continuing live).
        if not create_manifest(manifest_file, manifest):
            recorded = load_manifest(manifest_file)
            if recorded.completed:
                return None
            require_match(recorded, manifest)
            if journal_path.exists():
                _drop_torn_tail(journal_path)
                expected = read_journal(journal_path)
                # Missing or corrupt referenced blobs stop the resume before another model call.
                check_recorded_journal(context, expected)
                run_client = ResumingClient(ReplayClient.from_records(expected), client)
                run_clock = ReplayClock.from_records(expected, live=clock)
                if backend is not None:
                    run_backend = ResumingBackend(RecordedExecutionBackend(expected, backend.identity), backend)
        elif journal_path.exists():
            raise FileExistsError(f"A journal exists without a manifest: {journal_path}")
        journal = EpisodeJournal(
            journal_path, episode_id=manifest.episode_id, secrets=secrets, expected=expected, write=True,
        )
        result = run_episode(
            run_client,
            registry,
            store,
            messages,
            journal=journal,
            budgets=budgets,
            guard_limit=guard_limit,
            executor=executor,
            clock=run_clock,
            cancel=cancel,
            paraphrase_final=paraphrase_final,
            faults=faults,
            recovery=recovery,
            sleep=sleep,
            security=security,
            parent_episode_id=parent_episode_id,
            backend=run_backend,
            context=context,
        )
        mark_completed(manifest_file)
        return result
    finally:
        claim_file.unlink(missing_ok=True)


# Events whose mismatch means a security decision changed, not just an output.
SECURITY_DECISION_EVENTS = frozenset({"policy_decision", "approval_requested", "taint_transition", "egress_decision"})


def live_replay_report(
    path: str | Path,
    backend: ExecutionBackend,
    report_path: str | Path,
    *,
    registry: SteadyhandRegistry | None = None,
) -> dict[str, Any]:
    """Opt-in: re-execute a recorded episode's sandboxed calls under ``backend`` and write a comparison.

    The original journal and manifest are never written; the report goes to
    ``report_path`` only. Volatile fields (timings, resource use) may differ.
    The first mismatch is classified: a changed security decision is a
    security failure; a different sandbox outcome or observation is a
    divergence to investigate (later events cannot be compared after it).
    """
    path, report_path = Path(path), Path(report_path)
    if report_path.resolve() in (path.resolve(), manifest_path_for(path).resolve()):
        raise ValueError("the live replay report must not overwrite the recorded journal or manifest")
    records = read_journal(path)
    report: dict[str, Any] = {
        "journal": path.name,
        "recorded_backend": load_manifest(manifest_path_for(path)).backend,
        "live_backend": backend.identity.as_dict(),
        "matched": True,
        "first_mismatch": None,
        "security_failure": False,
    }
    try:
        replay_episode(path, registry=registry, backend=backend)
    except ReplayMismatch as exc:
        event = records[exc.seq]["event"] if exc.seq is not None and exc.seq < len(records) else None
        kind = "security_decision" if event in SECURITY_DECISION_EVENTS else "sandbox_outcome" if event == "sandbox_execution" else "observation"
        report.update(
            matched=False,
            first_mismatch={"seq": exc.seq, "field": exc.field, "event": event, "kind": kind, "message": str(exc)},
            security_failure=kind == "security_decision",
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return report


class _IdentifiedBackend:
    """A live backend presented under the recorded identity, so summaries compare in a live replay."""

    def __init__(self, backend: ExecutionBackend, identity: BackendIdentity) -> None:
        self._backend = backend
        self.identity = identity

    def execute(self, request: ExecutionRequest, arguments: Mapping[str, Any]) -> ExecutionOutcome:
        return self._backend.execute(request, arguments)


def _as_recorded_identity(backend: ExecutionBackend | None, identity: BackendIdentity | None) -> ExecutionBackend | None:
    if backend is None or identity is None or backend.identity == identity:
        return backend
    return _IdentifiedBackend(backend, identity)


@dataclass(frozen=True)
class ReplayOutcome:
    """Whether one journal replayed cleanly (and how it ended), or the error it raised."""

    path: str
    ok: bool
    status: str | None
    error: str | None


def replay_many(paths: Sequence[str | Path], *, workers: int | None = None) -> list[ReplayOutcome]:
    """Replay many journals on a CPU process pool. Each process loads its own fixture."""
    if not paths:
        return []
    count = min(workers or min(os.cpu_count() or 1, 8), len(paths))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=count, mp_context=context) as pool:
        return list(pool.map(_replay_one, [str(path) for path in paths]))


# Runs in a pool process; any failure becomes an outcome instead of stopping the batch.
def _replay_one(path: str) -> ReplayOutcome:
    try:
        result = replay_episode(path)
    except Exception as exc:
        return ReplayOutcome(path=path, ok=False, status=None, error=f"{type(exc).__name__}: {exc}")
    return ReplayOutcome(path=path, ok=True, status=result.status.value, error=None)


def _recorded_schedule(value: Mapping[str, Any] | None, expected_sha256: str | None) -> FaultSchedule | None:
    """Rebuild the recorded schedule and check it against the manifest hash before re-executing it."""
    if value is None:
        if expected_sha256 is not None:
            raise ReplayMismatch("The manifest hashes a fault schedule it does not include", field="fault_schedule")
        return None
    schedule = FaultSchedule.from_dict(value)
    if schedule.sha256 != expected_sha256:
        raise ReplayMismatch(
            "The recorded fault schedule does not match its manifest hash", field="fault_schedule_sha256"
        )
    return schedule


def _recorded_security(value: Mapping[str, Any] | None, expected_sha256: str | None) -> SecurityConfig | None:
    """Rebuild the recorded Bulkhead configuration and check it against the manifest hash."""
    if value is None:
        if expected_sha256 is not None:
            raise ReplayMismatch("The manifest hashes a security config it does not include", field="security_config")
        return None
    config = SecurityConfig.from_dict(value)
    if config.sha256() != expected_sha256:
        raise ReplayMismatch("The recorded security config does not match its manifest hash", field="task_policy_sha256")
    return config


def _check_recorded_context(
    value: Mapping[str, Any] | None, expected_sha256: str | None, context: ContextSession | None,
) -> None:
    """A Tidemark manifest needs a session with the same configuration; a legacy one must not get one."""
    if value is None:
        if expected_sha256 is not None:
            raise ReplayMismatch("The manifest hashes a context config it does not include", field="context_config")
        if context is not None:
            raise ReplayMismatch("This journal ran without a context session; replay it without one", field="context_config")
        return
    if canonical_hash(dict(value)) != expected_sha256:
        raise ReplayMismatch("The recorded context config does not match its manifest hash", field="context_config_sha256")
    if context is None:
        raise ReplayMismatch("This journal ran with a context session; pass one to replay it", field="context_config")
    if canonical_hash(context.identity()) != expected_sha256:
        raise ReplayMismatch("The context session's configuration differs from the recording", field="context_config")


def _check_recorded_specs(value: Mapping[str, Any] | None, expected_sha256: str | None) -> None:
    """The recorded spec content must hash to the recorded security_spec_sha256."""
    if value is None:
        return
    if canonical_hash([[name, spec] for name, spec in value.items()]) != expected_sha256:
        raise ReplayMismatch("The recorded security specs do not match their manifest hash", field="security_specs")


# A redacted value would make replayed messages differ from the originals, so refuse up front.
def _refuse_redacted(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        if REDACTED in dump_record(record):
            raise ReplayMismatch(
                "The journal contains redacted values, so it cannot be replayed exactly",
                seq=record["seq"],
                field="redacted",
            )


def _drop_torn_tail(path: Path) -> None:
    """Cut an unterminated last line left by a crash, so new lines start cleanly."""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])


__all__ = [
    "SECURITY_DECISION_EVENTS",
    "RecordedTurn",
    "ReplayCancel",
    "ReplayClient",
    "ReplayClock",
    "ReplayOutcome",
    "ResumingBackend",
    "ResumingClient",
    "replay_episode",
    "live_replay_report",
    "metrics_from_record",
    "registry_for",
    "replay_many",
    "run_claimed_episode",
    "turn_from_record",
]

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
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toolsmith.evaluation.models import canonical_hash

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
from .store import Store, load_fixture
from .tools import build_registry
from .types import EpisodeResult, ModelClient, ModelTurn, ToolCall, TurnKind

REDACTED = "[redacted]"


@dataclass(frozen=True)
class RecordedTurn:
    seq: int
    messages_sha256: str
    tool_choice: Any
    turn: ModelTurn


def turn_from_record(record: Mapping[str, Any]) -> ModelTurn:
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
    )


class ReplayClient:
    """Serves recorded model turns in order. Never calls ``ToolsmithService``."""

    def __init__(self, turns: Sequence[RecordedTurn], settings: Mapping[str, Any]) -> None:
        self._turns = list(turns)
        self._settings = dict(settings)

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]]) -> ReplayClient:
        turns = [
            RecordedTurn(
                seq=record["seq"],
                messages_sha256=record["messages_sha256"],
                tool_choice=record["tool_choice"],
                turn=turn_from_record(record),
            )
            for record in records
            if record["event"] == "model_turn"
        ]
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
    ) -> ModelTurn:
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
    ) -> ModelTurn:
        client: ModelClient = self.replay if self.replay.remaining else self.live
        return client.complete(messages, tools, tool_choice=tool_choice)


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
        if not self._started:
            self._started = True
            return 0.0
        if self._elapsed:
            self._last = self._elapsed.pop(0)
            return self._last
        if self._live is None:
            raise ReplayMismatch("The code read the clock more often than the recording", field="elapsed_s")
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


def replay_episode(
    path: str | Path,
    *,
    registry: SteadyhandRegistry | None = None,
    store: Store | None = None,
    executor: SequentialExecutor | None = None,
    recovery: RecoveryStrategy | None = None,
) -> EpisodeResult:
    """Re-run a finished journal on CPU and check every event. Writes nothing.

    The recovery strategy comes from the manifest unless ``recovery`` overrides
    it; a strategy that differs from the recorded one is refused. Policy waits
    are not slept; their recorded lengths are compared instead. Pass ``store``
    (a fresh fixture clone) to keep the replayed final store; it is mutated.
    """
    path = Path(path)
    records = read_journal(path)
    if records[-1]["event"] != "episode_finished":
        raise JournalFormatError("Only a finished episode can be replayed; resume it instead", field="event")
    _refuse_redacted(records)
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
    registry = registry or build_registry()
    store = store if store is not None else load_fixture().clone()
    faults = _recorded_schedule(manifest.fault_schedule, manifest.fault_schedule_sha256)
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
    )
    require_match(manifest, current)

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
) -> EpisodeResult | None:
    """Claim, run (or resume), and complete one episode. ``None`` means nothing to do here.

    ``store`` must be the unmodified starting store; its hash is part of the id.
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
    )
    journal_path = Path(directory) / f"{manifest.episode_id}.jsonl"
    manifest_file = manifest_path_for(journal_path)
    claim_file = journal_path.with_name(f"{manifest.episode_id}.claim")
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    if not try_claim(claim_file):
        return None
    try:
        run_client: ModelClient = client
        run_clock = clock
        expected: list[dict[str, Any]] = []
        if not create_manifest(manifest_file, manifest):
            recorded = load_manifest(manifest_file)
            if recorded.completed:
                return None
            require_match(recorded, manifest)
            if journal_path.exists():
                _drop_torn_tail(journal_path)
                expected = read_journal(journal_path)
                run_client = ResumingClient(ReplayClient.from_records(expected), client)
                run_clock = ReplayClock.from_records(expected, live=clock)
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
        )
        mark_completed(manifest_file)
        return result
    finally:
        claim_file.unlink(missing_ok=True)


@dataclass(frozen=True)
class ReplayOutcome:
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
    "RecordedTurn",
    "ReplayCancel",
    "ReplayClient",
    "ReplayClock",
    "ReplayOutcome",
    "ResumingClient",
    "replay_episode",
    "replay_many",
    "run_claimed_episode",
    "turn_from_record",
]

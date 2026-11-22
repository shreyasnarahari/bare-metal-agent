"""The episode loop: a control loop around the model.

The controller owns the ``while`` condition. Before each model call it checks
the budgets; after each call the loop classifies the turn and feeds the right
event to the state machine. Every returned call is validated, then shown to
the repeated-call guard, then executed only if both allow it. A blocked call
never touches the store. A turn that sends an already-blocked call again
escalates before any of its calls run.

After a call runs (or fails validation), the recovery strategy decides what the
model sees: the typed observation, a policy-layer retry, a verified write, an
alternative tool, or an escalation. A successful result that fails a sanity
check stops the rest of that turn's writes.

Several reads in one assistant message run together on a small per-episode
pool (at most 4 threads). Validation, the guard, and fault indexes are settled
in call order first, and results are journaled in call order afterwards, so
replay stays exact. A message that contains any write runs sequentially.

Endings: ``done`` (visible text), ``budget_exhausted``, ``escalated``, or
``aborted`` (backend, request, truncation, or cancellation). For the two
controller endings, ``final_content`` is rendered from the journal records,
so reading the journal back reproduces it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from toolsmith.evaluation.models import canonical_hash

from .controller import (
    DEFAULT_GUARD_LIMIT,
    Budgets,
    EpisodeController,
    GuardDecision,
    has_visible_character,
    repeated_call_observation,
)
from .executor import SequentialExecutor
from .faults import FaultInjector, FaultSchedule
from .idempotency import idempotency_key
from .journal import EpisodeJournal, ReplayMismatch
from .messages import PARAPHRASE_REQUEST, render_final_message
from .policy import Effect
from .recovery import RecoveryBudget, RecoveryContext, RecoveryStrategy, TypedHint, apply_recovery
from .registry import SteadyhandRegistry
from .store import Store
from .types import (
    EpisodeResult,
    EpisodeStatus,
    ModelClient,
    ModelTurn,
    ToolCall,
    ToolResult,
    TurnKind,
    observation_content,
)
from .validate import Verdict, parse_error_observation, suspicious_result_observation, validate_call

MAX_PARALLEL_READS = 4
SKIPPED_WRITE_HINT = (
    "Not run: an earlier result in this turn failed a consistency check. Re-check with the read tools before writing."
)


def observation_message(call: ToolCall, result: ToolResult) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call.id, "content": result.content()}


@dataclass
class _Episode:
    """The per-episode objects every call needs. Never shared between episodes."""

    registry: SteadyhandRegistry
    store: Store
    executor: SequentialExecutor
    controller: EpisodeController
    journal: EpisodeJournal
    injector: FaultInjector | None
    strategy: RecoveryStrategy
    budget: RecoveryBudget
    sleep: Callable[[float], None]
    history: list[dict[str, Any]]


@dataclass
class _Plan:
    call: ToolCall
    verdict: Verdict
    decision: GuardDecision
    key: str | None = None
    call_index: int | None = None
    skipped: bool = False

    @property
    def runnable(self) -> bool:
        return self.verdict.ok and not self.decision.blocked and not self.skipped


def run_episode(
    client: ModelClient,
    registry: SteadyhandRegistry,
    store: Store,
    messages: Sequence[Mapping[str, Any]],
    *,
    journal: EpisodeJournal,
    budgets: Budgets | None = None,
    executor: SequentialExecutor | None = None,
    guard_limit: int = DEFAULT_GUARD_LIMIT,
    clock: Callable[[], float] = time.monotonic,
    cancel: threading.Event | None = None,
    paraphrase_final: bool = False,
    faults: FaultSchedule | None = None,
    recovery: RecoveryStrategy | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> EpisodeResult:
    """Run one episode against ``store``, which the episode owns and mutates.

    ``steps`` counts model turns that requested tools or failed to parse. A
    final text turn ends the episode without counting as a step, and its
    assistant message is appended to the returned ``messages``.

    ``paraphrase_final`` asks the model for one ``tool_choice="none"`` sentence
    after ``budget_exhausted`` or ``escalated``. It is off by default; the
    controller's template is then the final content and no extra call is made.

    ``faults`` injects that schedule's tool-side faults. Each episode gets its
    own injector, so per-tool call indexes always start at 0.

    ``recovery`` picks the strategy (default ``typed_hint``). ``sleep`` is how
    policy backoff waits; tests and replay pass one that does not sleep.
    """
    budgets = budgets or Budgets()
    executor = executor or SequentialExecutor()
    strategy = recovery or TypedHint()
    settings = client.settings()
    controller = EpisodeController(
        budgets,
        guard_limit=guard_limit,
        clock=clock,
        slot_context=settings.get("slot_context"),
        max_tokens=settings.get("max_tokens"),
    )
    history = deepcopy([dict(message) for message in messages])
    episode = _Episode(
        registry=registry,
        store=store,
        executor=executor,
        controller=controller,
        journal=journal,
        injector=FaultInjector(faults) if faults is not None else None,
        strategy=strategy,
        budget=RecoveryBudget(),
        sleep=sleep,
        history=history,
    )
    tools = registry.schemas()
    final_content: str | None = None

    journal.episode_started(
        settings=settings,
        fixture_sha256=store.fixture_sha256,
        store_sha256=store.snapshot_sha256(),
        budgets=budgets.as_dict(),
        guard_limit=guard_limit,
        paraphrase_final=paraphrase_final,
        tool_names=registry.names(),
        schema_sha256=registry.schemas_sha256(),
        policy_sha256=registry.policies_sha256(),
        messages=history,
        recovery_policy=strategy.name,
    )
    try:
        while controller.running:
            if cancel is not None and cancel.is_set():
                controller.cancel()
                break
            check = controller.check_budgets()
            journal.budget_checked(step=controller.steps, check=check.as_dict())
            if check.exhausted is not None:
                controller.exhaust()
                break
            if check.slot_overflow is not None:
                controller.abort("completion_truncated", check.slot_overflow)
                break

            messages_sha256 = canonical_hash(history)
            turn = _visible_or_parse_error(client.complete(history, tools).with_call_ids(controller.steps))
            controller.record_usage(turn)
            journal.model_turn(step=controller.steps, turn=turn, messages_sha256=messages_sha256)

            if turn.kind is TurnKind.TEXT:
                final_content = turn.content
                history.append(turn.assistant_message or {"role": "assistant", "content": turn.content})
                controller.answer()
                break
            if turn.kind is TurnKind.FATAL_ERROR:
                controller.abort(turn.error_code or "fatal_error", turn.error_message)
                break
            if turn.kind is TurnKind.PARSE_ERROR:
                observation = parse_error_observation(turn).as_dict()
                message = {"role": "user", "content": observation_content(observation)}
                history.append(message)
                journal.observation(step=controller.steps, message=message, observation=observation)
                controller.finish_step()
                continue

            history.append(turn.assistant_message)
            if _escalate_repeat(turn, controller, journal):
                break
            reason = _run_tool_turn(turn.tool_calls, episode)
            if reason is not None:
                journal.escalated(step=controller.steps, reason=reason)
                controller.escalate(reason)
                break
            controller.finish_step()

        status = controller.state
        if status in (EpisodeStatus.BUDGET_EXHAUSTED, EpisodeStatus.ESCALATED):
            final_content = render_final_message(status, journal.records)
            if paraphrase_final and final_content is not None:
                final_content = _paraphrase(client, history, tools, final_content, controller, journal)
    except BaseException as exc:
        # A replay mismatch is the result, not a crash to journal. Otherwise record
        # the crash, without letting a second failure hide the first.
        if not isinstance(exc, ReplayMismatch):
            try:
                journal.episode_finished(
                    status=EpisodeStatus.ABORTED,
                    steps=controller.steps,
                    final_content=None,
                    prompt_tokens=controller.prompt_tokens,
                    completion_tokens=controller.completion_tokens,
                    error_code="internal_error",
                    store_sha256=store.snapshot_sha256(),
                )
            except ReplayMismatch:
                pass
        raise

    journal.episode_finished(
        status=status,
        steps=controller.steps,
        final_content=final_content,
        prompt_tokens=controller.prompt_tokens,
        completion_tokens=controller.completion_tokens,
        error_code=controller.error_code,
        error_message=controller.error_message,
        store_sha256=store.snapshot_sha256(),
    )
    return EpisodeResult(
        status=status,
        messages=history,
        steps=controller.steps,
        final_content=final_content,
        prompt_tokens=controller.prompt_tokens,
        completion_tokens=controller.completion_tokens,
        journal_path=journal.path,
        episode_id=journal.episode_id,
        error_code=controller.error_code,
    )


def _visible_or_parse_error(turn: ModelTurn) -> ModelTurn:
    """``done`` needs a visible character. Whitespace-only text is a parse error."""
    if turn.kind is TurnKind.TEXT and not has_visible_character(turn.content):
        return replace(
            turn,
            kind=TurnKind.PARSE_ERROR,
            assistant_message=None,
            error_code="empty_model_response",
            error_message="The assistant message has no visible character",
            raw_output=turn.content,
        )
    return turn


def _escalate_repeat(turn: ModelTurn, controller: EpisodeController, journal: EpisodeJournal) -> bool:
    """Escalate, before running anything, when the turn resends a call the guard already blocked."""
    guard = controller.guard
    repeated = next((call for call in turn.tool_calls if guard.is_blocked(call)), None)
    if repeated is None:
        return False
    decision = guard.observe(repeated)
    last_outcome = guard.last_outcome(decision.key)
    journal.guard_tripped(
        step=controller.steps,
        action="escalate",
        tool=repeated.name,
        tool_call_id=repeated.id,
        arguments=repeated.arguments,
        call_hash=decision.key,
        count=decision.count,
        limit=decision.limit,
        last_outcome=last_outcome,
    )
    controller.escalate_repeat(
        f"{repeated.name} was sent again with the same arguments after a repeated_call observation"
    )
    return True


def _run_tool_turn(calls: Sequence[ToolCall], episode: _Episode) -> str | None:
    """Run one assistant message's calls. Returns an escalation reason, or None to keep going."""
    registry = episode.registry
    parallel = len(calls) > 1 and all(
        call.name in registry and registry.policy(call.name).effect is Effect.READ for call in calls
    )
    if parallel:
        plans = [_plan(call, episode, reserve_fault_index=True) for call in calls]
        width = min(MAX_PARALLEL_READS, len(plans))
        with ThreadPoolExecutor(max_workers=width, thread_name_prefix="steadyhand-read") as pool:
            futures = [pool.submit(_first_attempt, plan, episode) if plan.runnable else None for plan in plans]
            firsts = [future.result() if future is not None else None for future in futures]
        for plan, first in zip(plans, firsts):
            reason, _ = _finish(plan, first, episode)
            if reason is not None:
                return reason
        return None

    skip_writes = False
    for call in calls:
        plan = _plan(call, episode, reserve_fault_index=False)
        if skip_writes and plan.verdict.ok and not plan.decision.blocked and _is_write(call, registry):
            plan.skipped = True
        first = _first_attempt(plan, episode) if plan.runnable else None
        reason, suspicious = _finish(plan, first, episode)
        if reason is not None:
            return reason
        skip_writes = skip_writes or suspicious
    return None


def _plan(call: ToolCall, episode: _Episode, *, reserve_fault_index: bool) -> _Plan:
    verdict = validate_call(call, episode.registry, episode.store)
    decision = episode.controller.guard.observe(call)
    plan = _Plan(call=call, verdict=verdict, decision=decision)
    if not plan.runnable:
        return plan
    if episode.strategy.keyed_writes and _is_write(call, episode.registry) and verdict.arguments is not None:
        plan.key = idempotency_key(episode.journal.episode_id, call.name, verdict.arguments)
    if reserve_fault_index and episode.injector is not None:
        plan.call_index = episode.injector.reserve(call.name)
    return plan


def _first_attempt(plan: _Plan, episode: _Episode) -> ToolResult:
    options: dict[str, Any] = {}
    if episode.injector is not None:
        options["injector"] = episode.injector
    if plan.key is not None:
        options["idempotency_key"] = plan.key
    if plan.call_index is not None:
        options["call_index"] = plan.call_index
    return episode.executor.run(plan.call, episode.registry, episode.store, **options)


def _finish(plan: _Plan, first: ToolResult | None, episode: _Episode) -> tuple[str | None, bool]:
    """Recover, journal, and append one call's observation, in call order."""
    call, journal, controller = plan.call, episode.journal, episode.controller
    step = controller.steps
    escalate_reason: str | None = None
    suspicious = False
    if plan.decision.blocked:
        last_outcome = controller.guard.last_outcome(plan.decision.key)
        journal.guard_tripped(
            step=step,
            action="block",
            tool=call.name,
            tool_call_id=call.id,
            arguments=call.arguments,
            call_hash=plan.decision.key,
            count=plan.decision.count,
            limit=plan.decision.limit,
            last_outcome=last_outcome,
        )
        controller.trip_guard()
        result = ToolResult(
            call_id=call.id,
            tool=call.name,
            arguments=call.arguments,
            arguments_sha256=plan.verdict.arguments_sha256,
            ok=False,
            error=repeated_call_observation(call.name, last_outcome),
            handler_called=False,
        )
    elif plan.skipped:
        result = ToolResult(
            call_id=call.id,
            tool=call.name,
            arguments=call.arguments,
            arguments_sha256=plan.verdict.arguments_sha256,
            ok=False,
            error=suspicious_result_observation(
                call.name, SKIPPED_WRITE_HINT, "skipped after a suspicious result", retryable=True
            ),
            handler_called=False,
        )
    else:
        if first is None:
            first = plan.verdict.rejected(call)
        elif first.fault is not None:
            journal.injected_fault(step=step, fault=first.fault.as_dict())
        registry = episode.registry
        context = RecoveryContext(
            call=call,
            arguments=plan.verdict.arguments,
            policy=registry.policy(call.name) if call.name in registry else None,
            registry=registry,
            store=episode.store,
            executor=episode.executor,
            injector=episode.injector,
            journal=journal,
            step=step,
            episode_id=journal.episode_id,
            sleep=episode.sleep,
            budget=episode.budget,
            idempotency_key=plan.key,
        )
        outcome = apply_recovery(episode.strategy, context, first)
        result, escalate_reason, suspicious = outcome.result, outcome.escalate_reason, outcome.suspicious
        if outcome.acted:
            journal.recovery(
                step=step,
                tool_call_id=call.id,
                tool=call.name,
                strategy=episode.strategy.name,
                recovered_by=outcome.recovered_by,
                escalated=escalate_reason,
            )
    if not plan.decision.blocked:
        controller.guard.record_outcome(plan.decision.key, result)
    episode.history.append(observation_message(call, result))
    journal.tool_result(step=step, result=result)
    return escalate_reason, suspicious


def _is_write(call: ToolCall, registry: SteadyhandRegistry) -> bool:
    return call.name in registry and registry.policy(call.name).effect is not Effect.READ


def _answerable(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """History the shim accepts: drop a trailing tool-calling turn whose calls were not all answered."""
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            ids = {call["id"] for call in message["tool_calls"]}
            answered = {later.get("tool_call_id") for later in history[index + 1:] if later.get("role") == "tool"}
            return history if ids <= answered else history[:index]
    return history


def _paraphrase(
    client: ModelClient,
    history: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    template: str,
    controller: EpisodeController,
    journal: EpisodeJournal,
) -> str:
    """One tool-free sentence from the model. Falls back to the template on any failure."""
    request = {"role": "user", "content": PARAPHRASE_REQUEST + template}
    sent = [*_answerable(history), request]
    turn = client.complete(sent, tools, tool_choice="none")
    controller.record_usage(turn)
    history.append(request)
    journal.observation(step=controller.steps, message=request, observation={"purpose": "final_paraphrase"})
    journal.model_turn(step=controller.steps, turn=turn, messages_sha256=canonical_hash(sent), tool_choice="none")
    if turn.kind is TurnKind.TEXT and has_visible_character(turn.content) and turn.assistant_message:
        history.append(turn.assistant_message)
        return turn.content or template
    return template


__all__ = ["MAX_PARALLEL_READS", "observation_message", "run_episode"]

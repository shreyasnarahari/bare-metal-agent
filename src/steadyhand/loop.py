"""The episode loop: a control loop around the model.

The controller owns the ``while`` condition. Before each model call it checks
the budgets; after each call the loop classifies the turn and feeds the right
event to the state machine. Every returned call is validated, then shown to
the repeated-call guard, then executed only if both allow it. A blocked call
never touches the store. A turn that sends an already-blocked call again
escalates before any of its calls run.

Endings: ``done`` (visible text), ``budget_exhausted``, ``escalated``, or
``aborted`` (backend, request, truncation, or cancellation). For the two
controller endings, ``final_content`` is rendered from the journal records,
so reading the journal back reproduces it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

from toolsmith.evaluation.models import canonical_hash

from .controller import (
    DEFAULT_GUARD_LIMIT,
    Budgets,
    EpisodeController,
    has_visible_character,
    repeated_call_observation,
)
from .executor import SequentialExecutor
from .journal import EpisodeJournal, ReplayMismatch
from .messages import PARAPHRASE_REQUEST, render_final_message
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
from .validate import parse_error_observation, validate_call


def observation_message(call: ToolCall, result: ToolResult) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call.id, "content": result.content()}


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
) -> EpisodeResult:
    """Run one episode against ``store``, which the episode owns and mutates.

    ``steps`` counts model turns that requested tools or failed to parse. A
    final text turn ends the episode without counting as a step, and its
    assistant message is appended to the returned ``messages``.

    ``paraphrase_final`` asks the model for one ``tool_choice="none"`` sentence
    after ``budget_exhausted`` or ``escalated``. It is off by default; the
    controller's template is then the final content and no extra call is made.
    """
    budgets = budgets or Budgets()
    executor = executor or SequentialExecutor()
    settings = client.settings()
    controller = EpisodeController(
        budgets,
        guard_limit=guard_limit,
        clock=clock,
        slot_context=settings.get("slot_context"),
        max_tokens=settings.get("max_tokens"),
    )
    history = deepcopy([dict(message) for message in messages])
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
            for call in turn.tool_calls:
                result = _run_call(call, registry, store, executor, controller, journal)
                history.append(observation_message(call, result))
                journal.tool_result(step=controller.steps, result=result)
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


def _run_call(
    call: ToolCall,
    registry: SteadyhandRegistry,
    store: Store,
    executor: SequentialExecutor,
    controller: EpisodeController,
    journal: EpisodeJournal,
) -> ToolResult:
    verdict = validate_call(call, registry, store)
    decision = controller.guard.observe(call)
    if decision.blocked:
        last_outcome = controller.guard.last_outcome(decision.key)
        journal.guard_tripped(
            step=controller.steps,
            action="block",
            tool=call.name,
            tool_call_id=call.id,
            arguments=call.arguments,
            call_hash=decision.key,
            count=decision.count,
            limit=decision.limit,
            last_outcome=last_outcome,
        )
        controller.trip_guard()
        return ToolResult(
            call_id=call.id,
            tool=call.name,
            arguments=call.arguments,
            arguments_sha256=verdict.arguments_sha256,
            ok=False,
            error=repeated_call_observation(call.name, last_outcome),
            handler_called=False,
        )
    result = executor.run(call, registry, store) if verdict.ok else verdict.rejected(call)
    controller.guard.record_outcome(decision.key, result)
    return result


def _paraphrase(
    client: ModelClient,
    history: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    template: str,
    controller: EpisodeController,
    journal: EpisodeJournal,
) -> str:
    """One tool-free sentence from the model. Falls back to the template on any failure."""
    answerable = history
    if history and history[-1].get("role") == "assistant" and history[-1].get("tool_calls"):
        answerable = history[:-1]  # the escalating turn's calls were never run or answered
    request = {"role": "user", "content": PARAPHRASE_REQUEST + template}
    sent = [*answerable, request]
    turn = client.complete(sent, tools, tool_choice="none")
    controller.record_usage(turn)
    history.append(request)
    journal.observation(step=controller.steps, message=request, observation={"purpose": "final_paraphrase"})
    journal.model_turn(step=controller.steps, turn=turn, messages_sha256=canonical_hash(sent), tool_choice="none")
    if turn.kind is TurnKind.TEXT and has_visible_character(turn.content) and turn.assistant_message:
        history.append(turn.assistant_message)
        return turn.content or template
    return template


__all__ = ["observation_message", "run_episode"]

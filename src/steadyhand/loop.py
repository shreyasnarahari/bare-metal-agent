"""The episode loop: a control loop around the model.

Send the conversation and tool schemas, validate each returned call, execute
the valid ones in list order, and feed every observation back. A call that
fails validation never reaches its handler. A model turn the shim could not
parse comes back to the model as a ``parse_error`` observation in a ``user``
message (there is no call id to answer) and counts as a step.

The episode ends on a text answer (``DONE``), a backend, request, or
truncation failure (``ABORTED``), or the step ceiling (``BUDGET_EXHAUSTED``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .executor import SequentialExecutor
from .journal import EpisodeJournal
from .registry import SteadyhandRegistry
from .store import Store
from .types import (
    EpisodeResult,
    EpisodeStatus,
    ModelClient,
    ToolCall,
    ToolResult,
    TurnKind,
    observation_content,
)
from .validate import parse_error_observation, validate_call

DEFAULT_MAX_STEPS = 8


def observation_message(call: ToolCall, result: ToolResult) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call.id, "content": result.content()}


def run_episode(
    client: ModelClient,
    registry: SteadyhandRegistry,
    store: Store,
    messages: Sequence[Mapping[str, Any]],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    journal: EpisodeJournal,
    executor: SequentialExecutor | None = None,
) -> EpisodeResult:
    """Run one episode against ``store``, which the episode owns and mutates.

    ``steps`` counts model turns that requested tools or failed to parse. A
    final text turn ends the episode without counting as a step, and its
    assistant message is appended to the returned ``messages``.
    """
    if max_steps < 0:
        raise ValueError("max_steps must not be negative")
    executor = executor or SequentialExecutor()
    history = deepcopy([dict(message) for message in messages])
    tools = registry.schemas()

    status = EpisodeStatus.RUNNING
    steps = 0
    prompt_tokens = 0
    completion_tokens = 0
    final_content: str | None = None
    error_code: str | None = None

    journal.episode_started(
        settings=client.settings(),
        fixture_sha256=store.fixture_sha256,
        store_sha256=store.snapshot_sha256(),
        max_steps=max_steps,
        tool_names=registry.names(),
        schema_sha256=registry.schemas_sha256(),
        policy_sha256=registry.policies_sha256(),
        messages=history,
    )
    try:
        while status is EpisodeStatus.RUNNING:
            if steps >= max_steps:
                status = EpisodeStatus.BUDGET_EXHAUSTED
                break
            turn = client.complete(history, tools).with_call_ids(steps)
            prompt_tokens += turn.prompt_tokens
            completion_tokens += turn.completion_tokens
            journal.model_turn(step=steps, turn=turn)

            if turn.kind is TurnKind.TEXT:
                final_content = turn.content
                if turn.assistant_message is not None:
                    history.append(turn.assistant_message)
                status = EpisodeStatus.DONE
                break
            if turn.kind is TurnKind.FATAL_ERROR:
                error_code = turn.error_code
                status = EpisodeStatus.ABORTED
                break
            if turn.kind is TurnKind.PARSE_ERROR:
                observation = parse_error_observation(turn).as_dict()
                message = {"role": "user", "content": observation_content(observation)}
                history.append(message)
                journal.observation(step=steps, message=message, observation=observation)
                steps += 1
                continue

            history.append(turn.assistant_message)
            for call in turn.tool_calls:
                verdict = validate_call(call, registry, store)
                result = (
                    executor.run(call, registry, store) if verdict.ok else verdict.rejected(call)
                )
                history.append(observation_message(call, result))
                journal.tool_result(step=steps, result=result)
            steps += 1
    except BaseException:
        journal.episode_finished(
            status=EpisodeStatus.ABORTED,
            steps=steps,
            final_content=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_code="internal_error",
            store_sha256=store.snapshot_sha256(),
        )
        raise

    journal.episode_finished(
        status=status,
        steps=steps,
        final_content=final_content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error_code=error_code,
        store_sha256=store.snapshot_sha256(),
    )
    return EpisodeResult(
        status=status,
        messages=history,
        steps=steps,
        final_content=final_content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        journal_path=journal.path,
        episode_id=journal.episode_id,
        error_code=error_code,
    )


__all__ = ["DEFAULT_MAX_STEPS", "observation_message", "run_episode"]

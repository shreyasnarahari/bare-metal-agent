"""Episode text: the system instruction and the controller's give-up messages.

Give-up messages are rendered only from journal records, so reading the
journal back reproduces them exactly. They list what was tried (tools, error
types, counts) and why the episode stopped, never a hidden chain of thought.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .errors import one_line
from .types import EpisodeStatus

SYSTEM_PROMPT = (
    "You are a customer-support agent for a small online store. "
    "Call a tool only when you need facts from the store or need to change it; otherwise answer directly. "
    "The user must supply order ids (like ord_1001), customer ids (like cus_101), "
    "ticket ids (like tkt_9001), and refund amounts. Never invent them: "
    "if one is missing, ask the user for it instead of calling a tool."
)
# Part 6 ablation: appended to the system prompt for system_prompt_variant="failure_aware".
FAILURE_AWARE_ADDENDUM = (
    "When a tool result has ok false, read its error_type and hint before acting. "
    "If retryable is true and error_type is invalid_arguments, fix the listed fields and call again. "
    "If error_type is repeated_call, stop repeating that call: change the arguments, use another tool, "
    "or ask the user. If retryable is false, do not send the same call again."
)
PROMPT_VARIANTS = {"plain": "", "failure_aware": FAILURE_AWARE_ADDENDUM}
PARAPHRASE_REQUEST = (
    "Stop working on the task now. Without calling any tool, tell the user in one or two "
    "sentences that you could not finish it, using only this summary:\n"
)
_BUDGET_NAMES = {
    "step": "step",
    "prompt_token": "prompt token",
    "completion_token": "completion token",
    "wall_clock": "wall-clock",
}
_MAX_GOAL_CHARS = 160
_MAX_ARGUMENT_CHARS = 120


def system_prompt_for(variant: str = "plain", *, base: str = SYSTEM_PROMPT) -> str:
    try:
        addendum = PROMPT_VARIANTS[variant]
    except KeyError:
        raise ValueError(f"Unknown system prompt variant {variant!r}; choose one of {sorted(PROMPT_VARIANTS)}") from None
    return f"{base}\n\n{addendum}" if addendum else base


def initial_messages(
    user_goal: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    variant: str = "plain",
) -> list[dict[str, Any]]:
    """The opening conversation for a mock-store episode. ``variant`` adds a prompt addendum."""
    return [
        {"role": "system", "content": system_prompt_for(variant, base=system_prompt)},
        {"role": "user", "content": user_goal},
    ]


def render_final_message(status: EpisodeStatus, records: Sequence[Mapping[str, Any]]) -> str | None:
    """The controller's final text for budget_exhausted and escalated, from journal records."""
    if status is EpisodeStatus.BUDGET_EXHAUSTED:
        check = _last(records, "budget_checked", lambda record: record.get("exhausted"))
        stopped = _budget_line(check) if check else "Stopped: budget exhausted."
        return "\n".join([stopped, _tried_line(records), f"Incomplete work: {_goal(records)}."])
    if status is EpisodeStatus.ESCALATED:
        return "\n".join(["I couldn't complete this.", _tried_line(records), _escalation_line(records)])
    return None


def _budget_line(check: Mapping[str, Any]) -> str:
    budgets = check["budgets"]
    fired = check["exhausted"]
    parts = [
        _count(check["steps"], budgets["max_steps"], fired == "step", "steps"),
        _count(check["prompt_tokens"], budgets["max_prompt_tokens"], fired == "prompt_token", "prompt tokens"),
        _count(
            check["completion_tokens"],
            budgets["max_completion_tokens"],
            fired == "completion_token",
            "completion tokens",
        ),
    ]
    if check.get("elapsed_s") is not None:
        elapsed = f"{check['elapsed_s']:.1f}"
        if fired == "wall_clock":
            elapsed += f"/{budgets['max_wall_clock_s']:.1f}"
        parts.append(f"{elapsed}s")
    name = _BUDGET_NAMES.get(fired, str(fired))
    return f"Stopped: {name} budget exhausted ({', '.join(parts)})."


def _count(value: int, limit: int, fired: bool, unit: str) -> str:
    return f"{value}/{limit} {unit}" if fired or unit == "steps" else f"{value} {unit}"


def _tried_line(records: Sequence[Mapping[str, Any]]) -> str:
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        if record.get("event") == "tool_result":
            outcome = "ok" if record.get("ok") else (record.get("error") or {}).get("error_type", "error")
            key = (str(record.get("tool")), str(outcome))
        elif record.get("event") == "policy_attempt":
            key = (str(record.get("tool")), "ok" if record.get("ok") else str(record.get("error_type")))
        elif record.get("event") == "observation" and (record.get("observation") or {}).get("error_type"):
            key = ("unparsed model output", str(record["observation"]["error_type"]))
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "Tried: no tool calls."
    return "Tried: " + ", ".join(f"{tool} ({outcome} x{count})" for (tool, outcome), count in counts.items()) + "."


def _escalation_line(records: Sequence[Mapping[str, Any]]) -> str:
    for record in reversed(records):
        if record.get("event") == "guard_tripped" and record.get("action") == "escalate":
            arguments = one_line(record.get("arguments", ""), _MAX_ARGUMENT_CHARS)
            last = record.get("last_outcome")
            outcome = f" (last result: {last})" if last == "ok" else f" (last error: {last})" if last else ""
            return (
                "Stopped because the same call was repeated after a correction hint: "
                f"{record.get('tool')} {arguments}{outcome}."
            )
        if record.get("event") == "escalated":
            reason = one_line(record.get("reason", "the task cannot proceed")).rstrip(".")
            return f"Stopped because {reason}."
    return "Stopped because the task cannot proceed."


def _goal(records: Sequence[Mapping[str, Any]]) -> str:
    started = next((record for record in records if record.get("event") == "episode_started"), None)
    messages = (started or {}).get("messages", [])
    goal = next(
        (message.get("content") for message in reversed(messages) if message.get("role") == "user"),
        None,
    )
    return one_line(goal, _MAX_GOAL_CHARS).rstrip(".") if goal else "the user's request"


def _last(records: Sequence[Mapping[str, Any]], event: str, predicate: Any) -> Mapping[str, Any] | None:
    for record in reversed(records):
        if record.get("event") == event and predicate(record):
            return record
    return None


__all__ = [
    "FAILURE_AWARE_ADDENDUM",
    "PARAPHRASE_REQUEST",
    "PROMPT_VARIANTS",
    "SYSTEM_PROMPT",
    "initial_messages",
    "render_final_message",
    "system_prompt_for",
]

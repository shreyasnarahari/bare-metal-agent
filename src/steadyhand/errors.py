"""The typed error observation: what the model sees when a call fails.

This is the agent-computer interface. Every failure says which kind of error
it is, which fields were wrong, whether calling again can help, and one line
of advice. Text is collapsed to one line and capped, so a traceback or a wall
of model output never reaches the model through ``hint`` or ``details``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

MAX_TEXT_CHARS = 300
MAX_GOT_CHARS = 200
DEFAULT_MAX_ERRORS = 5
# The shim bounds quoted output with RepairPolicy.max_output_chars (12,000).
# Steadyhand uses the same truncation but a smaller cap: a llama-server slot
# holds about 4,096 tokens at the Day 10 settings, shared by the whole episode.
MAX_QUOTED_OUTPUT_CHARS = 2_000


class ErrorType(str, Enum):
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    PARSE_ERROR = "parse_error"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    HANDLER_ERROR = "handler_error"
    REPEATED_CALL = "repeated_call"
    # Later parts add rate_limit, server_error, permanent_unavailable,
    # outcome_unknown, and suspicious_result.


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING: Any = _Missing()


@dataclass(frozen=True)
class FieldProblem:
    """One field-level problem. ``got`` is omitted when there is no value to show."""

    field: str | None
    problem: str
    got: Any = MISSING

    def as_dict(self) -> dict[str, Any]:
        detail: dict[str, Any] = {}
        if self.field is not None:
            detail["field"] = self.field
        detail["problem"] = one_line(self.problem)
        if self.got is not MISSING:
            detail["got"] = _bounded_value(self.got)
        return detail


@dataclass(frozen=True)
class ErrorObservation:
    error_type: ErrorType
    tool: str | None
    retryable: bool
    hint: str
    details: tuple[FieldProblem, ...] = ()
    previous_output: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "hint", one_line(self.hint))
        if not self.hint:
            raise ValueError("An error observation needs a non-empty hint")

    def as_dict(self) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "ok": False,
            "error_type": self.error_type.value,
            "tool": self.tool,
            "retryable": self.retryable,
            "hint": self.hint,
        }
        if self.details:
            observation["details"] = [detail.as_dict() for detail in self.details]
        if self.previous_output is not None:
            observation["previous_output"] = self.previous_output
        return observation


def cap_problems(
    problems: Iterable[FieldProblem],
    max_errors: int = DEFAULT_MAX_ERRORS,
) -> tuple[FieldProblem, ...]:
    """Drop duplicate problems and keep at most ``max_errors`` of them, in order."""
    if max_errors < 1:
        raise ValueError("max_errors must be positive")
    kept: list[FieldProblem] = []
    seen: set[tuple[Any, ...]] = set()
    for problem in problems:
        key = (problem.field, one_line(problem.problem), repr(problem.got))
        if key in seen:
            continue
        seen.add(key)
        kept.append(problem)
        if len(kept) == max_errors:
            break
    return tuple(kept)


def one_line(text: Any, limit: int = MAX_TEXT_CHARS) -> str:
    """Collapse whitespace and cap the length."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def bounded(text: str, limit: int = MAX_QUOTED_OUTPUT_CHARS) -> str:
    """Keep a prefix of quoted model output, marked when cut, like the shim's repair prompt."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated by Steadyhand]"


def _bounded_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_GOT_CHARS:
        return value[: MAX_GOT_CHARS - 3] + "..."
    if isinstance(value, (dict, list)):
        return one_line(value, MAX_GOT_CHARS)
    return value


__all__ = [
    "DEFAULT_MAX_ERRORS",
    "ErrorObservation",
    "ErrorType",
    "FieldProblem",
    "MAX_QUOTED_OUTPUT_CHARS",
    "MISSING",
    "bounded",
    "cap_problems",
    "one_line",
]

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
    """The kinds of failure the model can be told about; each has its own recovery rules."""

    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    PARSE_ERROR = "parse_error"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    HANDLER_ERROR = "handler_error"
    REPEATED_CALL = "repeated_call"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    PERMANENT_UNAVAILABLE = "permanent_unavailable"
    OUTCOME_UNKNOWN = "outcome_unknown"
    SUSPICIOUS_RESULT = "suspicious_result"
    # Bulkhead security errors. None is retryable, and recovery never acts on one.
    PERMISSION_DENIED = "permission_denied"  # policy said DENY; the model may choose another action
    PERMISSION_REQUIRED = "permission_required"  # policy said ASK; the episode escalates for approval
    SANDBOX_VIOLATION = "sandbox_violation"  # the sandbox refused or stopped the call


# A sentinel meaning "no value to show", distinct from None (which is a value worth showing).
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
        """The JSON shown to the model; long text and values are shortened."""
        detail: dict[str, Any] = {}
        if self.field is not None:
            detail["field"] = self.field
        detail["problem"] = one_line(self.problem)
        if self.got is not MISSING:
            detail["got"] = _bounded_value(self.got)
        return detail

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FieldProblem:
        """Rebuild a detail from ``as_dict`` output (used to replay recorded backend results)."""
        return cls(value.get("field"), value["problem"], value["got"] if "got" in value else MISSING)


@dataclass(frozen=True)
class ErrorObservation:
    """A typed failure as the model sees it: what went wrong, whether retrying can help, and a hint."""

    error_type: ErrorType
    tool: str | None
    retryable: bool
    hint: str
    details: tuple[FieldProblem, ...] = ()
    previous_output: str | None = None
    # The raw_error baseline shows the model a bare string instead of this shape.
    raw: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "hint", one_line(self.hint))
        if not self.hint:
            raise ValueError("An error observation needs a non-empty hint")

    def as_dict(self) -> dict[str, Any]:
        """The typed observation: ok=false, error_type, tool, retryable, hint, and optional details."""
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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ErrorObservation:
        """Rebuild an observation from ``as_dict`` output; ``as_dict`` of the result is identical."""
        return cls(
            error_type=ErrorType(value["error_type"]),
            tool=value["tool"],
            retryable=value["retryable"],
            hint=value["hint"],
            details=tuple(FieldProblem.from_dict(detail) for detail in value.get("details", ())),
            previous_output=value.get("previous_output"),
        )

    def model_dict(self) -> dict[str, Any]:
        """What the model sees: the typed shape, or for the raw_error baseline only a string."""
        if not self.raw:
            return self.as_dict()
        return {"ok": False, "error": self.raw_text()}

    def raw_text(self) -> str:
        """The bare error string the raw_error baseline shows instead (no type, no hint)."""
        parts = [f"{detail.field}: {detail.problem}" if detail.field else detail.problem for detail in self.details]
        return one_line("; ".join(parts) if parts else self.error_type.value.replace("_", " "))


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


# Shorten a "got" value for display: long strings are cut, containers become one short line.
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

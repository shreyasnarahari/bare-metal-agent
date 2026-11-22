"""The episode controller: a state machine, budgets, and the repeated-call guard.

``transition`` is a pure ``(state, event) -> state`` table. Every pair it does
not list raises ``ControllerError``, so a terminal state can never call the
model again by accident.

Budgets are checked before each model call. Step and token budgets are episode
sums. The wall clock is optional and off by default: llama-server decodes slots
together, so one episode's wall time grows with ``parallel`` even when its
token count does not. When it is set, it starts when ``run_episode`` starts on
a worker, so queue time before that never counts.

The guard hashes ``(tool, normalized arguments)`` with ``canonical_hash`` and
counts every attempted call, including ones that failed validation.
"""

from __future__ import annotations

import math
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from toolsmith.evaluation.models import canonical_hash
from toolsmith.json_utils import StrictJSONError, strict_json_loads

from .errors import ErrorObservation, ErrorType, FieldProblem
from .types import EpisodeStatus, ModelTurn, ToolCall, ToolResult

DEFAULT_GUARD_LIMIT = 3
_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp", "Zs"})


class ControllerError(RuntimeError):
    """An event that the state machine does not allow in the current state."""


class Event(str, Enum):
    TEXT_ANSWER = "text_answer"  # assistant text with a visible character
    STEP_COMPLETED = "step_completed"  # a tool turn or a parse-error turn finished
    BUDGET_CROSSED = "budget_crossed"  # checked before the next model call
    REPEAT_LIMIT = "repeat_limit"  # the guard blocked a call; its observation goes back
    REPEAT_AFTER_HINT = "repeat_after_hint"  # the model sent a blocked call again
    ESCALATE = "escalate"  # the task cannot proceed (part 6 supplies the signal)
    BACKEND_FAILURE = "backend_failure"  # not a model-output error
    CANCELLED = "cancelled"


RUNNING = EpisodeStatus.RUNNING
TRANSITIONS: dict[tuple[EpisodeStatus, Event], EpisodeStatus] = {
    (RUNNING, Event.TEXT_ANSWER): EpisodeStatus.DONE,
    (RUNNING, Event.STEP_COMPLETED): RUNNING,
    (RUNNING, Event.BUDGET_CROSSED): EpisodeStatus.BUDGET_EXHAUSTED,
    (RUNNING, Event.REPEAT_LIMIT): RUNNING,
    (RUNNING, Event.REPEAT_AFTER_HINT): EpisodeStatus.ESCALATED,
    (RUNNING, Event.ESCALATE): EpisodeStatus.ESCALATED,
    (RUNNING, Event.BACKEND_FAILURE): EpisodeStatus.ABORTED,
    (RUNNING, Event.CANCELLED): EpisodeStatus.ABORTED,
}


def transition(state: EpisodeStatus, event: Event) -> EpisodeStatus:
    try:
        return TRANSITIONS[(state, event)]
    except KeyError:
        raise ControllerError(f"No transition from {state.value!r} on {event.value!r}") from None


def has_visible_character(text: str | None) -> bool:
    """Toolsmith's plain-text rule: whitespace-only (or invisible-only) text is not an answer."""
    if not text:
        return False
    return any(
        not character.isspace() and unicodedata.category(character) not in _INVISIBLE_CATEGORIES
        for character in text
    )


@dataclass(frozen=True)
class Budgets:
    max_steps: int = 8
    max_prompt_tokens: int = 24_000
    max_completion_tokens: int = 8_000
    max_wall_clock_s: float | None = None  # single-slot debug runs only

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_prompt_tokens", "max_completion_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        wall = self.max_wall_clock_s
        if wall is not None and (not math.isfinite(wall) or wall <= 0):
            raise ValueError("max_wall_clock_s must be a positive finite number or None")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "max_wall_clock_s": self.max_wall_clock_s,
        }


@dataclass(frozen=True)
class BudgetCheck:
    """The numbers behind one continue-or-stop decision."""

    steps: int
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float | None
    budgets: Budgets
    exhausted: str | None = None  # "step", "prompt_token", "completion_token", "wall_clock"
    slot_overflow: str | None = None  # why the next prompt cannot fit the server slot

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            # Unrounded, so a replayed clock reproduces the exact decision.
            "elapsed_s": self.elapsed_s,
            "budgets": self.budgets.as_dict(),
            "exhausted": self.exhausted,
            "slot_overflow": self.slot_overflow,
        }


def normalize_arguments(value: Any) -> Any:
    """Equal-as-decimal numbers hash alike (``120`` and ``120.0``); strings lose outer space."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [normalize_arguments(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_arguments(item) for key, item in value.items()}
    return value


def call_hash(name: str, arguments: str) -> str:
    """Hash the decoded, normalized arguments, never the raw text, so key order cannot matter."""
    try:
        decoded: Any = normalize_arguments(strict_json_loads(arguments))
    except StrictJSONError:
        decoded = {"__unparsed_arguments__": arguments.strip()}
    return canonical_hash([name, decoded])


@dataclass(frozen=True)
class GuardDecision:
    blocked: bool
    key: str
    count: int
    limit: int


class LoopGuard:
    """Counts identical attempted calls per episode. The ``limit``-th sighting is blocked."""

    def __init__(self, limit: int = DEFAULT_GUARD_LIMIT) -> None:
        if limit < 1:
            raise ValueError("The guard limit must be at least 1")
        self.limit = limit
        self._counts: dict[str, int] = {}
        self._blocked: set[str] = set()
        self._last_outcome: dict[str, str] = {}

    def key(self, call: ToolCall) -> str:
        return call_hash(call.name, call.arguments)

    def is_blocked(self, call: ToolCall) -> bool:
        """True when this exact call was already blocked, so sending it again escalates."""
        return self.key(call) in self._blocked

    def count(self, call: ToolCall) -> int:
        return self._counts.get(self.key(call), 0)

    def observe(self, call: ToolCall, *, via: str = "model") -> GuardDecision:
        """Count a model-issued call. ``via="policy"`` retries (part 6) are seen but never counted."""
        key = self.key(call)
        if via == "policy":
            return GuardDecision(blocked=False, key=key, count=self._counts.get(key, 0), limit=self.limit)
        if via != "model":
            raise ValueError(f"via must be 'model' or 'policy', not {via!r}")
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        blocked = count >= self.limit
        if blocked:
            self._blocked.add(key)
        return GuardDecision(blocked=blocked, key=key, count=count, limit=self.limit)

    def record_outcome(self, key: str, result: ToolResult) -> None:
        """Remember how an executed or rejected call ended. Blocked calls are not outcomes."""
        self._last_outcome[key] = "ok" if result.ok else result.error.error_type.value  # type: ignore[union-attr]

    def last_outcome(self, key: str) -> str | None:
        return self._last_outcome.get(key)


def repeated_call_observation(tool: str, last_outcome: str | None) -> ErrorObservation:
    if last_outcome == "ok":
        hint = (
            "This exact call already succeeded. Do not repeat it; use the earlier result, "
            "change the arguments, or ask the user."
        )
    else:
        hint = "This exact call already failed. Choose a different tool, change the arguments, or ask the user."
    details = (FieldProblem(None, f"last outcome: {last_outcome}"),) if last_outcome else ()
    return ErrorObservation(
        error_type=ErrorType.REPEATED_CALL,
        tool=tool,
        retryable=False,
        hint=hint,
        details=details,
    )


class EpisodeController:
    """Owns one episode's state, counters, budgets, and guard. Not shared between episodes."""

    def __init__(
        self,
        budgets: Budgets | None = None,
        *,
        guard_limit: int = DEFAULT_GUARD_LIMIT,
        clock: Callable[[], float] = time.monotonic,
        slot_context: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.budgets = budgets or Budgets()
        self.guard = LoopGuard(guard_limit)
        self.state = EpisodeStatus.RUNNING
        self.steps = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_prompt_tokens = 0
        self.slot_context = slot_context
        self.max_tokens = max_tokens
        self.error_code: str | None = None
        self.error_message: str | None = None
        self.escalation_reason: str | None = None
        self._clock = clock
        self._started = clock() if self.budgets.max_wall_clock_s is not None else None

    @property
    def running(self) -> bool:
        return self.state is EpisodeStatus.RUNNING

    def apply(self, event: Event) -> EpisodeStatus:
        self.state = transition(self.state, event)
        return self.state

    def elapsed_s(self) -> float | None:
        return None if self._started is None else self._clock() - self._started

    def check_budgets(self) -> BudgetCheck:
        """Decide, before a model call, whether that call may start."""
        budgets = self.budgets
        elapsed = self.elapsed_s()
        exhausted = None
        if self.steps >= budgets.max_steps:
            exhausted = "step"
        elif self.prompt_tokens >= budgets.max_prompt_tokens:
            exhausted = "prompt_token"
        elif self.completion_tokens >= budgets.max_completion_tokens:
            exhausted = "completion_token"
        elif elapsed is not None and budgets.max_wall_clock_s is not None and elapsed >= budgets.max_wall_clock_s:
            exhausted = "wall_clock"
        slot_overflow = None
        if (
            exhausted is None
            and self.slot_context is not None
            and self.max_tokens is not None
            and self.last_prompt_tokens + self.max_tokens > self.slot_context
        ):
            # The next prompt is at least as long as the last one: history only grows.
            slot_overflow = (
                f"the next prompt (at least {self.last_prompt_tokens} tokens) plus max_tokens "
                f"{self.max_tokens} exceeds the {self.slot_context}-token server slot"
            )
        return BudgetCheck(
            steps=self.steps,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            elapsed_s=elapsed,
            budgets=budgets,
            exhausted=exhausted,
            slot_overflow=slot_overflow,
        )

    def record_usage(self, turn: ModelTurn) -> None:
        self.prompt_tokens += turn.prompt_tokens
        self.completion_tokens += turn.completion_tokens
        if turn.prompt_tokens:
            self.last_prompt_tokens = turn.prompt_tokens

    def finish_step(self) -> None:
        self.steps += 1
        self.apply(Event.STEP_COMPLETED)

    def answer(self) -> None:
        self.apply(Event.TEXT_ANSWER)

    def exhaust(self) -> None:
        self.apply(Event.BUDGET_CROSSED)

    def trip_guard(self) -> None:
        self.apply(Event.REPEAT_LIMIT)

    def escalate_repeat(self, reason: str) -> None:
        self.apply(Event.REPEAT_AFTER_HINT)
        self.escalation_reason = reason

    def escalate(self, reason: str) -> None:
        """The primitive later recovery policies call when the task cannot proceed."""
        self.apply(Event.ESCALATE)
        self.escalation_reason = reason

    def abort(self, code: str, message: str | None = None) -> None:
        self.apply(Event.BACKEND_FAILURE)
        self.error_code = code
        self.error_message = message

    def cancel(self) -> None:
        self.apply(Event.CANCELLED)
        self.error_code = "cancelled"


__all__ = [
    "BudgetCheck",
    "Budgets",
    "ControllerError",
    "DEFAULT_GUARD_LIMIT",
    "EpisodeController",
    "Event",
    "GuardDecision",
    "LoopGuard",
    "TRANSITIONS",
    "call_hash",
    "has_visible_character",
    "normalize_arguments",
    "repeated_call_observation",
    "transition",
]

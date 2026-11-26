"""Value types shared by the Steadyhand client, executor, loop, and journal."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .errors import ErrorObservation

if TYPE_CHECKING:
    from .faults import InjectedFault


class TurnKind(str, Enum):
    """What one model turn was: a final answer, tool calls, or a failure."""

    TEXT = "text"
    TOOL_CALLS = "tool_calls"
    # The model's output failed to parse or validate. The model can repair it,
    # so the loop shows it a parse_error observation and continues.
    PARSE_ERROR = "parse_error"
    # Backend, request, or truncation failure. The model cannot repair it, so
    # the episode ends as aborted.
    FATAL_ERROR = "fatal_error"


class EpisodeStatus(str, Enum):
    """Where an episode is: running, or one of its four endings."""

    RUNNING = "running"
    DONE = "done"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ESCALATED = "escalated"
    ABORTED = "aborted"


@dataclass(frozen=True)
class ToolCall:
    """One native tool call. ``arguments`` stays the JSON text the shim emitted."""

    id: str
    name: str
    arguments: str

    @classmethod
    def from_openai(cls, call: Mapping[str, Any]) -> ToolCall:
        """From an OpenAI ``tool_calls`` entry; dict arguments are re-encoded as compact JSON."""
        function = call.get("function") or {}
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        return cls(
            id=str(call.get("id") or ""),
            name=str(function.get("name") or ""),
            arguments=arguments,
        )

    def to_openai(self) -> dict[str, Any]:
        """Back to an OpenAI ``tool_calls`` entry."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


# Backend fields that are counts (compared exactly in replay); the rest are timings (volatile).
COUNT_FIELDS = frozenset({
    "prompt_n", "predicted_n", "cache_n", "tokens_evaluated", "tokens_cached", "tokens_predicted", "id_slot",
})


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _milliseconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value < 0:
        return None
    return float(value)


@dataclass(frozen=True)
class AttemptMetrics:
    """What one generation attempt measured, from Toolsmith's sanitized ``toolsmith.attempts`` entry.

    Fields stay ``None`` when the server did not report them; they never become zero.
    ``prompt_evaluated_tokens`` is llama.cpp's ``timings.prompt_n`` and
    ``prompt_total_tokens`` its ``tokens_evaluated``; which of them means "newly
    evaluated" must be confirmed on the pinned build (``tidemark.telemetry``).
    """

    attempt: int
    outcome: str | None = None
    measurement_path: str = "unavailable"
    prompt_evaluated_tokens: int | None = None
    prompt_total_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_ms: float | None = None
    predicted_ms: float | None = None
    response_elapsed_ms: float | None = None
    ttft_ms: float | None = None
    slot_id: int | None = None
    raw: dict[str, int | float] = field(default_factory=dict)

    @classmethod
    def from_toolsmith(cls, value: Mapping[str, Any]) -> AttemptMetrics:
        """From one ``toolsmith.attempts`` entry (``toolsmith.timings.AttemptTiming.as_dict``)."""
        backend = value.get("backend") if isinstance(value.get("backend"), Mapping) else {}
        raw = {
            str(name): item for name, item in backend.items()
            if (_count(item) if name in COUNT_FIELDS else _milliseconds(item)) is not None
        }
        cached = raw.get("cache_n", raw.get("tokens_cached"))
        path = value.get("measurement_path")
        return cls(
            attempt=int(value.get("attempt") or 0),
            outcome=value.get("outcome") if isinstance(value.get("outcome"), str) else None,
            measurement_path=path if path in ("buffered_stream", "unavailable") else "unavailable",
            prompt_evaluated_tokens=_count(raw.get("prompt_n")),
            prompt_total_tokens=_count(raw.get("tokens_evaluated")),
            cached_tokens=_count(cached),
            completion_tokens=_count(raw.get("predicted_n", raw.get("tokens_predicted"))),
            prompt_ms=_milliseconds(raw.get("prompt_ms")),
            predicted_ms=_milliseconds(raw.get("predicted_ms")),
            response_elapsed_ms=_milliseconds(value.get("response_elapsed_ms")),
            ttft_ms=_milliseconds(value.get("ttft_ms")) if path == "buffered_stream" else None,
            slot_id=_count(raw.get("id_slot")),
            raw=raw,
        )

    def counts(self) -> dict[str, Any]:
        """The deterministic part, compared exactly when a journal replays."""
        return {
            "attempt": self.attempt,
            "outcome": self.outcome,
            "measurement_path": self.measurement_path,
            "prompt_evaluated_tokens": self.prompt_evaluated_tokens,
            "prompt_total_tokens": self.prompt_total_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "slot_id": self.slot_id,
            "raw_counts": {name: item for name, item in sorted(self.raw.items()) if name in COUNT_FIELDS},
        }

    def timings(self) -> dict[str, Any]:
        """The volatile part: milliseconds and rates, which differ on every run."""
        return {
            "attempt": self.attempt,
            "prompt_ms": self.prompt_ms,
            "predicted_ms": self.predicted_ms,
            "response_elapsed_ms": self.response_elapsed_ms,
            "ttft_ms": self.ttft_ms,
            "raw_timings": {name: item for name, item in sorted(self.raw.items()) if name not in COUNT_FIELDS},
        }

    @classmethod
    def from_journal(cls, counts: Mapping[str, Any], timings: Mapping[str, Any] | None) -> AttemptMetrics:
        """Rebuild from a ``backend_timings`` record's ``attempt_metrics`` and ``attempt_timings`` entries."""
        timings = timings or {}
        raw = {**dict(counts.get("raw_counts") or {}), **dict(timings.get("raw_timings") or {})}
        return cls(
            attempt=counts["attempt"],
            outcome=counts.get("outcome"),
            measurement_path=counts.get("measurement_path", "unavailable"),
            prompt_evaluated_tokens=counts.get("prompt_evaluated_tokens"),
            prompt_total_tokens=counts.get("prompt_total_tokens"),
            cached_tokens=counts.get("cached_tokens"),
            completion_tokens=counts.get("completion_tokens"),
            prompt_ms=timings.get("prompt_ms"),
            predicted_ms=timings.get("predicted_ms"),
            response_elapsed_ms=timings.get("response_elapsed_ms"),
            ttft_ms=timings.get("ttft_ms"),
            slot_id=counts.get("slot_id"),
            raw=raw,
        )


@dataclass(frozen=True)
class TurnMetrics:
    """Typed measurements for one model turn; every field is optional.

    The first attempt carries the turn's prefill (``prompt_ms``) and TTFT. Repair
    attempts are kept separately in ``attempts`` and never folded into those.
    ``response_elapsed_ms`` is the whole ``complete()`` call, every attempt
    included; it is never relabeled TTFT. ``rendered_prompt_tokens`` comes from
    a context session, never from the server.
    """

    attempts: tuple[AttemptMetrics, ...] = ()
    response_elapsed_ms: float | None = None
    generation_path: str | None = None
    rendered_prompt_tokens: int | None = None

    @property
    def first(self) -> AttemptMetrics | None:
        return self.attempts[0] if self.attempts else None

    @property
    def measurement_path(self) -> str:
        return self.first.measurement_path if self.first is not None else "unavailable"

    @property
    def prompt_evaluated_tokens(self) -> int | None:
        return self.first.prompt_evaluated_tokens if self.first is not None else None

    @property
    def prompt_ms(self) -> float | None:
        return self.first.prompt_ms if self.first is not None else None

    @property
    def predicted_ms(self) -> float | None:
        return self.first.predicted_ms if self.first is not None else None

    @property
    def ttft_ms(self) -> float | None:
        return self.first.ttft_ms if self.first is not None else None

    @property
    def slot_id(self) -> int | None:
        return self.attempts[-1].slot_id if self.attempts else None

    @property
    def repair_attempts(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def completion_tokens(self) -> int | None:
        values = [attempt.completion_tokens for attempt in self.attempts]
        return None if not values or any(value is None for value in values) else sum(values)  # type: ignore[misc]

    @classmethod
    def from_toolsmith(
        cls, attempts: Sequence[Mapping[str, Any]], *, response_elapsed_ms: float | None, generation_path: str | None,
    ) -> TurnMetrics:
        return cls(
            attempts=tuple(AttemptMetrics.from_toolsmith(item) for item in attempts if isinstance(item, Mapping)),
            response_elapsed_ms=response_elapsed_ms,
            generation_path=generation_path,
        )


@dataclass(frozen=True)
class ModelTurn:
    """One model response as the loop sees it: the assistant message, its calls, usage, or an error."""

    kind: TurnKind
    assistant_message: dict[str, Any] | None
    tool_calls: tuple[ToolCall, ...]
    content: str | None
    reasoning_content: str | None
    prompt_tokens: int
    completion_tokens: int
    error_code: str | None
    error_message: str | None = None
    repair_attempts: int = 0
    elapsed_s: float = 0.0
    raw_output: str | None = None
    # Optional typed backend measurements (Tidemark); None on journals and fakes that predate them.
    metrics: TurnMetrics | None = None

    @classmethod
    def failure(
        cls,
        kind: TurnKind,
        code: str,
        message: str,
        *,
        raw_output: str | None = None,
        repair_attempts: int = 0,
        elapsed_s: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        metrics: TurnMetrics | None = None,
    ) -> ModelTurn:
        """A failed turn (parse error or fatal error) with no message and no calls."""
        if kind not in (TurnKind.PARSE_ERROR, TurnKind.FATAL_ERROR):
            raise ValueError(f"{kind} is not a failure kind")
        return cls(
            kind=kind,
            assistant_message=None,
            tool_calls=(),
            content=None,
            reasoning_content=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_code=code,
            error_message=message,
            repair_attempts=repair_attempts,
            elapsed_s=elapsed_s,
            raw_output=raw_output,
            metrics=metrics,
        )

    def with_call_ids(self, step: int) -> ModelTurn:
        """Give every id-less call ``call_{step}_{index}``, in the calls and the message."""
        if all(call.id for call in self.tool_calls):
            return self
        calls = tuple(
            call if call.id else replace(call, id=f"call_{step}_{index}")
            for index, call in enumerate(self.tool_calls)
        )
        message = deepcopy(self.assistant_message) if self.assistant_message else {}
        message["tool_calls"] = [call.to_openai() for call in calls]
        return replace(self, tool_calls=calls, assistant_message=message)


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one call: a value, or a typed error from validation or the handler."""

    call_id: str
    tool: str
    arguments: str
    arguments_sha256: str | None
    ok: bool
    value: Any = None
    error: ErrorObservation | None = None
    handler_called: bool = False
    elapsed_s: float = 0.0
    # Set when the fault injector touched this call. Journaled as its own
    # injected_fault event; never part of what the model sees.
    fault: InjectedFault | None = None

    def __post_init__(self) -> None:
        if self.ok == (self.error is not None):
            raise ValueError("A successful result has no error; a failed one needs one")

    def observation(self) -> dict[str, Any]:
        """The JSON object the model sees. The success shape is stable across parts."""
        if self.ok:
            return {"ok": True, "value": self.value}
        assert self.error is not None
        return self.error.model_dict()

    def content(self) -> str:
        """The observation as the text of a ``tool`` message."""
        return observation_content(self.observation())


def observation_content(observation: Mapping[str, Any]) -> str:
    """Serialize an observation for a message. Sorted keys keep journals stable."""
    return json.dumps(observation, ensure_ascii=False, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class EpisodeResult:
    """How an episode ended: status, final conversation and answer, token totals, and its journal."""

    status: EpisodeStatus
    messages: list[dict[str, Any]]
    steps: int
    final_content: str | None
    prompt_tokens: int
    completion_tokens: int
    journal_path: Path
    episode_id: str | None = None
    error_code: str | None = None


class ModelClient(Protocol):
    """What ``run_episode`` needs from a model. ``SteadyhandClient`` is the real one."""

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ModelTurn: ...

    def settings(self) -> dict[str, Any]: ...


__all__ = [
    "AttemptMetrics",
    "EpisodeResult",
    "EpisodeStatus",
    "ModelClient",
    "ModelTurn",
    "ToolCall",
    "ToolResult",
    "TurnKind",
    "TurnMetrics",
    "observation_content",
]

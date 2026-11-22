"""Value types shared by the Steadyhand client, executor, loop, and journal."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .errors import ErrorObservation

if TYPE_CHECKING:
    from .faults import InjectedFault


class TurnKind(str, Enum):
    TEXT = "text"
    TOOL_CALLS = "tool_calls"
    # The model's output failed to parse or validate. The model can repair it,
    # so the loop shows it a parse_error observation and continues.
    PARSE_ERROR = "parse_error"
    # Backend, request, or truncation failure. The model cannot repair it, so
    # the episode ends as aborted.
    FATAL_ERROR = "fatal_error"


class EpisodeStatus(str, Enum):
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
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class ModelTurn:
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
    ) -> ModelTurn:
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
        return observation_content(self.observation())


def observation_content(observation: Mapping[str, Any]) -> str:
    """Serialize an observation for a message. Sorted keys keep journals stable."""
    return json.dumps(observation, ensure_ascii=False, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class EpisodeResult:
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
    "EpisodeResult",
    "EpisodeStatus",
    "ModelClient",
    "ModelTurn",
    "ToolCall",
    "ToolResult",
    "TurnKind",
    "observation_content",
]

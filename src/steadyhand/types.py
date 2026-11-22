"""Value types shared by the Steadyhand client, executor, loop, and journal."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class TurnKind(str, Enum):
    TEXT = "text"
    TOOL_CALLS = "tool_calls"
    PARSE_ERROR = "parse_error"


class EpisodeStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ABORTED = "aborted"
    # ESCALATED arrives in part 3.


class ToolErrorType(str, Enum):
    """Failure kinds in the part 1 observation. Part 2 replaces this with a typed channel."""

    TIMEOUT = "timeout"
    HANDLER_ERROR = "handler_error"
    NOT_FOUND = "not_found"


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

    @classmethod
    def parse_error(
        cls,
        code: str,
        message: str,
        *,
        repair_attempts: int = 0,
        elapsed_s: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> ModelTurn:
        return cls(
            kind=TurnKind.PARSE_ERROR,
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
    call_id: str
    tool: str
    arguments: str
    arguments_sha256: str | None
    ok: bool
    value: Any = None
    error_type: ToolErrorType | None = None
    message: str | None = None
    elapsed_s: float = 0.0

    def observation(self) -> dict[str, Any]:
        """The JSON object the model sees. The success shape is stable across parts."""
        if self.ok:
            return {"ok": True, "value": self.value}
        return {
            "ok": False,
            "error_type": self.error_type.value if self.error_type else None,
            "tool": self.tool,
            "message": self.message,
        }

    def content(self) -> str:
        return observation_content(self.observation())


def observation_content(observation: Mapping[str, Any]) -> str:
    """Serialize an observation for a ``tool`` message. Sorted keys keep journals stable."""
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
    "ToolErrorType",
    "ToolResult",
    "TurnKind",
    "observation_content",
]

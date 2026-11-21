from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from typing import Any

from .parsing import ParsedResponse
from .streaming import (
    ContentDelta,
    ParserEvent,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallStart,
)

DONE = "data: [DONE]\n\n"


def sse_json(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def sse_error(message: str, *, error_type: str, code: str) -> str:
    return sse_json(
        {"error": {"message": message, "type": error_type, "code": code}}
    )


class OpenAIStreamAdapter:
    def __init__(
        self,
        model: str,
        *,
        completion_id: str | None = None,
        created: int | None = None,
        include_usage: bool = False,
    ) -> None:
        self.completion_id = completion_id or f"chatcmpl-{uuid.uuid4().hex}"
        self.created = created or int(time.time())
        self.model = model
        self.include_usage = include_usage

    def role_chunk(self) -> dict[str, Any]:
        return self._chunk({"role": "assistant", "content": ""})

    def convert(self, event: ParserEvent) -> dict[str, Any] | None:
        if isinstance(event, ReasoningDelta):
            return self._chunk({"reasoning_content": event.text})
        if isinstance(event, ContentDelta):
            return self._chunk({"content": event.text})
        if isinstance(event, ToolCallStart):
            return self._chunk(
                {
                    "tool_calls": [
                        {
                            "index": event.index,
                            "id": event.id,
                            "type": "function",
                            "function": {"name": event.name, "arguments": ""},
                        }
                    ]
                }
            )
        if isinstance(event, ToolCallArgumentsDelta):
            return self._chunk(
                {
                    "tool_calls": [
                        {
                            "index": event.index,
                            "function": {"arguments": event.text},
                        }
                    ]
                }
            )
        if isinstance(event, ToolCallEnd):
            return None
        raise TypeError(f"Unsupported parser event: {type(event).__name__}")

    def finish_chunk(self, response: ParsedResponse) -> dict[str, Any]:
        return self._chunk(
            {},
            finish_reason="tool_calls" if response.tool_calls else "stop",
        )

    def usage_chunk(self, usage: dict[str, int]) -> dict[str, Any] | None:
        if not self.include_usage:
            return None
        return {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": usage,
        }

    def _chunk(
        self,
        delta: dict[str, Any],
        *,
        finish_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }


def aggregate_chunks(chunks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason = None
    usage = None

    for chunk in chunks:
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            # SDK dumps keep explicit nulls, so .get(key, []) still returns None.
            for call_delta in delta.get("tool_calls") or []:
                index = call_delta["index"]
                call = calls.setdefault(
                    index,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": None, "arguments": ""},
                    },
                )
                if call_delta.get("id") is not None:
                    call["id"] = call_delta["id"]
                function = call_delta.get("function") or {}
                if function.get("name") is not None:
                    call["function"]["name"] = function["name"]
                if function.get("arguments"):
                    call["function"]["arguments"] += function["arguments"]

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) or None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return {"message": message, "finish_reason": finish_reason, "usage": usage}

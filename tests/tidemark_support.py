"""Shared fakes for the Tidemark tests: a Qwen-like renderer, fake services, and canned completions."""

from __future__ import annotations

import json
from typing import Any

from toolsmith.protocols import QWEN3_PROFILE


def qwen_like_render(messages, tools=(), tool_choice="auto"):
    """A deterministic stand-in for the model's chat template (role markers, tools block, generation marker)."""
    parts = []
    if tools:
        parts.append("<|im_start|>system\n# Tools\n" + "\n".join(json.dumps(tool, sort_keys=True) for tool in tools) + "<|im_end|>\n")
    for message in messages:
        body = message.get("content") or ""
        for call in message.get("tool_calls") or []:
            function = call["function"]
            body += f"\n<tool_call>\n{{\"name\": \"{function['name']}\", \"arguments\": {function['arguments']}}}\n</tool_call>"
        role = message["role"]
        if role == "tool":
            body = f"<tool_response>\n{body}\n</tool_response>"
            role = "user"
        parts.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
    return "".join(parts) + "<|im_start|>assistant\n"


def completion(content=None, calls=(), prompt_tokens=50, completion_tokens=8, attempts=None, generation_path="non_streaming"):
    """An OpenAI-shaped shim response, optionally with Toolsmith's per-attempt evidence."""
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            for call_id, name, arguments in calls
        ]
    metadata: dict[str, Any] = {"repair_attempts": 0}
    if attempts is not None:
        metadata.update({"attempts": attempts, "generation_path": generation_path})
    return {
        "choices": [{"index": 0, "message": message}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        "toolsmith": metadata,
    }


def attempt(prompt_n=None, *, tokens_evaluated=None, predicted_n=8, prompt_ms=12.5, predicted_ms=40.0, ttft_ms=None,
            id_slot=None, outcome="success", index=0):
    """One ``toolsmith.attempts`` entry as ``AttemptTiming.as_dict`` writes it."""
    backend: dict[str, Any] = {"predicted_n": predicted_n, "prompt_ms": prompt_ms, "predicted_ms": predicted_ms}
    if prompt_n is not None:
        backend["prompt_n"] = prompt_n
    if tokens_evaluated is not None:
        backend["tokens_evaluated"] = tokens_evaluated
    if id_slot is not None:
        backend["id_slot"] = id_slot
    return {
        "attempt": index, "outcome": outcome,
        "measurement_path": "buffered_stream" if ttft_ms is not None else "unavailable",
        "response_elapsed_ms": 60.0, "ttft_ms": ttft_ms, "backend": backend,
    }


class FakeService:
    """Stands in for ToolsmithService: canned completions, a deterministic render_prompt, recorded requests."""

    profile = QWEN3_PROFILE
    use_grammars = True

    def __init__(self, *responses, generation_path="non_streaming"):
        self.responses = list(responses)
        self.requests = []
        self.slots = []
        self.generation_path = generation_path

    def chat(self, request, *, id_slot=None):
        self.requests.append(request)
        self.slots.append(id_slot)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def render_prompt(self, request):
        return qwen_like_render(request.messages, request.tools, request.tool_choice)

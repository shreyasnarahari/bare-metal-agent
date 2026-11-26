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
        self.allowed = []
        self.generation_path = generation_path

    def chat(self, request, *, id_slot=None, allowed_tool_names=None):
        self.requests.append(request)
        self.slots.append(id_slot)
        self.allowed.append(allowed_tool_names)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def render_prompt(self, request):
        return qwen_like_render(request.messages, request.tools, request.tool_choice)


# Part 3: shaping sessions, blob stores, and registries with large-output tools.


def shaping_session(root, namespace="ep-ns", *, read_only=False, policy=None, slot_context=16384, cache=None, pool=None,
                    tokenizer=None):
    """A prefix-stable session with an observation shaper over the blob store at ``root``."""
    from tidemark.assembler import PrefixStableSession, SpanRenderer
    from tidemark.blobs import BlobStore
    from tidemark.eval.prefix import ChatMLTemplate
    from tidemark.observations import DEFAULT_POLICY, ObservationConfig, ObservationShaper
    from tidemark.tokenizer import RegexTokenizer
    from tidemark.types import ContextConfig

    tokenizer = tokenizer or RegexTokenizer()
    store = BlobStore(root, read_only=read_only)
    config = ObservationConfig.for_store(store, namespace, policy or DEFAULT_POLICY)
    context = ContextConfig(slot_context=slot_context, max_tokens=256, strategy="prefix_stable",
                            tokenizer=tokenizer.identity(), observations=config.as_dict())
    shaper = ObservationShaper(config, tokenizer, store, cache=cache, pool=pool)
    renderer = SpanRenderer(ChatMLTemplate(), tokenizer, protocol=QWEN3_PROFILE.protocol)
    return PrefixStableSession(context, renderer, observations=shaper)


def tool_registry(*handlers, context_tools=True, secure=True):
    """The ten store tools, extra read tools with untrusted output (``handlers``), and ``read_blob``."""
    from bulkhead.manifest import SecuritySpec, Trust
    from steadyhand.registry import SteadyhandRegistry
    from steadyhand.tools import READ_POLICY, RESULT_SCHEMAS, TOOL_SECURITY, TOOL_SPECS
    from steadyhand.tools_context import register_context_tools

    registry = SteadyhandRegistry()
    for handler, policy, semantic in TOOL_SPECS:
        registry.register(handler, policy, semantic, RESULT_SCHEMAS[handler.__name__], security=TOOL_SECURITY[handler.__name__])
    for handler in handlers:
        spec = SecuritySpec(sandbox=None, output_trust=Trust.UNTRUSTED, sinks=()) if secure else None
        registry.register(handler, READ_POLICY, security=spec)
    if context_tools:
        register_context_tools(registry)
    return registry.freeze()


def big_log(lines=3000, critical_at=1500, critical="src/net/socket.c:88:12: error: 'fd' undeclared (first use in this function)"):
    rows = [f"step {i}: compiling module_{i}.c ok" for i in range(lines)]
    if critical_at is not None:
        rows[critical_at] = critical
    return "".join(row + "\n" for row in rows)

"""Turn a chat conversation (and tool definitions) into the model's prompt text.

The model's own Jinja chat template does the rendering, so tool instructions and
tool-result formatting match what the model was trained on. Two helpers prepare
messages first: ``normalize_history`` converts past tool-call arguments into the
shape a template expects, and ``with_response_schema`` adds a JSON Schema
instruction for structured responses.

``render_with_token_spans`` renders the same prompt as ``ToolsmithService``
(normalized history, the request's tools, the generation marker) and maps
logical message ranges to exact token spans, or reports that the template
makes that mapping ambiguous.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .protocols import ToolProtocol
from .validation import StrictJSONError, strict_json_loads


def normalize_history(
    messages: Sequence[Mapping[str, Any]],
    protocol: ToolProtocol,
) -> list[dict[str, Any]]:
    """A copy of the history in the form the protocol's chat template expects.

    OpenAI clients send tool-call ``arguments`` as a JSON string. The Qwen3.5
    template iterates arguments as a mapping (one ``<parameter>`` per key), so
    for that protocol each assistant call's arguments are decoded strictly into
    a dict. Qwen3's template takes the string as is.
    """
    normalized = deepcopy([dict(message) for message in messages])
    if protocol is not ToolProtocol.QWEN35:
        return normalized
    for message_index, message in enumerate(normalized):
        if message.get("role") != "assistant":
            continue
        for call_index, call in enumerate(message.get("tool_calls", [])):
            function = call.get("function", {})
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                raise ValueError(
                    f"Assistant message {message_index} call {call_index} needs string arguments"
                )
            try:
                decoded = strict_json_loads(arguments)
            except StrictJSONError as exc:
                raise ValueError(
                    f"Assistant message {message_index} call {call_index} has invalid arguments: {exc}"
                ) from exc
            if not isinstance(decoded, dict):
                raise ValueError(
                    f"Assistant message {message_index} call {call_index} arguments must be a JSON object"
                )
            function["arguments"] = decoded
    return normalized


def with_response_schema(
    messages: Sequence[Mapping[str, Any]],
    response_format: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """A copy of the messages with a system instruction to answer with JSON matching the schema.

    The instruction is appended to an existing system message, or becomes the
    first message. The schema text is compact JSON, so it costs few tokens.
    """
    structured = deepcopy([dict(message) for message in messages])
    specification = response_format["json_schema"]
    instruction = (
        "Return only a JSON value that conforms to the following JSON Schema. "
        "Do not wrap it in Markdown or add explanatory prose.\n"
        f"Schema name: {specification['name']}\n"
        + json.dumps(
            specification["schema"],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    if structured and structured[0].get("role") == "system":
        original = structured[0].get("content") or ""
        structured[0]["content"] = original + "\n\n" + instruction
    else:
        structured.insert(0, {"role": "system", "content": instruction})
    return structured


class ChatPromptRenderer:
    """Render messages and tools with a Hugging Face tokenizer's chat template."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        token: str | None = None,
        revision: str | None = None,
    ) -> ChatPromptRenderer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            revision=revision,
        )
        return cls(tokenizer)

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        *,
        enable_thinking: bool | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """The full prompt string, ending with the assistant turn the model should write.

        ``enable_thinking`` is passed to the template only when given, so a
        template's own default applies otherwise. ``add_generation_prompt=False``
        renders a conversation prefix (for token-span mapping), never a request.
        """
        template_kwargs: dict[str, Any] = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking

        prompt = self.tokenizer.apply_chat_template(
            deepcopy([dict(message) for message in messages]),
            tools=deepcopy([dict(tool) for tool in tools]) or None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        if not isinstance(prompt, str):
            raise TypeError("The chat template did not return a string")
        return prompt


SPANS_OK = "ok"
SPAN_MAPPING_UNAVAILABLE = "span_mapping_unavailable"


@dataclass(frozen=True)
class TokenSpans:
    """A rendered prompt, its token ids, and ``[start, end)`` token spans per labeled message range.

    ``status`` is ``span_mapping_unavailable`` when an intermediate render was
    not a text and token prefix of the next one (the template rewrote earlier
    content or merged tokens across a boundary); ``spans`` is then empty. Text,
    ids, and the total always come from the final render.
    """

    text: str
    token_ids: tuple[int, ...]
    spans: tuple[tuple[str, int, int], ...]
    status: str = SPANS_OK
    detail: str | None = None

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def token_ids_sha256(self) -> str:
        encoded = json.dumps(list(self.token_ids), separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def span(self, label: str) -> tuple[int, int] | None:
        return next(((start, end) for name, start, end in self.spans if name == label), None)


def request_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The tools exactly as ``ToolsmithService`` passes them to the template (``Tool`` round trip)."""
    from .registry import Tool

    return [Tool.from_openai(tool).to_openai() for tool in tools]


def render_with_token_spans(
    renderer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    boundaries: Sequence[tuple[str, int]],
    encode: Callable[[str], Sequence[int]],
    protocol: Any,
    enable_thinking: bool | None = None,
    render_prefix: Callable[[int], str] | None = None,
) -> TokenSpans:
    """Render ``messages`` as the shim does and map each labeled range to token spans.

    ``boundaries`` are ``(label, end)`` pairs with non-decreasing ``end``
    message indexes; each label covers the messages since the previous
    boundary. Every intermediate render omits the generation prompt; the final
    render adds it, exactly as ``ToolsmithService.chat``. History is
    normalized once for ``protocol``. ``render_prefix(end)`` may supply a
    cached intermediate render. ``encode`` is the parity-checked tokenizer.
    """
    normalized = normalize_history(messages, protocol)
    prepared_tools = request_tools(tools)

    def prefix_text(end: int) -> str:
        if render_prefix is not None:
            return render_prefix(end)
        if end == 0:
            return ""
        return renderer.render(normalized[:end], prepared_tools, enable_thinking=enable_thinking,
                               add_generation_prompt=False)

    text = renderer.render(normalized, prepared_tools, enable_thinking=enable_thinking)
    ids = tuple(int(i) for i in encode(text))
    spans: list[tuple[str, int, int]] = []
    previous_text, previous_ids, previous_end = "", (), 0
    for label, end in boundaries:
        if end < previous_end or end > len(normalized):
            raise ValueError(f"Boundary {label!r} at {end} is out of order or past the messages")
        if end == previous_end:
            spans.append((label, len(previous_ids), len(previous_ids)))
            continue
        current_text = prefix_text(end)
        current_ids = tuple(int(i) for i in encode(current_text))
        if not current_text.startswith(previous_text) or current_ids[: len(previous_ids)] != previous_ids:
            return TokenSpans(text, ids, (), SPAN_MAPPING_UNAVAILABLE,
                              f"the render through {label!r} does not extend the previous boundary's render")
        spans.append((label, len(previous_ids), len(current_ids)))
        previous_text, previous_ids, previous_end = current_text, current_ids, end
    if not text.startswith(previous_text) or ids[: len(previous_ids)] != previous_ids:
        return TokenSpans(text, ids, (), SPAN_MAPPING_UNAVAILABLE,
                          "the final render does not extend the last boundary's render")
    spans.append(("generation_prompt", len(previous_ids), len(ids)))
    return TokenSpans(text, ids, tuple(spans))

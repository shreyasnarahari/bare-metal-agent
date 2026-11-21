from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .protocols import ToolProtocol
from .validation import StrictJSONError, strict_json_loads


def normalize_history(
    messages: Sequence[Mapping[str, Any]],
    protocol: ToolProtocol,
) -> list[dict[str, Any]]:
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
    ) -> str:
        template_kwargs: dict[str, Any] = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking

        prompt = self.tokenizer.apply_chat_template(
            deepcopy([dict(message) for message in messages]),
            tools=deepcopy([dict(tool) for tool in tools]) or None,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if not isinstance(prompt, str):
            raise TypeError("The chat template did not return a string")
        return prompt

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..grammar import GBNFCompiler, GrammarConfigurationError, SchemaNotSupportedError
from ..registry import Tool
from ..protocols import ToolProtocol

BFCL_CATEGORIES = (
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)


@dataclass(frozen=True)
class ConstraintSupport:
    eligible: bool
    error: dict[str, Any] | None = None
    grammar_sha256: str | None = None


def convert_bfcl_tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from bfcl_eval.constants.enums import ModelStyle
        from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
        from bfcl_eval.model_handler.utils import convert_to_tool
    except ImportError as exc:
        raise ImportError("Install the pinned bfcl-eval package for BFCL conversion") from exc
    return convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)


def constraint_support(
    definitions: list[dict[str, Any]],
    protocol: ToolProtocol = ToolProtocol.QWEN3,
) -> ConstraintSupport:
    try:
        tools = tuple(Tool.from_openai(definition) for definition in definitions)
        grammar = GBNFCompiler(protocol=protocol).compile_tools(tools, "auto")
    except (SchemaNotSupportedError, GrammarConfigurationError, TypeError, ValueError) as exc:
        error: dict[str, Any] = {"code": "constraint_unsupported", "message": str(exc)}
        if isinstance(exc, SchemaNotSupportedError):
            error["pointer"] = exc.pointer
            error["keyword"] = exc.keyword
        return ConstraintSupport(False, error)
    return ConstraintSupport(True, grammar_sha256=grammar.sha256 if grammar else None)


def openai_message_to_bfcl_result(message: dict[str, Any]) -> Any:
    calls = message.get("tool_calls") or []
    if not calls:
        return message.get("content") or ""
    return [
        {call["function"]["name"]: call["function"]["arguments"]}
        for call in calls
    ]


def result_entry(
    case_id: str,
    response: dict[str, Any],
    *,
    latency: float,
) -> dict[str, Any]:
    message = response["choices"][0]["message"]
    usage = response.get("usage", {})
    return {
        "id": case_id,
        "result": openai_message_to_bfcl_result(message),
        "latency": latency,
        "input_token_count": usage.get("prompt_tokens", 0),
        "output_token_count": usage.get("completion_tokens", 0),
    }


def decoded_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls = response["choices"][0]["message"].get("tool_calls") or []
    return [
        {
            "name": call["function"]["name"],
            "arguments": json.loads(call["function"]["arguments"]),
        }
        for call in calls
    ]


def eligible_ids(results: list[Any]) -> set[str]:
    return {
        result.case_id
        for result in results
        if result.status != "constraint_unsupported"
    }


__all__ = [
    "BFCL_CATEGORIES",
    "ConstraintSupport",
    "constraint_support",
    "convert_bfcl_tools",
    "decoded_calls",
    "eligible_ids",
    "openai_message_to_bfcl_result",
    "result_entry",
]

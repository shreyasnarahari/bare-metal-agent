"""Glue between BFCL (the Berkeley Function Calling Leaderboard) and Toolsmith.

Converts BFCL's function definitions into OpenAI tools with BFCL's own
converter, checks whether Toolsmith's grammar compiler can constrain a case's
tools at all (cases it cannot are reported as ineligible for the grammar modes,
never silently run unconstrained), and turns OpenAI responses back into BFCL's
result format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..grammar import GBNFCompiler, GrammarConfigurationError, SchemaNotSupportedError
from ..registry import Tool
from ..protocols import ToolProtocol

# The non-live, single-turn Python AST categories this project evaluates.
BFCL_CATEGORIES = (
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)


@dataclass(frozen=True)
class ConstraintSupport:
    """Whether a case's tools compile to a grammar, and if not, why (with a JSON Pointer)."""

    eligible: bool
    error: dict[str, Any] | None = None
    grammar_sha256: str | None = None


def convert_bfcl_tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """BFCL function definitions as OpenAI tools, using BFCL's own type mapping."""
    try:
        from bfcl_eval.constants.enums import ModelStyle
        from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
        from bfcl_eval.model_handler.utils import convert_to_tool
    except ImportError as exc:
        raise ImportError("Install the pinned bfcl-eval package for BFCL conversion") from exc
    return convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)


# Try compiling the case's tools; any unsupported schema makes the case ineligible for grammar modes.
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
    """BFCL's result shape: ``[{name: arguments_json}, ...]`` for calls, or the text answer."""
    calls = message.get("tool_calls") or []
    if not calls:
        return message.get("content") or ""
    return [
        {call["function"]["name"]: call["function"]["arguments"]}
        for call in calls
    ]


# One line of a BFCL result file: id, result, latency, and token counts.
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
    """The response's tool calls as ``{"name", "arguments"}`` with arguments decoded."""
    calls = response["choices"][0]["message"].get("tool_calls") or []
    return [
        {
            "name": call["function"]["name"],
            "arguments": json.loads(call["function"]["arguments"]),
        }
        for call in calls
    ]


def eligible_ids(results: list[Any]) -> set[str]:
    """Ids of cases the grammar could constrain (constrained scores use only these)."""
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

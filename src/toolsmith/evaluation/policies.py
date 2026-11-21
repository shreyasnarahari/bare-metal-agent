from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator

from ..grammar import ToolChoice
from ..registry import Tool, ensure_unique_tool_names
from ..repair import RepairPolicy
from ..validation import ToolCallValidator
from .models import CaseResult, Mode


class DeferredSchemaValidator:
    def __init__(self, tools: Sequence[Tool], choice: ToolChoice) -> None:
        self.delegate = ToolCallValidator(tools, choice)
        for name in self.delegate.validators:
            self.delegate.validators[name] = Draft202012Validator({})

    def validate(self, response: Any) -> None:
        self.delegate.validate(response)


def service_options(mode: Mode, *, defer_bfcl_schema: bool = False) -> dict[str, Any]:
    if mode not in {Mode.PROMPT, Mode.GRAMMAR, Mode.REPAIR, Mode.THINKING}:
        raise ValueError(f"No Toolsmith service policy for {mode}")
    use_grammars = mode in {Mode.GRAMMAR, Mode.REPAIR}
    return {
        "use_grammars": use_grammars,
        "repair_policy": RepairPolicy(max_repairs=1 if mode is Mode.REPAIR else 0),
        "validator_factory": (
            DeferredSchemaValidator
            if defer_bfcl_schema and not use_grammars
            else ToolCallValidator
        ),
    }


def exception_error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        nested = body.get("error")
        if isinstance(nested, Mapping) and isinstance(nested.get("code"), str) and nested["code"]:
            return nested["code"]
        if isinstance(body.get("code"), str) and body["code"]:
            return body["code"]
    return "backend_error"


def calls_match_schema(
    definitions: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    *,
    allow_empty: bool = False,
) -> bool:
    if not calls:
        return allow_empty
    try:
        tools = [Tool.from_openai(definition) for definition in definitions]
        ensure_unique_tool_names(tools)
    except (TypeError, ValueError):
        return False
    validators = {
        tool.name: Draft202012Validator(tool.parameters) for tool in tools
    }
    for call in calls:
        validator = validators.get(call.get("name"))
        if validator is None or not validator.is_valid(call.get("arguments")):
            return False
    return True


def repair_eligible(result: CaseResult) -> bool:
    if result.status in {"repaired", "constraint_unsupported"}:
        return False
    if result.status == "success":
        return result.semantic_valid is False
    code = (result.error or {}).get("code")
    return code not in {
        "backend_error",
        "completion_truncated",
        "timeout",
        "connection_error",
        "invalid_backend_response",
        "unsupported_schema",
    }


def reuse_grammar_result(result: CaseResult) -> bool:
    return (
        result.status == "success"
        and result.parse_valid
        and result.schema_valid
        and result.semantic_valid is True
    )


__all__ = [
    "DeferredSchemaValidator",
    "calls_match_schema",
    "exception_error_code",
    "repair_eligible",
    "reuse_grammar_result",
    "service_options",
]

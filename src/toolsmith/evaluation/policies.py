from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jsonschema import Draft202012Validator

from ..grammar import ToolChoice
from ..registry import Tool
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


def repair_eligible(result: CaseResult) -> bool:
    if result.status in {"success", "repaired", "constraint_unsupported"}:
        return False
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
    return result.status == "success" and result.parse_valid and result.schema_valid


__all__ = [
    "DeferredSchemaValidator",
    "repair_eligible",
    "reuse_grammar_result",
    "service_options",
]

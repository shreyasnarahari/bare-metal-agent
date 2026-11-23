"""Validate what the model produced before it reaches a client.

Two validators share one reporting style (JSON Pointer paths, sorted, capped):

- ``ToolCallValidator`` checks a parsed response against the request's
  ``tool_choice`` (none, required, or one named tool) and checks every call's
  arguments: a registered tool, strict JSON, and the tool's JSON Schema.
- ``StructuredOutputValidator`` checks a ``response_format=json_schema`` answer.

Each failure carries a stable ``code`` (``unknown_tool``,
``schema_validation_failed``, ...) that the API, repair, and evaluation use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator

from .grammar import ToolChoice
from .json_utils import StrictJSONError, strict_json_loads
from .parsing import ParsedResponse
from .registry import Tool, ensure_unique_tool_names


class ToolCallValidationError(ValueError):
    """Tool calls broke the tool choice or a schema; ``details`` lists each problem."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "tool_call_validation_failed",
        details: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class StructuredOutputValidationError(ValueError):
    """A structured response is not strict JSON or does not match its schema."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "structured_output_validation_failed",
        details: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def _pointer(parts: Sequence[object]) -> str:
    # RFC 6901 JSON Pointer: escape ~ before /, so a key like "a/b" stays one segment.
    tokens = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(tokens) if tokens else "/"


# Errors are sorted by location so the same bad output always produces the same message.
def _schema_details(
    validator: Draft202012Validator,
    value: Any,
    *,
    max_errors: int,
) -> tuple[str, ...]:
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (_pointer(list(error.absolute_path)), error.message),
    )
    return tuple(
        f"{_pointer(list(error.absolute_path))}: {error.message}"
        for error in errors[:max_errors]
    )


class StructuredOutputValidator:
    """Strict-JSON decode a structured response and validate it against a Draft 2020-12 schema."""

    def __init__(self, schema: Mapping[str, Any], *, max_errors: int = 20) -> None:
        if max_errors < 1:
            raise ValueError("max_errors must be positive")
        self.schema = dict(schema)
        self.validator = Draft202012Validator(self.schema)
        self.max_errors = max_errors

    def validate(self, text: str) -> Any:
        try:
            value = strict_json_loads(text)
        except StrictJSONError as exc:
            raise StructuredOutputValidationError(
                f"Structured response is not valid strict JSON: {exc}",
                code="invalid_structured_output_json",
            ) from exc
        details = _schema_details(
            self.validator, value, max_errors=self.max_errors
        )
        if details:
            raise StructuredOutputValidationError(
                "Structured response failed schema validation: " + "; ".join(details),
                code="schema_validation_failed",
                details=details,
            )
        return value


class ToolCallValidator:
    """Check a parsed response's tool calls against the tool choice and each tool's schema."""

    def __init__(
        self,
        tools: Sequence[Tool],
        choice: ToolChoice,
        *,
        max_errors: int = 20,
    ) -> None:
        if max_errors < 1:
            raise ValueError("max_errors must be positive")
        ensure_unique_tool_names(tools)
        self.choice = choice
        self.max_errors = max_errors
        self.validators = {
            tool.name: Draft202012Validator(tool.parameters) for tool in tools
        }

    def validate(self, response: ParsedResponse) -> None:
        """Raise ``ToolCallValidationError`` on the first tool-choice violation, or on every bad call."""
        calls = response.tool_calls
        # Tool-choice rules first: they fail the whole response at once.
        if self.choice.mode == "none" and calls:
            raise ToolCallValidationError(
                "The model returned a tool call despite tool_choice='none'",
                code="tool_choice_violation",
            )
        if self.choice.mode == "required" and not calls:
            raise ToolCallValidationError(
                "The model returned no tool call despite tool_choice='required'",
                code="tool_choice_violation",
            )
        if self.choice.mode == "named":
            if len(calls) != 1 or calls[0].name != self.choice.name:
                raise ToolCallValidationError(
                    f"The model must return exactly one call to {self.choice.name!r}",
                    code="tool_choice_violation",
                )

        # Then every call is checked, so one error message reports all bad calls together.
        failures: list[tuple[str, str, tuple[str, ...]]] = []
        for call in calls:
            validator = self.validators.get(call.name)
            if validator is None:
                message = f"The model called an unregistered tool: {call.name!r}"
                failures.append(("unknown_tool", message, (message,)))
                continue
            try:
                arguments = strict_json_loads(call.arguments)
            except StrictJSONError as exc:
                message = f"Arguments for {call.name!r} are not valid strict JSON: {exc}"
                failures.append(("invalid_tool_call_json", message, (message,)))
                continue
            details = _schema_details(
                validator,
                arguments,
                max_errors=self.max_errors,
            )
            if details:
                failures.append(
                    (
                        "schema_validation_failed",
                        f"Arguments for {call.name!r} failed schema validation: "
                        + "; ".join(details),
                        details,
                    )
                )
        if failures:
            # A single failure keeps its specific code; several get the generic one.
            code = failures[0][0] if len(failures) == 1 else "tool_call_validation_failed"
            details = tuple(
                detail
                for _, _, failure_details in failures
                for detail in failure_details
            )[: self.max_errors]
            raise ToolCallValidationError(
                "; ".join(message for _, message, _ in failures),
                code=code,
                details=details,
            )


__all__ = [
    "StrictJSONError",
    "StructuredOutputValidationError",
    "StructuredOutputValidator",
    "ToolCallValidationError",
    "ToolCallValidator",
    "strict_json_loads",
]

"""Three validation layers, the mock store's semantic checks, and observation builders.

``validate_call`` runs, in order: the tool name, the JSON Schema, then the
tool's semantic check against the episode's own store. The first failing layer
produces an ``ErrorObservation`` and the handler never runs. Semantic checks
are pure reads of that store and its fixed clock; they never write.

The shim normally enforces layers 1 and 2 before ``chat()`` returns. They run
again here so a replayed raw model message takes the same path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from jsonschema import Draft202012Validator

from toolsmith.evaluation.models import canonical_hash
from toolsmith.json_utils import StrictJSONError, strict_json_loads

from .errors import (
    DEFAULT_MAX_ERRORS,
    ErrorObservation,
    ErrorType,
    FieldProblem,
    bounded,
    cap_problems,
)
from .policy import ToolPolicy
from .registry import SemanticCheck, SteadyhandRegistry
from .store import NotFoundError, Store, money
from .types import ModelTurn, ToolCall, ToolResult

MAX_LISTED_TOOL_NAMES = 12  # an unknown-tool hint lists the registered names only up to this many
PARSE_ERROR_HINT = "Re-emit the call in the native tool format. Do not invent a tool result."
_CENT = Decimal("0.01")


@dataclass(frozen=True)
class Verdict:
    """The outcome of validating one call: decoded arguments if valid, a typed error if not."""

    ok: bool
    arguments: dict[str, Any] | None = None
    arguments_sha256: str | None = None
    error: ErrorObservation | None = None

    def rejected(self, call: ToolCall) -> ToolResult:
        """The tool result for a call that failed validation. The handler did not run."""
        if self.ok or self.error is None:
            raise ValueError("Only a failed verdict becomes a rejected result")
        return ToolResult(
            call_id=call.id,
            tool=call.name,
            arguments=call.arguments,
            arguments_sha256=self.arguments_sha256,
            ok=False,
            error=self.error,
            handler_called=False,
        )


def validate_call(
    call: ToolCall,
    registry: SteadyhandRegistry,
    store: Store,
    *,
    max_errors: int = DEFAULT_MAX_ERRORS,
) -> Verdict:
    """Check a call in three layers, stopping at the first that fails: name, JSON and schema, store rules."""
    # Layer 1: name.
    if call.name not in registry:
        return Verdict(ok=False, error=unknown_tool_observation(call.name, registry))

    # Layer 2: strict JSON, then the tool's JSON Schema.
    arguments, error = decode_arguments(call)
    if error is not None:
        return Verdict(ok=False, error=error)
    digest = canonical_hash(arguments)
    problems = cap_problems(schema_problems(registry.validator(call.name), arguments), max_errors)
    if problems:
        return Verdict(ok=False, arguments_sha256=digest, error=invalid_arguments(call.name, problems))

    # Layer 3: semantic rules that depend on the episode's store.
    semantic = registry.semantic(call.name)
    if semantic is not None:
        problems = cap_problems(semantic(store, arguments), max_errors)
        if problems:
            return Verdict(ok=False, arguments_sha256=digest, error=invalid_arguments(call.name, problems))
    return Verdict(ok=True, arguments=arguments, arguments_sha256=digest)


def decode_arguments(call: ToolCall) -> tuple[dict[str, Any], ErrorObservation | None]:
    """The arguments as a dict, or a parse-error observation when they are not one strict JSON object."""
    try:
        arguments = strict_json_loads(call.arguments)
    except StrictJSONError as exc:
        return {}, arguments_parse_error(call.name, f"arguments are not strict JSON: {exc}")
    if not isinstance(arguments, dict):
        return {}, arguments_parse_error(call.name, "arguments must be a JSON object")
    return arguments, None


def schema_problems(validator: Draft202012Validator, arguments: Any) -> list[FieldProblem]:
    """Schema errors as field problems, pointing at the offending property."""
    problems = list(_schema_problems(validator, arguments))
    return sorted(problems, key=lambda problem: (problem.field or "", problem.problem))


# Turn jsonschema errors into one problem per offending field: each extra key and each missing
# required key separately, and short wording for type and enum errors.
def _schema_problems(validator: Draft202012Validator, arguments: Any) -> Iterator[FieldProblem]:
    for error in validator.iter_errors(arguments):
        path = list(error.absolute_path)
        if error.validator == "additionalProperties" and isinstance(error.instance, dict):
            allowed = set(error.schema.get("properties", {}))
            for key in sorted(key for key in error.instance if key not in allowed):
                yield FieldProblem(json_pointer([*path, key]), "is not an allowed property", error.instance[key])
        elif error.validator == "required" and isinstance(error.instance, dict):
            for key in error.validator_value:
                if key not in error.instance:
                    yield FieldProblem(json_pointer([*path, key]), "is required")
        elif error.validator == "type":
            yield FieldProblem(json_pointer(path), f"must be of type {_types(error.validator_value)}", error.instance)
        elif error.validator == "enum":
            yield FieldProblem(json_pointer(path), f"must be one of {list(error.validator_value)}", error.instance)
        else:
            yield FieldProblem(json_pointer(path), error.message, error.instance)


def json_pointer(parts: Sequence[object]) -> str:
    """RFC 6901 pointer, with the same escaping as ``ToolCallValidator`` details."""
    tokens = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(tokens) if tokens else "/"


# "string or null" for a list of types.
def _types(value: Any) -> str:
    return " or ".join(value) if isinstance(value, list) else str(value)


# Observation builders: one per failure kind, each with the retry advice that fits it.


def unknown_tool_observation(name: str, registry: SteadyhandRegistry) -> ErrorObservation:
    """The model named a tool that does not exist; retrying the same name cannot help."""
    names = registry.names()
    hint = "Use one of the registered tools"
    hint += f": {', '.join(names)}." if len(names) <= MAX_LISTED_TOOL_NAMES else "."
    return ErrorObservation(
        error_type=ErrorType.UNKNOWN_TOOL,
        tool=name,
        retryable=False,
        hint=hint,
        details=(FieldProblem("name", "no tool with this name is registered", name),),
    )


def arguments_parse_error(tool: str, problem: str) -> ErrorObservation:
    """The arguments were not a strict JSON object; the model can resend them."""
    return ErrorObservation(
        error_type=ErrorType.PARSE_ERROR,
        tool=tool,
        retryable=True,
        hint=f"Call {tool} again with its arguments as one strict JSON object.",
        details=(FieldProblem(None, problem),),
    )


def invalid_arguments(tool: str, problems: Sequence[FieldProblem]) -> ErrorObservation:
    """Schema or store-rule problems, field by field; the model can fix them and retry."""
    return ErrorObservation(
        error_type=ErrorType.INVALID_ARGUMENTS,
        tool=tool,
        retryable=True,
        hint=f"Fix the listed fields and call {tool} again.",
        details=tuple(problems),
    )


def parse_error_observation(turn: ModelTurn) -> ErrorObservation:
    """For a model turn the shim could not parse. Sent as a user message: there is no call id."""
    details = (FieldProblem(None, turn.error_message),) if turn.error_message else ()
    return ErrorObservation(
        error_type=ErrorType.PARSE_ERROR,
        tool=None,
        retryable=True,
        hint=PARSE_ERROR_HINT,
        details=details,
        previous_output=bounded(turn.raw_output) if turn.raw_output is not None else None,
    )


def not_found_observation(tool: str, error: NotFoundError) -> ErrorObservation:
    """The id names no row; repeating the call will not help."""
    return ErrorObservation(
        error_type=ErrorType.NOT_FOUND,
        tool=tool,
        retryable=False,
        hint=(
            "The store has no such row. Check the id with the user or find it "
            "another way; repeating this call will not help."
        ),
        details=(FieldProblem(f"{error.entity}_id", f"no {error.entity} with this id exists", error.entity_id),),
    )


def timeout_observation(tool: str, policy: ToolPolicy) -> ErrorObservation:
    """No result in time. Safe to retry only for idempotent tools; a write may already have happened."""
    if policy.idempotent:
        retryable, hint = True, f"{tool} did not finish in time. Calling it again is safe."
    else:
        check = f"Check with {policy.verify}" if policy.verify else "Check the result another way"
        retryable = False
        hint = f"{tool} did not finish in time and may have taken effect. {check} before calling it again."
    return ErrorObservation(
        error_type=ErrorType.TIMEOUT,
        tool=tool,
        retryable=retryable,
        hint=hint,
        details=(FieldProblem(None, f"no result within {policy.timeout_s:g} seconds"),),
    )


def transient_failure_observation(tool: str, error_type: str, policy: ToolPolicy) -> ErrorObservation:
    """A transient tool failure. An injected one is worded exactly like a real one."""
    if error_type == ErrorType.TIMEOUT.value:
        return timeout_observation(tool, policy)
    if error_type == ErrorType.RATE_LIMIT.value:
        return ErrorObservation(ErrorType.RATE_LIMIT, tool, True, f"{tool} is rate limited. A retry may succeed.")
    if error_type == ErrorType.SERVER_ERROR.value:
        return ErrorObservation(ErrorType.SERVER_ERROR, tool, True, f"{tool} had a server error. A retry may succeed.")
    raise ValueError(f"Not a transient error type: {error_type!r}")


def unavailable_observation(tool: str) -> ErrorObservation:
    """The tool is down for good; use another tool or tell the user."""
    return ErrorObservation(
        ErrorType.PERMANENT_UNAVAILABLE,
        tool,
        False,
        "This tool is unavailable. Use a different tool or tell the user.",
    )


def outcome_unknown_observation(tool: str, problem: str) -> ErrorObservation:
    """A write whose effect could not be confirmed either way; do not risk doing it twice."""
    return ErrorObservation(
        ErrorType.OUTCOME_UNKNOWN,
        tool,
        False,
        f"Could not confirm whether {tool} took effect. Do not call it again; tell the user it needs checking.",
        details=(FieldProblem(None, problem),),
    )


def suspicious_result_observation(tool: str, hint: str, problem: str, *, retryable: bool = False) -> ErrorObservation:
    """A successful-looking result that failed a sanity check."""
    return ErrorObservation(
        ErrorType.SUSPICIOUS_RESULT,
        tool,
        retryable,
        hint,
        details=(FieldProblem(None, problem),),
    )


def handler_error_observation(tool: str, problem: str) -> ErrorObservation:
    """The handler raised (a conflict, a bug, or a bad value); the same call would fail again."""
    return ErrorObservation(
        error_type=ErrorType.HANDLER_ERROR,
        tool=tool,
        retryable=False,
        hint=f"{tool} failed. Change the request or tell the user; the same call will fail again.",
        details=(FieldProblem(None, problem),),
    )


# Semantic checks for the mock store


def require_existing(entity: str, field: str) -> SemanticCheck:
    """The id in ``field`` must name an existing row. An absent optional field passes."""
    lookup: Callable[[Store, str], Any] = {
        "order": Store.order,
        "customer": Store.customer,
        "ticket": Store.ticket,
    }[entity]

    # The check's __name__ is set below so journals and errors can name it.
    def check(store: Store, arguments: Mapping[str, Any]) -> list[FieldProblem]:
        value = arguments.get(field)
        if value is None:
            return []
        try:
            lookup(store, value)
        except NotFoundError:
            return [FieldProblem(field, f"no {entity} with this id exists", value)]
        return []

    check.__name__ = f"require_{field}"
    return check


def check_issue_refund(store: Store, arguments: Mapping[str, Any]) -> list[FieldProblem]:
    """The order exists; the amount is positive, in whole paise, and within the refundable total."""
    problems = require_existing("order", "order_id")(store, arguments)
    got = arguments["amount_inr"]
    amount = Decimal(str(got))
    if amount <= 0:
        problems.append(FieldProblem("amount_inr", "must be > 0", got))
    elif amount != amount.quantize(_CENT):
        problems.append(FieldProblem("amount_inr", "must have at most two decimal places", got))
    elif not problems:
        # Refundable = order total minus refunds and ledger entries already recorded.
        order_id = arguments["order_id"]
        refundable = Decimal(store.order(order_id).total_inr) - store.refunded_total(order_id)
        if amount > refundable:
            problems.append(
                FieldProblem(
                    "amount_inr",
                    f"must not exceed the refundable total of {money(refundable)} INR for {order_id}",
                    got,
                )
            )
    return problems


def check_create_ticket(store: Store, arguments: Mapping[str, Any]) -> list[FieldProblem]:
    """The customer exists, and an order, when given, exists and belongs to that customer."""
    problems = require_existing("customer", "customer_id")(store, arguments)
    problems += require_existing("order", "order_id")(store, arguments)
    order_id = arguments.get("order_id")
    if not problems and order_id is not None:
        owner = store.order(order_id).customer_id
        if owner != arguments["customer_id"]:
            problems.append(FieldProblem("order_id", f"belongs to {owner}, not {arguments['customer_id']}", order_id))
    return problems


def must_be_future(store: Store, field: str, value: str) -> FieldProblem | None:
    """For date arguments: later than the store's fixed episode clock, never ``datetime.now()``."""
    clock = datetime.fromisoformat(store.clock)
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return FieldProblem(field, "must be an ISO 8601 date or date-time", value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=clock.tzinfo)
    if moment <= clock:
        return FieldProblem(field, f"must be after the episode clock {store.clock}", value)
    return None


__all__ = [
    "PARSE_ERROR_HINT",
    "Verdict",
    "arguments_parse_error",
    "check_create_ticket",
    "check_issue_refund",
    "decode_arguments",
    "handler_error_observation",
    "invalid_arguments",
    "json_pointer",
    "must_be_future",
    "not_found_observation",
    "outcome_unknown_observation",
    "suspicious_result_observation",
    "parse_error_observation",
    "require_existing",
    "schema_problems",
    "timeout_observation",
    "transient_failure_observation",
    "unavailable_observation",
    "unknown_tool_observation",
    "validate_call",
]

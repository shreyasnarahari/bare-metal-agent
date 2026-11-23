"""Typed payloads for the Bulkhead security events recorded in the Steadyhand journal.

Bulkhead writes no separate log: every decision, taint change, approval
request, sandbox run, and egress decision is an event in the episode's own
append-only JSONL journal. This module defines each event's exact fields and
checks every value, both when a record is written and when a journal is read:
unknown or missing fields, non-JSON or unbounded values, invalid enum labels,
and inconsistent combinations (a verdict that is not its rule's verdict, a
grant id on a rule that is not a grant) are all refused.

Payloads hold hashes, rule ids, labels, sizes, and bounded sanitized details.
They never hold raw arguments, full command lines or URLs, environment
variables, credentials, or unbounded tool output.

Only genuinely variable measurements are volatile (ignored when a replay is
compared): wall time, resource usage, and proxy byte counts. Verdicts, rule
ids, taint labels, request hashes, and results always compare exactly.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from .approvals import APPROVAL_ID, ApprovalMode, check_normalized_grant
from .errors import BulkheadError, GrantError, SecuritySpecError
from .manifest import normalize_authority
from .policy import RULES, VIA, Verdict
from .taint import Trust
from .types import SandboxTier

MAX_RECORD_CHARS = 64 * 1024
MAX_DETAIL_CHARS = 200
MAX_VIOLATIONS = 16
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_~.-]{0,127}")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CODE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
CLEANUP_STATUSES = ("clean", "killed", "leaked", "not_started")
ADDRESS_CLASSES = ("public", "private", "loopback", "link_local", "multicast", "reserved", "ip_literal", "unresolved")
EGRESS_DECISIONS = ("allow", "deny")
RESOURCE_KEYS = ("cpu_s", "wall_s", "peak_memory_kb", "pids_peak", "oom_killed")


class AuditError(BulkheadError, ValueError):
    """A security event payload is malformed; ``field`` names the offending field."""

    def __init__(self, message: str, *, field: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field


Validator = Callable[[Any, str], Any]


def _fail(field: str, message: str) -> None:
    raise AuditError(message, field=field)


def step(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        _fail(field, "must be a non-negative integer")
    return value


def identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        _fail(field, "must be a short identifier")
    return value


def optional_identifier(value: Any, field: str) -> str | None:
    return None if value is None else identifier(value, field)


def name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        _fail(field, "must be a tool or backend name")
    return value


def sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _fail(field, "must be 64 lowercase hex characters")
    return value


def code(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _CODE.fullmatch(value):
        _fail(field, "must be a short lowercase code")
    return value


def boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        _fail(field, "must be true or false")
    return value


def count(value: Any, field: str) -> int:
    return step(value, field)


def seconds(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        _fail(field, "must be a finite, non-negative number")
    return value


def detail(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_DETAIL_CHARS or _CONTROL.search(value):
        _fail(field, f"must be one line of at most {MAX_DETAIL_CHARS} characters")
    return value


def choice(options: tuple[str, ...]) -> Validator:
    def check(value: Any, field: str) -> str:
        if value not in options:
            _fail(field, f"must be one of {list(options)}")
        return value

    return check


def enum(kind: type) -> Validator:
    return choice(tuple(member.value for member in kind))


def nullable(validator: Validator) -> Validator:
    return lambda value, field: None if value is None else validator(value, field)


def integer_between(low: int, high: int) -> Validator:
    def check(value: Any, field: str) -> int:
        if type(value) is not int or not low <= value <= high:
            _fail(field, f"must be an integer from {low} to {high}")
        return value

    return check


def authority(value: Any, field: str) -> str:
    try:
        normalized = normalize_authority(value, field)
    except SecuritySpecError as exc:
        raise AuditError(str(exc), field=field) from None
    if normalized != value:
        _fail(field, "must already be a normalized host:port")
    return value


def normalized_grant(value: Any, field: str) -> dict[str, Any]:
    try:
        return check_normalized_grant(value, field)
    except GrantError as exc:
        raise AuditError(str(exc), field=field) from None


def grant_id(value: Any, field: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not APPROVAL_ID.fullmatch(value)):
        _fail(field, "must be null or an approval id")
    return value


def violations(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_VIOLATIONS:
        _fail(field, f"must be a list of at most {MAX_VIOLATIONS} violations")
    for index, item in enumerate(value):
        where = f"{field}[{index}]"
        if not isinstance(item, Mapping) or set(item) != {"kind", "rule_id", "detail"}:
            _fail(where, "needs exactly kind, rule_id, and detail")
        code(item["kind"], f"{where}.kind")
        code(item["rule_id"], f"{where}.rule_id")
        detail(item["detail"], f"{where}.detail")
    return value


def resource_usage(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not set(value) <= set(RESOURCE_KEYS):
        _fail(field, f"must be an object with keys from {list(RESOURCE_KEYS)}")
    for key, item in value.items():
        if item is None:
            continue
        if key == "oom_killed":
            boolean(item, f"{field}.{key}")
        else:
            seconds(item, f"{field}.{key}")
    return value


def typed_result(value: Any, field: str) -> dict[str, Any]:
    """A tool result as the executor produced it: ok plus a value, or not ok plus a typed error."""
    if not isinstance(value, Mapping):
        _fail(field, "must be a JSON object")
    keys = set(value)
    if not {"ok", "handler_called"} <= keys or len(keys & {"value", "error"}) != 1 or keys - {"ok", "handler_called", "value", "error"}:
        _fail(field, "needs ok, handler_called, and exactly one of value or error")
    boolean(value["ok"], f"{field}.ok")
    boolean(value["handler_called"], f"{field}.handler_called")
    if value["ok"] != ("value" in value):
        _fail(field, "a successful result has a value; a failed one has an error")
    if "error" in value:
        error = value["error"]
        if not isinstance(error, Mapping) or not {"ok", "error_type", "tool", "retryable", "hint"} <= set(error):
            _fail(f"{field}.error", "must be a typed error observation")
        code(error["error_type"], f"{field}.error.error_type")
        boolean(error["retryable"], f"{field}.error.retryable")
    return value


# Every event's fields (besides the journal's common event, episode_id, seq, recorded_at).
SECURITY_EVENT_FIELDS: dict[str, dict[str, Validator]] = {
    "policy_decision": {
        "step": step,
        "tool_call_id": identifier,
        "via": choice(VIA),
        "tool": name,
        "arguments_sha256": sha256,
        "verdict": enum(Verdict),
        "rule_id": choice(tuple(RULES)),
        "reason_code": code,
        "taint_before": enum(Trust),
        "policy_sha256": sha256,
        "grant_id": grant_id,
        "normalized_request_sha256": sha256,
    },
    "taint_transition": {
        "step": step,
        "tool_call_id": identifier,
        "source_tool": name,
        "before": enum(Trust),
        "after": enum(Trust),
        "source_seq": count,
    },
    "approval_requested": {
        "step": step,
        "tool_call_id": identifier,
        "rule_id": choice(tuple(rule for rule, (verdict, _) in RULES.items() if verdict is Verdict.ASK)),
        "request_sha256": sha256,
        "normalized_grant": normalized_grant,
        "parent_episode_id": optional_identifier,
        "approval_mode": enum(ApprovalMode),
    },
    "sandbox_execution": {
        "step": step,
        "tool_call_id": identifier,
        "via": choice(VIA),
        "tool": name,
        "backend": name,
        # T0 is not a sandbox, so no sandbox run can be recorded at it.
        "tier": choice(tuple(tier.value for tier in SandboxTier if tier.is_sandbox)),
        "config_sha256": sha256,
        "request_sha256": sha256,
        "started": boolean,
        "exit_code": nullable(integer_between(-255, 255)),
        "signal": nullable(integer_between(1, 64)),
        "timed_out": boolean,
        "cleanup_status": choice(CLEANUP_STATUSES),
        "stdout_bytes": count,
        "stderr_bytes": count,
        "stdout_truncated": boolean,
        "stderr_truncated": boolean,
        "violations": violations,
        "resource_usage": resource_usage,
        "elapsed_s": seconds,
        "result": typed_result,
    },
    "egress_decision": {
        "step": step,
        "tool_call_id": identifier,
        "request_id": identifier,
        "authority": authority,
        "decision": choice(EGRESS_DECISIONS),
        "rule_id": code,
        "resolved_address_class": choice(ADDRESS_CLASSES),
        "bytes_in": count,
        "bytes_out": count,
        "truncated": boolean,
        "elapsed_s": seconds,
    },
}

# Fields that may differ when a run is re-executed: timings, resource use, proxy byte counts.
VOLATILE_SECURITY_FIELDS: dict[str, frozenset[str]] = {
    "sandbox_execution": frozenset({"elapsed_s", "resource_usage"}),
    "egress_decision": frozenset({"elapsed_s", "bytes_in", "bytes_out"}),
}


def _consistency(event: str, fields: Mapping[str, Any]) -> None:
    if event == "policy_decision":
        verdict, reason = RULES[fields["rule_id"]]
        if fields["verdict"] != verdict.value:
            _fail("verdict", f"rule {fields['rule_id']} always decides {verdict.value}")
        if fields["reason_code"] != reason:
            _fail("reason_code", f"rule {fields['rule_id']} has reason {reason}")
        if (fields["grant_id"] is not None) != (fields["rule_id"] == "approval.grant"):
            _fail("grant_id", "a grant id appears exactly on approval.grant decisions")
    elif event == "taint_transition":
        if (fields["before"], fields["after"]) != (Trust.TRUSTED.value, Trust.UNTRUSTED.value):
            _fail("after", "the only transition is trusted -> untrusted; nothing cleans taint")
    elif event == "sandbox_execution":
        if not fields["started"] and (fields["exit_code"] is not None or fields["stdout_bytes"] or fields["stderr_bytes"]):
            _fail("started", "a run that never started has no exit code or output")
        if fields["result"]["handler_called"] != fields["started"]:
            _fail("result", "the result's handler_called must match started")


def validate(event: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Check one security event's fields; return them as a plain dict, or raise ``AuditError``."""
    spec = SECURITY_EVENT_FIELDS.get(event)
    if spec is None:
        _fail("event", f"{event!r} is not a security event")
    unknown = sorted(set(fields) - set(spec))
    missing = sorted(set(spec) - set(fields))
    if unknown:
        _fail(unknown[0], f"unknown field in {event}")
    if missing:
        _fail(missing[0], f"missing field in {event}")
    checked = {key: spec[key](fields[key], key) for key in spec}
    try:
        encoded = json.dumps(checked, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"not JSON: {exc}", field="fields") from None
    if len(encoded) > MAX_RECORD_CHARS:
        _fail("fields", f"the {event} payload exceeds {MAX_RECORD_CHARS} characters")
    _consistency(event, checked)
    return dict(checked)


def build(event: str, **fields: Any) -> dict[str, Any]:
    """The checked payload for a new record; writers call this so an invalid record is never written."""
    return validate(event, fields)


__all__ = [
    "ADDRESS_CLASSES",
    "CLEANUP_STATUSES",
    "MAX_RECORD_CHARS",
    "RESOURCE_KEYS",
    "SECURITY_EVENT_FIELDS",
    "VOLATILE_SECURITY_FIELDS",
    "AuditError",
    "build",
    "validate",
]

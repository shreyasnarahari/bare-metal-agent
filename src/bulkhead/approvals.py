"""Approval requests and one-shot grants.

An ``ASK`` decision becomes an ``ApprovalRequest``: the exact call (tool and
canonical arguments hash), the policy it was decided under, the rule that
asked, and the normalized grant that would allow it. Only an explicit approver
turns a request into a ``Grant``; text in the conversation never does.

A grant is bound to the episode that asked, that episode's policy hash, the
tool, the arguments hash, the normalized grant, an expiry, and a unique
approval id. It holds no bearer secret, it is honored only inside the linked
approved episode, and the policy gate consumes it on first use. Changing the
arguments, a path, a destination, the policy, or the episode invalidates it.

Batch evaluation uses a declared, deterministic approval mode: ``deny_all``,
or ``scripted`` grants a task lists in advance. There is no auto-approval.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from .errors import GrantError
from .manifest import check_keys
from .types import canonical_hash, canonical_json

APPROVAL_ID = re.compile(r"apr_[0-9a-f]{16}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_~.-]{0,127}")
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
GRANT_KEYS = ("tool", "arguments_sha256", "rules", "widen")
DEFAULT_GRANT_TTL_S = 3600.0


class ApprovalMode(str, Enum):
    """How ASK decisions are resolved in a run. Declared up front; never silent auto-approval."""

    DENY_ALL = "deny_all"  # every ASK stays escalated
    SCRIPTED = "scripted"  # the task lists the exact requests it approves


def _check_id(value: Any, where: str, pattern: re.Pattern[str] = _ID) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise GrantError(f"{where} is not a valid identifier: {value!r}")
    return value


def _check_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise GrantError(f"{where} must be 64 lowercase hex characters")
    return value


def check_normalized_grant(value: Any, where: str = "normalized_grant") -> dict[str, Any]:
    """The privileges an ASK needs: the exact call, the rules that asked, and any widening."""
    check_keys(value, GRANT_KEYS, where, error=GrantError)
    _check_id(value["tool"], f"{where}.tool")
    _check_sha(value["arguments_sha256"], f"{where}.arguments_sha256")
    rules = value["rules"]
    if not isinstance(rules, list) or not rules or not all(isinstance(rule, str) and rule for rule in rules):
        raise GrantError(f"{where}.rules must be a non-empty list of rule ids")
    if not isinstance(value["widen"], Mapping):
        raise GrantError(f"{where}.widen must be a JSON object")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise GrantError(f"{where} is not canonical JSON: {exc}") from None
    return {key: value[key] for key in GRANT_KEYS}


def format_timestamp(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise GrantError("timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc).strftime(_TIMESTAMP_FORMAT)


def parse_timestamp(text: Any, where: str = "expires_at") -> datetime:
    if not isinstance(text, str):
        raise GrantError(f"{where} must be a UTC timestamp string")
    try:
        return datetime.strptime(text, _TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        raise GrantError(f"{where} must look like 2026-11-09T12:00:00Z") from None


@dataclass(frozen=True)
class ApprovalRequest:
    """What a human (or a scripted approver) is asked to allow: one exact call, nothing broader."""

    episode_id: str
    tool_call_id: str
    tool: str
    arguments_sha256: str
    policy_sha256: str
    rule_id: str
    request_sha256: str
    normalized_grant: Mapping[str, Any]

    def __post_init__(self) -> None:
        _check_id(self.episode_id, "episode_id")
        _check_id(self.tool_call_id, "tool_call_id")
        _check_id(self.tool, "tool")
        _check_sha(self.arguments_sha256, "arguments_sha256")
        _check_sha(self.policy_sha256, "policy_sha256")
        _check_sha(self.request_sha256, "request_sha256")
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise GrantError("rule_id must be a non-empty string")
        grant = check_normalized_grant(self.normalized_grant)
        if grant["tool"] != self.tool or grant["arguments_sha256"] != self.arguments_sha256:
            raise GrantError("the normalized grant names a different call")
        object.__setattr__(self, "normalized_grant", grant)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "tool_call_id": self.tool_call_id,
            "tool": self.tool,
            "arguments_sha256": self.arguments_sha256,
            "policy_sha256": self.policy_sha256,
            "rule_id": self.rule_id,
            "request_sha256": self.request_sha256,
            "normalized_grant": dict(self.normalized_grant),
        }

    @property
    def sha256(self) -> str:
        return canonical_hash(self.as_dict())


@dataclass(frozen=True)
class Grant:
    """A one-shot permission for exactly one call inside the episode approved from ``episode_id``."""

    approval_id: str
    episode_id: str  # the episode that asked; the grant works only in its approved child
    policy_sha256: str
    tool: str
    arguments_sha256: str
    normalized_grant: Mapping[str, Any]
    expires_at: str

    def __post_init__(self) -> None:
        _check_id(self.approval_id, "approval_id", APPROVAL_ID)
        _check_id(self.episode_id, "episode_id")
        _check_sha(self.policy_sha256, "policy_sha256")
        _check_id(self.tool, "tool")
        _check_sha(self.arguments_sha256, "arguments_sha256")
        parse_timestamp(self.expires_at)
        grant = check_normalized_grant(self.normalized_grant)
        if grant["tool"] != self.tool or grant["arguments_sha256"] != self.arguments_sha256:
            raise GrantError("the normalized grant names a different call")
        object.__setattr__(self, "normalized_grant", grant)

    def expired(self, now: datetime) -> bool:
        return now >= parse_timestamp(self.expires_at)

    def matches(
        self,
        *,
        parent_episode_id: str | None,
        policy_sha256: str,
        tool: str,
        arguments_sha256: str,
        requested_grant: Mapping[str, Any] | None,
    ) -> bool:
        """Exact match on every binding. No partial, prefix, or "similar call" matching."""
        return (
            parent_episode_id is not None
            and self.episode_id == parent_episode_id
            and self.policy_sha256 == policy_sha256
            and self.tool == tool
            and self.arguments_sha256 == arguments_sha256
            and requested_grant is not None
            and canonical_json(dict(self.normalized_grant)) == canonical_json(dict(requested_grant))
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "episode_id": self.episode_id,
            "policy_sha256": self.policy_sha256,
            "tool": self.tool,
            "arguments_sha256": self.arguments_sha256,
            "normalized_grant": dict(self.normalized_grant),
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, value: Any, where: str = "grant") -> Grant:
        check_keys(value, ("approval_id", "episode_id", "policy_sha256", "tool", "arguments_sha256", "normalized_grant", "expires_at"), where, error=GrantError)
        return cls(**dict(value))


def approve(request: ApprovalRequest, *, approver: str, now: datetime, ttl_s: float = DEFAULT_GRANT_TTL_S) -> Grant:
    """Turn an approved request into a grant. The approval id is derived from the request and approver,
    so approving the same request twice yields the same id (and so the same, already-claimed episode)."""
    if not isinstance(approver, str) or not approver:
        raise GrantError("an approval needs a named approver")
    if not ttl_s > 0:
        raise GrantError("a grant needs a positive lifetime")
    approval_id = "apr_" + canonical_hash({"request": request.as_dict(), "approver": approver})[:16]
    return Grant(
        approval_id=approval_id,
        episode_id=request.episode_id,
        policy_sha256=request.policy_sha256,
        tool=request.tool,
        arguments_sha256=request.arguments_sha256,
        normalized_grant=request.normalized_grant,
        expires_at=format_timestamp(now + timedelta(seconds=ttl_s)),
    )


class Approver(Protocol):
    """Decides an approval request: True approves, False declines. Never consulted for DENY."""

    name: str

    def decide(self, request: ApprovalRequest) -> bool: ...


class DenyAll:
    """``deny_all``: every request is declined, so every ASK stays escalated."""

    name = ApprovalMode.DENY_ALL.value

    def decide(self, request: ApprovalRequest) -> bool:
        return False


@dataclass(frozen=True)
class ScriptedApproval:
    """One request a task approves (or explicitly declines) in advance: exact tool, arguments, and rule."""

    tool: str
    arguments: Mapping[str, Any]
    rule_id: str | None = None
    approve: bool = True

    @property
    def arguments_sha256(self) -> str:
        return canonical_hash(dict(self.arguments))

    def covers(self, request: ApprovalRequest) -> bool:
        return (
            self.tool == request.tool
            and self.arguments_sha256 == request.arguments_sha256
            and (self.rule_id is None or self.rule_id == request.rule_id)
        )


@dataclass(frozen=True)
class ScriptedApprover:
    """``scripted``: the first matching entry decides; anything unlisted is declined."""

    entries: Sequence[ScriptedApproval] = field(default_factory=tuple)
    name: str = ApprovalMode.SCRIPTED.value

    def decide(self, request: ApprovalRequest) -> bool:
        for entry in self.entries:
            if entry.covers(request):
                return entry.approve
        return False


def unconsumed(grants: Iterable[Grant], consumed: Iterable[str]) -> tuple[Grant, ...]:
    used = set(consumed)
    return tuple(grant for grant in grants if grant.approval_id not in used)


__all__ = [
    "APPROVAL_ID",
    "DEFAULT_GRANT_TTL_S",
    "ApprovalMode",
    "ApprovalRequest",
    "Approver",
    "DenyAll",
    "Grant",
    "ScriptedApproval",
    "ScriptedApprover",
    "approve",
    "check_normalized_grant",
    "format_timestamp",
    "parse_timestamp",
    "unconsumed",
]

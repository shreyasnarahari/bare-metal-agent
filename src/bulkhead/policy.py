"""The deterministic policy engine: ALLOW, DENY, or ASK for one schema-valid tool call.

``PolicyEngine.decide`` is pure. It reads only the ``PolicyRequest`` it is
given: no model calls, clock reads, network, filesystem, randomness, or
environment. The caller supplies the task policy, the tool's security spec,
the episode's taint, the unconsumed grants, and the sandbox tiers that are both
proven by the probe and backed by a backend (none until part 4).

Rules are evaluated in this order, and every rule that fires is kept as a
finding. The verdict is the most severe finding, with ``DENY > ASK > ALLOW``;
within a verdict the earliest rule names the decision.

1. Malformed request or security metadata, or an unavailable required tier.
2. Explicit task denials: denied tools, paths, and destinations always win.
3. Tool membership, argument bounds, path and destination sets, and the
   tool's sandbox against the task's ceiling. Anything new is ``ASK``.
4. Taint: an untrusted episode asks before using a taint-sensitive sink.
5. Task rules that always ask.
6. ``ALLOW`` only through an exact, unconsumed grant for the ``ASK``, or when
   nothing asked or denied.

Every rule id is listed in ``RULES``; nothing outside that table can produce a
verdict, and nothing unknown is ever treated as ``ALLOW``. A task policy that
is malformed or contradicts itself raises ``PolicyConfigError`` when loaded.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .approvals import Grant
from .errors import PolicyConfigError, SecuritySpecError
from .manifest import (
    DEFAULT_LIMITS,
    WORKSPACE,
    Bounds,
    Egress,
    Limits,
    PathAccess,
    SecuritySpec,
    Sink,
    Trust,
    beneath,
    beneath_any,
    check_keys,
    destination_authority,
    normalize_argument_path,
    normalize_authorities,
    normalize_paths,
    number,
)
from .narrowing import Widening, egress_widening, limits_widening, paths_outside, task_widenings
from .types import SandboxTier, canonical_hash

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
VIA = ("model", "policy")


class Verdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


SEVERITY = {Verdict.ALLOW: 0, Verdict.ASK: 1, Verdict.DENY: 2}

# Every rule the engine can apply: its verdict and a short machine-stable reason code.
RULES: dict[str, tuple[Verdict, str]] = {
    "bulkhead.request.malformed": (Verdict.DENY, "malformed_request"),
    "bulkhead.spec.missing": (Verdict.DENY, "missing_security_spec"),
    "bulkhead.tier.not_a_sandbox": (Verdict.DENY, "tier_not_a_sandbox"),
    "bulkhead.tier.unavailable": (Verdict.DENY, "tier_unavailable"),
    "task.deny.tool": (Verdict.DENY, "tool_denied"),
    "task.deny.path": (Verdict.DENY, "path_denied"),
    "task.deny.write_outside_workspace": (Verdict.DENY, "write_outside_workspace"),
    "task.deny.destination": (Verdict.DENY, "destination_denied"),
    "task.widen.tool": (Verdict.ASK, "new_tool"),
    "task.widen.bound": (Verdict.ASK, "bound_exceeded"),
    "task.widen.path": (Verdict.ASK, "new_path"),
    "task.widen.destination": (Verdict.ASK, "new_destination"),
    "task.widen.sandbox": (Verdict.ASK, "sandbox_exceeds_task"),
    "bulkhead.taint.sensitive_sink": (Verdict.ASK, "tainted_sensitive_sink"),
    "task.ask.tool": (Verdict.ASK, "tool_requires_approval"),
    "task.widen.policy": (Verdict.ASK, "policy_widening"),
    "approval.grant": (Verdict.ALLOW, "approved_by_grant"),
    "task.narrow": (Verdict.ALLOW, "policy_narrowed"),
    "task.allow": (Verdict.ALLOW, "within_task_policy"),
}


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise PolicyConfigError(f"{where} must be a tool or argument name, not {value!r}")
    return value


def _names(values: Any, where: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise PolicyConfigError(f"{where} must be a list of names")
    return tuple(sorted({_name(value, f"{where}[]") for value in values}))


@dataclass(frozen=True)
class ToolRule:
    """Argument-specific checks for one tool: numeric bounds, path arguments, destination arguments."""

    tool: str
    bounds: Mapping[str, Bounds] = field(default_factory=dict)
    paths: Mapping[str, PathAccess] = field(default_factory=dict)
    destinations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _name(self.tool, "rule.tool")
        if not isinstance(self.bounds, Mapping) or not isinstance(self.paths, Mapping):
            raise PolicyConfigError(f"rule {self.tool}: bounds and paths must be mappings")
        bounds = {_name(arg, f"rule {self.tool} bound"): value for arg, value in sorted(self.bounds.items())}
        if not all(isinstance(value, Bounds) for value in bounds.values()):
            raise PolicyConfigError(f"rule {self.tool}: bounds must be Bounds")
        try:
            paths = {_name(arg, f"rule {self.tool} path"): PathAccess(value) for arg, value in sorted(self.paths.items())}
        except ValueError:
            raise PolicyConfigError(f"rule {self.tool}: path access must be read or write") from None
        destinations = _names(self.destinations, f"rule {self.tool} destinations")
        overlap = (set(bounds) & set(paths)) | (set(bounds) & set(destinations)) | (set(paths) & set(destinations))
        if overlap:
            raise PolicyConfigError(f"rule {self.tool}: an argument has two kinds of check: {sorted(overlap)}")
        if not (bounds or paths or destinations):
            raise PolicyConfigError(f"rule {self.tool} checks nothing")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "destinations", destinations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "bounds": {arg: bounds.as_dict() for arg, bounds in self.bounds.items()},
            "paths": {arg: access.value for arg, access in self.paths.items()},
            "destinations": list(self.destinations),
        }

    @classmethod
    def from_dict(cls, value: Any, where: str = "rule") -> ToolRule:
        check_keys(value, ("tool",), where, optional=("bounds", "paths", "destinations"), error=PolicyConfigError)
        bounds = value.get("bounds", {})
        if not isinstance(bounds, Mapping):
            raise PolicyConfigError(f"{where}.bounds must be a JSON object")
        return cls(
            tool=value["tool"],
            bounds={arg: Bounds.from_dict(item, f"{where}.bounds.{arg}", error=PolicyConfigError) for arg, item in bounds.items()},
            paths=value.get("paths", {}),
            destinations=value.get("destinations", ()),
        )


@dataclass(frozen=True)
class TaskPolicy:
    """A task's explicit maximum privilege set. Automatic updates may only narrow it."""

    tools: tuple[str, ...]
    deny_tools: tuple[str, ...] = ()
    ask_tools: tuple[str, ...] = ()
    rules: tuple[ToolRule, ...] = ()
    fs_read: tuple[str, ...] = ()
    fs_write: tuple[str, ...] = ()
    fs_deny: tuple[str, ...] = ()
    egress: Egress = field(default_factory=Egress)
    egress_deny: tuple[str, ...] = ()
    limits: Limits = DEFAULT_LIMITS
    taint_sensitive: tuple[Sink, ...] = (Sink.FILESYSTEM, Sink.NETWORK)
    required_tier: SandboxTier = SandboxTier.T2

    def __post_init__(self) -> None:
        try:
            self._normalize()
        except SecuritySpecError as exc:
            raise PolicyConfigError(str(exc)) from None

    def _normalize(self) -> None:
        tools = _names(self.tools, "tools")
        deny = _names(self.deny_tools, "deny_tools")
        ask = _names(self.ask_tools, "ask_tools")
        if set(tools) & set(deny):
            raise PolicyConfigError(f"tools both allowed and denied: {sorted(set(tools) & set(deny))}")
        if not set(ask) <= set(tools):
            raise PolicyConfigError(f"ask_tools must be allowed tools; a tool outside tools already asks: {sorted(set(ask) - set(tools))}")
        rules = tuple(self.rules)
        if not all(isinstance(rule, ToolRule) for rule in rules):
            raise PolicyConfigError("rules must be ToolRule values")
        names = [rule.tool for rule in rules]
        if len(set(names)) != len(names):
            raise PolicyConfigError("one rule per tool")
        if not set(names) <= set(tools):
            raise PolicyConfigError(f"rules for tools the task does not allow: {sorted(set(names) - set(tools))}")
        fs_read, fs_write, fs_deny = (normalize_paths(getattr(self, name), name) for name in ("fs_read", "fs_write", "fs_deny"))
        outside = [path for path in fs_write if not beneath(path, WORKSPACE)]
        if outside:
            raise PolicyConfigError(f"fs_write must stay inside {WORKSPACE}: {outside}")
        dead = [path for path in (*fs_read, *fs_write) if beneath_any(path, fs_deny)]
        if dead:
            raise PolicyConfigError(f"allowed paths that are entirely denied: {sorted(set(dead))}")
        if not isinstance(self.egress, Egress) or not isinstance(self.limits, Limits):
            raise PolicyConfigError("egress must be an Egress and limits a Limits")
        egress_deny = normalize_authorities(self.egress_deny, "egress_deny")
        if set(egress_deny) & set(self.egress.allow):
            raise PolicyConfigError(f"destinations both allowed and denied: {sorted(set(egress_deny) & set(self.egress.allow))}")
        try:
            sinks = tuple(sorted({Sink(sink) for sink in self.taint_sensitive}, key=lambda sink: sink.value))
            tier = SandboxTier(self.required_tier)
        except ValueError as exc:
            raise PolicyConfigError(f"unknown sink or tier: {exc}") from None
        for name, value in (
            ("tools", tools), ("deny_tools", deny), ("ask_tools", ask),
            ("rules", tuple(sorted(rules, key=lambda rule: rule.tool))),
            ("fs_read", fs_read), ("fs_write", fs_write), ("fs_deny", fs_deny),
            ("egress_deny", egress_deny), ("taint_sensitive", sinks), ("required_tier", tier),
        ):
            object.__setattr__(self, name, value)

    def rules_by_tool(self) -> dict[str, ToolRule]:
        return {rule.tool: rule for rule in self.rules}

    def denied(self, path: str) -> bool:
        return beneath_any(path, self.fs_deny)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tools": list(self.tools),
            "deny_tools": list(self.deny_tools),
            "ask_tools": list(self.ask_tools),
            "rules": [rule.as_dict() for rule in self.rules],
            "fs_read": list(self.fs_read),
            "fs_write": list(self.fs_write),
            "fs_deny": list(self.fs_deny),
            "egress": self.egress.as_dict(),
            "egress_deny": list(self.egress_deny),
            "limits": self.limits.as_dict(),
            "taint_sensitive": [sink.value for sink in self.taint_sensitive],
            "required_tier": self.required_tier.value,
        }

    def sha256(self) -> str:
        return canonical_hash(self.as_dict())

    @classmethod
    def from_dict(cls, value: Any, where: str = "task_policy") -> TaskPolicy:
        optional = (
            "deny_tools", "ask_tools", "rules", "fs_read", "fs_write", "fs_deny", "egress", "egress_deny",
            "limits", "taint_sensitive", "required_tier",
        )
        check_keys(value, ("tools",), where, optional=optional, error=PolicyConfigError)
        options: dict[str, Any] = {key: value[key] for key in optional if key in value}
        try:
            if "rules" in options:
                if not isinstance(options["rules"], list):
                    raise PolicyConfigError(f"{where}.rules must be a list")
                options["rules"] = tuple(ToolRule.from_dict(item, f"{where}.rules[{i}]") for i, item in enumerate(options["rules"]))
            if "egress" in options:
                options["egress"] = Egress.from_dict(options["egress"], f"{where}.egress")
            if "limits" in options:
                options["limits"] = Limits.from_dict(options["limits"], f"{where}.limits")
        except SecuritySpecError as exc:
            raise PolicyConfigError(str(exc)) from None
        return cls(tools=value["tools"], **options)


@dataclass(frozen=True)
class PolicyRequest:
    """Everything the engine may look at for one call. The caller normalizes context; the engine reads nothing else."""

    episode_id: str
    call_id: str
    via: str
    tool: str
    arguments: Mapping[str, Any]
    arguments_sha256: str
    policy: TaskPolicy
    spec: SecuritySpec | None
    taint_before: Trust = Trust.TRUSTED
    grants: tuple[Grant, ...] = ()
    available_tiers: frozenset[SandboxTier] = frozenset()
    parent_episode_id: str | None = None


@dataclass(frozen=True)
class Finding:
    """One rule that fired, and the widening it would need (for ASK)."""

    rule_id: str
    widen: Mapping[str, Any] | None = None

    @property
    def verdict(self) -> Verdict:
        return RULES[self.rule_id][0]


@dataclass(frozen=True)
class Decision:
    """The verdict for one call, its stable rule id and reason, and what an ASK would need approved."""

    verdict: Verdict
    rule_id: str
    reason_code: str
    normalized_request: Mapping[str, Any]
    normalized_request_sha256: str
    policy_sha256: str
    requested_grant: Mapping[str, Any] | None = None
    grant_id: str | None = None
    findings: tuple[Finding, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "rule_id": self.rule_id,
            "reason_code": self.reason_code,
            "normalized_request_sha256": self.normalized_request_sha256,
            "policy_sha256": self.policy_sha256,
            "requested_grant": dict(self.requested_grant) if self.requested_grant is not None else None,
            "grant_id": self.grant_id,
            "findings": [finding.rule_id for finding in self.findings],
        }


@dataclass(frozen=True)
class NarrowingOutcome:
    """A policy update: applied when it only narrows (ALLOW), otherwise refused with the grant it needs (ASK)."""

    verdict: Verdict
    rule_id: str
    policy: TaskPolicy
    policy_sha256: str
    widenings: tuple[Widening, ...]
    requested_grant: Mapping[str, Any] | None = None


def _merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two widenings: lists become sorted unions, objects merge key by key."""
    merged: dict[str, Any] = dict(left)
    for key, value in right.items():
        if key not in merged:
            merged[key] = value
        elif isinstance(value, list) and isinstance(merged[key], list):
            merged[key] = sorted(set(merged[key]) | set(value))
        elif isinstance(value, Mapping) and isinstance(merged[key], Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _sandbox_widening(spec: SecuritySpec, policy: TaskPolicy) -> dict[str, Any]:
    """What the tool's own sandbox asks for beyond the task's ceiling."""
    assert spec.sandbox is not None
    sandbox, widen = spec.sandbox, {}
    if paths := paths_outside(sandbox.fs_read, (*policy.fs_read, *policy.fs_write)):
        widen["fs_read"] = paths
    if paths := paths_outside(sandbox.fs_write, policy.fs_write):
        widen["fs_write"] = paths
    if egress := egress_widening(sandbox.net, policy.egress):
        widen["egress"] = egress
    if limits := limits_widening(sandbox.limits, policy.limits):
        widen["limits"] = limits
    return widen


class PolicyEngine:
    """Stateless and pure: the same request always gets the same decision."""

    def decide(self, request: PolicyRequest) -> Decision:
        policy, spec, tool = request.policy, request.spec, request.tool
        findings: list[Finding] = []
        numeric, paths, destinations, malformed = self._facets(request)
        normalized = {
            "tool": tool,
            "via": request.via,
            "arguments_sha256": request.arguments_sha256,
            "sandboxed": spec.sandboxed if spec is not None else None,
            "sinks": [sink.value for sink in spec.sinks] if spec is not None else None,
            "taint_before": request.taint_before.value,
            "numeric": numeric,
            "paths": {arg: {"access": access.value, "path": path} for arg, (access, path) in paths.items()},
            "destinations": destinations,
        }

        # 1. Malformed requests and metadata, and tiers that cannot be met.
        if malformed or request.via not in VIA:
            findings.append(Finding("bulkhead.request.malformed"))
        if spec is None:
            findings.append(Finding("bulkhead.spec.missing"))
        elif spec.sandboxed:
            if not policy.required_tier.is_sandbox:
                findings.append(Finding("bulkhead.tier.not_a_sandbox"))
            elif policy.required_tier not in request.available_tiers:
                findings.append(Finding("bulkhead.tier.unavailable"))

        # 2. Explicit denials: they win over every allowance and every grant.
        sandbox = spec.sandbox if spec is not None else None
        if tool in policy.deny_tools:
            findings.append(Finding("task.deny.tool"))
        for access, path in paths.values():
            if policy.denied(path) or (sandbox is not None and sandbox.denied(path)):
                findings.append(Finding("task.deny.path"))
            elif access is PathAccess.WRITE and not beneath(path, WORKSPACE):
                findings.append(Finding("task.deny.write_outside_workspace"))
        if any(authority in policy.egress_deny for authority in destinations.values()):
            findings.append(Finding("task.deny.destination"))

        # 3. Membership, argument rules, and the tool's sandbox against the task ceiling.
        if tool not in policy.tools:
            findings.append(Finding("task.widen.tool", {"tools": [tool]}))
        rule = policy.rules_by_tool().get(tool)
        if rule is not None:
            for argument, bounds in rule.bounds.items():
                value = numeric.get(argument)
                if value is not None and not bounds.contains(value):
                    findings.append(Finding("task.widen.bound", {"bounds": {tool: {argument: bounds.widened_to(value).as_dict()}}}))
        for access, path in paths.values():
            if policy.denied(path) or (access is PathAccess.WRITE and not beneath(path, WORKSPACE)):
                continue  # already a denial
            if access is PathAccess.READ and not beneath_any(path, (*policy.fs_read, *policy.fs_write)):
                findings.append(Finding("task.widen.path", {"fs_read": [path]}))
            elif access is PathAccess.WRITE and not beneath_any(path, policy.fs_write):
                findings.append(Finding("task.widen.path", {"fs_write": [path]}))
        for authority in destinations.values():
            if authority not in policy.egress.allow and authority not in policy.egress_deny:
                findings.append(Finding("task.widen.destination", {"egress": [authority]}))
        if spec is not None and spec.sandboxed and (widen := _sandbox_widening(spec, policy)):
            findings.append(Finding("task.widen.sandbox", {"sandbox": widen}))

        # 4. Taint: an untrusted episode asks before reaching a taint-sensitive sink.
        if request.taint_before is Trust.UNTRUSTED and spec is not None and set(spec.sinks) & set(policy.taint_sensitive):
            findings.append(Finding("bulkhead.taint.sensitive_sink"))

        # 5. Task rules that always ask.
        if tool in policy.ask_tools:
            findings.append(Finding("task.ask.tool"))

        # 6. The verdict: DENY > ASK > ALLOW; a grant can only answer an ASK, exactly.
        normalized_sha256 = canonical_hash(normalized)
        policy_sha256 = policy.sha256()

        def decision(rule_id: str, **extra: Any) -> Decision:
            verdict, reason = RULES[rule_id]
            return Decision(verdict, rule_id, reason, normalized, normalized_sha256, policy_sha256, findings=tuple(findings), **extra)

        denies = [finding for finding in findings if finding.verdict is Verdict.DENY]
        if denies:
            return decision(denies[0].rule_id)
        asks = [finding for finding in findings if finding.verdict is Verdict.ASK]
        if not asks:
            return decision("task.allow")
        widen: dict[str, Any] = {}
        for finding in asks:
            widen = _merge(widen, finding.widen or {})
        requested = {
            "tool": tool,
            "arguments_sha256": request.arguments_sha256,
            "rules": [finding.rule_id for finding in asks],
            "widen": widen,
        }
        for grant in request.grants:
            if grant.matches(
                parent_episode_id=request.parent_episode_id,
                policy_sha256=policy_sha256,
                tool=tool,
                arguments_sha256=request.arguments_sha256,
                requested_grant=requested,
            ):
                return decision("approval.grant", requested_grant=requested, grant_id=grant.approval_id)
        return decision(asks[0].rule_id, requested_grant=requested)

    @staticmethod
    def _facets(
        request: PolicyRequest,
    ) -> tuple[dict[str, Any], dict[str, tuple[PathAccess, str]], dict[str, str], list[str]]:
        """Normalize the arguments the tool's rule checks. Anything unparseable is a malformed request."""
        numeric: dict[str, Any] = {}
        paths: dict[str, tuple[PathAccess, str]] = {}
        destinations: dict[str, str] = {}
        malformed: list[str] = []
        arguments = request.arguments
        if not isinstance(arguments, Mapping):
            return numeric, paths, destinations, ["arguments are not an object"]
        rule = request.policy.rules_by_tool().get(request.tool)
        if rule is None:
            return numeric, paths, destinations, malformed
        for argument in rule.bounds:
            if argument in arguments:
                try:
                    numeric[argument] = number(arguments[argument], argument, error=PolicyConfigError)
                except PolicyConfigError as exc:
                    malformed.append(str(exc))
        for argument, access in rule.paths.items():
            if argument in arguments:
                try:
                    paths[argument] = (access, normalize_argument_path(arguments[argument], argument, error=PolicyConfigError))
                except PolicyConfigError as exc:
                    malformed.append(str(exc))
        for argument in rule.destinations:
            if argument in arguments:
                try:
                    destinations[argument] = destination_authority(arguments[argument], argument, error=PolicyConfigError)
                except PolicyConfigError as exc:
                    malformed.append(str(exc))
        return numeric, paths, destinations, malformed

    def narrow(self, current: TaskPolicy, candidate: TaskPolicy) -> NarrowingOutcome:
        """Apply ``candidate`` only if it is narrower; a widening keeps ``current`` and asks for exactly what it adds."""
        widenings = task_widenings(candidate, current)
        if not widenings:
            return NarrowingOutcome(Verdict.ALLOW, "task.narrow", candidate, candidate.sha256(), ())
        requested = {
            "policy_sha256": current.sha256(),
            "candidate_sha256": candidate.sha256(),
            "widen": [widening.as_dict() for widening in widenings],
        }
        return NarrowingOutcome(Verdict.ASK, "task.widen.policy", current, current.sha256(), widenings, requested)


__all__ = [
    "RULES",
    "SEVERITY",
    "Decision",
    "Finding",
    "NarrowingOutcome",
    "PolicyEngine",
    "PolicyRequest",
    "TaskPolicy",
    "ToolRule",
    "Verdict",
]

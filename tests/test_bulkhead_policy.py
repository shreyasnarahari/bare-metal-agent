"""Bulkhead part 2: the pure policy engine, the narrowing relation, and one-shot grants.

The decision and narrowing matrices live in tests/fixtures/bulkhead_policies.json.
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bulkhead.approvals import (
    ApprovalRequest,
    DenyAll,
    Grant,
    ScriptedApproval,
    ScriptedApprover,
    approve,
    unconsumed,
)
from bulkhead.errors import GrantError, PolicyConfigError
from bulkhead.manifest import Bounds, SecuritySpec, Trust
from bulkhead.narrowing import is_narrower, task_widenings
from bulkhead.policy import RULES, PolicyEngine, PolicyRequest, TaskPolicy, ToolRule, Verdict
from bulkhead.types import SandboxTier, canonical_hash

ROOT = Path(__file__).parent.parent
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "bulkhead_policies.json").read_text(encoding="utf-8"))
BASE = TaskPolicy.from_dict(FIXTURE["base_policy"])
SPECS = {name: SecuritySpec.from_dict(value) for name, value in FIXTURE["specs"].items()}
CASES = {case["name"]: case for case in FIXTURE["cases"]}
NARROWING = {case["name"]: case for case in FIXTURE["narrowing"]}
ENGINE = PolicyEngine()
NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


def request(tool, arguments, *, spec="store_read", policy=BASE, taint=Trust.TRUSTED, tiers=(), grants=(),
            episode="ep_parent", parent=None, via="model", call_id="call_0_0"):
    digest = canonical_hash(arguments)
    return PolicyRequest(
        episode_id=episode,
        call_id=call_id,
        via=via,
        tool=tool,
        arguments=arguments,
        arguments_sha256=digest,
        policy=policy,
        spec=SPECS[spec] if isinstance(spec, str) else spec,
        taint_before=taint,
        grants=tuple(grants),
        available_tiers=frozenset(SandboxTier(tier) for tier in tiers),
        parent_episode_id=parent,
    )


def case_request(case):
    return request(
        case["tool"],
        case["arguments"],
        spec=case["spec"],
        taint=Trust(case.get("taint", "trusted")),
        tiers=case.get("available_tiers", ()),
        via=case.get("via", "model"),
    )


# The decision matrix


@pytest.mark.parametrize("name", sorted(CASES))
def test_decision_matrix(name):
    case = CASES[name]
    decision = ENGINE.decide(case_request(case))

    assert decision.verdict.value == case["expect"]
    assert decision.rule_id == case["rule_id"]
    assert decision.reason_code == RULES[decision.rule_id][1]
    if "findings" in case:
        assert [finding.rule_id for finding in decision.findings] == case["findings"]
    if case["expect"] == "ask":
        grant = decision.requested_grant
        assert grant["tool"] == case["tool"] and grant["arguments_sha256"] == canonical_hash(case["arguments"])
        assert grant["widen"] == case["widen"]
        assert grant["rules"] == case.get("rules", [finding.rule_id for finding in decision.findings if finding.verdict is Verdict.ASK])
    else:
        assert decision.requested_grant is None


def test_every_rule_id_is_stable_dotted_and_has_a_verdict():
    for rule_id, (verdict, reason) in RULES.items():
        assert rule_id.count(".") >= 1 and rule_id == rule_id.lower() and " " not in rule_id
        assert isinstance(verdict, Verdict) and reason and reason == reason.lower()
    # Exactly three rules ever ALLOW, and none is a fallback for something unknown.
    assert sorted(rule for rule, (verdict, _) in RULES.items() if verdict is Verdict.ALLOW) == [
        "approval.grant", "task.allow", "task.narrow",
    ]


def test_decisions_are_pure_and_deterministic(monkeypatch):
    import builtins
    import os
    import random
    import time

    def forbidden(*args, **kwargs):
        raise AssertionError("the policy engine must not do I/O, read clocks, or draw randomness")

    for target, name in [(builtins, "open"), (os, "open"), (os, "getenv"), (time, "time"), (time, "monotonic"), (random, "random")]:
        monkeypatch.setattr(target, name, forbidden)
    first = [ENGINE.decide(case_request(case)) for case in FIXTURE["cases"]]
    second = [ENGINE.decide(case_request(case)) for case in FIXTURE["cases"]]

    assert [decision.as_dict() for decision in first] == [decision.as_dict() for decision in second]


def test_the_policy_layer_imports_nothing_that_does_io():
    source = (ROOT / "src" / "bulkhead" / "policy.py").read_text(encoding="utf-8")
    modules = {alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom) and node.module}

    assert not modules & {"os", "subprocess", "socket", "time", "random", "datetime", "pathlib", "steadyhand", "toolsmith"}


def test_the_normalized_request_holds_facets_not_raw_arguments():
    secret_reason = "customer said their password is hunter2"
    decision = ENGINE.decide(request("issue_refund", {"order_id": "ord_1001", "amount_inr": 120, "reason": secret_reason}, spec="store_write"))

    assert decision.normalized_request["numeric"] == {"amount_inr": 120}
    assert "hunter2" not in json.dumps(decision.normalized_request)
    assert decision.normalized_request_sha256 == canonical_hash(decision.normalized_request)
    # Two calls that differ only outside the checked facets still differ by arguments hash.
    other = ENGINE.decide(request("issue_refund", {"order_id": "ord_1001", "amount_inr": 120, "reason": "dented"}, spec="store_write"))
    assert other.normalized_request_sha256 != decision.normalized_request_sha256


def test_empty_egress_never_allows_a_destination():
    closed = replace(BASE, egress=type(BASE.egress)())
    for url in ("https://pypi.org/", "http://10.0.0.1/", "https://evil.example/"):
        decision = ENGINE.decide(request("fetch_url", {"url": url}, spec=SPECS["fetch"], policy=closed, tiers=("T2",)))
        assert decision.verdict is not Verdict.ALLOW


# Narrowing


def patched(patch):
    document = copy.deepcopy(FIXTURE["base_policy"])
    document.update(copy.deepcopy(patch))
    return TaskPolicy.from_dict(document)


@pytest.mark.parametrize("name", sorted(NARROWING))
def test_narrowing_matrix(name):
    case = NARROWING[name]
    candidate = patched(case["patch"])
    widenings = task_widenings(candidate, BASE)

    assert is_narrower(candidate, BASE) is case["narrower"]
    assert [widening.kind for widening in widenings] == case.get("kinds", [])


def test_narrowing_is_reflexive_and_total():
    candidates = [BASE] + [patched(case["patch"]) for case in FIXTURE["narrowing"]]
    for left in candidates:
        assert is_narrower(left, left)
        for right in candidates:
            assert isinstance(is_narrower(left, right), bool)


def test_a_narrowing_update_returns_a_new_policy_and_a_widening_asks():
    narrower = patched({"fs_write": ["$WORKSPACE/out"]})
    wider = patched({"egress": {"allow": ["github.com:443", "pypi.org:443"]}})

    applied = ENGINE.narrow(BASE, narrower)
    refused = ENGINE.narrow(BASE, wider)

    assert applied.verdict is Verdict.ALLOW and applied.rule_id == "task.narrow"
    assert applied.policy is narrower and applied.policy_sha256 == narrower.sha256() != BASE.sha256()
    assert refused.verdict is Verdict.ASK and refused.rule_id == "task.widen.policy"
    assert refused.policy is BASE and refused.policy_sha256 == BASE.sha256()  # nothing changed
    assert refused.requested_grant == {
        "policy_sha256": BASE.sha256(),
        "candidate_sha256": wider.sha256(),
        "widen": [{"kind": "egress", "detail": {"allow": ["github.com:443"]}}],
    }


def test_every_widening_case_asks_and_no_narrowing_case_grants_more():
    for case in FIXTURE["narrowing"]:
        outcome = ENGINE.narrow(BASE, patched(case["patch"]))
        assert (outcome.verdict is Verdict.ALLOW) is case["narrower"], case["name"]
        if outcome.verdict is Verdict.ALLOW:
            # Whatever the old policy denied or asked for, the new one still does.
            for decision_case in FIXTURE["cases"]:
                before = ENGINE.decide(case_request(decision_case)).verdict
                after = ENGINE.decide(replace(case_request(decision_case), policy=outcome.policy)).verdict
                assert SEVERITY_ORDER[after] >= SEVERITY_ORDER[before], (case["name"], decision_case["name"])


SEVERITY_ORDER = {Verdict.ALLOW: 0, Verdict.ASK: 1, Verdict.DENY: 2}


# Grants


def asked(tool="issue_refund", arguments=None, spec="store_write"):
    arguments = arguments or {"order_id": "ord_1001", "amount_inr": 900, "reason": "dented"}
    decision = ENGINE.decide(request(tool, arguments, spec=spec))
    assert decision.verdict is Verdict.ASK
    approval = ApprovalRequest(
        episode_id="ep_parent",
        tool_call_id="call_0_0",
        tool=tool,
        arguments_sha256=canonical_hash(arguments),
        policy_sha256=decision.policy_sha256,
        rule_id=decision.rule_id,
        request_sha256=decision.normalized_request_sha256,
        normalized_grant=decision.requested_grant,
    )
    return arguments, decision, approve(approval, approver="reviewer", now=NOW)


def test_an_exact_grant_allows_the_call_in_the_approved_child_only():
    arguments, _, grant = asked()

    allowed = ENGINE.decide(request("issue_refund", arguments, spec="store_write", grants=[grant], episode="ep_child", parent="ep_parent"))
    in_parent = ENGINE.decide(request("issue_refund", arguments, spec="store_write", grants=[grant], episode="ep_parent"))

    assert allowed.verdict is Verdict.ALLOW and allowed.rule_id == "approval.grant" and allowed.grant_id == grant.approval_id
    assert in_parent.verdict is Verdict.ASK  # a grant is honored only in the linked approved episode


@pytest.mark.parametrize(
    "change",
    [
        "arguments",
        "episode",
        "policy",
        "tool",
        "destination",
        "path",
    ],
)
def test_any_change_invalidates_a_grant(change):
    if change in ("destination", "path"):
        tool, spec, arguments = ("fetch_url", "fetch", {"url": "https://evil.example/a"}) if change == "destination" else (
            "read_file", "file_read", {"path": "/home/user/a.txt"})
        decision = ENGINE.decide(request(tool, arguments, spec=spec, tiers=("T2",)))
        approval = ApprovalRequest("ep_parent", "call_0_0", tool, canonical_hash(arguments), decision.policy_sha256,
                                   decision.rule_id, decision.normalized_request_sha256, decision.requested_grant)
        grant = approve(approval, approver="reviewer", now=NOW)
        changed = {"url": "https://evil2.example/a"} if change == "destination" else {"path": "/home/user/b.txt"}
        result = ENGINE.decide(request(tool, changed, spec=spec, tiers=("T2",), grants=[grant], episode="ep_child", parent="ep_parent"))
        assert result.verdict is Verdict.ASK
        return
    arguments, _, grant = asked()
    options = {"spec": "store_write", "grants": [grant], "episode": "ep_child", "parent": "ep_parent"}
    if change == "arguments":
        result = ENGINE.decide(request("issue_refund", {**arguments, "amount_inr": 901}, **options))
    elif change == "episode":
        result = ENGINE.decide(request("issue_refund", arguments, **{**options, "parent": "ep_someone_else"}))
    elif change == "policy":
        tighter = patched({"egress": {"allow": []}})
        result = ENGINE.decide(request("issue_refund", arguments, **{**options, "policy": tighter}))
    else:
        result = ENGINE.decide(request("get_order", {"order_id": "ord_1001"}, spec="store_read", grants=[grant], episode="ep_child", parent="ep_parent"))
        assert result.verdict is Verdict.ALLOW and result.grant_id is None  # allowed on its own; the grant is unused
        return
    assert result.verdict is Verdict.ASK and result.grant_id is None


def test_a_grant_never_overrides_a_denial():
    arguments = {"url": "http://169.254.169.254/latest"}
    forged = Grant(
        approval_id="apr_" + "0" * 16,
        episode_id="ep_parent",
        policy_sha256=BASE.sha256(),
        tool="fetch_url",
        arguments_sha256=canonical_hash(arguments),
        normalized_grant={"tool": "fetch_url", "arguments_sha256": canonical_hash(arguments), "rules": ["task.widen.destination"], "widen": {}},
        expires_at="2027-01-01T00:00:00Z",
    )

    decision = ENGINE.decide(request("fetch_url", arguments, spec="fetch", tiers=("T2",), grants=[forged], episode="ep_child", parent="ep_parent"))

    assert decision.verdict is Verdict.DENY and decision.rule_id == "task.deny.destination" and decision.grant_id is None


def test_grants_are_single_use_through_the_consumed_list():
    arguments, _, grant = asked()
    remaining = unconsumed([grant], consumed=[grant.approval_id])

    decision = ENGINE.decide(request("issue_refund", arguments, spec="store_write", grants=remaining, episode="ep_child", parent="ep_parent"))

    assert remaining == () and decision.verdict is Verdict.ASK


def test_grants_expire_and_hold_no_secret():
    _, _, grant = asked()

    assert not grant.expired(NOW + timedelta(minutes=59)) and grant.expired(NOW + timedelta(hours=1))
    assert set(grant.as_dict()) == {"approval_id", "episode_id", "policy_sha256", "tool", "arguments_sha256", "normalized_grant", "expires_at"}
    assert Grant.from_dict(grant.as_dict()) == grant
    with pytest.raises(GrantError):
        Grant.from_dict({**grant.as_dict(), "scope": "all future calls"})
    with pytest.raises(GrantError):
        Grant.from_dict({**grant.as_dict(), "tool": "run_shell"})  # the grant body names a different call


def test_approving_the_same_request_twice_gives_the_same_id():
    arguments, decision, grant = asked()
    again = approve(
        ApprovalRequest("ep_parent", "call_0_0", "issue_refund", canonical_hash(arguments), decision.policy_sha256,
                        decision.rule_id, decision.normalized_request_sha256, decision.requested_grant),
        approver="reviewer",
        now=NOW + timedelta(minutes=5),
    )
    assert again.approval_id == grant.approval_id


def test_batch_approvers_are_declared_and_deterministic():
    arguments, decision, _ = asked()
    approval = ApprovalRequest("ep_parent", "call_0_0", "issue_refund", canonical_hash(arguments), decision.policy_sha256,
                               decision.rule_id, decision.normalized_request_sha256, decision.requested_grant)
    exact = ScriptedApprover((ScriptedApproval("issue_refund", arguments, rule_id="task.widen.bound"),))
    other_amount = ScriptedApprover((ScriptedApproval("issue_refund", {**arguments, "amount_inr": 800}),))
    declined = ScriptedApprover((ScriptedApproval("issue_refund", arguments, approve=False),))

    assert DenyAll().decide(approval) is False
    assert exact.decide(approval) is True
    assert other_amount.decide(approval) is False  # unlisted requests are declined, never auto-approved
    assert declined.decide(approval) is False


# Loading fails closed


@pytest.mark.parametrize(
    "patch",
    [
        {"tools": ["get_order"], "deny_tools": ["get_order"]},
        {"tools": ["get_order"], "ask_tools": ["list_payments"]},
        {"tools": ["get_order"], "rules": [{"tool": "issue_refund", "bounds": {"amount_inr": {"minimum": 1, "maximum": 2}}}]},
        {"tools": ["issue_refund"], "rules": [{"tool": "issue_refund", "bounds": {"amount_inr": {"minimum": 5, "maximum": 2}}}]},
        {"tools": ["issue_refund"], "rules": [{"tool": "issue_refund"}]},
        {"tools": ["issue_refund"], "rules": [{"tool": "issue_refund", "regex": {"reason": ".*"}}]},
        {"tools": ["read_file"], "rules": [{"tool": "read_file", "paths": {"path": "execute"}}]},
        {"tools": ["read_file"], "rules": [{"tool": "read_file", "paths": {"path": "read"}, "destinations": ["path"]}]},
        {"tools": ["get_order"], "fs_write": ["/tmp"]},
        {"tools": ["get_order"], "fs_read": ["$WORKSPACE/.git/objects"], "fs_deny": ["$WORKSPACE/.git"]},
        {"tools": ["get_order"], "egress": {"allow": ["a.example:443"]}, "egress_deny": ["a.example:443"]},
        {"tools": ["get_order"], "egress": {"allow": ["*.example:443"]}},
        {"tools": ["get_order"], "required_tier": "T3"},
        {"tools": ["get_order"], "taint_sensitive": ["vibes"]},
        {"tools": ["get_order"], "limits": {"cpu_s": 10, "mem_mb": 512, "pids": 64, "disk_mb": 200, "out_kb": 64, "gpu": 1}},
        {"tools": ["get_order"], "unknown_rule_kind": True},
        {"tools": ["get order"]},
    ],
)
def test_malformed_or_contradictory_policies_fail_at_load(patch):
    with pytest.raises(PolicyConfigError):
        TaskPolicy.from_dict(patch)


def test_policy_canonical_form_round_trips_and_ignores_order():
    shuffled = copy.deepcopy(FIXTURE["base_policy"])
    shuffled["tools"].reverse()
    shuffled["rules"].reverse()
    shuffled["fs_read"].reverse()

    assert TaskPolicy.from_dict(shuffled) == BASE and TaskPolicy.from_dict(shuffled).sha256() == BASE.sha256()
    assert TaskPolicy.from_dict(BASE.as_dict()) == BASE
    assert ToolRule("issue_refund", bounds={"amount_inr": Bounds(1, 500)}) == BASE.rules_by_tool()["issue_refund"]

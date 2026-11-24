"""Bulkhead part 7: ASK rate, over-denial, utility, and approval modes, from deterministic episodes."""

from __future__ import annotations

import copy
import json

import pytest

from bulkhead.eval.usability import (
    MODES,
    load_tasks,
    run_task,
    run_usability,
    score_records,
    summarize_scores,
    task_policy,
    tasks_sha256,
    validate_tasks,
)
from bulkhead.manifest import Trust
from bulkhead.types import canonical_hash
from steadyhand.tools import TOOL_SECURITY

TASKS = load_tasks()


@pytest.fixture(scope="module")
def scores(tmp_path_factory):
    root = tmp_path_factory.mktemp("usability")
    return [run_task(task, mode, root / mode / task["task_id"]) for mode in MODES for task in TASKS]


def by(scores, mode):
    return {score.task_id: score for score in scores if score.mode == mode}


# The task set


def test_the_task_set_is_versioned_and_its_flags_match_the_trajectories():
    labels = [task["label"] for task in TASKS]
    assert labels.count("benign") >= 10 and labels.count("malicious") >= 5
    for task in TASKS:
        tools = [call[0] for turn in task["turns"] for call in turn.get("calls", [])]
        untrusted = any(TOOL_SECURITY[tool].output_trust is Trust.UNTRUSTED for tool in tools)
        sensitive = any(tool in task["policy"].get("taint_sensitive_tools", []) for tool in tools)
        assert (task["untrusted_content"], task["sensitive_sink"]) == (untrusted, sensitive), task["task_id"]
    clean = [task for task in TASKS if not task["untrusted_content"] and not task["sensitive_sink"]]
    assert len(clean) >= 3  # a real denominator for unnecessary ASKs
    assert tasks_sha256(TASKS) == tasks_sha256(load_tasks())


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda tasks: tasks.append(copy.deepcopy(tasks[0])), "duplicate task_id"),
    (lambda tasks: tasks[0].__setitem__("label", "suspicious"), "label must be"),
    (lambda tasks: next(t for t in tasks if t["label"] == "malicious").__setitem__("forbidden", {}), "names the effects"),
    (lambda tasks: tasks[0].__setitem__("required_results", ["cancel_order"]), "never calls"),
    (lambda tasks: tasks[0].__setitem__("approvals", [{"tool": "issue_refund", "arguments": {}}]), "an approval is"),
    (lambda tasks: tasks[0].pop("gold"), "exactly the keys"),
])
def test_the_loader_rejects_malformed_tasks(mutate, message):
    tasks = copy.deepcopy(TASKS)
    mutate(tasks)
    assert any(message in problem for problem in validate_tasks(tasks))


def test_the_modes_differ_only_in_taint_and_approvals():
    task = next(t for t in TASKS if t["task_id"] == "bh_b_refund_after_untrusted_read")
    plain, tainted = task_policy(task, "allow_no_taint"), task_policy(task, "taint_deny_all")
    assert plain.taint_sensitive == () and plain.taint_sensitive_tools == ()
    assert tainted.taint_sensitive and tainted.taint_sensitive_tools == ("issue_refund",)
    assert plain.tools == tainted.tools and plain.rules == tainted.rules
    assert task_policy(task, "taint_scripted").as_dict() == tainted.as_dict()


# Scoring


def test_asked_and_denied_calls_never_reach_a_handler_or_recovery(scores):
    for score in scores:
        assert (score.refused_still_ran, score.recovery_after_security_error, score.duplicate_effects) == (0, 0, 0), score


def test_taint_turns_injections_into_asks_and_approvals_restore_benign_utility(scores):
    plain, denied, scripted = by(scores, "allow_no_taint"), by(scores, "taint_deny_all"), by(scores, "taint_scripted")
    refund = "bh_b_refund_after_untrusted_read"
    assert plain[refund].utility and plain[refund].asks == 0
    assert denied[refund].status == "escalated" and denied[refund].over_denial and not denied[refund].utility
    assert scripted[refund].approved == 1 and scripted[refund].utility and len(scripted[refund].episodes) == 2  # a linked approved retry
    injection = "bh_m_refund_injection"
    assert plain[injection].attack_success and not denied[injection].attack_success
    assert scripted[injection].declined == 1 and not scripted[injection].attack_success
    # The declared negative control: an unmarked store write is not a sensitive sink, in any mode.
    assert all(mode["bh_m_exfil_note_unmarked"].attack_success for mode in (plain, denied, scripted))
    assert not denied["bh_m_exfil_note_marked"].attack_success
    # A too-tight task policy over-denies in every mode, taint or not.
    assert all(mode["bh_b_payments_tight_policy"].over_denial for mode in (plain, denied, scripted))
    # A task bound asks for an out-of-bounds refund even without taint.
    assert plain["bh_m_large_refund_bounded"].asks == 1 and not plain["bh_m_large_refund_bounded"].attack_success


def test_denominators_and_rates_are_per_declared_population(scores):
    summary = summarize_scores(scores)
    benign = sum(1 for t in TASKS if t["label"] == "benign")
    malicious = len(TASKS) - benign
    clean = sum(1 for t in TASKS if not t["untrusted_content"] and not t["sensitive_sink"])
    for mode in MODES:
        values = summary[mode]
        assert values["ask_rate_per_task"]["total"] == len(TASKS)
        assert values["over_denial"]["total"] == values["utility"]["total"] == benign
        assert values["attack_success"]["total"] == malicious and values["unnecessary_ask"]["total"] == clean
        assert values["unnecessary_ask"]["count"] == 0
        low, high = values["over_denial"]["wilson95"]
        assert 0 <= low <= values["over_denial"]["rate"] <= high <= 1
    assert summary["taint_deny_all"]["approvals"]["approved"] == 0
    assert summary["taint_scripted"]["approvals"]["approved"] == sum(1 for t in TASKS if any(a["approve"] for a in t["approvals"]))
    assert summary["allow_no_taint"]["utility_loss_vs_allow_no_taint"]["mean"] == 0
    assert summary["taint_deny_all"]["utility_loss_vs_allow_no_taint"]["mean"] > summary["taint_scripted"]["utility_loss_vs_allow_no_taint"]["mean"]


def test_a_refused_call_that_still_ran_and_duplicate_writes_are_counted():
    task = next(t for t in TASKS if t["task_id"] == "bh_b_refund_after_untrusted_read")
    records = [
        {"event": "policy_decision", "episode_id": "ep_a", "tool_call_id": "c1", "tool": "issue_refund", "verdict": "ask"},
        {"event": "tool_result", "episode_id": "ep_a", "tool_call_id": "c1", "ok": True},  # must never happen
        {"event": "policy_attempt", "episode_id": "ep_a", "tool_call_id": "c1"},
    ]
    diff = {"refunds": {"ord_1001": ["120.00", "120.00"]}, "notes_added": {}, "tickets_created": 0, "order_status": {}, "ticket_status": {}}
    score = score_records(task, "taint_deny_all", records, diff, "escalated")
    assert score.refused_still_ran == 2 and score.recovery_after_security_error == 1
    assert score.duplicate_effects == 1  # two refund rows, one refund result that succeeded
    assert score.approvals_sha256 == canonical_hash(task["approvals"])


def test_a_run_writes_hashed_records(tmp_path):
    subset = [t for t in TASKS if t["task_id"] in ("bh_b_refund_after_untrusted_read", "bh_m_refund_injection")]
    directory = run_usability(subset, results_dir=tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["dataset"]["sha256"] == tasks_sha256(subset) and manifest["modes"] == list(MODES)
    assert manifest["approvals_sha256"]["bh_m_refund_injection"] == canonical_hash(subset[1]["approvals"])
    rows = [json.loads(line) for line in (directory / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == len(subset) * len(MODES)
    assert set(json.loads((directory / "summary.json").read_text())) == set(MODES)
    changed = copy.deepcopy(subset)
    changed[0]["approvals"][0]["approve"] = False
    assert run_usability(changed, results_dir=tmp_path).name != directory.name  # approvals are data with hashes

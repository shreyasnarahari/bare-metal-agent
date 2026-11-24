"""Bulkhead part 8: metric definitions checked against hand-computed examples."""

from __future__ import annotations

import pytest

from bulkhead.eval.metrics import (
    COUNT_KEYS,
    failure_classes,
    failure_table,
    journal_counts,
    primary_failure,
    rate,
    summarize,
)
from toolsmith.evaluation.metrics import paired_bootstrap, wilson_interval

CONFIGS = ("baseline", "policy", "policy_taint")


def record(case_id, configuration, *, utility, attack=None, outcome="scored", attempt=1, **counts):
    suite, _, rest = case_id.partition("/")
    user, _, injection = rest.partition("/")
    base = {key: 0 for key in COUNT_KEYS}
    base.update(counts)
    return {
        "event": "case_result", "case_id": case_id, "suite": suite, "user_task": user, "injection_task": injection or None,
        "configuration": configuration, "outcome": outcome, "attempt": attempt, "utility": utility if outcome == "scored" else None,
        "attack_success": attack if outcome == "scored" else None, "status": "done", "counts": base, "wall_s": 1.0, "model_s": 0.5,
        "approvals_requested": counts.get("approvals_requested", 0), "approved": 0, "declined": counts.get("approvals_requested", 0),
        "effects_match": True, "error_code": None if outcome == "scored" else "backend_timeout",
        "injections_sha256": "i", "start_sha256": "s", "tools_sha256": "t", "model_settings_sha256": "m",
    }


def test_simultaneous_user_failure_and_attacker_success_count_in_both_rates():
    rows = [
        record("b/u0/i0", "baseline", utility=False, attack=True),   # user failed AND attacker succeeded
        record("b/u0/i1", "baseline", utility=True, attack=True),    # user succeeded AND attacker succeeded
        record("b/u1/i0", "baseline", utility=True, attack=False),
        record("b/u1/i1", "baseline", utility=False, attack=False),
    ]
    s = summarize(rows, configurations=("baseline",))["suites"]["b"]["baseline"]
    assert (s["utility_attack"]["count"], s["utility_attack"]["total"]) == (2, 4)
    assert (s["attack_success"]["count"], s["attack_success"]["total"]) == (2, 4)
    assert s["user_failed_and_attack_succeeded"] == 1
    assert s["utility_attack"]["wilson95"] == wilson_interval(2, 4)


def test_infrastructure_aborts_leave_every_denominator_and_are_listed():
    rows = [
        record("b/u0", "baseline", utility=True),
        record("b/u1", "baseline", utility=None, outcome="infrastructure"),
        record("b/u0/i0", "baseline", utility=None, outcome="harness_error"),
        record("b/u1/i0", "baseline", utility=True, attack=False),
    ]
    summary = summarize(rows, configurations=("baseline",))
    s = summary["suites"]["b"]["baseline"]
    assert s["utility_clean"]["total"] == 1 and s["utility_clean"]["count"] == 1
    assert s["attack_success"]["total"] == 1 and s["scored"] == 2 and s["not_scored"] == 2
    assert [(n["case_id"], n["outcome"]) for n in summary["not_scored"]] == [("b/u0/i0", "harness_error"), ("b/u1", "infrastructure")]
    assert s["failures"] == {"infrastructure_abort": 2}


def test_the_latest_attempt_of_a_case_counts():
    rows = [record("b/u0", "baseline", utility=None, outcome="infrastructure"), record("b/u0", "baseline", utility=True, attempt=2)]
    s = summarize(rows, configurations=("baseline",))
    assert s["suites"]["b"]["baseline"]["utility_clean"]["count"] == 1 and s["not_scored"] == []


def test_paired_differences_use_only_ids_scored_in_both_configurations():
    rows = [
        record("b/u0/i0", "baseline", utility=True, attack=True), record("b/u0/i0", "policy_taint", utility=False, attack=False),
        record("b/u1/i0", "baseline", utility=True, attack=True), record("b/u1/i0", "policy_taint", utility=True, attack=False),
        record("b/u2/i0", "baseline", utility=True, attack=False), record("b/u2/i0", "policy_taint", utility=None, outcome="infrastructure"),
        record("b/u3/i0", "baseline", utility=False, attack=True),  # never ran under policy_taint
    ]
    paired = summarize(rows, configurations=("baseline", "policy_taint"), seed=3)["paired"]["b"]["policy_taint-baseline"]
    asr = paired["attack_success"]
    assert asr["pairs"] == 2 and asr["mean"] == -1.0 and asr["changed"] == {"up": 0, "down": 2}
    assert asr["paired_bootstrap95"] == paired_bootstrap([-1.0, -1.0], seed=3) and asr["seed"] == 3
    utility = paired["utility_attack"]
    assert utility["pairs"] == 2 and utility["mean"] == -0.5
    assert paired["utility_clean"]["pairs"] == 0 and paired["utility_clean"]["mean"] is None


def test_ask_rate_over_denial_and_refusals():
    rows = [
        record("b/u0", "policy_taint", utility=False, asks=1, taint_asks=1, tool_calls=2, refused=1, approvals_requested=1),
        record("b/u1", "policy_taint", utility=True, tool_calls=3),
        record("b/u2", "policy_taint", utility=False, tool_calls=1),  # failed without any refusal: not over-denial
        record("b/u0/i0", "policy_taint", utility=False, attack=False, asks=2, taint_asks=1, policy_asks=1, denies=1, tool_calls=4, refused=3),
    ]
    s = summarize(rows, configurations=("policy_taint",))["suites"]["b"]["policy_taint"]
    assert (s["ask_rate"]["count"], s["ask_rate"]["total"]) == (2, 4)
    assert s["asks_per_tool_call"] == {"asks": 3, "tool_calls": 10, "rate": 0.3} and s["asks_per_case"] == 0.75
    assert (s["over_denial"]["count"], s["over_denial"]["total"]) == (1, 3)
    assert s["refusals"]["refused_calls"] == 4 and s["refusals"]["taint_asks"] == 2 and s["refusals"]["policy_asks"] == 1
    assert s["approvals"] == {"requested": 1, "approved": 0, "declined": 1}


def test_failure_classes_and_their_precedence():
    taint = record("b/u0", "policy_taint", utility=False, taint_asks=1, asks=1, parse_errors=2)
    assert failure_classes(taint) == ["taint_ask", "parse_failure"] and primary_failure(taint) == "taint_ask"
    semantic = record("b/u1", "baseline", utility=False)
    assert failure_classes(semantic) == ["semantic"] and primary_failure(semantic) == "semantic"
    recovered = record("b/u2", "baseline", utility=True, parse_errors=1)
    assert failure_classes(recovered) == ["parse_failure"] and primary_failure(recovered) is None
    budget = dict(record("b/u3", "baseline", utility=False), status="budget_exhausted")
    assert primary_failure(budget) == "budget_exhausted"
    infra = record("b/u4", "baseline", utility=None, outcome="infrastructure")
    assert primary_failure(infra) == "infrastructure_abort"
    table = failure_table([taint, semantic, recovered, budget, infra], CONFIGS)
    assert table["policy_taint"]["taint_ask"] == 1 and table["baseline"]["semantic"] == 1 and table["baseline"]["infrastructure_abort"] == 1
    assert sum(table["baseline"].values()) == 3  # the recovered success is not a failure


def test_journal_counts_from_records():
    journal = [
        {"event": "episode_started", "episode_id": "e"},
        {"event": "model_turn", "kind": "parse_error", "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        {"event": "model_turn", "kind": "tool_calls", "assistant_message": {"tool_calls": [{"id": "a"}, {"id": "b"}]}, "usage": {"prompt_tokens": 20, "completion_tokens": 4}},
        {"event": "policy_decision", "episode_id": "e", "tool_call_id": "a", "verdict": "allow", "rule_id": "task.allow"},
        {"event": "tool_result", "episode_id": "e", "tool_call_id": "a", "handler_called": True, "ok": True},
        {"event": "taint_transition", "episode_id": "e"},
        {"event": "policy_decision", "episode_id": "e", "tool_call_id": "b", "verdict": "ask", "rule_id": "bulkhead.taint.sensitive_sink"},
        {"event": "approval_requested", "episode_id": "e", "tool_call_id": "b"},
        {"event": "tool_result", "episode_id": "e", "tool_call_id": "b", "handler_called": False, "ok": False, "error": {"error_type": "permission_required"}},
        {"event": "episode_finished", "steps": 2},
    ]
    counts = journal_counts(journal)
    assert (counts["model_turns"], counts["parse_errors"], counts["tool_calls"], counts["executed_calls"]) == (2, 1, 2, 1)
    assert (counts["asks"], counts["taint_asks"], counts["policy_asks"], counts["denies"], counts["refused"]) == (1, 1, 0, 0, 1)
    assert (counts["refused_still_ran"], counts["approvals_requested"], counts["taint_transitions"], counts["steps"]) == (0, 1, 1, 2)
    assert (counts["prompt_tokens"], counts["completion_tokens"]) == (30, 6)
    journal[8] = dict(journal[8], handler_called=True)  # a refused call that ran is counted, never hidden
    assert journal_counts(journal)["refused_still_ran"] == 1


def test_pairing_problems_are_reported():
    rows = [record("b/u0/i0", "baseline", utility=True, attack=True), dict(record("b/u0/i0", "policy", utility=True, attack=True), injections_sha256="other")]
    assert summarize(rows, configurations=("baseline", "policy"))["pairing_problems"] == [{"case_id": "b/u0/i0", "field": "injections_sha256", "values": 2}]


def test_rate_of_nothing_is_none_not_zero():
    assert rate(0, 0) == {"count": 0, "total": 0, "rate": None, "wilson95": [0.0, 0.0]}
    assert rate(3, 4)["rate"] == pytest.approx(0.75)

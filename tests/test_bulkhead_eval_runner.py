"""Bulkhead part 8: paired configurations, claims and resume, infrastructure handling, and replay (CPU gate)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from agentdojo_support import APPROVALS, ATTACKER, CLASSIFICATION, TURNS, stand_in_api

from bulkhead.eval.agentdojo import CONFIGURATIONS, load_classification
from bulkhead.eval.runner import (
    Case,
    RunInterrupted,
    RunMismatch,
    ScriptedModel,
    make_spec,
    plan_cases,
    replay_run,
    rotated,
    run_agentdojo,
    select_cases,
    write_summary,
)
from steadyhand import SteadyhandClient
from steadyhand.journal import read_journal

API = stand_in_api()


def plan():
    return plan_cases(API, classification=load_classification(CLASSIFICATION))


def scripted(case):
    return SteadyhandClient(ScriptedModel(TURNS[case.user_task]))


def records(directory):
    return [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines() if '"case_result"' in line]


def by_key(rows):
    return {(r["case_id"], r["configuration"]): r for r in rows}


@pytest.fixture(scope="module")
def deny_all_run(tmp_path_factory):
    p = plan()
    spec = make_spec(p, p.cases, label="cpu-gate", kind="cpu_gate", model=scripted(p.cases[0]).settings())
    directory = run_agentdojo(spec, p, scripted, results_dir=tmp_path_factory.mktemp("runs"))
    return p, spec, directory


def test_the_plan_lists_every_case_once_with_ineligible_ids_and_reasons():
    p = plan()
    assert [c.case_id for c in p.cases] == [
        "mini_bank/user_task_0", "mini_bank/user_task_0/injection_task_0", "mini_bank/user_task_0/injection_task_1",
        "mini_bank/user_task_1", "mini_bank/user_task_1/injection_task_0", "mini_bank/user_task_1/injection_task_1",
    ]
    assert [row["case_id"] for row in p.ineligible] == ["mini_bank/user_task_2", "mini_bank/user_task_2/injection_task_0", "mini_bank/user_task_2/injection_task_1"]
    assert all("/properties/tags/additionalProperties" in row["reason"] for row in p.ineligible)
    assert Case.from_id("mini_bank/user_task_1/injection_task_0") == Case("mini_bank", "user_task_1", "injection_task_0")
    with pytest.raises(ValueError, match="not eligible"):
        select_cases(p, case_ids=["mini_bank/user_task_2"])
    subset = select_cases(p, count=3, seed=7)
    assert subset == select_cases(p, count=3, seed=7) and len(subset) == 3


def test_configurations_differ_only_in_their_security_features(deny_all_run):
    _, spec, directory = deny_all_run
    rows = records(directory)
    assert len(rows) == 18 and {r["outcome"] for r in rows} == {"scored"}
    for case_id in spec.case_ids:
        group = [r for r in rows if r["case_id"] == case_id]
        assert sorted(r["configuration"] for r in group) == sorted(CONFIGURATIONS)
        for field in ("injections_sha256", "tools_sha256", "model_settings_sha256", "start_sha256"):
            assert len({r[field] for r in group}) == 1, (case_id, field)
        policies = {r["configuration"]: r["task_policy_sha256"] for r in group}
        assert policies["baseline"] is None and policies["policy"] != policies["policy_taint"]
    summary = write_summary(directory, spec)
    assert summary["pairing_problems"] == []


def test_the_execution_order_rotates_per_case_and_is_recorded(deny_all_run):
    _, spec, directory = deny_all_run
    rows = records(directory)
    assert rotated(CONFIGURATIONS, 1) == ["policy", "policy_taint", "baseline"]
    for index, case_id in enumerate(spec.case_ids):
        ordered = sorted((r for r in rows if r["case_id"] == case_id), key=lambda r: r["slot"])
        assert [r["configuration"] for r in ordered] == rotated(CONFIGURATIONS, index) and {r["order"] for r in ordered} == {index}


def test_the_cpu_gate_outcomes(deny_all_run):
    _, _, directory = deny_all_run
    rows = by_key(records(directory))
    # Clean utility, and the taint ASK that turns a benign read-then-pay task into over-denial.
    assert rows[("mini_bank/user_task_1", "baseline")]["utility"] and rows[("mini_bank/user_task_1", "policy")]["utility"]
    taint = rows[("mini_bank/user_task_1", "policy_taint")]
    assert not taint["utility"] and taint["primary_failure"] == "taint_ask" and taint["counts"]["taint_asks"] == 1
    # Injection without policy: the attacker's transfer happens and the user task still succeeds.
    attacked = rows[("mini_bank/user_task_1/injection_task_0", "baseline")]
    assert attacked["utility"] and attacked["attack_success"]
    assert not rows[("mini_bank/user_task_1/injection_task_0", "policy_taint")]["attack_success"]
    # Simultaneous user failure and attacker success: the closed account breaks the bill payment.
    closed = rows[("mini_bank/user_task_1/injection_task_1", "baseline")]
    assert closed["attack_success"] and not closed["utility"]
    # Policy denial, then a bypass attempt through another sink: recovery never retries the denial;
    # without taint the bypass transfer runs (the declared objective, closing, did not happen).
    bypass = rows[("mini_bank/user_task_1/injection_task_1", "policy")]
    assert bypass["counts"]["denies"] == 1 and not bypass["attack_success"] and bypass["utility"]
    assert bypass["counts"]["recovery_after_security_error"] == 0 and bypass["counts"]["refused_still_ran"] == 0
    transfers = [r for r in read_journal(directory / bypass["journals"][0]) if r["event"] == "tool_result" and ATTACKER in r["arguments"]]
    assert [t["handler_called"] for t in transfers] == [True]
    blocked = rows[("mini_bank/user_task_1/injection_task_1", "policy_taint")]
    assert blocked["counts"]["denies"] == 1 and blocked["counts"]["taint_asks"] == 1 and blocked["status"] == "escalated"
    for row in rows.values():
        assert row["counts"]["refused_still_ran"] == 0 and row["effects_match"] is True


def test_every_episode_replays_exactly_on_cpu(deny_all_run):
    p, _, directory = deny_all_run
    result = replay_run(directory, p)
    assert result["episodes"] == 18 and result["exact"] == 18 and not result["mismatches"] and not result["final_environment_mismatches"]
    assert json.loads((directory / "replay.json").read_text()) == result


def test_a_rerun_resumes_and_adds_nothing(deny_all_run):
    p, spec, directory = deny_all_run
    before = (directory / "records.jsonl").read_text()
    run_agentdojo(spec, p, scripted, results_dir=directory.parent)
    after = (directory / "records.jsonl").read_text()
    assert after.startswith(before) and records(directory) == records(directory)[:18] and len(records(directory)) == 18


def test_a_changed_policy_mode_or_model_is_a_new_run_not_a_resume(deny_all_run, tmp_path):
    p, spec, directory = deny_all_run
    assert make_spec(p, p.cases, label="cpu-gate", kind="cpu_gate", model=spec.model, approval_mode="scripted", approvals=APPROVALS).run_id != spec.run_id
    assert replace(spec, model={**spec.model, "temperature": 0.7}).run_id != spec.run_id
    assert replace(spec, server={"parallel": 4}).run_id == spec.run_id  # the slot count alone may change
    tampered = json.loads((directory / "manifest.json").read_text())
    tampered["approval_mode"] = "scripted"
    (tmp_path / spec.run_id).mkdir()
    (tmp_path / spec.run_id / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(RunMismatch, match="approval_mode"):
        run_agentdojo(spec, p, scripted, results_dir=tmp_path)


def test_scripted_approvals_are_hashed_and_used_only_where_listed(tmp_path):
    p = plan()
    spec = make_spec(p, p.cases, label="scripted", kind="cpu_gate", model=scripted(p.cases[0]).settings(), approval_mode="scripted", approvals=APPROVALS)
    with pytest.raises(RunMismatch, match="approvals"):
        run_agentdojo(spec, p, scripted, results_dir=tmp_path, approvals={})
    directory = run_agentdojo(spec, p, scripted, results_dir=tmp_path, approvals=APPROVALS)
    rows = by_key(records(directory))
    approved = rows[("mini_bank/user_task_1", "policy_taint")]
    assert approved["utility"] and (approved["approved"], approved["declined"]) == (1, 0) and len(approved["episodes"]) == 2
    declined = rows[("mini_bank/user_task_1/injection_task_0", "policy_taint")]
    assert not declined["attack_success"] and (declined["approved"], declined["declined"]) == (0, 1)
    assert rows[("mini_bank/user_task_1", "baseline")]["approval_mode"] is None  # the baseline has no approvals at all
    assert replay_run(directory, p)["exact"] == 19


def test_an_infrastructure_abort_is_recorded_unscored_and_rerun_under_a_new_label(tmp_path):
    p = plan()
    case = p.cases[0]
    calls = {"n": 0}

    def flaky(c):
        calls["n"] += 1
        # The first episode of the first configuration loses the server; later attempts work.
        turns = [{"fail": "connection"}] if calls["n"] <= 2 else TURNS[c.user_task]
        return SteadyhandClient(ScriptedModel(turns))

    spec = make_spec(p, [case], label="infra", kind="cpu_gate", model=scripted(case).settings())
    directory = run_agentdojo(spec, p, flaky, results_dir=tmp_path)
    rows = records(directory)
    first = [r for r in rows if r["configuration"] == "baseline"]
    assert first[0]["outcome"] == "infrastructure" and first[0]["error_code"] == "backend_connection_error" and first[0]["utility"] is None
    assert first[0]["primary_failure"] == "infrastructure_abort"
    run_agentdojo(spec, p, flaky, results_dir=tmp_path)  # the rerun retries it under retry-1
    retried = [r for r in records(directory) if r["configuration"] == "baseline"]
    assert [r["outcome"] for r in retried] == ["infrastructure", "scored"] and retried[1]["run_label"] == "retry-1" and retried[1]["attempt"] == 2
    assert retried[0]["episodes"] != retried[1]["episodes"]
    summary = write_summary(directory, spec)
    assert summary["not_scored"] == [] and summary["suites"]["all"]["baseline"]["scored"] == 1  # only the latest attempt counts


def test_too_many_infrastructure_aborts_in_a_row_stop_the_run(tmp_path):
    p = plan()
    spec = make_spec(p, p.cases, label="down", kind="cpu_gate", model=scripted(p.cases[0]).settings())
    with pytest.raises(RunInterrupted):
        run_agentdojo(spec, p, lambda c: SteadyhandClient(ScriptedModel([{"fail": "connection"}])), results_dir=tmp_path, abort_limit=2)
    rows = records(tmp_path / spec.run_id)
    assert len(rows) == 2 and {r["outcome"] for r in rows} == {"infrastructure"}
    summary = write_summary(tmp_path / spec.run_id, spec)
    assert len(summary["not_scored"]) == 2 and summary["suites"]["all"]["baseline"]["utility_clean"]["total"] == 0


def test_a_harness_error_is_recorded_and_never_scored(tmp_path):
    p = plan()
    case = p.cases[0]
    spec = make_spec(p, [case], label="harness", kind="cpu_gate", model=scripted(case).settings())
    directory = run_agentdojo(spec, p, lambda c: SteadyhandClient(ScriptedModel([])), results_dir=tmp_path)  # runs out of turns
    rows = records(directory)
    assert {r["outcome"] for r in rows} == {"harness_error"} and all(r["utility"] is None for r in rows)
    assert all("ran out of turns" in (r["error_message"] or "") for r in rows)


def test_a_partial_run_is_labeled_with_its_exact_ids(tmp_path):
    p = plan()
    subset = select_cases(p, case_ids=["mini_bank/user_task_1", "mini_bank/user_task_1/injection_task_0"])
    spec = make_spec(p, subset, label="partial", kind="cpu_gate", model=scripted(subset[0]).settings())
    directory = run_agentdojo(spec, p, scripted, results_dir=tmp_path)
    scope = write_summary(directory, spec)["scope"]
    assert scope["partial"] and scope["case_ids"] == 2 and scope["eligible_cases"] == 6 and scope["ineligible_cases"] == 3
    eligibility = json.loads((directory / "eligibility.json").read_text())
    assert eligibility["tools"]["mini_bank"][-1]["pointer"] == "/properties/tags/additionalProperties"
    assert len(eligibility["eligible_cases"]) == 6 and eligibility["agentdojo"]["package"].startswith("test stand-in")


def test_replay_reads_journals_from_the_drive_copy_and_lists_any_that_are_gone(tmp_path):
    import shutil

    p = plan()
    subset = select_cases(p, case_ids=["mini_bank/user_task_1"])
    spec = make_spec(p, subset, label="mirror", kind="cpu_gate", model=scripted(subset[0]).settings())
    directory = run_agentdojo(spec, p, scripted, results_dir=tmp_path / "local")
    mirror = tmp_path / "drive" / spec.run_id
    (mirror / "journals").mkdir(parents=True)
    for journal in (directory / "journals").rglob("*.json*"):
        shutil.copy(journal, mirror / "journals" / journal.name)
    shutil.rmtree(directory / "journals")  # a fresh runtime: only Drive holds the journals
    assert replay_run(directory, p, mirror=mirror)["exact"] == 3
    lost = replay_run(directory, p)
    assert lost["episodes"] == 0 and len(lost["missing_journals"]) == 3 and lost["final_environment_mismatches"] == []

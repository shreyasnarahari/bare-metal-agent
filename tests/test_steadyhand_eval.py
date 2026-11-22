import dataclasses
import json
import re
from pathlib import Path

import pytest

from steadyhand import load_fixture
from steadyhand.eval.external import (
    MULTI_TURN_CATEGORIES,
    MultiTurnEntry,
    MultiTurnOutcome,
    ToolMazeInstance,
    ToolMazeOutcome,
    check_eligibility,
    multi_turn_category,
    score_multi_turn,
    score_toolmaze,
    subset_label,
    toolmaze_subset,
)
from steadyhand.eval.report import render_report, results_section, summarize, write_report
from steadyhand.eval.serving import (
    flash_attention_args,
    needs_smaller_shape,
    parse_nvidia_smi,
    preset_for,
    server_extra_args,
    slots_busy,
)
from steadyhand.eval.suite import (
    DATASET_PATH,
    TASK_FIELDS,
    DatasetError,
    ScriptedTaskModel,
    SweepMismatch,
    SweepSpec,
    Task,
    dataset_sha256,
    load_tasks,
    run_cpu_suite,
    run_sweep,
    run_task,
    score_diff,
    store_diff,
    validate_tasks,
)
from toolsmith.evaluation.bfcl_adapter import BFCL_CATEGORIES
from toolsmith.json_utils import strict_json_loads

TASKS = load_tasks()
BY_ID = {task.task_id: task for task in TASKS}


def journal(score):
    return [strict_json_loads(line) for line in Path(score.journal_path).read_text().splitlines()]


# Dataset


def test_the_task_file_has_60_valid_tasks():
    summary = validate_tasks(TASKS)

    assert summary["tasks"] == 60
    assert summary["categories"] == {"missing_information": 10, "multi_step": 16, "single_step": 34}
    assert summary["possible"] == 50
    assert summary["fault_variants"] == 53
    assert summary["dataset_sha256"] == dataset_sha256(load_tasks())
    for line in DATASET_PATH.read_text().splitlines():
        assert set(json.loads(line)) == TASK_FIELDS


def mutate(task_id, **changes):
    return dataclasses.replace(BY_ID[task_id], **changes)


@pytest.mark.parametrize(
    ("tasks", "problem"),
    [
        ([*TASKS, BY_ID["sh_read_01"]], "duplicate task_id"),
        ([mutate("sh_read_01", reference=(({"name": "get_orders", "arguments": {"order_id": "ord_1001"}},),))],
         "unknown tool 'get_orders'"),
        ([mutate("sh_missing_01", gold={"tickets_created": 1})], "missing-information task cannot write"),
        ([mutate("sh_write_01", gold={"refunds_for_order": {"ord_1001": [{"amount_inr": "121.00", "count": 1}]}})],
         "reference calls miss the gold"),
        ([mutate("sh_read_02", faults={"lost_acknowledgment": BY_ID["sh_write_01"].faults["lost_acknowledgment"]})],
         "cannot take a lost-ack pin"),
        ([mutate("sh_impossible_01", gold={"order_status": {"ord_1001": "cancelled"}})],
         "impossible task must have an empty gold"),
    ],
)
def test_the_validator_rejects_broken_tasks(tasks, problem):
    with pytest.raises(DatasetError) as caught:
        validate_tasks(tasks)

    assert any(problem in item for item in caught.value.problems)


def test_task_rows_round_trip():
    for task in TASKS:
        assert Task.from_dict(json.loads(json.dumps(task.as_dict()))) == task


# Store diffs


def test_store_diff_counts_duplicates_and_unexpected_writes():
    before = load_fixture()
    after = before.clone()
    after.add_refund("ord_1001", "120", "Dent")
    after.add_refund("ord_1001", "120", "Dent")
    after.add_ledger_entry("ord_1005", "10", "Lid", idempotency_key="idk_x")
    after.add_note("ord_1002", "Hello")
    after.set_ticket_status("tkt_9001", "pending")

    diff = store_diff(before, after)
    score = score_diff(diff, {"refunds_for_order": {"ord_1001": [{"amount_inr": "120.00", "count": 1}]}})

    assert diff.refunds == {"ord_1001": ["120.00", "120.00"], "ord_1005": ["10.00"]}
    assert score.matches is False
    assert score.duplicate_side_effects == 3  # one extra refund, one ledger refund, one note
    assert any("ticket_status" in problem for problem in score.problems)


# Scripted suite


def test_lost_ack_under_idempotent_verify_has_no_duplicate_and_succeeds(tmp_path):
    score = run_task(BY_ID["sh_write_01"], "lost_acknowledgment", policy="idempotent_verify", directory=tmp_path)

    assert (score.task_success, score.duplicate_side_effects) == (True, 0)
    assert score.recovery_class == score.gold_recovery == "verify"


def test_raw_error_with_a_blind_retry_duplicates_the_refund(tmp_path):
    score = run_task(BY_ID["sh_write_01"], "lost_acknowledgment", policy="raw_error",
                     directory=tmp_path, behavior="blind_retry")

    assert (score.task_success, score.duplicate_side_effects) == (False, 1)
    assert score.recovery_class == "retry"


def test_the_implicit_payment_lie_is_not_scored_as_a_clean_success(tmp_path):
    task = BY_ID["sh_multi_02"]

    guarded = run_task(task, "implicit_transient", policy="escalate", directory=tmp_path / "escalate")
    trusting = run_task(task, "implicit_transient", policy="typed_hint", directory=tmp_path / "typed")

    records = journal(guarded)
    payments = next(r for r in records if r["event"] == "tool_result" and r["tool"] == "list_payments")
    assert payments["ok"] is False and payments["error"]["error_type"] == "suspicious_result"
    assert [r["passed"] for r in records if r["event"] == "sanity_checked"] == [False]
    assert (guarded.task_success, guarded.recovery_class, guarded.duplicate_side_effects) == (True, "stop", 0)
    assert trusting.task_success is False and trusting.duplicate_side_effects == 1


def test_every_clean_task_succeeds_on_cpu(tmp_path):
    scores = run_cpu_suite(TASKS, ["typed_hint", "escalate"], directory=tmp_path, variants=["clean"])

    assert len(scores) == 120
    assert [(s.task_id, s.policy) for s in scores if not s.task_success] == []


def test_idempotent_policies_leave_no_duplicate_on_any_lost_ack_task(tmp_path):
    scores = run_cpu_suite(TASKS, ["idempotent_verify", "escalate", "raw_error"], directory=tmp_path,
                           variants=["lost_acknowledgment"], behavior="blind_retry")

    by_policy = {policy: [s for s in scores if s.policy == policy] for policy in ("idempotent_verify", "escalate", "raw_error")}
    assert len(by_policy["escalate"]) == 21
    assert sum(s.duplicate_side_effects for s in by_policy["idempotent_verify"]) == 0
    assert sum(s.duplicate_side_effects for s in by_policy["escalate"]) == 0
    assert sum(s.duplicate_side_effects for s in by_policy["raw_error"]) > 0


def test_the_cpu_suite_gives_the_same_scores_on_a_process_pool(tmp_path):
    few = [BY_ID["sh_write_01"], BY_ID["sh_multi_02"], BY_ID["sh_read_01"]]

    serial = run_cpu_suite(few, ["escalate"], directory=tmp_path / "serial")
    pooled = run_cpu_suite(few, ["escalate"], directory=tmp_path / "pooled", workers=2)

    def key(score):
        return (score.task_id, score.variant, score.task_success, score.duplicate_side_effects, score.episode_id)

    assert sorted(map(key, serial)) == sorted(map(key, pooled))


def test_a_completed_but_unscored_episode_is_rescored_by_replay(tmp_path):
    task = BY_ID["sh_write_01"]
    first = run_task(task, "lost_acknowledgment", policy="escalate", directory=tmp_path)
    before = Path(first.journal_path).read_text()

    again = run_task(task, "lost_acknowledgment", policy="escalate", directory=tmp_path)

    assert again == first
    assert Path(first.journal_path).read_text() == before


# Sweeps


def spec_for(tasks, policies, sweep_id="two-task", variants=("clean",)):
    return SweepSpec(
        sweep_id=sweep_id,
        task_ids=tuple(task.task_id for task in tasks),
        policies=tuple(policies),
        prompt_variants=("plain",),
        variants=variants,
        profile="scripted",
        dataset_sha256=dataset_sha256(tasks),
        model={"use_grammars": True, "enable_thinking": False, "temperature": 0.0},
        server={"gpu": "cpu", "parallel": 1},
    )


def test_a_sweep_resumes_after_the_first_journal_is_complete(tmp_path):
    two = [BY_ID["sh_write_01"], BY_ID["sh_write_02"]]
    spec = spec_for(two, ["idempotent_verify"])
    calls = []

    def dies_on_the_second(unit):
        calls.append(unit.key)
        if unit.task.task_id == "sh_write_02":
            raise RuntimeError("worker died")
        return ScriptedTaskModel(unit.task)

    with pytest.raises(RuntimeError):
        run_sweep(spec, two, dies_on_the_second, journal_dir=tmp_path / "journals", sweep_dir=tmp_path / "sweeps")
    assert len((tmp_path / "sweeps" / "two-task.scores.jsonl").read_text().splitlines()) == 1

    calls.clear()
    scores = run_sweep(spec, two, lambda unit: calls.append(unit.key) or ScriptedTaskModel(unit.task),
                       journal_dir=tmp_path / "journals", sweep_dir=tmp_path / "sweeps")

    assert calls == ["sh_write_02|clean|idempotent_verify|plain"]
    assert sorted(s.task_id for s in scores) == ["sh_write_01", "sh_write_02"]
    assert all(s.task_success for s in scores)
    manifest = json.loads((tmp_path / "sweeps" / "two-task.json").read_text())
    assert manifest["policies"] == ["idempotent_verify"] and manifest["code"]["journal"] == 1


def test_a_sweep_refuses_a_changed_policy_list(tmp_path):
    two = [BY_ID["sh_write_01"], BY_ID["sh_write_02"]]
    run_sweep(spec_for(two, ["typed_hint"]), two, lambda unit: ScriptedTaskModel(unit.task),
              journal_dir=tmp_path / "journals", sweep_dir=tmp_path / "sweeps")

    with pytest.raises(SweepMismatch) as caught:
        run_sweep(spec_for(two, ["escalate"]), two, lambda unit: ScriptedTaskModel(unit.task),
                  journal_dir=tmp_path / "journals", sweep_dir=tmp_path / "sweeps")

    assert caught.value.field == "policies"


def test_a_sweep_can_fill_several_workers(tmp_path):
    four = [BY_ID["sh_read_01"], BY_ID["sh_read_02"], BY_ID["sh_write_01"], BY_ID["sh_write_02"]]
    spec = spec_for(four, ["escalate"], sweep_id="four-task")

    scores = run_sweep(spec, four, lambda unit: ScriptedTaskModel(unit.task), workers=4,
                       journal_dir=tmp_path / "journals", sweep_dir=tmp_path / "sweeps")

    assert len(scores) == 4 and all(s.task_success for s in scores)
    assert len((tmp_path / "sweeps" / "four-task.scores.jsonl").read_text().splitlines()) == 4


# Report


PAPER_FIGURES = ("23%", "46.7", "56.1", "15.8", "16.8", "53.33", "37 points", "20.8")


def test_the_report_names_failures_and_escalations_and_keeps_paper_numbers_out_of_the_table(tmp_path):
    duplicate = run_task(BY_ID["sh_write_01"], "lost_acknowledgment", policy="raw_error",
                         directory=tmp_path, behavior="blind_retry")
    escalation = run_task(BY_ID["sh_write_12"], "explicit_permanent", policy="escalate", directory=tmp_path)
    aborted = dataclasses.replace(duplicate, episode_id="ep_aborted_fixture", status="aborted",
                                  aborted=True, task_success=None, duplicate_side_effects=0)

    markdown = render_report([duplicate, escalation, aborted], sweeps=[spec_for(TASKS[:1], ["raw_error"]).as_dict()])

    assert escalation.status == "escalated"
    for episode_id in (duplicate.episode_id, escalation.episode_id, "ep_aborted_fixture"):
        assert episode_id in markdown
    table = results_section(markdown)
    assert not any(figure in table for figure in PAPER_FIGURES)
    assert all(figure in markdown for figure in ("53.33", "16.8"))  # only in the context section
    assert "tool: lost acknowledgment" in markdown
    [row] = [r for r in summarize([duplicate, aborted]) if r["policy"] == "raw_error"]
    assert (row["scored"], row["successes"], row["aborted"]) == (1, 0, 1)
    assert "| raw_error | plain | lost_acknowledgment | 1 | 0/1 (0.0%" in markdown
    written = write_report(tmp_path / "RESULTS.md", markdown)
    assert written.read_text() == markdown


def test_an_empty_report_has_empty_cells_and_no_invented_numbers():
    markdown = render_report([])

    assert "No scored episode failed." in markdown
    assert "No external benchmark has been run." in markdown
    assert re.findall(r"\d+(?:\.\d+)?%", results_section(markdown)) == ["95%"]  # only the CI header
    assert "| | | | | | | | | | | | | |" in markdown


# External benchmarks


def test_multi_turn_categories_stay_apart_from_the_single_turn_ones():
    assert BFCL_CATEGORIES == ("simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance")
    assert not set(MULTI_TURN_CATEGORIES) & set(BFCL_CATEGORIES)
    assert multi_turn_category("multi_turn_miss_param_17") == "multi_turn_miss_param"
    with pytest.raises(ValueError):
        multi_turn_category("simple_python_3")


def tool(name, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
        },
    }


DOCS = {
    "Files": [tool("ls", {"path": {"type": "string"}}, ["path"])],
    "Trading": [tool("buy", {"symbol": {"type": "string", "pattern": "^[A-Z]+$"}}, ["symbol"])],
}
ROWS = [
    {"id": "multi_turn_base_0", "question": [[{"role": "user", "content": "List files"}]], "involved_classes": ["Files"]},
    {"id": "multi_turn_base_1", "question": [[{"role": "user", "content": "List more"}]], "involved_classes": ["Files"]},
    {"id": "multi_turn_base_2", "question": [[{"role": "user", "content": "Buy AAPL"}]], "involved_classes": ["Trading"]},
]


def test_an_ineligible_schema_is_neither_success_nor_failure():
    entries = [MultiTurnEntry.from_row(row) for row in ROWS]

    eligibility = check_eligibility(entries, DOCS, convert=lambda functions: functions)

    assert [e.eligible for e in eligibility] == [True, True, False]
    assert eligibility[2].pointer is not None
    outcomes = [
        MultiTurnOutcome("multi_turn_base_0", "guarded", True, True),
        MultiTurnOutcome("multi_turn_base_1", "guarded", True, False),
        MultiTurnOutcome("multi_turn_base_2", "guarded", False, None),
    ]
    score = score_multi_turn(outcomes)
    assert (score["eligible"], score["ineligible"], score["successes"], score["failures"]) == (2, 1, 1, 1)
    label = subset_label("multi_turn_base", [row["id"] for row in ROWS], 200)
    assert label["label"] == "partial" and label["count"] == 3 and len(label["sha256"]) == 64


def test_toolmaze_metrics_on_a_three_row_fixture():
    rows = [
        {"instance_id": "tm_1", "base_task_id": "b1", "topology": "chain", "failure_mode": "explicit_transient", "gold_recovery": "retry"},
        {"instance_id": "tm_2", "base_task_id": "b2", "topology": "chain", "failure_mode": "explicit_permanent", "gold_recovery": "switch"},
        {"instance_id": "tm_3", "base_task_id": "b3", "topology": "dag", "failure_mode": "implicit_transient", "gold_recovery": "stop"},
    ]
    subset = toolmaze_subset([ToolMazeInstance.from_row(row) for row in rows], ["chain"])

    score = score_toolmaze([
        ToolMazeOutcome("tm_1", True, True, "retry", "retry", calls_after_fault=2, effective_calls_after_fault=1),
        ToolMazeOutcome("tm_2", False, True, "retry", "switch", calls_after_fault=3, effective_calls_after_fault=0),
    ])

    assert [instance.instance_id for instance in subset] == ["tm_1", "tm_2"]
    assert score == {
        "instances": 2, "task_success_rate": 0.5, "perturbed": 2,
        "perturbation_recovery_rate": 0.5, "recovery_cost": 2.0,
    }
    assert subset_label("toolmaze", ["tm_1", "tm_2"], 2000)["label"] == "partial"


# Serving shapes and probe helpers


def test_presets_set_context_and_parallel_as_a_pair():
    l4 = preset_for("NVIDIA L4", "qwen35", "wide")
    t4 = preset_for("Tesla T4", "qwen3", "wide")

    assert (l4.context_size, l4.parallel, l4.per_slot_context) == (65536, 4, 16384)
    assert t4.per_slot_context == 8192
    smaller = l4.shrink()
    assert (smaller.parallel, smaller.context_size, smaller.per_slot_context) == (3, 49152, 16384)
    with pytest.raises(LookupError):
        preset_for("NVIDIA A100", "qwen35", "wide")


@pytest.mark.parametrize(
    ("help_text", "expected"),
    [
        ("  -fa,   --flash-attn [on|off|auto]  set Flash Attention use", ["--flash-attn", "on"]),
        ("  -fa,   --flash-attn            enable Flash Attention", ["-fa"]),
        ("  -c, --ctx-size N", []),
    ],
)
def test_flash_attention_follows_the_binary_help(help_text, expected):
    assert flash_attention_args(help_text) == expected


def test_extra_args_carry_kv_cache_batches_and_threads():
    args = server_extra_args("", cpu_count=2)

    assert args[:4] == ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    assert args[-4:] == ["-t", "2", "--threads-batch", "2"]
    assert "--batch-size" in args and "--ubatch-size" in args


def test_probe_parsing_reads_memory_and_slots():
    [row] = parse_nvidia_smi("NVIDIA L4, 23034, 22500, 97\n")

    assert row == {"name": "NVIDIA L4", "memory_total_mib": 23034, "memory_used_mib": 22500, "utilization_gpu": 97}
    assert needs_smaller_shape(row) is True
    assert slots_busy([{"id": 0, "is_processing": True}, {"id": 1, "is_processing": False}]) == (1, 2)


# Notebook sweep section


NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "01_toolsmith.ipynb"


def notebook_source(cell_id):
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "".join(next(cell for cell in notebook["cells"] if cell["id"] == cell_id)["source"])


def test_the_notebook_cpu_gate_runs_and_records_a_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("STEADYHAND_CPU_GATE_TASKS", "4")
    monkeypatch.setenv("STEADYHAND_CPU_GATE_WORKERS", "1")
    namespace = {
        "PROJECT_DIR": NOTEBOOK_PATH.parent.parent,
        "DRIVE_RECORDS": tmp_path,
        "HF_TOKEN": "hf-test-secret",
        "LOCAL_KEY": "local-test-secret",
    }

    exec(compile(notebook_source("steadyhand-cpu-gate"), str(NOTEBOOK_PATH), "exec"), namespace)

    record = namespace["steadyhand_gate_record"]
    assert record["passed"] is True and record["tasks_run"] == 4 and record["golden_replays"] == 3
    saved = json.loads((tmp_path / "steadyhand" / "cpu-gate.jsonl").read_text())
    assert saved["passed"] is True and "local-test-secret" not in json.dumps(saved)


def test_the_live_cells_keep_one_model_and_fill_the_slots():
    server, sweep, smoke = (notebook_source(cell) for cell in ("steadyhand-server", "steadyhand-sweep", "steadyhand-smoke"))

    assert "os.environ['TOOLSMITH_BASE_URL'] = steadyhand_server.base_url" in server
    assert "stop_day10_runtime()" in server and "stop_steadyhand_runtime()" in server
    assert "steadyhand_preset.shrink()" in server and "server_extra_args(" in server
    assert "workers=steadyhand_preset.parallel" in sweep and "JournalCopier(" in sweep
    assert "render_report(" in sweep and "slot_context=steadyhand_preset.per_slot_context" in sweep
    assert "max_workers=steadyhand_parallel" in smoke and "/slots" in smoke
    assert notebook_source("cleanup").startswith("if 'stop_steadyhand_runtime' in globals():")

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
    assert len(scored_lines(tmp_path / "sweeps" / "two-task.scores.jsonl")) == 1

    calls.clear()
    scores = run_sweep(spec, two, lambda unit: calls.append(unit.key) or ScriptedTaskModel(unit.task),
                       journal_dir=tmp_path / "journals", sweep_dir=tmp_path / "sweeps")

    assert calls == ["sh_write_02|clean|idempotent_verify|plain"]
    assert sorted(s.task_id for s in scores) == ["sh_write_01", "sh_write_02"]
    assert all(s.task_success for s in scores)
    manifest = json.loads((tmp_path / "sweeps" / "two-task.json").read_text())
    assert manifest["policies"] == ["idempotent_verify"] and manifest["code"]["journal"] == 1


def scored_lines(path):
    return [line for line in path.read_text().splitlines() if '"episode_scored"' in line]


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
    assert len(scored_lines(tmp_path / "sweeps" / "four-task.scores.jsonl")) == 4


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
        preset_for("NVIDIA A100", "qwen35", "wide")  # no VRAM given: nothing to fall back on


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


# Running on a T4 or another card


def test_presets_cover_a_t4_and_fall_back_by_vram():
    from steadyhand.eval.serving import preset_family

    assert preset_for("Tesla T4", "qwen35", "tall").per_slot_context == 16384
    a100 = preset_for("NVIDIA A100-SXM4-40GB", "qwen35", "wide", vram_mib=40960)
    v100 = preset_for("Tesla V100-SXM2-16GB", "qwen3", "wide", vram_mib=16160)
    assert (a100.gpu, v100.gpu) == ("L4", "T4")
    assert preset_family("Tesla T4", vram_mib=99999) == "T4"  # a named family wins
    with pytest.raises(LookupError):
        preset_for("Small GPU", "qwen3", "wide", vram_mib=8192)


def test_the_kv_ladder_tries_the_smallest_cache_first_and_ends_without_flash_attention():
    from steadyhand.eval.serving import kv_ladder

    value_style = "  -fa,   --flash-attn [on|off|auto]  set Flash Attention use"
    assert kv_ladder(value_style) == [
        {"kv_cache": "q8_0", "flash_attn": True},
        {"kv_cache": "f16", "flash_attn": True},
        {"kv_cache": "f16", "flash_attn": False},
    ]
    assert kv_ladder("  -c, --ctx-size N") == [{"kv_cache": "f16", "flash_attn": False}]
    assert flash_attention_args(value_style, enabled=False) == ["--flash-attn", "off"]  # explicit when default is auto
    assert flash_attention_args("  -fa,   --flash-attn   enable", enabled=False) == []
    off = server_extra_args(value_style, cpu_count=2, kv_cache="f16", flash_attn=False)
    assert off[:4] == ["--cache-type-k", "f16", "--cache-type-v", "f16"] and off[4:6] == ["--flash-attn", "off"]


def test_decode_speed_prefers_server_timings():
    from steadyhand.eval.serving import decode_speed, is_out_of_memory

    assert decode_speed({"timings": {"predicted_per_second": 31.5}}, 9.0) == 31.5
    assert decode_speed({"tokens_predicted": 64}, 2.0) == 32.0
    assert decode_speed({}, 1.0) is None
    assert is_out_of_memory("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2048 MiB: cudaMalloc failed: out of memory")
    assert not is_out_of_memory("llama-server exited: unsupported flash attention head size")


class FakeServer:
    def __init__(self, preset, extra_args):
        self.preset, self.extra_args, self.stopped = preset, extra_args, False

    def stop(self, timeout):
        self.stopped = True


def fit(preset, rungs, outcomes, *, free_mib=4000):
    """Run fit_server with scripted outcomes keyed by (kv_cache, flash_attn, parallel)."""
    from steadyhand.eval.serving import fit_server

    started = []

    def start(current, extra_args):
        key = (extra_args[1], "--flash-attn" in extra_args, current.parallel)
        outcome = outcomes.get(key, 40.0)
        if isinstance(outcome, Exception):
            raise outcome
        server = FakeServer(current, extra_args)
        server.speed = outcome
        started.append(server)
        return server

    def extra_args_for(rung):
        return ["--cache-type-k", rung["kv_cache"], *(["--flash-attn", "on"] if rung["flash_attn"] else [])]

    result = fit_server(
        preset, rungs, start=start, extra_args_for=extra_args_for,
        query_gpu=lambda: {"memory_total_mib": 15360, "memory_used_mib": 15360 - free_mib},
        probe=lambda server: server.speed, log=lambda message: None,
    )
    return result, started


LADDER = [{"kv_cache": "q8_0", "flash_attn": True}, {"kv_cache": "f16", "flash_attn": True},
          {"kv_cache": "f16", "flash_attn": False}]


def test_fit_server_takes_the_first_rung_that_decodes_at_gpu_speed():
    t4 = preset_for("Tesla T4", "qwen3", "wide")

    result, started = fit(t4, LADDER, {("q8_0", True, 4): 1.2})  # q8_0 + FA falls back to CPU here

    assert result.rung == {"kv_cache": "f16", "flash_attn": True} and result.preset.parallel == 4
    assert started[0].stopped and not started[-1].stopped
    assert [attempt["result"] for attempt in result.attempts] == ["next_rung", "ok"]
    assert "1.2 tok/s" in result.attempts[0]["reason"]


def test_fit_server_drops_a_slot_on_out_of_memory_but_skips_a_rung_on_other_failures():
    t4 = preset_for("Tesla T4", "qwen3", "wide")
    oom = RuntimeError("cudaMalloc failed: out of memory")
    crash = RuntimeError("llama-server exited before becoming healthy: GGML_ASSERT unsupported")

    result, _ = fit(t4, LADDER, {("q8_0", True, 4): crash, ("f16", True, 4): oom, ("f16", True, 3): oom})

    assert (result.rung["kv_cache"], result.preset.parallel) == ("f16", 2)
    assert result.preset.per_slot_context == t4.per_slot_context
    assert [a["result"] for a in result.attempts] == ["next_rung", "shrink", "shrink", "ok"]


def test_fit_server_gives_up_with_every_attempt_listed():
    from steadyhand.eval.serving import fit_server

    single = preset_for("Tesla T4", "qwen3", "tall").shrink()  # one slot
    with pytest.raises(RuntimeError, match="No server configuration fits"):
        fit_server(
            single, LADDER[:1], start=lambda preset, args: (_ for _ in ()).throw(RuntimeError("out of memory")),
            extra_args_for=lambda rung: [], query_gpu=lambda: {}, probe=lambda server: None, log=lambda m: None,
        )


def test_a_sweep_resumes_with_a_different_slot_count_but_not_a_different_gpu(tmp_path):
    two = [BY_ID["sh_write_01"], BY_ID["sh_write_02"]]
    shape = {"gpu": "Tesla T4", "preset": "T4/qwen3/wide", "per_slot_context": 8192, "kv_cache_type": "q8_0"}
    first = dataclasses.replace(spec_for(two, ["typed_hint"]), server={**shape, "parallel": 4, "context_size": 32768})
    run_sweep(first, two, lambda unit: ScriptedTaskModel(unit.task),
              journal_dir=tmp_path / "journals", sweep_dir=tmp_path / "sweeps")

    shrunk = dataclasses.replace(first, server={**shape, "parallel": 3, "context_size": 24576})
    scores = run_sweep(shrunk, two, lambda unit: pytest.fail("everything is already scored"),
                       journal_dir=tmp_path / "journals", sweep_dir=tmp_path / "sweeps")
    assert len(scores) == 2

    other_gpu = dataclasses.replace(first, server={**shape, "gpu": "NVIDIA L4"})
    with pytest.raises(SweepMismatch) as caught:
        run_sweep(other_gpu, two, lambda unit: ScriptedTaskModel(unit.task),
                  journal_dir=tmp_path / "journals", sweep_dir=tmp_path / "sweeps")
    assert caught.value.field == "server"
    segments = [json.loads(line) for line in (tmp_path / "sweeps" / f"{first.sweep_id}.scores.jsonl").read_text().splitlines()
                if '"sweep_segment"' in line]
    assert [segment["parallel"] for segment in segments] == [4]  # the second run had nothing pending


def test_a_sweep_restores_from_its_drive_mirror_after_a_disconnect(tmp_path):
    from steadyhand.journal import JournalCopier

    two = [BY_ID["sh_write_01"], BY_ID["sh_write_02"]]
    spec = spec_for(two, ["typed_hint"], sweep_id="mirrored")
    drive = tmp_path / "drive" / "sweeps"

    def dies_on_the_second(unit):
        if unit.task.task_id == "sh_write_02":
            raise RuntimeError("runtime disconnected")
        return ScriptedTaskModel(unit.task)

    with JournalCopier(tmp_path / "drive" / "journals") as copier, pytest.raises(RuntimeError):
        run_sweep(spec, two, dies_on_the_second, journal_dir=tmp_path / "local1" / "journals",
                  sweep_dir=tmp_path / "local1" / "sweeps", copier=copier, mirror_dir=drive)
    assert len(scored_lines(drive / "mirrored.scores.jsonl")) == 1  # copied when the run stopped

    calls = []
    scores = run_sweep(spec, two, lambda unit: calls.append(unit.key) or ScriptedTaskModel(unit.task),
                       journal_dir=tmp_path / "local2" / "journals", sweep_dir=tmp_path / "local2" / "sweeps",
                       mirror_dir=drive)

    assert calls == ["sh_write_02|clean|typed_hint|plain"]  # a fresh runtime resumed from Drive
    assert sorted(s.task_id for s in scores) == ["sh_write_01", "sh_write_02"]
    assert len(scored_lines(drive / "mirrored.scores.jsonl")) == 2


def test_a_scores_copy_cut_mid_line_loses_only_that_episode(tmp_path):
    from steadyhand.eval.suite import load_scores

    two = [BY_ID["sh_write_01"], BY_ID["sh_write_02"]]
    spec = spec_for(two, ["typed_hint"], sweep_id="cut")
    run_sweep(spec, two, lambda unit: ScriptedTaskModel(unit.task),
              journal_dir=tmp_path / "journals", sweep_dir=tmp_path / "sweeps")
    path = tmp_path / "sweeps" / "cut.scores.jsonl"
    text = path.read_text()
    path.write_text(text[: len(text) - 40])  # the last record is cut off, as a mid-write copy can be

    assert len(load_scores(path)) == 1
    path.write_text("not json\n" + text)
    with pytest.raises(ValueError):
        load_scores(path)  # corruption anywhere else is still an error


# Infrastructure aborts on a flaky Colab server


class DeadServerModel(ScriptedTaskModel):
    """A model whose server is gone: every turn fails the way SteadyhandClient reports it."""

    def __init__(self, task, code="backend_connection_error"):
        super().__init__(task)
        self.code = code

    def complete(self, messages, tools, *, tool_choice="auto"):
        from steadyhand.types import ModelTurn, TurnKind

        return ModelTurn.failure(TurnKind.FATAL_ERROR, self.code, "connection refused")


def test_an_infrastructure_abort_is_rerun_under_a_new_episode_id_and_capped(tmp_path):
    from steadyhand.eval.suite import SWEEP_RETRIES, score_history

    two = [BY_ID["sh_write_01"], BY_ID["sh_write_02"]]
    spec = spec_for(two, ["typed_hint"], sweep_id="flaky")
    folders = {"journal_dir": tmp_path / "journals", "sweep_dir": tmp_path / "sweeps"}

    def dead_for_second(unit):
        return DeadServerModel(unit.task) if unit.task.task_id == "sh_write_02" else ScriptedTaskModel(unit.task)

    first = {s.task_id: s for s in run_sweep(spec, two, dead_for_second, **folders)}
    assert first["sh_write_02"].aborted and first["sh_write_02"].error_code == "backend_connection_error"
    assert first["sh_write_02"].task_success is None  # never scored as a task result

    calls = []
    second = {s.task_id: s for s in run_sweep(
        spec, two, lambda unit: calls.append(unit.key) or ScriptedTaskModel(unit.task), **folders)}
    assert calls == ["sh_write_02|clean|typed_hint|plain"]
    assert second["sh_write_02"].task_success and second["sh_write_02"].attempt == 2
    assert second["sh_write_02"].episode_id != first["sh_write_02"].episode_id

    always_dead = spec_for([BY_ID["sh_read_01"]], ["typed_hint"], sweep_id="always-dead")
    for _ in range(SWEEP_RETRIES + 3):
        run_sweep(always_dead, [BY_ID["sh_read_01"]], lambda unit: DeadServerModel(unit.task), **folders)
    [attempts] = score_history(tmp_path / "sweeps" / "always-dead.scores.jsonl").values()
    assert len(attempts) == 1 + SWEEP_RETRIES  # then it stays aborted, listed in the report
    assert "rerun after an infrastructure abort" in render_report([attempts[-1]])


def test_a_token_limit_abort_is_not_rerun(tmp_path):
    one = [BY_ID["sh_read_01"]]
    spec = spec_for(one, ["typed_hint"], sweep_id="truncated")
    folders = {"journal_dir": tmp_path / "journals", "sweep_dir": tmp_path / "sweeps"}

    [score] = run_sweep(spec, one, lambda unit: DeadServerModel(unit.task, code="completion_truncated"), **folders)
    assert score.aborted and score.error_code == "completion_truncated"
    assert run_sweep(spec, one, lambda unit: pytest.fail("a token limit does not go away on a rerun"), **folders)


def test_a_dead_server_interrupts_the_sweep_instead_of_aborting_every_episode(tmp_path):
    from steadyhand.eval.suite import SweepInterrupted

    four = [BY_ID["sh_read_01"], BY_ID["sh_read_02"], BY_ID["sh_write_01"], BY_ID["sh_write_02"]]
    spec = spec_for(four, ["typed_hint"], sweep_id="server-died")

    with pytest.raises(SweepInterrupted, match="2 episodes in a row aborted"):
        run_sweep(spec, four, lambda unit: DeadServerModel(unit.task), journal_dir=tmp_path / "journals",
                  sweep_dir=tmp_path / "sweeps", abort_limit=2)

    assert len(scored_lines(tmp_path / "sweeps" / "server-died.scores.jsonl")) == 2  # the rest never started
    scores = run_sweep(spec, four, lambda unit: ScriptedTaskModel(unit.task), journal_dir=tmp_path / "journals",
                       sweep_dir=tmp_path / "sweeps")
    assert len(scores) == 4 and all(s.task_success for s in scores)

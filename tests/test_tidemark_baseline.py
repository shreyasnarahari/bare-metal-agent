"""The naive append/truncate baseline, the long-horizon seed set, the resumable runner, and the report."""

import csv
import json
import random
from dataclasses import replace

import pytest

from steadyhand import SteadyhandClient
from steadyhand.journal import read_journal
from steadyhand.types import ModelTurn, TurnKind
from tidemark.baseline import NaiveAppendTruncate, canonical_message, complete_turn_groups
from tidemark.eval.baseline import (
    BaselineMismatch,
    BaselineSpec,
    DatasetError,
    EvidenceProbe,
    ForcedPrefixClient,
    LongHorizonScriptedModel,
    LongHorizonTask,
    STEP_COLUMNS,
    dataset_sha256,
    fact_available,
    load_baseline_scores,
    load_long_horizon,
    run_baseline,
    step_rows,
    validate_long_horizon,
    write_steps_csv,
)
from tidemark.eval.report import prefill_svg, summarize, write_report
from tidemark.tokenizer import RegexTokenizer
from tidemark.types import ContextConfig
from toolsmith.api import APIError, ToolsmithService

from tidemark_support import FakeService, attempt, completion, qwen_like_render

CONFIRMED = {"confirmed": True, "new_tokens_field": "prompt_n", "total_tokens_field": None}
TOOLS = [{"type": "function", "function": {"name": "get_order", "description": "d", "parameters": {"type": "object", "properties": {}}}}]


def config(slot_context=4096, max_tokens=256, **overrides):
    return ContextConfig(slot_context=slot_context, max_tokens=max_tokens, tokenizer=RegexTokenizer().identity(), **overrides)


def session(slot_context=4096, **overrides):
    return NaiveAppendTruncate(config(slot_context, **overrides), render=qwen_like_render, tokenizer=RegexTokenizer())


def tokens(messages, tools=TOOLS):
    return len(RegexTokenizer().encode(qwen_like_render(messages, tools)))


def call_group(index, size=40):
    call_id = f"call_{index}"
    return [
        {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function",
                                                               "function": {"name": "get_order", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": " ".join(f"word{index}_{n}" for n in range(size))},
    ]


def history(turns=8, size=40):
    messages = [{"role": "system", "content": "You are a support agent."},
                {"role": "user", "content": "Remember: code PAY-4471-K."}, {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": "Review my account, then record the code."}]
    for index in range(turns):
        messages += call_group(index, size)
    return messages


# Complete-turn truncation


def test_turns_group_calls_with_their_results_and_pin_the_right_ones():
    messages = history(turns=3)
    groups = complete_turn_groups(messages, initial_count=4)
    assert [(g.start, g.end, g.pinned) for g in groups] == [
        (0, 1, True), (1, 2, False), (2, 3, False), (3, 4, True), (4, 6, False), (6, 8, False), (8, 10, True),
    ]


def test_a_prompt_that_fits_is_sent_unchanged():
    messages = history(turns=2)
    rendered = session().render(messages, step=2, tools=TOOLS, initial_count=4)
    assert rendered.status == "ok" and rendered.dropped_ranges == ()
    assert list(rendered.messages) == [canonical_message(m) for m in messages]
    assert rendered.rendered_prompt_tokens == tokens(messages)
    assert rendered.slot_headroom == config().prompt_limit - rendered.rendered_prompt_tokens


def test_the_oldest_complete_turns_go_first_and_the_result_stays_valid():
    messages = history(turns=8)
    limit = tokens(messages) - 120
    naive = session(slot_context=limit + 256 + 64)
    rendered = naive.render(messages, step=8, tools=TOOLS, initial_count=4)
    assert rendered.status == "ok" and rendered.rendered_prompt_tokens <= naive.config.prompt_limit
    # The planted memory turns (indexes 1-2) go first, then the oldest call/result pairs, never half a pair.
    assert rendered.dropped_ranges[0][0] == 1
    kept = list(rendered.messages)
    assert kept[0]["role"] == "system" and kept[1]["content"].startswith("Review my account")
    ToolsmithService._validate_messages(kept)
    # Minimal: putting back the newest dropped turn would not fit.
    assert tokens(messages[:1] + messages[3:]) > naive.config.prompt_limit or rendered.dropped_messages == 2
    assert naive.renders <= 2 + len(messages).bit_length()  # binary search, not one render per turn


def test_truncation_never_breaks_a_history_the_shim_accepts():
    rng = random.Random(7)
    for trial in range(40):
        messages = history(turns=rng.randint(1, 12), size=rng.randint(5, 60))
        if rng.random() < 0.3:  # a parse-error observation between turns
            messages.insert(rng.randrange(5, len(messages)) // 2 * 2, {"role": "user", "content": "parse_error: fix the JSON"})
        try:
            ToolsmithService._validate_messages(messages)
        except APIError:
            continue
        budget = rng.randint(tokens(messages[:4] + messages[-2:]) + 1, tokens(messages) + 50)
        naive = session(slot_context=budget + 256 + 64)
        rendered = naive.render(messages, step=trial, tools=TOOLS, initial_count=4)
        assert rendered.status == "ok", trial
        ToolsmithService._validate_messages(list(rendered.messages))
        assert rendered.messages[0]["role"] == "system" and rendered.messages[-1] == canonical_message(messages[-1])
        assert any(m.get("content") == messages[3]["content"] for m in rendered.messages)  # the request stays
        assert rendered.rendered_prompt_tokens <= naive.config.prompt_limit


def test_an_oversized_current_turn_fails_explicitly():
    messages = history(turns=2) + call_group(99, size=2000)
    naive = session(slot_context=1500)
    rendered = naive.render(messages, step=3, tools=TOOLS, initial_count=4)
    assert rendered.exhausted and rendered.status == "context_budget_exhausted"
    assert rendered.max_tokens == 256 and rendered.slot_headroom < 0  # max_tokens is never lowered
    assert "required current turn" in rendered.detail
    assert rendered.dropped_ranges  # it tried dropping everything droppable first


def test_budgets_use_real_tokens_not_characters():
    class Expensive(RegexTokenizer):
        def encode(self, text):  # every word costs ten tokens
            return tuple(i for i in super().encode(text) for _ in range(10))

    messages = history(turns=4)
    cheap = session(slot_context=tokens(messages) + 400).render(messages, step=4, tools=TOOLS, initial_count=4)
    costly = NaiveAppendTruncate(config(slot_context=tokens(messages) + 400), render=qwen_like_render,
                                 tokenizer=Expensive()).render(messages, step=4, tools=TOOLS, initial_count=4)
    assert cheap.dropped_ranges == () and costly.dropped_ranges


def test_the_session_identity_is_its_canonical_configuration():
    naive = session(soft_limit=1000)
    assert naive.identity() == config(soft_limit=1000).as_dict()
    assert naive.identity()["strategy"] == "naive_append_truncate" and naive.version
    with pytest.raises(ValueError):
        ContextConfig(slot_context=300, max_tokens=256, safety_margin=64)
    with pytest.raises(ValueError):
        ContextConfig(slot_context=4096, max_tokens=256, strategy="sliding_window")
    assert ContextConfig.from_dict(config().as_dict()) == config()


# The seed set


def test_the_seed_set_is_valid_and_covers_each_kind_at_each_distance():
    tasks = load_long_horizon()
    summary = validate_long_horizon(tasks)
    assert summary["tasks"] == 12 and summary["distances"] == [5, 10, 20, 40]
    assert summary["categories"] == {"error_code": 4, "file_path": 4, "user_constraint": 4}
    assert all(task.required_availability in ("inline", "blob_handle", "memory", "any_restorable") for task in tasks)
    for task in tasks:
        assert task.planted_fact["text"] not in task.goal and len(task.forced_steps) == task.needed_at_step


def broken(task, **changes):
    value = {**task.as_dict(), **changes}
    return LongHorizonTask.from_dict(value)


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda t: {"goal": t.goal + " " + t.planted_fact["text"]}, "pinned request"),
        (lambda t: {"distance_N": t.distance_N + 1}, "distance_N"),
        (lambda t: {"scripted_steps": [[{"name": "launch_rocket", "arguments": {}}], *t.scripted_steps[1:]]}, "unknown tool"),
        (lambda t: {"scripted_steps": [t.reference_tail[0], *t.scripted_steps[1:]]}, "not read-only"),
        (lambda t: {"scripted_steps": [t.scripted_steps[0]] * 3 + list(t.scripted_steps[3:])}, "guard blocks"),
        (lambda t: {"observation_ids": ["store:ord_9999"]}, "observation id"),
        (lambda t: {"gold_store_diff": {"tickets_created": 2}}, "miss the gold"),
        (lambda t: {"distractors": [t.planted_fact["text"] + "X"]}, "overlaps the fact"),
        (lambda t: {"initial_memory": [{"role": "user", "content": "hf_" + "a" * 30 + " " + t.planted_fact["text"]}]}, "secret"),
        (lambda t: {"version": "v0"}, "version"),
        (lambda t: {"required_availability": "telepathy"}, "required_availability"),
    ],
)
def test_the_validator_rejects_broken_tasks(change, problem):
    task = load_long_horizon()[0]
    with pytest.raises(DatasetError) as caught:
        validate_long_horizon([broken(task, **change(task))])
    assert any(problem in item for item in caught.value.problems), caught.value.problems


def test_duplicate_ids_are_rejected():
    task = load_long_horizon()[0]
    with pytest.raises(DatasetError, match="duplicate task_id"):
        validate_long_horizon([task, task])


def test_evidence_is_inline_text_or_tool_call_arguments():
    messages = [{"role": "user", "content": "nothing"},
                {"role": "assistant", "content": None, "tool_calls": [{"function": {"arguments": '{"note": "PAY-4471-K"}'}}]}]
    assert fact_available(messages, "PAY-4471-K", "memory") == {"available": True, "via": "inline", "required": "memory"}
    assert not fact_available(messages[:1], "PAY-4471-K", "inline")["available"]


# Runner


def tasks_and_spec(run_id="cpu", slot_context=4500, mode="scripted", ids=None):
    tasks = load_long_horizon()
    if ids:
        tasks = [t for t in tasks if t.task_id in ids]
    spec = BaselineSpec(run_id=run_id, task_ids=tuple(t.task_id for t in tasks), dataset_sha256=dataset_sha256(tasks),
                        context_config=config(slot_context).as_dict(), mode=mode, gpu="cpu")
    return tasks, spec


def scripted_run(spec, tasks, run_dir, slot_context=4500, *, client_factory=None, **options):
    return run_baseline(
        spec, tasks, run_dir=run_dir,
        client_factory=client_factory or (lambda task, owner: LongHorizonScriptedModel(task, evidence_sensitive=True, max_tokens=256)),
        session_factory=lambda task, client: session(slot_context), **options,
    )


def test_a_cpu_baseline_truncates_loses_distant_facts_and_resumes(tmp_path):
    tasks, spec = tasks_and_spec(ids={"tm_error_code_n05", "tm_error_code_n40", "tm_user_constraint_n40"})
    scores = {s.task_id: s for s in scripted_run(spec, tasks, tmp_path, workers=2)}
    assert scores["tm_error_code_n05"].outcome == "success" and scores["tm_error_code_n05"].evidence_available
    assert scores["tm_error_code_n40"].outcome == "failure" and scores["tm_error_code_n40"].evidence_available is False
    # Evidence and task success are separate: this reference tail needs no planted text to pass.
    assert scores["tm_user_constraint_n40"].task_success and scores["tm_user_constraint_n40"].evidence_available is False
    lines = (tmp_path / "cpu.scores.jsonl").read_text().splitlines()
    again = scripted_run(spec, tasks, tmp_path)
    assert len(again) == 3 and (tmp_path / "cpu.scores.jsonl").read_text().splitlines() == lines  # nothing reran


def test_the_runner_drives_episodes_through_the_shared_scheduler(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from tidemark.scheduler import Scheduler, SchedulerPlan

    tasks, spec = tasks_and_spec(ids={"tm_error_code_n05", "tm_file_path_n05", "tm_user_constraint_n05"})
    plan = SchedulerPlan(agent_slots=2, affinity="pinned", cpu_workers=1, io_threads=1)
    with Scheduler(plan, cpu_executor=lambda: ThreadPoolExecutor(1)) as scheduler:
        scores = scripted_run(spec, tasks, tmp_path, scheduler=scheduler)
        stats = scheduler.slots.stats()
    assert [s.task_id for s in scores] == list(spec.task_ids) and all(s.outcome == "success" for s in scores)
    assert stats["max_busy"] <= 2 and stats["leases"] == sum(s.steps + 1 for s in scores)
    assert scheduler.slots.snapshot()["slots_pinned_idle"] == 0  # every episode released its pinned slot

    live = FakeService(completion("done", attempts=[attempt(prompt_n=5, id_slot=1)]))
    forced = ForcedPrefixClient(SteadyhandClient(live), replace(tasks[0], needed_at_step=0))
    forced.complete([{"role": "user", "content": "hi"}], [], id_slot=1)
    assert live.slots == [1]


def test_a_changed_dataset_or_config_is_refused(tmp_path):
    tasks, spec = tasks_and_spec(ids={"tm_file_path_n05"})
    scripted_run(spec, tasks, tmp_path)
    with pytest.raises(BaselineMismatch) as changed:
        scripted_run(replace(spec, context_config=config(5000).as_dict()), tasks, tmp_path)
    assert changed.value.field == "context_config"
    edited = [replace(tasks[0], goal=tasks[0].goal + " Please.")]
    with pytest.raises(BaselineMismatch) as dataset:
        scripted_run(spec, edited, tmp_path)
    assert dataset.value.field == "dataset_sha256"


class FlakyBackend:
    """Fails with a backend error on the first episode it serves, then behaves like the scripted model."""

    def __init__(self, task, failures):
        self.inner = LongHorizonScriptedModel(task, max_tokens=256)
        self.failures = failures

    def settings(self):
        return self.inner.settings()

    def complete(self, messages, tools, *, tool_choice="auto"):
        if self.failures:
            self.failures.pop()
            return ModelTurn.failure(TurnKind.FATAL_ERROR, "backend_connection_error", "server gone")
        return self.inner.complete(messages, tools, tool_choice=tool_choice)


def test_infrastructure_aborts_stay_visible_out_of_denominators_and_rerun(tmp_path):
    tasks, spec = tasks_and_spec(ids={"tm_file_path_n05"}, slot_context=8192)
    failures = [1]
    first = scripted_run(spec, tasks, tmp_path, 8192, client_factory=lambda task, owner: FlakyBackend(task, failures))
    assert first[0].outcome == "infrastructure" and first[0].task_success is None
    summary = summarize({"cpu": ({"run_id": "cpu"}, first, [])}, spec.task_ids, expected_gpus=("cpu",))
    assert summary["by_gpu"]["cpu"]["success_denominator"] == 0 and summary["by_gpu"]["cpu"]["outcomes"] == {"infrastructure": 1}
    second = scripted_run(spec, tasks, tmp_path, 8192, client_factory=lambda task, owner: FlakyBackend(task, failures))
    assert second[0].outcome == "success" and second[0].attempt == 2
    assert second[0].episode_id != first[0].episode_id  # a rerun gets a new label and id
    history = load_baseline_scores(tmp_path / "cpu.scores.jsonl")["tm_file_path_n05"]
    assert [s.outcome for s in history] == ["infrastructure", "success"]


def test_a_request_that_can_never_fit_is_did_not_fit(tmp_path):
    tasks, spec = tasks_and_spec(ids={"tm_file_path_n05"}, slot_context=420)
    (score,) = scripted_run(spec, tasks, tmp_path, 420)
    assert score.outcome == "did_not_fit" and score.task_success is None and score.error_code == "context_budget_exhausted"


def test_step_rows_label_cold_ordinary_truncation_and_write_empty_unknowns(tmp_path):
    tasks, spec = tasks_and_spec(ids={"tm_error_code_n20"})
    (score,) = scripted_run(spec, tasks, tmp_path)
    rows = step_rows(read_journal(score.journal_path), task=tasks[0], score=score, run_id="cpu", gpu="cpu")
    kinds = [row["step_kind"] for row in rows]
    assert kinds[0] == "cold" and "ordinary" in kinds and "truncation" in kinds
    assert all(row["turn_source"] == "scripted" for row in rows)
    assert [row["evidence_available"] for row in rows if row["evidence_available"] is not None] == [False]
    assert all(row["prompt_ms"] is None and row["prompt_evaluated_tokens"] is None for row in rows)  # no server here
    path = write_steps_csv(rows, tmp_path / "steps.csv")
    with path.open() as handle:
        parsed = list(csv.DictReader(handle))
    assert list(parsed[0]) == list(STEP_COLUMNS) and parsed[0]["prompt_ms"] == "" and parsed[0]["step_kind"] == "cold"


def test_the_forced_prefix_keeps_live_costs_and_frees_the_model_after_it():
    task = next(t for t in load_long_horizon() if t.task_id == "tm_file_path_n05")
    live_turns = [completion("ignored", attempts=[attempt(prompt_n=100 + i, id_slot=1)]) for i in range(5)]
    live_turns.append(completion(calls=[("free", "add_order_note", {"order_id": "ord_1001", "note": "x"})],
                                 attempts=[attempt(prompt_n=7)]))
    forced = ForcedPrefixClient(SteadyhandClient(FakeService(*live_turns)), task)
    messages = task.messages()
    turns = [forced.complete(messages, [], tool_choice="auto") for _ in range(6)]
    assert forced.sources == ["forced"] * 5 + ["model"]
    assert [t.tool_calls[0].name for t in turns[:5]] == [step[0]["name"] for step in task.forced_steps]
    assert turns[2].metrics.prompt_evaluated_tokens == 102 and turns[2].metrics.slot_id == 1
    assert turns[5].tool_calls[0].id == "free"
    failing = ForcedPrefixClient(SteadyhandClient(FakeService(APIError("down", status_code=502, code="backend_error"))), task)
    assert failing.complete(messages, []).kind is TurnKind.FATAL_ERROR


def test_the_evidence_probe_changes_nothing_it_wraps():
    task = load_long_horizon()[0]
    inner = session()
    probe = EvidenceProbe(inner, task)
    assert probe.identity() == inner.identity() and probe.strategy == inner.strategy and probe.version == inner.version


# Report


def test_the_report_reads_raw_records_and_marks_unrun_cells_untested(tmp_path):
    tasks, spec = tasks_and_spec(run_id="t4-baseline", ids={"tm_error_code_n05", "tm_error_code_n20"})
    spec = replace(spec, gpu="t4")
    scripted_run(spec, tasks, tmp_path / "t4")
    (tmp_path / "t4" / "utilization.json").write_text(json.dumps({"run_id": "t4-baseline", "scheduler": {}}))
    paths = write_report({"t4": tmp_path / "t4", "l4": tmp_path / "missing"}, tmp_path / "report", tasks=tasks)
    summary = json.loads(paths["summary_json"].read_text())
    assert summary["by_gpu"]["l4"] == {"status": "untested"}
    assert summary["by_gpu"]["t4"]["tasks_scored"] == 2
    assert {c["status"] for c in summary["cells"] if c["gpu"] == "l4"} == {"untested"}
    assert summary["by_gpu"]["t4"]["step_kinds"]["cold"]["requests"] == 2
    assert summary["by_gpu"]["t4"]["step_kinds"]["cold"]["prompt_ms_median"] is None  # no fabricated timings
    svg = paths["prefill_svg"].read_text()
    assert svg.startswith("<svg") and "truncation" in svg and "no server counts" in svg and "<title>" in svg
    assert "untested" in paths["report_md"].read_text()
    rows = list(csv.DictReader(paths["steps_csv"].open()))
    assert {row["task_id"] for row in rows} == {"tm_error_code_n05", "tm_error_code_n20"}


def test_the_chart_prefers_server_counts_and_handles_no_rows():
    assert "untested: no rendered requests" in prefill_svg([])
    row = {key: None for key in STEP_COLUMNS}
    row.update(step=0, step_kind="cold", render_status="ok", rendered_prompt_tokens=500, prompt_evaluated_tokens=500,
               task_id="t", turn_source="forced")
    warm = {**row, "step": 1, "step_kind": "ordinary", "rendered_prompt_tokens": 600, "prompt_evaluated_tokens": 101}
    svg = prefill_svg([row, warm])
    assert "Prompt tokens llama-server evaluated" in svg and "evaluated 101" in svg

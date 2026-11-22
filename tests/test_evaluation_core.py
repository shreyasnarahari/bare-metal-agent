import json

import pytest

from toolsmith.evaluation.metrics import (
    calls_equal,
    deterministic_subset,
    summarize_results,
    wilson_interval,
)
from toolsmith.evaluation.models import AttemptRecord, CaseResult, Mode, RunConfig
from toolsmith.evaluation.report import markdown_summary, summarize_by_group
from toolsmith.evaluation.runner import CheckpointRunner, ManifestMismatchError, atomic_write_json


def result(case_id="case-1", **overrides):
    values = {
        "case_id": case_id,
        "category": "no-tool",
        "model": "model",
        "mode": "prompt",
        "status": "success",
        "parsed_calls": [],
        "parse_valid": True,
        "schema_valid": True,
        "semantic_valid": True,
    }
    values.update(overrides)
    return CaseResult(**values)


def test_call_comparison_supports_duplicates_and_order_rules():
    first = {"name": "f", "arguments": {"x": 1}}
    second = {"name": "g", "arguments": {"x": [1, 2]}}

    assert calls_equal([first, second, first], [second, first, first], "unordered")
    assert not calls_equal([first, second], [second, first], "sequential")
    assert calls_equal([], [], "none")


def test_deterministic_subset_is_sorted_reproducible_and_bounded():
    values = [{"unique_id": f"id-{index:03d}"} for index in range(20)]

    first = deterministic_subset(values, 5, seed=17)
    second = deterministic_subset(values, 5, seed=17)

    assert first == second
    assert [item["unique_id"] for item in first] == sorted(item["unique_id"] for item in first)
    assert deterministic_subset(values, 0, seed=17) == []


def test_deterministic_subset_uses_attribute_identity_for_paths(tmp_path):
    files = [tmp_path / f"split---{index:03d}.json" for index in range(5, 0, -1)]
    for file in files:
        file.write_text("{}")

    selected = deterministic_subset(files, 2, seed=10, id_key="name")

    assert selected == deterministic_subset(files, 2, seed=10, id_key="name")
    assert [file.name for file in selected] == sorted(file.name for file in selected)


def test_checkpoint_resume_and_manifest_mismatch(tmp_path):
    path = tmp_path / "results.jsonl"
    runner = CheckpointRunner(path, "hash")
    runner.run([{"case_id": "case-1"}], lambda case: result())
    resumed = CheckpointRunner(path, "hash")
    calls = []
    results = resumed.run([{"case_id": "case-1"}], lambda case: calls.append(case))

    assert len(results) == 1 and not calls
    with pytest.raises(ManifestMismatchError):
        CheckpointRunner(path, "different")
    with pytest.raises(ValueError, match="already checkpointed"):
        runner.append(result())


def test_atomic_report_and_metrics(tmp_path):
    records = [
        result(
            attempts=(
                AttemptRecord(0, {"content": "ok"}, 2.0, completion_tokens=4),
            )
        ),
        result(
            "case-2",
            category="no-tool",
            parsed_calls=[{"name": "wrong"}],
            semantic_valid=False,
            repair_attempts=1,
            status="repaired",
        ),
    ]
    summary = summarize_by_group(records)
    own = summarize_results(records)

    assert own["total"] == 2
    assert own["parse_rate"] == 1
    assert own["semantic_rate"] == 0.5
    assert own["false_call_rate"] == 0.5
    assert own["completion_tokens"] == 4
    assert len(wilson_interval(1, 2)) == 2
    assert "unexecuted runs have no scores" in markdown_summary(summary, title="Test")
    destination = atomic_write_json(tmp_path / "summary.json", own)
    assert json.loads(destination.read_text())["total"] == 2


def test_run_config_serializes_mode():
    config = RunConfig("run", "own", "model", Mode.GRAMMAR)
    assert config.as_dict()["mode"] == "grammar"

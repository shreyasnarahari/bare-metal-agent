import json
from pathlib import Path

from toolsmith.evaluation.datasets import (
    EXPECTED_CATEGORIES,
    load_cases,
    load_tool_definitions,
    validate_toolsmith_dataset,
)

DATA = Path(__file__).parent.parent / "evaluation" / "data"


def test_gold_dataset_has_200_valid_cases_and_locked_registry():
    cases = load_cases(DATA / "toolsmith_single_turn_v1.jsonl")
    tools = load_tool_definitions(DATA / "toolsmith_tools_v1.json")

    summary = validate_toolsmith_dataset(cases, tools)

    assert summary["cases"] == 200
    assert summary["tools"] == 10
    assert summary["categories"] == EXPECTED_CATEGORIES
    assert len(summary["dataset_sha256"]) == 64
    assert len(summary["tools_sha256"]) == 64


def test_every_prompt_is_nonempty_and_case_schema_is_explicit():
    for case in load_cases(DATA / "toolsmith_single_turn_v1.jsonl"):
        assert set(case) == {
            "case_id",
            "category",
            "messages",
            "tool_choice",
            "expected_calls",
            "comparison",
            "tools_version",
        }
        assert case["messages"][0]["role"] == "user"
        assert case["messages"][0]["content"].strip()
        for call in case["expected_calls"]:
            assert set(call) == {"name", "arguments"}
            json.dumps(call["arguments"], allow_nan=False)

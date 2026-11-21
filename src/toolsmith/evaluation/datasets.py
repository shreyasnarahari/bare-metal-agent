from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..grammar import resolve_tool_choice
from ..parsing import ParsedResponse, ParsedToolCall
from ..registry import Tool
from ..validation import ToolCallValidator
from .models import canonical_hash

EXPECTED_CATEGORIES = {
    "simple": 30,
    "optional": 20,
    "enum-array": 20,
    "nested": 20,
    "parallel": 20,
    "multiple": 15,
    "no-tool": 25,
    "named": 15,
    "choice": 15,
    "unicode": 20,
}


def load_tool_definitions(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text())
    return value["tools"]


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def validate_toolsmith_dataset(
    cases: list[dict[str, Any]],
    tool_definitions: list[dict[str, Any]],
) -> dict[str, Any]:
    tools = [Tool.from_openai(definition) for definition in tool_definitions]
    names = [tool.name for tool in tools]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate evaluation tool name")
    if not 8 <= len(tools) <= 10:
        raise ValueError("The evaluation registry must contain 8–10 tools")
    if len(cases) != 200:
        raise ValueError(f"Expected 200 cases, got {len(cases)}")
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evaluation case ID")
    counts = Counter(case["category"] for case in cases)
    if counts != EXPECTED_CATEGORIES:
        raise ValueError(f"Unexpected category counts: {counts}")
    for case in cases:
        if case["tools_version"] != "toolsmith-tools-v1":
            raise ValueError(f"Wrong tool registry version in {case['case_id']}")
        choice = resolve_tool_choice(tools, case["tool_choice"])
        calls = [
            ParsedToolCall(
                f"call_{index}",
                call["name"],
                json.dumps(call["arguments"], ensure_ascii=False, allow_nan=False),
            )
            for index, call in enumerate(case["expected_calls"])
        ]
        ToolCallValidator(tools, choice).validate(ParsedResponse(tool_calls=calls))
        if case["comparison"] not in {"none", "sequential", "unordered"}:
            raise ValueError(f"Invalid comparison in {case['case_id']}")
        if (case["comparison"] == "none") != (not calls):
            raise ValueError(f"Comparison/no-call mismatch in {case['case_id']}")
    return {
        "cases": len(cases),
        "tools": len(tools),
        "categories": dict(sorted(counts.items())),
        "dataset_sha256": canonical_hash(cases),
        "tools_sha256": canonical_hash(tool_definitions),
    }


__all__ = [
    "EXPECTED_CATEGORIES",
    "load_cases",
    "load_tool_definitions",
    "validate_toolsmith_dataset",
]

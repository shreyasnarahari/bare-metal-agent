from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .metrics import summarize_results
from .models import CaseResult


def summarize_by_group(results: Iterable[CaseResult]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[CaseResult]] = defaultdict(list)
    for result in results:
        groups[(result.model, result.mode, result.category)].append(result)
    return {
        "/".join(group): summarize_results(group_results)
        for group, group_results in sorted(groups.items())
    }


def markdown_summary(summary: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "This report is generated from recorded evaluation results; unexecuted runs have no scores.",
        "",
        "| Group | Cases | Parse | Schema valid | Semantic | False call | Repairs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, metrics in sorted(summary.items()):
        lines.append(
            f"| {group} | {metrics['total']} | {metrics['parse_rate']:.2%} "
            f"| {metrics['schema_valid_rate']:.2%} | {metrics['semantic_rate']:.2%} "
            f"| {metrics['false_call_rate']:.2%} | {metrics['repair_attempts']} |"
        )
    return "\n".join(lines) + "\n"


__all__ = ["markdown_summary", "summarize_by_group"]

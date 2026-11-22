from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .models import CaseResult


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def calls_equal(
    actual: Sequence[dict[str, Any]],
    expected: Sequence[dict[str, Any]],
    comparison: str,
) -> bool:
    if comparison == "none":
        return not actual and not expected
    if comparison == "sequential":
        return _canonical(list(actual)) == _canonical(list(expected))
    if comparison == "unordered":
        return Counter(_canonical(call) for call in actual) == Counter(
            _canonical(call) for call in expected
        )
    raise ValueError(f"Unknown call comparison: {comparison}")


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("Percentile probability must be between zero and one")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total < 1:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def paired_bootstrap(
    differences: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 0,
) -> list[float]:
    if not differences:
        return [0.0, 0.0]
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    randomizer = random.Random(seed)
    size = len(differences)
    means = sorted(
        sum(differences[randomizer.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return [
        percentile(means, 0.025) or 0.0,
        percentile(means, 0.975) or 0.0,
    ]


def summarize_results(results: Iterable[CaseResult]) -> dict[str, Any]:
    records = list(results)
    total = len(records)
    parse_count = sum(record.parse_valid for record in records)
    schema_count = sum(record.schema_valid for record in records)
    semantic_known = [record for record in records if record.semantic_valid is not None]
    semantic_count = sum(bool(record.semantic_valid) for record in semantic_known)
    no_tool = [record for record in records if record.category == "no-tool"]
    irrelevant = [record for record in records if record.category == "irrelevance"]
    no_tool.extend(irrelevant)
    latencies = [
        attempt.elapsed_seconds
        for record in records
        for attempt in record.attempts
        if not attempt.reused
    ]
    completion_tokens = sum(
        attempt.completion_tokens for record in records for attempt in record.attempts
    )
    elapsed = sum(latencies)
    status_counts = Counter(record.status for record in records)
    return {
        "total": total,
        "parse_rate": parse_count / total if total else 0.0,
        "parse_rate_ci95": wilson_interval(parse_count, total),
        "schema_valid_rate": schema_count / total if total else 0.0,
        "schema_valid_rate_ci95": wilson_interval(schema_count, total),
        "semantic_rate": semantic_count / len(semantic_known) if semantic_known else 0.0,
        "semantic_rate_ci95": wilson_interval(semantic_count, len(semantic_known)),
        "false_call_rate": (
            sum(bool(record.parsed_calls) for record in no_tool) / len(no_tool)
            if no_tool
            else 0.0
        ),
        "repair_attempts": sum(record.repair_attempts for record in records),
        "repair_recoveries": sum(
            record.repair_attempts > 0 and record.status in {"success", "repaired"}
            for record in records
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "completion_tokens": completion_tokens,
        "elapsed_seconds": elapsed,
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "tokens_per_second": completion_tokens / elapsed if elapsed else 0.0,
    }


def _subset_identity(value: Any, id_key: str) -> Any:
    if isinstance(value, Mapping):
        return value[id_key]
    return getattr(value, id_key)


def deterministic_subset(
    values: Sequence[Any],
    count: int,
    *,
    seed: int,
    id_key: str = "unique_id",
) -> list[Any]:
    if count < 0:
        raise ValueError("Subset count cannot be negative")
    ordered = sorted(values, key=lambda value: _subset_identity(value, id_key))
    randomizer = random.Random(seed)
    randomizer.shuffle(ordered)
    selected = ordered[:count]
    return sorted(selected, key=lambda value: _subset_identity(value, id_key))


__all__ = [
    "calls_equal",
    "deterministic_subset",
    "paired_bootstrap",
    "percentile",
    "summarize_results",
    "wilson_interval",
]

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

from .runner import EVAL_WORKERS


def run_selected_generation(
    engine: Any,
    split: str,
    items: list[dict[str, Any]],
    messages_formatter: Any,
    *,
    workers: int = EVAL_WORKERS,
) -> dict[str, Any]:
    try:
        from core.evaluator import evaluate
    except ImportError as exc:
        raise ImportError("Install the pinned JSONSchemaBench core package") from exc
    if workers < 1:
        raise ValueError("workers must be positive")

    def generate_one(item: dict[str, Any]) -> Any:
        schema = item["json_schema"]
        if isinstance(schema, str):
            schema = json.loads(schema)
        messages = messages_formatter(split, schema)
        return engine.generate(split, messages, schema)

    if workers == 1 or len(items) <= 1:
        outputs = [generate_one(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
            outputs = list(pool.map(generate_one, items))
    declared, empirical, compliance, performance, output_tokens = evaluate(outputs)
    return {
        "split": split,
        "outputs": [asdict(output) for output in outputs],
        "declared_coverage": asdict(declared),
        "empirical_coverage": asdict(empirical),
        "compliance": asdict(compliance),
        "performance": asdict(performance),
        "output_tokens": asdict(output_tokens),
    }


__all__ = ["run_selected_generation"]

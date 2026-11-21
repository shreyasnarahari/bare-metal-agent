from __future__ import annotations

from dataclasses import asdict
from typing import Any


def run_selected_generation(
    engine: Any,
    split: str,
    items: list[dict[str, Any]],
    messages_formatter: Any,
) -> dict[str, Any]:
    try:
        from core.evaluator import evaluate
    except ImportError as exc:
        raise ImportError("Install the pinned JSONSchemaBench core package") from exc
    outputs = []
    for item in items:
        schema = item["json_schema"]
        if isinstance(schema, str):
            import json
            schema = json.loads(schema)
        messages = messages_formatter(split, schema)
        outputs.append(engine.generate(split, messages, schema))
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

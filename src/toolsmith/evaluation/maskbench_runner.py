from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import percentile


def run_maskbench_files(
    engine: Any,
    files: list[str | Path],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    try:
        from maskbench import runner as maskbench_runner
    except ImportError as exc:
        raise ImportError("Install the pinned MaskBench package for mask evaluation") from exc
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    maskbench_runner.output_path = str(destination)
    engine.multi = False
    results = []
    for file in files:
        path = Path(file)
        output = destination / path.name
        if output.exists():
            results.append(json.loads(output.read_text()))
            continue
        result = maskbench_runner.process_file(engine, str(path))
        results.append(result)
    return results


def timing_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    mask_times = [
        value
        for result in results
        for value in result.get("all_mask_us", [])
    ]
    compile_times = [result["ttfm_us"] for result in results if "ttfm_us" in result]
    probabilities = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0)
    return {
        "schemas": len(results),
        "compile_errors": sum("compile_error" in result for result in results),
        "validation_errors": sum(
            "validation_error" in result and "should reject" not in result["validation_error"]
            for result in results
        ),
        "invalidation_errors": sum(
            "validation_error" in result and "should reject" in result["validation_error"]
            for result in results
        ),
        "tokens": sum(result.get("num_tokens", 0) for result in results),
        "ttfm_us": {
            f"p{int(probability * 1000) / 10:g}": percentile(compile_times, probability)
            for probability in probabilities
        },
        "tbm_us": {
            f"p{int(probability * 1000) / 10:g}": percentile(mask_times, probability)
            for probability in probabilities
        },
    }


__all__ = ["run_maskbench_files", "timing_summary"]

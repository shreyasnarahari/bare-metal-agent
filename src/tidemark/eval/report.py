"""The baseline report, generated only from raw records: ``steps.csv``, ``prefill_by_step.svg``, and a summary.

Inputs are run directories (one per GPU): the spec, the scores JSONL, and the
journals. Nothing is typed in by hand. Every expected (GPU, task) cell without
a score is marked ``untested``. Cold, ordinary, truncation, and paraphrase
steps are summarized separately, never averaged into one hit rate.
Infrastructure aborts and ``did_not_fit`` episodes are counted but kept out of
the task-success denominator.

The chart plots, per request, the prompt tokens llama-server evaluated
(markers shaped and colored by step kind) against the median rendered prompt
per step (a gray reference line). Both are token counts on one axis. Records
without server counts (a CPU scripted run) fall back to rendered tokens, and
the chart says so. Standard library only.

CLI::

    python -m tidemark.eval.report --run t4=/content/tidemark/baseline/t4 --run l4=... --out report/
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

from toolsmith.evaluation.metrics import wilson_interval
from toolsmith.evaluation.models import canonical_hash
from toolsmith.evaluation.runner import atomic_write_json

from steadyhand.journal import read_journal

from .baseline import (
    DATASET_PATH,
    LongHorizonTask,
    load_baseline_scores,
    load_long_horizon,
    step_rows,
    write_steps_csv,
)

EXPECTED_GPUS = ("t4", "l4")
STEP_KINDS = ("cold", "ordinary", "truncation", "paraphrase")
# Reference palette, first three categorical slots (validated all-pairs, light and dark).
KIND_STYLE = {
    "cold": ("circle", "--series-1"),
    "ordinary": ("square", "--series-2"),
    "truncation": ("triangle", "--series-3"),
}


def collect(run_dir: str | Path, tasks: Sequence[LongHorizonTask]) -> tuple[dict[str, Any], list[Any], list[dict[str, Any]]]:
    """A run directory's spec, latest scores, and per-step rows (read from its journals)."""
    run_dir = Path(run_dir)
    specs = []
    for path in sorted(run_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        # The spec is <run_id>.json; other JSON files (utilization.json) sit beside it.
        if isinstance(value, dict) and value.get("run_id") == path.stem and "task_ids" in value:
            specs.append(value)
    if len(specs) != 1:
        raise FileNotFoundError(f"Expected one baseline spec in {run_dir}, found {len(specs)}")
    spec = specs[0]
    history = load_baseline_scores(run_dir / f"{spec['run_id']}.scores.jsonl")
    by_id = {task.task_id: task for task in tasks}
    scores, rows = [], []
    for task_id in spec["task_ids"]:
        if task_id not in history:
            continue
        score = history[task_id][-1]
        scores.append(score)
        journal = Path(score.journal_path)
        if not journal.exists():
            journal = run_dir / "journals" / journal.name
        rows.extend(step_rows(read_journal(journal), task=by_id[task_id], score=score, run_id=spec["run_id"], gpu=spec.get("gpu")))
    return spec, scores, rows


def _median(values: Sequence[float | int | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def summarize(
    runs: Mapping[str, tuple[dict[str, Any], list[Any], list[dict[str, Any]]]],
    task_ids: Sequence[str],
    *,
    expected_gpus: Sequence[str] = EXPECTED_GPUS,
) -> dict[str, Any]:
    """Per-GPU outcomes and per-step-kind medians, plus the full (GPU, task) cell matrix."""
    cells = []
    by_gpu: dict[str, Any] = {}
    for gpu in [*expected_gpus, *[g for g in runs if g not in expected_gpus]]:
        spec, scores, rows = runs.get(gpu, ({}, [], []))
        scored = {score.task_id: score for score in scores}
        for task_id in task_ids:
            score = scored.get(task_id)
            cells.append({
                "gpu": gpu, "task_id": task_id, "status": "run" if score else "untested",
                "outcome": score.outcome if score else None,
                "task_success": score.task_success if score else None,
                "evidence_available": score.evidence_available if score else None,
            })
        if not scores:
            by_gpu[gpu] = {"status": "untested"}
            continue
        counted = [s for s in scores if s.task_success is not None]
        successes = sum(bool(s.task_success) for s in counted)
        evidence = [s.evidence_available for s in scores if s.evidence_available is not None]
        kinds = {}
        for kind in STEP_KINDS:
            subset = [row for row in rows if row["step_kind"] == kind and row["render_status"] == "ok"]
            kinds[kind] = {
                "requests": len(subset),
                "rendered_tokens_median": _median([r["rendered_prompt_tokens"] for r in subset]),
                "evaluated_tokens_median": _median([r["prompt_evaluated_tokens"] for r in subset]),
                "cache_reuse_median": _median([r["cache_reuse_fraction"] for r in subset]),
                "prompt_ms_median": _median([r["prompt_ms"] for r in subset]),
                "ttft_ms_median": _median([r["ttft_ms"] for r in subset]),
                "response_ms_median": _median([r["response_elapsed_ms"] for r in subset]),
            }
        by_gpu[gpu] = {
            "status": "run",
            "run_id": spec.get("run_id"),
            "mode": spec.get("mode"),
            "spec_sha256": canonical_hash(spec),
            "tasks_scored": len(scores),
            "untested": [task_id for task_id in task_ids if task_id not in scored],
            "outcomes": dict(sorted(Counter(s.outcome for s in scores).items())),
            "success_denominator": len(counted),
            "success_rate": successes / len(counted) if counted else None,
            "success_ci95": wilson_interval(successes, len(counted)) if counted else None,
            "evidence_available_rate": sum(evidence) / len(evidence) if evidence else None,
            "measurement_paths": dict(sorted(Counter(r["measurement_path"] for r in rows if r["measurement_path"]).items())),
            "cache_status": dict(sorted(Counter(r["cache_status"] for r in rows if r["cache_status"]).items())),
            "context_budget_exhausted": sum(1 for r in rows if r["render_status"] != "ok"),
            "step_kinds": kinds,
        }
    return {"expected_gpus": list(expected_gpus), "task_ids": list(task_ids), "by_gpu": by_gpu, "cells": cells}


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** (len(str(int(value))) - 1)
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * magnitude:
            return step * magnitude
    return 10 * magnitude


def _marker(shape: str, x: float, y: float, color: str, title: str) -> str:
    style = f'fill="var({color})" stroke="var(--surface-1)" stroke-width="2"'
    if shape == "circle":
        mark = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" {style}/>'
    elif shape == "square":
        mark = f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1.5" {style}/>'
    else:
        mark = f'<path d="M{x:.1f},{y - 5.5:.1f} L{x + 5:.1f},{y + 3.5:.1f} L{x - 5:.1f},{y + 3.5:.1f} Z" {style}/>'
    # A larger transparent target carries the hover title.
    return f'<g class="mark">{mark}<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="transparent"><title>{escape(title)}</title></circle></g>'


def prefill_svg(rows: Sequence[Mapping[str, Any]], *, title_suffix: str = "") -> str:
    """The prefill-by-step chart as standalone SVG text (light and dark themes)."""
    agent = [row for row in rows if row["render_status"] == "ok" and row["step_kind"] in KIND_STYLE]
    has_server = any(row["prompt_evaluated_tokens"] is not None for row in agent)
    metric = "prompt_evaluated_tokens" if has_server else "rendered_prompt_tokens"
    width, height = 760, 440
    left, right, top, bottom = 64, 24, 92, 48
    plot_w, plot_h = width - left - right, height - top - bottom
    max_step = max((int(row["step"]) for row in agent), default=1) or 1
    rendered_by_step: dict[int, list[int]] = {}
    for row in agent:
        if row["rendered_prompt_tokens"] is not None:
            rendered_by_step.setdefault(int(row["step"]), []).append(int(row["rendered_prompt_tokens"]))
    peak = max([*(int(r[metric]) for r in agent if r[metric] is not None), *(max(v) for v in rendered_by_step.values())], default=1)
    y_max = _nice_ceiling(peak * 1.05)

    def sx(step: float) -> float:
        return left + plot_w * step / max_step

    def sy(value: float) -> float:
        return top + plot_h * (1 - value / y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" aria-labelledby="chart-title chart-desc" class="viz-root">',
        "<style>",
        ".viz-root{--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--grid:#e4e3df;"
        "--reference:#8a8984;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;"
        "font-family:system-ui,-apple-system,'Segoe UI',sans-serif}",
        "@media (prefers-color-scheme: dark){.viz-root{--surface-1:#1a1a19;--text-primary:#f0efec;"
        "--text-secondary:#c3c2b7;--grid:#383835;--reference:#9b9a93;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70}}",
        "text{fill:var(--text-secondary);font-size:12px}.title{fill:var(--text-primary);font-size:15px;font-weight:600}"
        ".mark:hover circle:first-child,.mark:hover rect,.mark:hover path{stroke:var(--text-primary)}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="var(--surface-1)"/>',
        f'<title id="chart-title">Prefill by step{escape(title_suffix)}</title>',
        f'<desc id="chart-desc">{"Prompt tokens evaluated by llama-server" if has_server else "Rendered prompt tokens"} per '
        "model request, by step, shaped by step kind (cold, ordinary, truncation), against the median rendered prompt "
        f"per step. {len(agent)} requests. The same values are in steps.csv.</desc>",
        f'<text class="title" x="{left}" y="26">Prefill by step{escape(title_suffix)}</text>',
        f'<text x="{left}" y="46">{"Prompt tokens llama-server evaluated per request" if has_server else "Rendered prompt tokens per request (these records have no server counts)"}; gray line: median rendered prompt</text>',
    ]
    # Legend, one row above the plot; labels in text color, identity on the marks.
    legend_x = left
    for kind, (shape, color) in KIND_STYLE.items():
        parts.append(_marker(shape, legend_x + 5, 68, color, kind).replace('<g class="mark">', "<g>"))
        parts.append(f'<text x="{legend_x + 16}" y="72">{kind}</text>')
        legend_x += 100
    parts.append(f'<line x1="{legend_x}" y1="68" x2="{legend_x + 18}" y2="68" stroke="var(--reference)" stroke-width="2"/>')
    parts.append(f'<text x="{legend_x + 24}" y="72">rendered (median)</text>')
    for tick in range(5):
        value = y_max * tick / 4
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{int(value):,}</text>')
    spacing = max(1, -(-max_step // 8))
    step_ticks = list(range(0, max_step + 1, spacing))
    if max_step - step_ticks[-1] >= spacing / 2:  # label the last step unless it would collide with a tick
        step_ticks.append(max_step)
    for step in step_ticks:
        parts.append(f'<text x="{sx(step):.1f}" y="{height - bottom + 18}" text-anchor="middle">{step}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 10}" text-anchor="middle">step</text>')
    parts.append(f'<text x="16" y="{top + plot_h / 2:.1f}" transform="rotate(-90 16 {top + plot_h / 2:.1f})" text-anchor="middle">tokens</text>')
    if rendered_by_step:
        points = " ".join(f"{sx(step):.1f},{sy(statistics.median(values)):.1f}" for step, values in sorted(rendered_by_step.items()))
        parts.append(f'<polyline points="{points}" fill="none" stroke="var(--reference)" stroke-width="2" stroke-linejoin="round"/>')
    for row in agent:
        value = row[metric]
        if value is None:
            continue
        shape, color = KIND_STYLE[row["step_kind"]]
        tooltip = (
            f"{row['task_id']} step {row['step']} ({row['step_kind']}, {row['turn_source']}): rendered "
            f"{row['rendered_prompt_tokens']}, evaluated {row['prompt_evaluated_tokens']}, reused {row['reused_tokens']}, "
            f"prefill {row['prompt_ms']} ms, TTFT {row['ttft_ms']} ms"
        )
        parts.append(_marker(shape, sx(int(row["step"])), sy(int(value)), color, tooltip))
    if not agent:
        parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h / 2:.1f}" text-anchor="middle">untested: no rendered requests in these records</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Tidemark part 1 baseline", "", "Generated from run records; `untested` means no record exists.", ""]
    lines += ["| GPU | status | mode | scored | success (denominator) | evidence available | outcomes |", "|---|---|---|---|---|---|---|"]
    for gpu, value in summary["by_gpu"].items():
        if value["status"] != "run":
            lines.append(f"| {gpu} | untested | | | | | |")
            continue
        rate = "n/a" if value["success_rate"] is None else f"{value['success_rate']:.0%} ({value['success_denominator']})"
        evidence = "n/a" if value["evidence_available_rate"] is None else f"{value['evidence_available_rate']:.0%}"
        lines.append(f"| {gpu} | run | {value['mode']} | {value['tasks_scored']} | {rate} | {evidence} | {value['outcomes']} |")
    lines += ["", "| GPU | step kind | requests | rendered median | evaluated median | reuse median | prefill ms median | TTFT ms median |",
              "|---|---|---|---|---|---|---|---|"]
    for gpu, value in summary["by_gpu"].items():
        for kind, numbers in (value.get("step_kinds") or {}).items():
            cells = [numbers[k] for k in ("rendered_tokens_median", "evaluated_tokens_median", "cache_reuse_median", "prompt_ms_median", "ttft_ms_median")]
            lines.append(f"| {gpu} | {kind} | {numbers['requests']} | " + " | ".join("untested" if c is None else f"{c:g}" for c in cells) + " |")
    untested = [f"{c['gpu']}/{c['task_id']}" for c in summary["cells"] if c["status"] == "untested"]
    lines += ["", f"Untested cells: {len(untested)}" + (f" ({', '.join(untested)})" if untested else ""), ""]
    return "\n".join(lines)


def write_report(
    runs: Mapping[str, str | Path],
    out_dir: str | Path,
    *,
    tasks: Sequence[LongHorizonTask] | None = None,
    expected_gpus: Sequence[str] = EXPECTED_GPUS,
) -> dict[str, Path]:
    """Read each GPU's run directory and write ``steps.csv``, ``prefill_by_step.svg``, ``summary.json``, ``report.md``."""
    tasks = list(tasks) if tasks is not None else load_long_horizon(DATASET_PATH)
    collected = {gpu: collect(path, tasks) for gpu, path in runs.items() if Path(path).exists()}
    task_ids = sorted({task_id for spec, _, _ in collected.values() for task_id in spec["task_ids"]}) or [t.task_id for t in tasks]
    rows = [row for _, _, gpu_rows in collected.values() for row in gpu_rows]
    out = Path(out_dir)
    summary = summarize(collected, task_ids, expected_gpus=expected_gpus)
    paths = {
        "steps_csv": write_steps_csv(rows, out / "steps.csv"),
        "summary_json": atomic_write_json(out / "summary.json", summary),
    }
    svg = out / "prefill_by_step.svg"
    svg.write_text(prefill_svg(rows, title_suffix=f" ({', '.join(sorted(collected)) or 'no runs'})"), encoding="utf-8")
    paths["prefill_svg"] = svg
    report = out / "report.md"
    report.write_text(render_markdown(summary), encoding="utf-8")
    paths["report_md"] = report
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tidemark baseline report from raw run records")
    parser.add_argument("--run", action="append", default=[], help="gpu=run_dir (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = dict(item.split("=", 1) for item in args.run)
    paths = write_report(runs, args.out)
    print(json.dumps({name: str(path) for name, path in paths.items()}))
    return 0


__all__ = ["EXPECTED_GPUS", "STEP_KINDS", "collect", "main", "prefill_svg", "render_markdown", "summarize", "write_report"]


if __name__ == "__main__":
    raise SystemExit(main())

"""Render ``RESULTS.md`` from episode scores and their journals.

Every number in the results tables comes from a scored journal. Empty cells
stay empty. Infrastructure aborts are listed on their own and never enter a
denominator. Figures from other papers appear only in the context section,
with the harness named, and never in a results table.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from toolsmith.evaluation.metrics import wilson_interval

from ..journal import read_journal
from .suite import CLEAN, EpisodeScore

CONTEXT_CITATIONS = (
    "ReAct (HotpotQA, that paper's harness): uninformative search results were involved in 23% of error cases.",
    "Bench2Robust (that harness): 69 of 70 model/task pairs degraded under injected tool failures, by up to "
    "46.7 points; Qwen3-4B fell from 56.1% to 15.8% on tau2-bench retail; a structured recovery context "
    "recovered 16.8 points.",
    "UndoBench (the local models they ran): naive retry produced duplicate external effects in 53.33% of "
    "lost-acknowledgment trials.",
    "ToolMaze (their composite): recovery on implicit failures averaged 37 points lower than on explicit ones; "
    "a failure-aware prompt helped the models they tested by 1.5 to 20.8 points.",
)
FAILURE_CLASSES = (
    "model: parse_error",
    "model: unknown_tool",
    "model: invalid_arguments",
    "model: repeated_call",
    "tool: explicit fault",
    "tool: implicit fault",
    "tool: lost acknowledgment",
    "budget",
    "outcome only",
)
_MODEL_ERRORS = ("unknown_tool", "invalid_arguments", "repeated_call", "parse_error")

JournalReader = Callable[[EpisodeScore], Sequence[Mapping[str, Any]]]


def _default_reader(score: EpisodeScore) -> Sequence[Mapping[str, Any]]:
    return read_journal(score.journal_path)


def summarize(scores: Sequence[EpisodeScore]) -> list[dict[str, Any]]:
    """One row per (profile, policy, prompt variant, fault variant). Aborts are counted, not scored."""
    groups: dict[tuple[str, str, str, str], list[EpisodeScore]] = defaultdict(list)
    for score in scores:
        groups[(score.profile or "", score.policy, score.prompt_variant, score.variant)].append(score)
    rows = []
    for (profile, policy, prompt_variant, variant), group in sorted(groups.items()):
        scored = [score for score in group if not score.aborted]
        successes = sum(bool(score.task_success) for score in scored)
        escalations = [score for score in scored if score.escalated]
        rows.append({
            "profile": profile,
            "policy": policy,
            "prompt_variant": prompt_variant,
            "variant": variant,
            "scored": len(scored),
            "successes": successes,
            "success_rate": successes / len(scored) if scored else None,
            "success_interval": wilson_interval(successes, len(scored)) if scored else None,
            "duplicate_episodes": sum(1 for score in scored if score.duplicate_side_effects),
            "duplicate_effects": sum(score.duplicate_side_effects for score in scored),
            "escalations": len(escalations),
            "escalation_precision": (
                sum(1 for score in escalations if not score.possible) / len(escalations) if escalations else None
            ),
            "false_escalations": sum(1 for score in escalations if score.possible),
            "recovery_match_rate": (
                sum(score.recovery_match for score in scored) / len(scored) if scored and variant != CLEAN else None
            ),
            "mean_steps": mean(score.steps for score in scored) if scored else None,
            "mean_tokens": mean(score.prompt_tokens + score.completion_tokens for score in scored) if scored else None,
            "aborted": len(group) - len(scored),
        })
    return rows


def failure_classes(score: EpisodeScore, records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Why a failed episode failed, from its journal. Model-side and tool-side classes stay apart."""
    found: set[str] = set()
    for record in records:
        event = record["event"]
        if event == "observation" and (record.get("observation") or {}).get("error_type") == "parse_error":
            found.add("model: parse_error")
        elif event == "tool_result" and not record["ok"]:
            error_type = (record.get("error") or {}).get("error_type")
            if error_type in _MODEL_ERRORS:
                found.add(f"model: {error_type}")
        elif event == "injected_fault":
            mode = record["pinned_mode"]
            if mode == "lost_acknowledgment":
                found.add("tool: lost acknowledgment")
            elif mode.startswith("explicit"):
                found.add("tool: explicit fault")
            else:
                found.add("tool: implicit fault")
    if score.status == "budget_exhausted":
        found.add("budget")
    if not found:
        found.add("outcome only")
    return [name for name in FAILURE_CLASSES if name in found]


def excerpt(records: Sequence[Mapping[str, Any]], limit: int = 8) -> list[str]:
    """A few journal lines that show what happened, without prompts or tool payloads."""
    lines = []
    for record in records:
        event = record["event"]
        if event == "model_turn":
            calls = [call["function"]["name"] for call in (record.get("assistant_message") or {}).get("tool_calls") or []]
            detail = f"calls {', '.join(calls)}" if calls else f"{record['kind']}"
            if record.get("error_code"):
                detail += f" ({record['error_code']})"
        elif event == "tool_result":
            outcome = "ok" if record["ok"] else (record.get("error") or {}).get("error_type", "error")
            detail = f"{record['tool']} -> {outcome}"
        elif event == "injected_fault":
            detail = f"{record['tool']} call {record['call_index']}: {record['mode']} (committed={record['committed']})"
        elif event == "policy_attempt":
            outcome = "ok" if record["ok"] else record["error_type"]
            detail = f"{record['purpose']} via {record['tool']} -> {outcome}"
        elif event in ("guard_tripped", "sanity_checked", "escalated", "episode_finished"):
            detail = {
                "guard_tripped": lambda r: f"{r['action']} {r['tool']} (count {r['count']})",
                "sanity_checked": lambda r: f"{r['tool']} passed={r['passed']}",
                "escalated": lambda r: r["reason"],
                "episode_finished": lambda r: f"status {r['status']}",
            }[event](record)
        else:
            continue
        lines.append(f"seq {record['seq']:>3} {event}: {detail}")
    return lines[-limit:] if len(lines) > limit else lines


def render_report(
    scores: Sequence[EpisodeScore],
    *,
    sweeps: Sequence[Mapping[str, Any]] = (),
    smoke: Sequence[Mapping[str, Any]] = (),
    external: Sequence[Mapping[str, Any]] = (),
    journal_reader: JournalReader = _default_reader,
    title: str = "Steadyhand results",
) -> str:
    rows = summarize(scores)
    out: list[str] = [f"# {title}", ""]
    out += ["Generated from episode journals by `steadyhand.eval.report`. Every number below traces to a "
            "journal, a sweep manifest, a task-id list, and a seed.", ""]

    out += ["## 1. Setup", ""]
    if sweeps:
        for sweep in sweeps:
            server = sweep.get("server") or {}
            model = sweep.get("model") or {}
            out.append(
                f"- Sweep `{sweep.get('sweep_id')}`: profile `{sweep.get('profile')}`, "
                f"grammars={model.get('use_grammars')}, thinking={model.get('enable_thinking')}, "
                f"temperature={model.get('temperature')}, tasks={len(sweep.get('task_ids', []))} "
                f"(dataset `{str(sweep.get('dataset_sha256', ''))[:12]}`), "
                f"policies={', '.join(sweep.get('policies', []))}, prompts={', '.join(sweep.get('prompt_variants', []))}, "
                f"server: {server.get('gpu', '')} -c {server.get('context_size', '')} "
                f"--parallel {server.get('parallel', '')} (per slot {server.get('per_slot_context', '')}), "
                f"KV {server.get('kv_cache_type', '')}."
            )
    else:
        out.append("- No sweep manifest was supplied.")
    out.append("")

    if smoke:
        out += ["### Run description (not a score)", "",
                "| GPU | Preset | Single-stream tok/s | Aggregate tok/s | Slots busy | Peak memory (MiB) |",
                "|---|---|---|---|---|---|"]
        for line in smoke:
            out.append(
                f"| {line.get('gpu', '')} | {line.get('preset', '')} | {_number(line.get('single_stream_tps'))} | "
                f"{_number(line.get('aggregate_tps'))} | {line.get('slots_busy', '')} | {line.get('peak_memory_mib', '')} |"
            )
        out.append("")

    out += ["## 2. Results", "",
            "Success uses the final store against the gold diff. Aborted episodes are excluded from every "
            "denominator and listed below.", "",
            "| Profile | Policy | Prompt | Variant | Scored | Success (95% CI) | Duplicate-effect episodes | "
            "Escalation precision | False escalations | Recovery match | Mean steps | Mean tokens | Aborted |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        success = ""
        if row["success_rate"] is not None:
            low, high = row["success_interval"]
            success = f"{row['successes']}/{row['scored']} ({row['success_rate']:.1%}, {low:.1%}-{high:.1%})"
        duplicates = f"{row['duplicate_episodes']}/{row['scored']}" if row["scored"] else ""
        out.append(
            f"| {row['profile']} | {row['policy']} | {row['prompt_variant']} | {row['variant']} | {row['scored']} | "
            f"{success} | {duplicates} | {_percent(row['escalation_precision'])} | "
            f"{row['false_escalations'] if row['escalations'] else ''} | {_percent(row['recovery_match_rate'])} | "
            f"{_number(row['mean_steps'])} | {_number(row['mean_tokens'])} | {row['aborted'] or ''} |"
        )
    if not rows:
        out.append("| | | | | | | | | | | | | |")
    out.append("")

    escalated = [score for score in scores if score.escalated and not score.aborted]
    out += ["### Escalations", ""]
    out += [
        f"- `{score.episode_id}` ({score.task_id}, {score.variant}, {score.policy}): "
        f"{'task was not possible' if not score.possible else 'false escalation on a possible task'}"
        for score in escalated
    ] or ["- None."]
    out.append("")

    aborted = [score for score in scores if score.aborted]
    out += ["### Infrastructure aborts", ""]
    out += [
        f"- `{score.episode_id}` ({score.task_id}, {score.variant}, {score.policy}"
        f"{', ' + score.error_code if score.error_code else ''})"
        for score in aborted
    ] or ["- None."]
    retried = [score for score in scores if score.attempt > 1]
    if retried:
        out += ["", f"{len(retried)} episode(s) were rerun after an infrastructure abort; each row above and in the "
                "tables is the latest attempt. Earlier attempts stay in the scores file and on Drive."]
    out.append("")

    out += ["## 3. Failure taxonomy", ""]
    failed = [score for score in scores if not score.aborted and score.task_success is False]
    by_class: dict[str, list[tuple[EpisodeScore, Sequence[Mapping[str, Any]]]]] = defaultdict(list)
    for score in failed:
        records = journal_reader(score)
        for name in failure_classes(score, records):
            by_class[name].append((score, records))
    if not failed:
        out += ["No scored episode failed.", ""]
    for name in FAILURE_CLASSES:
        if name not in by_class:
            continue
        examples = by_class[name]
        score, records = examples[0]
        out += [f"### {name} ({len(examples)} episodes)", "",
                f"Example `{score.episode_id}`: {score.task_id} / {score.variant} / {score.policy}. "
                f"Problems: {'; '.join(score.problems) or 'none recorded'}.", "", "```text"]
        out += excerpt(records)
        out += ["```", ""]

    out += ["## 4. Which policy addressed which failure", ""]
    policies = sorted({score.policy for score in scores})
    variants = sorted({score.variant for score in scores}, key=lambda v: (v != CLEAN, v))
    if policies:
        out += ["Successes per scored episode, by fault variant. Lower numbers are results too.", "",
                "| Variant | " + " | ".join(policies) + " |", "|---|" + "---|" * len(policies)]
        for variant in variants:
            cells = []
            for policy in policies:
                group = [s for s in scores if s.variant == variant and s.policy == policy and not s.aborted]
                cells.append(f"{sum(bool(s.task_success) for s in group)}/{len(group)}" if group else "")
            out.append(f"| {variant} | " + " | ".join(cells) + " |")
        out.append("")
        lost = [s for s in scores if s.variant == "lost_acknowledgment" and not s.aborted]
        if lost:
            out += ["Duplicate side effects on lost-acknowledgment tasks (the UndoBench-shaped slice on this store):", ""]
            for policy in policies:
                group = [s for s in lost if s.policy == policy]
                if group:
                    out.append(f"- {policy}: {sum(s.duplicate_side_effects for s in group)} extra effects over "
                               f"{len(group)} episodes")
            out.append("")
    else:
        out += ["No episodes yet.", ""]

    out += ["## 5. External benchmarks", ""]
    if external:
        for run in external:
            subset = run.get("subset") or {}
            out.append(
                f"- {run.get('name')} [{subset.get('label', 'partial')}]: {subset.get('count', 0)} ids "
                f"(sha256 `{str(subset.get('sha256', ''))[:12]}`), eligible {run.get('eligible', '')}, "
                f"ineligible {run.get('ineligible', '')}, success {run.get('successes', '')}/{run.get('scored', '')}."
            )
    else:
        out.append("- No external benchmark has been run.")
    out.append("")

    out += ["## 6. Context from other harnesses", "",
            "These figures come from other papers and harnesses. They explain why the loop matters. They are not "
            "Steadyhand measurements and do not appear in the tables above.", ""]
    out += [f"- {citation}" for citation in CONTEXT_CITATIONS]
    out.append("")
    return "\n".join(out)


def results_section(markdown: str) -> str:
    """The text between the results heading and the context section (for checks)."""
    start = markdown.index("## 2. Results")
    end = markdown.index("## 6. Context")
    return markdown[start:end]


def write_report(path: str | Path, markdown: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")
    return destination


def _percent(value: float | None) -> str:
    return "" if value is None else f"{value:.1%}"


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


__all__ = [
    "CONTEXT_CITATIONS",
    "FAILURE_CLASSES",
    "excerpt",
    "failure_classes",
    "render_report",
    "results_section",
    "summarize",
    "write_report",
]

"""Bulkhead parts 7 and 8: ``results/bulkhead/RESULTS.md``, generated only from recorded runs.

Inputs are run directories written by ``attacks``, ``overhead``,
``usability``, and the AgentDojo ``runner``. Every number in the report is
read from their manifests, results, samples, records, and summaries; nothing
is typed in here. Profiles a run could not build, and cells it did not run,
are shown as untested with the recorded reason. Several attack runs (say, one
on the bare host and one inside a delegated cgroup scope) are merged: for each
(attack, profile) the latest record in the given order wins. An AgentDojo run
made with the scripted model on the local fixture (``kind: cpu_gate``) is
labeled as a harness check, never as a benchmark result.

Sections follow the part 8 write-up: threat model and architecture,
environment coverage, escape results, AgentDojo paired results, usability and
overhead, failure taxonomy, replay evidence, limitations, context references,
and handoff.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .attacks import PROFILE_TIERS, ROOT, read_records

RESULTS_ROOT = ROOT / "results" / "bulkhead"
VERDICT_COLUMNS = ("blocked", "different_boundary", "escaped", "setup_failure", "infrastructure_failure", "not_applicable", "untested")
ABBREVIATION = {
    "blocked": "B", "different_boundary": "DB", "escaped": "**ESC**", "setup_failure": "SF", "infrastructure_failure": "IF",
    "not_applicable": "n/a", "passed": "P", "untested": "—",
}
CONTEXT_NOTE = (
    "Related work named in the project brief is context only: Progent (privilege control with a policy language), "
    "CaMeL (capability-tracking interpreter), AgentDojo (the benchmark adapted in section 4; its paper's numbers come "
    "from its own pipelines and models), Anthropic's sandbox-runtime and the Codex Linux sandbox (process sandboxes for "
    "coding agents), the Linux kernel documentation for Landlock, seccomp, and cgroup v2, the \"lethal trifecta\" framing "
    "of private data, untrusted content, and outside communication, GrantBox and MCP Security Bench (permission and MCP "
    "attack suites), and SandboxEscapeBench (container-escape tasks). Any figure they report belongs to their own "
    "harness and setup. None is reproduced here, and none is a Bulkhead measurement."
)
ARCHITECTURE = (
    "Bulkhead sits between Steadyhand's validated tool calls and anything that runs. Assets: the host, the user's files "
    "and secrets, the mock store or benchmark state, and outside services. The attacker controls tool output (files, "
    "web pages, messages, tool results) and the programs a model asks to run; the model itself may follow injected "
    "instructions. Each call passes validate -> repeated-call guard -> policy decision (ALLOW, DENY, or ASK, from the "
    "task's monotonic policy) -> fault injector -> handler or sandbox backend. Taint makes an episode untrusted the "
    "first time it reads output not declared trusted, and never lowers it; afterwards a call to a taint-sensitive sink "
    "asks. ASK escalates under `deny_all`; an approval is a single-use grant for one exact call, retried as a linked "
    "episode. Exec, filesystem, and network tools run only in a sandbox: tier 2 is bubblewrap namespaces, seccomp, "
    "no_new_privs, rlimits, and a delegated cgroup leaf; tier 1 is Landlock with the same seccomp, rlimit, and cgroup "
    "layers; T0 is not a sandbox and runs nothing. Network leaves only through a per-call outside CONNECT proxy with "
    "exact host:port rules and byte budgets; output is bounded, sanitized, redacted, and wrapped in untrusted-output "
    "delimiters before the model or the journal sees it. Every decision, approval, taint change, sandbox run, egress "
    "request, and gate result is a journal event. See `Bulkhead/THREAT_MODEL.md`."
)
REPLAY_NOTE = (
    "Control-plane replay re-runs an episode from its journal with the model served from the recording: validation, "
    "the guard, every policy decision, approval request, taint transition, recovery step, observation, and message "
    "must reproduce exactly, and mock-store or AgentDojo tools re-run only on fresh in-memory copies. Sandboxed calls "
    "are served from their recorded `sandbox_execution` events, so replay spawns no process and opens no socket. "
    "Replay therefore proves the decisions and messages, not kernel enforcement: only live attack reruns under the "
    "real backend (section 3) show that the sandbox holds."
)
HANDOFF = (
    "Fields later stages can consume without trusting model prose: `policy_decision` (verdict, rule_id, reason_code, "
    "taint_before, policy_sha256, normalized_request_sha256, grant_id), `approval_requested`, `taint_transition` "
    "(the source tool, call id, and journal seq that made the episode untrusted), `tool_result.handler_called` (the "
    "evidence that anything ran), `sandbox_execution`, `egress_decision` (authority and byte counts only), and "
    "`output_gate` (what sanitizing removed), plus each episode manifest's security_config and security_spec hashes. "
    "Context and memory (Day 4) can use the taint label and its source seq to keep untrusted observations out of "
    "anything summarized as trusted. Observability and guardrails (Day 9) can count ASK, DENY, and taint events per "
    "tool and rule from the journal, and the AgentDojo records' counts and failure classes, instead of parsing answers."
)
LIMITATIONS = (
    "Output redaction matches declared secret values literally; an encoded secret (base64, split, reversed) is not redacted. "
    "The defense is that no secret is readable inside the sandbox; redaction is a last line.",
    "Task-policy bounds are per call: several calls inside a bound can add up to more than the bound. Cumulative limits are not modeled.",
    "Taint only protects sinks the task marks sensitive; an unmarked store write after an untrusted read is allowed (see the usability negative control).",
    "Tier 1 hides content, not names: stat succeeds on undeclared paths, and a directory holding a denied entry cannot be listed.",
    "Tier-2 expectations (bubblewrap) are only verified on hosts where the probe proves bubblewrap; elsewhere those cells are untested, and the network cases were exercised only through a test network namespace that is not a sandbox.",
    "Overhead numbers describe one host and its filesystem. Per-call cold start is dominated by interpreter start-up for the pre-exec helper and launcher (compare the supervised time with the sandbox wall time), and the helper imports Bulkhead from the repository, so a slow filesystem (such as a 9P-mounted Windows drive under WSL2) shows up directly in it.",
    "Stopping a call's egress proxy waits for its accept loop's poll interval, which dominates the measured proxy start/stop cost.",
    "Kernel zero-days, a compromised host kernel, and host root are out of scope: no measured result here is an absolute proof of security.",
    "Taint is per episode, not per value: one untrusted read makes every later sensitive call ask, including calls whose arguments came from trusted data. That costs benign utility (sections 4 and 5).",
    "AgentDojo tools are simulated in memory; the AgentDojo ablation measures the policy and taint layers, not the OS sandbox. The sandbox's strength comes from the attack suite in section 3.",
    "AgentDojo tools whose schemas Toolsmith's grammar cannot constrain are withheld, and the cases that need them are ineligible (listed with their JSON Pointer reason); a partial run covers only its listed case ids and is not a full benchmark score.",
    "AgentDojo output trust is declared per tool from the benchmark's field layout and checked with canaries over every ground-truth call; a tool no ground truth calls stays declared but unverified.",
    "Static task policies cannot tell an attacker's transfer from a user's with the same tool and arguments shape; only taint (or a person) separates them.",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_runs(kind: str, root: Path = RESULTS_ROOT) -> list[Path]:
    """Run directories of one kind, oldest first by manifest modification time."""
    base = root / kind
    if not base.is_dir():
        return []
    runs = [path for path in base.iterdir() if (path / "manifest.json").is_file()]
    return sorted(runs, key=lambda path: (path / "manifest.json").stat().st_mtime)


def _ms(value: Any) -> str:
    return "—" if value is None else f"{value * 1000:.1f}"


def _rate(entry: Mapping[str, Any] | None) -> str:
    if not entry or entry.get("total") in (None, 0):
        return "—"
    low, high = entry["wilson95"]
    return f"{entry['count']}/{entry['total']} = {entry['rate']:.2f} [{low:.2f}, {high:.2f}]"


def merge_attack_runs(runs: Sequence[Path]) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    """The latest record per (attack, profile) across ``runs`` (in order), and each run's manifest."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    manifests = []
    for run in runs:
        manifest = _load(run / "manifest.json")
        manifest["_directory"] = str(run)
        manifests.append(manifest)
        for record in read_records(run / "results.jsonl"):
            if record.get("event") in ("attack_result", "attack_untested"):
                record["_run"] = run.name
                key = (record["attack_id"], record["profile"])
                # A real result never gets replaced by a later run's "untested" for the same cell.
                if record["event"] == "attack_untested" and latest.get(key, {}).get("event") == "attack_result":
                    continue
                latest[key] = record
    return latest, manifests


def _label(record: Mapping[str, Any] | None) -> str:
    if record is None or record.get("event") == "attack_untested":
        return "untested"
    return record["verdict"]


def attack_section(runs: Sequence[Path]) -> list[str]:
    lines = ["## 3. Escape results", "", "### Attack results per profile", ""]
    if not runs:
        return lines + ["No attack runs were found: every cell is untested.", ""]
    latest, manifests = merge_attack_runs(runs)
    datasets = {m["identity"]["dataset"]["sha256"] for m in manifests}
    attack_ids = sorted({key[0] for key in latest})
    kinds = {record["attack_id"]: record["kind"] for record in latest.values()}
    scored = [a for a in attack_ids if kinds.get(a) not in ("control", "harness_control")]
    controls = [a for a in attack_ids if kinds.get(a) in ("control", "harness_control")]
    profiles = list(PROFILE_TIERS)
    if len(datasets) > 1:
        lines += [f"Warning: these runs used {len(datasets)} different dataset hashes; cells may come from different dataset versions.", ""]
    lines += [f"{len(scored)} attacks and {len(controls)} positive controls. Counts are attacks only (controls are listed separately).", ""]
    lines += ["| Profile | Tier | " + " | ".join(VERDICT_COLUMNS) + " |", "|---|---|" + "---|" * len(VERDICT_COLUMNS)]
    for profile in profiles:
        counts = {column: 0 for column in VERDICT_COLUMNS}
        for attack_id in scored:
            label = _label(latest.get((attack_id, profile)))
            counts[label] = counts.get(label, 0) + 1
        lines.append(f"| {profile} | {PROFILE_TIERS[profile]} | " + " | ".join(str(counts[c]) for c in VERDICT_COLUMNS) + " |")
    lines += ["", "Positive controls:", "", "| Control | " + " | ".join(profiles) + " |", "|---|" + "---|" * len(profiles)]
    for attack_id in controls:
        lines.append(f"| {attack_id} | " + " | ".join(ABBREVIATION.get(_label(latest.get((attack_id, p))), "?") for p in profiles) + " |")
    lines += ["", "Full matrix (B blocked, DB different boundary, **ESC** escaped, SF setup failure, IF infrastructure failure, n/a not applicable, — untested):", "",
              "| Attack | " + " | ".join(profiles) + " |", "|---|" + "---|" * len(profiles)]
    for attack_id in scored:
        lines.append(f"| {attack_id} | " + " | ".join(ABBREVIATION.get(_label(latest.get((attack_id, p))), "?") for p in profiles) + " |")
    lines.append("")
    return lines


def untested_section(runs: Sequence[Path]) -> list[str]:
    latest, _ = merge_attack_runs(runs)
    reasons: dict[str, dict[str, int]] = {}
    for (attack_id, profile), record in latest.items():
        if record.get("event") == "attack_untested":
            reasons.setdefault(profile, {}).setdefault(record["reason"], 0)
            reasons[profile][record["reason"]] += 1
    lines = ["### Untested cells", ""]
    if not reasons:
        return lines + ["Every cell of every profile in these runs was run or declared not applicable.", ""]
    for profile in PROFILE_TIERS:
        for reason, count in sorted(reasons.get(profile, {}).items()):
            lines.append(f"- **{profile}**: {count} cell(s) (attacks and controls) untested: {reason}")
    return lines + [""]


def environment_section(attack_runs: Sequence[Path], overhead_runs: Sequence[Path], usability_runs: Sequence[Path], agentdojo_runs: Sequence[Path] = ()) -> list[str]:
    lines = ["## 2. Environment coverage", ""]
    for run in attack_runs:
        m = _load(run / "manifest.json")
        host = m["host"]
        lines += [
            f"**Attack run `{m['run_id']}`**: {host['os']} {host['kernel']} ({host['architecture']}), {host['cpu_count']} CPUs, Python {host['python']}, load {host.get('loadavg')}.",
            f"Dataset `{m['identity']['dataset']['name']}` v{m['identity']['dataset']['version']} sha256 `{m['identity']['dataset']['sha256'][:16]}…` "
            f"({m['dataset']['attacks']} attacks, {m['dataset']['controls']} controls, {len(m['dataset']['scripts'])} hashed scripts); "
            f"helper `{m['identity']['helper_version']}`; probe `{(m['identity'].get('probe_sha256') or 'none')[:16]}`; "
            f"code `{(m['code'].get('git_head') or 'unknown')[:12]}`{' (+ uncommitted src changes)' if m['code'].get('git_dirty_src') else ''}, sources `{m['code']['sources_sha256'][:16]}…`.",
            f"Options: {json.dumps(m.get('options', {}), sort_keys=True)}.",
            "",
            "| Profile | Available | Backend | Config sha256 | Reason |",
            "|---|---|---|---|---|",
        ]
        for profile, target in m["targets"].items():
            backend = target.get("backend") or {}
            lines.append(f"| {profile} | {'yes' if target['available'] else 'no'} | {backend.get('name', '—')} {backend.get('version', '')} | `{target['config_sha256'][:12]}` | {target.get('reason') or ''} |")
        lines.append("")
    for run in overhead_runs:
        m = _load(run / "manifest.json")
        lines.append(f"**Overhead run `{m['run_id']}`**: {m['host']['kernel']}, {m['host']['cpu_count']} CPUs, load before {m['host'].get('loadavg')} / after {m.get('host_after', {}).get('loadavg')}, "
                     f"{m['concurrency']['mode']}, {m['samples']} measured samples per cell after {m['warmup']} warm-up calls, {m['episode_repeats']} episode repeats.")
    for run in usability_runs:
        m = _load(run / "manifest.json")
        lines.append(f"**Usability run `{m['run_id']}`**: tasks sha256 `{m['dataset']['sha256'][:16]}…` ({m['dataset']['tasks']} tasks), modes {', '.join(m['modes'])}, approval clock {m['approval_clock']}.")
    for run in agentdojo_runs:
        m = _load(run / "manifest.json")
        pin = m["agentdojo"]
        lines.append(f"**AgentDojo run `{m['run_id']}`** ({m['kind']}, label `{m['label']}`): {pin.get('package')} {pin.get('version', '')} "
                     f"commit `{(pin.get('commit') or 'unrecorded')[:12]}`, benchmark {pin.get('benchmark_version')}, attack `{pin.get('attack')}`; "
                     f"model {m['model'].get('model')} (grammars {m['model'].get('use_grammars')}, thinking {m['model'].get('enable_thinking')}, "
                     f"temperature {m['model'].get('temperature')}, seed {m['model'].get('seed')}); server {json.dumps(m.get('server', {}), sort_keys=True)}; "
                     f"code `{(m['code'].get('git_head') or 'unknown')[:12]}`.")
    if not (attack_runs or overhead_runs or usability_runs or agentdojo_runs):
        lines.append("No runs were found.")
    return lines + [""] + untested_section(attack_runs)


def escapes_section(runs: Sequence[Path]) -> list[str]:
    lines = ["### Escapes and failures, with first evidence", ""]
    latest, _ = merge_attack_runs(runs)
    failing = [r for r in latest.values() if r.get("verdict") in ("escaped", "setup_failure", "infrastructure_failure")]
    if not failing:
        return lines + ["No escape, setup failure, or infrastructure failure is recorded in these runs.", ""]
    order = {"escaped": 0, "setup_failure": 1, "infrastructure_failure": 2}
    for record in sorted(failing, key=lambda r: (order[r["verdict"]], r["attack_id"], r["profile"])):
        lines.append(f"### {record['verdict']}: `{record['attack_id']}` on {record['profile']} (run `{record['_run']}`, key `{record['key'][:12]}`)")
        lines.append("")
        for reason in record.get("reasons", [])[:6]:
            lines.append(f"- {reason}")
        evidence = record.get("evidence", {})
        excerpt = {k: evidence[k] for k in ("final", "decisions", "egress", "connections", "survivors", "store_diff", "audit") if k in evidence}
        if excerpt:
            text = json.dumps(excerpt, sort_keys=True)
            lines += ["", "```json", text[:1500] + ("…" if len(text) > 1500 else ""), "```"]
        if record.get("evidence_file"):
            lines.append(f"Full bounded evidence: `{record['_run']}/evidence/{record['evidence_file']}`.")
        lines.append("")
    return lines


def overhead_section(runs: Sequence[Path]) -> list[str]:
    lines = ["### Overhead", ""]
    if not runs:
        return lines + ["No overhead runs were found: overhead is untested.", ""]
    for run in runs:
        summary = _load(run / "summary.json")
        manifest = _load(run / "manifest.json")
        lines += [f"Run `{manifest['run_id']}` (raw samples: `{run.name}/samples.jsonl`). Times in milliseconds; percentiles are nearest-rank over every measured sample; failures are excluded from them and counted.", "",
                  "| Profile | Workload | Measured | Failures | First call | Median | p95 | p99 | Max | Prepare (median) | Cleanup (median) | Peak memory, KiB (median) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for profile, workloads in summary["cold_start"].items():
            for workload, s in workloads.items():
                lines.append(f"| {profile} | {workload} | {s['measured']} | {s['failures']} | {_ms(s['first_total_s'])} | {_ms(s['total_s']['median'])} | {_ms(s['total_s']['p95'])} | "
                             f"{_ms(s['total_s']['p99'])} | {_ms(s['total_s']['max'])} | {_ms(s['prepare_s']['median'])} | {_ms(s['cleanup_s']['median'])} | "
                             f"{s.get('usage_peak_memory_kb', {}).get('median', '—')} |")
        lines += ["", "Peak memory is per call, as each profile measures it: a cgroup profile reports the leaf's `memory.peak` (memory charged after the call entered its leaf); "
                  "a profile without a cgroup reports the leader's maximum RSS from `wait4`, which includes the pre-exec helper's interpreter before it execs. "
                  "T0 and the proxy rows have none."]
        tier2 = [s for p, w in summary["cold_start"].items() if PROFILE_TIERS.get(p) == "T2" for s in w.values()]
        target = manifest.get("target_s")
        if target is None:
            pass
        elif tier2:
            worst = max(s["total_s"]["median"] or 0 for s in tier2)
            lines += ["", f"The brief's example target (tier 2 under {target * 1000:.0f} ms per call) was {'met' if worst < target else 'not met'}: worst tier-2 median {_ms(worst)} ms."]
        else:
            lines += ["", f"The brief's example target (tier 2 under {target * 1000:.0f} ms per call) is untested: no tier-2 profile was measured in this run."]
        lines += ["", "Whole episodes (scripted model; wall time per episode, and the sandboxed calls' parts summed per episode):", "",
                  "| Episode | Mode | Calls (sandboxed) | Repeats | Median wall | p95 wall | Sandbox wall | Supervised | Backend overhead | Output gate | Proxy | Workspace prepare + cleanup |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in summary["episodes"]:
            def median(key: str) -> str:
                return _ms(e[key]["median"]) if key in e else "—"
            prep = (e["sandbox_prepare_s"]["median"] or 0) + (e["sandbox_workspace_cleanup_s"]["median"] or 0) if "sandbox_prepare_s" in e else None
            lines.append(f"| {e['episode']} | {e['profile'] or e['mode']} | {e['calls']} ({e.get('sandboxed_calls', 0)}) | {e['repeats']} | {_ms(e['wall_s']['median'])} | {_ms(e['wall_s']['p95'])} | "
                         f"{median('sandbox_wall_s')} | {median('sandbox_supervised_s')} | {median('sandbox_backend_overhead_s')} | {median('sandbox_gate_s')} | {median('sandbox_proxy_s')} | {_ms(prep)} |")
        lines += ["", "The exec episode has no policy-only row: the policy never lets an exec tool run without a sandbox tier."]
        for item in summary.get("untested", []):
            lines.append(f"- Untested: **{item['profile']}**: {item['reason']}")
        resource = summary.get("resource", {})
        if resource:
            lines.append(f"- Supervisor peak RSS {resource['supervisor_peak_rss_kb'] / 1024:.1f} MiB; largest child peak RSS {resource['children_peak_rss_kb'] / 1024:.1f} MiB (separate from any tool's memory limit).")
        lines.append("")
    return lines


def usability_section(runs: Sequence[Path]) -> list[str]:
    lines = ["### Approval fatigue, over-denial, and utility (scripted security tasks)", ""]
    if not runs:
        return lines + ["No usability runs were found: usability is untested.", ""]
    for run in runs:
        summary = _load(run / "summary.json")
        manifest = _load(run / "manifest.json")
        lines += [f"Run `{manifest['run_id']}` ({manifest['dataset']['tasks']} tasks). Rates are count/total = rate [Wilson 95%].", "",
                  "| Mode | Approval mode | ASK rate per task | ASKs per tool call | Approvals (requested / approved / declined) | Benign over-denial | Unnecessary ASK | Benign utility | Utility loss vs allow_no_taint | Malicious attack success | Duplicate effects | Refused calls that ran |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        modes = {"allow_no_taint": "deny_all (no taint ASKs)", "taint_deny_all": "deny_all", "taint_scripted": "scripted, hashed one-shot approvals"}
        for mode, s in summary.items():
            loss = s["utility_loss_vs_allow_no_taint"]
            loss_text = "—" if loss["mean"] is None else f"{loss['mean']:.2f} [{loss['paired_bootstrap95'][0]:.2f}, {loss['paired_bootstrap95'][1]:.2f}] ({loss['pairs']} pairs)"
            asks = s["asks_per_tool_call"]
            lines.append(f"| {mode} | {modes.get(mode, mode)} | {_rate(s['ask_rate_per_task'])} | {asks['asks']}/{asks['tool_calls']} | "
                         f"{s['approvals']['requested']} / {s['approvals']['approved']} / {s['approvals']['declined']} | {_rate(s['over_denial'])} | {_rate(s['unnecessary_ask'])} | "
                         f"{_rate(s['utility'])} | {loss_text} | {_rate(s['attack_success'])} | {s['duplicate_effects']} | {s['refused_still_ran']} |")
        scores = [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines() if line.strip()]
        successes = sorted({(s["task_id"], s["mode"]) for s in scores if s.get("attack_success")})
        if successes:
            lines += ["", "Malicious tasks whose forbidden effect happened: " + ", ".join(f"`{task}` ({mode})" for task, mode in successes) + "."]
        lines.append("")
    return lines


def negatives_section(attack_runs: Sequence[Path], usability_runs: Sequence[Path], overhead_runs: Sequence[Path], agentdojo_runs: Sequence[Path] = ()) -> list[str]:
    lines = ["## 8. Limitations and negative results", ""]
    latest, manifests = merge_attack_runs(attack_runs)
    escaped = sorted({f"`{a}` on {p}" for (a, p), r in latest.items() if r.get("verdict") == "escaped"})
    lines.append(f"- Escapes: {', '.join(escaped) if escaped else 'none recorded'}.")
    available = sorted({p for m in manifests for p, t in m["targets"].items() if t["available"]})
    unavailable = sorted({f"{p} ({t.get('reason')})" for m in manifests for p, t in m["targets"].items() if not t["available"] and p not in available})
    if unavailable:
        lines.append(f"- Profiles not built on these hosts: {'; '.join(unavailable)}." + (f" Built: {', '.join(available)}." if available else ""))
    for run in usability_runs:
        scores = [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines() if line.strip()]
        always = sorted({s["task_id"] for s in scores if s.get("attack_success")} - {s["task_id"] for s in scores if s.get("attack_success") is False})
        if always:
            lines.append(f"- Malicious usability tasks that succeeded in every mode: {', '.join(f'`{t}`' for t in always)}.")
    for run in agentdojo_runs:
        manifest = _load(run / "manifest.json")
        if manifest["kind"] != "live":
            continue
        summary = _agentdojo_summary(run)
        for configuration in manifest["configurations"]:
            asr = summary["suites"]["all"][configuration]["attack_success"]
            if asr["count"]:
                lines.append(f"- AgentDojo run `{manifest['run_id']}`: {asr['count']} of {asr['total']} attacked cases reached the attacker's objective under {configuration}.")
    if not any(_load(run / "manifest.json")["kind"] == "live" for run in agentdojo_runs):
        lines.append("- No live AgentDojo run is recorded: the three configurations are untested on a real model, and no AgentDojo score is claimed.")
    for limitation in LIMITATIONS:
        lines.append(f"- {limitation}")
    return lines + ["", "## 9. Context references (not Bulkhead measurements)", "", CONTEXT_NOTE, "", "## 10. Handoff", "", HANDOFF, ""]


def _agentdojo_summary(run: Path) -> dict[str, Any]:
    """The run's summary, recomputed from its records (``summary.json`` is a cache of the same function)."""
    from .runner import compute_summary

    return compute_summary(run, _load(run / "manifest.json"))


def _diff(entry: Mapping[str, Any]) -> str:
    if entry["mean"] is None:
        return "— (0 pairs)"
    low, high = entry["paired_bootstrap95"]
    return f"{entry['mean']:+.2f} [{low:+.2f}, {high:+.2f}] ({entry['pairs']} pairs; {entry['changed']['up']} up, {entry['changed']['down']} down)"


TRAJECTORY_FIELDS = ("event", "step", "kind", "tool", "verdict", "rule_id", "taint_before", "ok", "handler_called", "status", "error_code")


def trajectory(run: Path, record: Mapping[str, Any], limit: int = 24) -> list[str]:
    """Bounded journal events of a case's last episode: names, verdicts, and rules only; no arguments, outputs, or model text."""
    from steadyhand.journal import read_journal

    journal = run / record["journals"][-1]
    if not journal.exists():
        return [f"(journal `{record['journals'][-1]}` is not in this copy of the run)"]
    lines = []
    for event in read_journal(journal):
        if event["event"] in ("budget_checked", "episode_started"):
            continue
        fields = {key: event[key] for key in TRAJECTORY_FIELDS if key in event and event[key] is not None}
        if event["event"] == "model_turn":
            fields["calls"] = [call["function"]["name"] for call in (event.get("assistant_message") or {}).get("tool_calls") or []]
        if event["event"] == "tool_result" and not event.get("ok"):
            fields["error_type"] = (event.get("error") or {}).get("error_type")
        if event["event"] == "taint_transition":
            fields.update(source_tool=event.get("source_tool"), after=event.get("after"))
        lines.append(json.dumps(fields, sort_keys=True))
    return lines[:limit] + ([f"… {len(lines) - limit} more events"] if len(lines) > limit else [])


RESIDUALS = (
    ("An attack succeeded under a secured configuration (the policy allowed the attacker's action)",
     lambda r: r["configuration"] != "baseline" and r.get("attack_success")),
    ("An ASK cost benign utility", lambda r: r["injection_task"] is None and not r.get("utility") and r["counts"]["asks"]),
    ("Model output was truncated", lambda r: r["counts"]["output_truncations"]),
    ("An attack succeeded without Bulkhead (baseline)", lambda r: r["configuration"] == "baseline" and r.get("attack_success")),
)


def agentdojo_section(runs: Sequence[Path]) -> list[str]:
    from .metrics import latest

    lines = ["## 4. AgentDojo paired results", ""]
    if not runs:
        return lines + ["No AgentDojo runs were found: the paired configurations are untested and no AgentDojo result is claimed.", ""]
    for run in runs:
        manifest = _load(run / "manifest.json")
        summary = _agentdojo_summary(run)
        scope = summary["scope"]
        eligibility = _load(run / "eligibility.json") if (run / "eligibility.json").exists() else {"tools": {}, "ineligible_cases": []}
        lines.append(f"### Run `{manifest['run_id']}` ({manifest['kind']}, label `{manifest['label']}`)")
        lines.append("")
        if manifest["kind"] == "cpu_gate":
            lines += ["**CPU correctness gate:** a scripted model on the local fixture. These rows check the harness, the pairing, and the metrics; "
                      "they are not AgentDojo benchmark results and say nothing about a real model.", ""]
        label = "partial run" if scope["partial"] else "every eligible case"
        lines += [
            f"Scope: {scope['case_ids']} case ids (sha256 `{scope['case_ids_sha256'][:16]}…`), {label}: {scope['eligible_cases']} eligible and "
            f"{scope['ineligible_cases']} ineligible cases in the planned suites; benchmark {scope['benchmark_version']}; approval mode "
            f"`{manifest['approval_mode']}` for the secured configurations (the baseline has none). Same case ids, injections, tools, model settings, "
            f"and (empty) fault schedule in every configuration; execution order rotates per case.",
            "",
            "Rates are count/total = rate [Wilson 95%]. Utility under attack and attack success are separate: one case can count in both.",
            "",
        ]
        for scope_name, configurations in summary["suites"].items():
            lines += [f"**{scope_name}**", "", "| Configuration | Clean utility | Utility under attack | Attack success | User failed and attack succeeded | ASK rate | ASKs / tool call | Over-denial | Refused calls (ran) | Effect mismatches | Approvals (req / ok / no) | Not scored |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for configuration, s in configurations.items():
                asks = s["asks_per_tool_call"]
                refusals = s["refusals"]
                approvals = s["approvals"]
                lines.append(f"| {configuration} | {_rate(s['utility_clean'])} | {_rate(s['utility_attack'])} | {_rate(s['attack_success'])} | {s['user_failed_and_attack_succeeded']} | "
                             f"{_rate(s['ask_rate'])} | {asks['asks']}/{asks['tool_calls']} | {_rate(s['over_denial'])} | {refusals['refused_calls']} ({refusals['refused_still_ran']}) | "
                             f"{refusals['effect_mismatches']} | {approvals['requested']} / {approvals['approved']} / {approvals['declined']} | {s['not_scored']} |")
            lines.append("")
        lines += ["Paired differences (later configuration minus earlier, over case ids scored in both; paired bootstrap 95%, seed recorded):", "",
                  "| Comparison | Clean utility | Utility under attack | Attack success |", "|---|---|---|---|"]
        for comparison, metrics in summary["paired"]["all"].items():
            lines.append(f"| {comparison} | {_diff(metrics['utility_clean'])} | {_diff(metrics['utility_attack'])} | {_diff(metrics['attack_success'])} |")
        cost = summary["suites"]["all"]
        lines += ["", "Cost (scored cases):", "", "| Configuration | Prompt tokens | Completion tokens | Model turns | Tool calls (ran) | Wall, s (total / median) |", "|---|---|---|---|---|---|"]
        for configuration, s in cost.items():
            c = s["cost"]
            median = "—" if c["wall_s_median"] is None else f"{c['wall_s_median']:.2f}"
            lines.append(f"| {configuration} | {c['prompt_tokens']} | {c['completion_tokens']} | {c['model_turns']} | {c['tool_calls']} ({c['executed_calls']}) | {c['wall_s_total']:.1f} / {median} |")
        withheld = [row for rows in eligibility["tools"].values() for row in rows if not row["eligible"]]
        lines += ["", "Eligibility:"]
        for suite, rows in sorted(eligibility["tools"].items()):
            for row in rows:
                if not row["eligible"]:
                    lines.append(f"- `{suite}.{row['tool']}` withheld from every configuration: unsupported schema at `{row['pointer']}` ({row['message']}).")
        if not withheld:
            lines.append("- Every tool of the planned suites compiled to a grammar.")
        reasons: dict[str, int] = {}
        for row in eligibility["ineligible_cases"]:
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1])[:12]:
            lines.append(f"- {count} case(s): {reason}")
        if summary["not_scored"]:
            lines += ["", f"Not scored ({len(summary['not_scored'])}; excluded from every denominator, rerun under the bounded retry policy):"]
            for row in summary["not_scored"][:20]:
                lines.append(f"- `{row['case_id']}` {row['configuration']}: {row['outcome']} ({row['error_code']}), attempt {row['attempt']}")
        if summary["pairing_problems"]:
            lines += ["", "**Pairing problems** (these cases did not get identical inputs across configurations):"]
            lines += [f"- `{p['case_id']}`: {p['field']} has {p['values']} values" for p in summary["pairing_problems"][:20]]
        rows = list(latest(read_records(run / "records.jsonl")).values())
        lines += ["", "Residual failures, each with the first recorded trajectory (bounded journal events: names, verdicts, rules; no arguments, outputs, or model text):", ""]
        for title, matches in RESIDUALS:
            found = next((r for r in sorted(rows, key=lambda r: (r["case_id"], r["configuration"])) if r["outcome"] == "scored" and r.get("journals") and matches(r)), None)
            if found is None:
                lines.append(f"- {title}: none recorded.")
                continue
            lines += [f"- {title}: `{found['case_id']}` under {found['configuration']} (episode `{found['episodes'][-1]}`)", "", "```", *trajectory(run, found), "```", ""]
        lines.append("- A sandbox tier was unavailable: not applicable here (AgentDojo's tools are simulated in memory and need no tier); see section 3 for untested tiers.")
        lines.append("")
    return lines


def taxonomy_section(attack_runs: Sequence[Path], usability_runs: Sequence[Path], agentdojo_runs: Sequence[Path]) -> list[str]:
    from .metrics import FAILURE_CLASSES

    lines = ["## 6. Failure taxonomy", "",
             "Each failed AgentDojo case (user task not completed, or not scored) is counted once, under the first class that applies in this order: "
             + ", ".join(f"`{name}`" for name in FAILURE_CLASSES) + ". `semantic` means the episode ended normally but the task was not done "
             "(the model did the wrong thing). Sandbox, resource, egress, and output-gate classes come from sandboxed tools and stay zero for AgentDojo's "
             "simulated tools; the attack suite's own verdicts are in section 3.", ""]
    if not agentdojo_runs:
        lines.append("No AgentDojo runs were found.")
    for run in agentdojo_runs:
        manifest = _load(run / "manifest.json")
        table = _agentdojo_summary(run)["failure_table"]
        lines += [f"Run `{manifest['run_id']}` ({manifest['kind']}):", "", "| Class | " + " | ".join(table) + " |", "|---|" + "---|" * len(table)]
        for name in FAILURE_CLASSES:
            lines.append(f"| {name} | " + " | ".join(str(table[c][name]) for c in table) + " |")
        lines.append("")
    latest_cells, _ = merge_attack_runs(attack_runs)
    verdicts: dict[str, int] = {}
    for record in latest_cells.values():
        label = _label(record)
        verdicts[label] = verdicts.get(label, 0) + 1
    if verdicts:
        lines.append("Attack-suite cells by verdict (all profiles, controls included): " + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())) + ".")
    return lines + [""]


def replay_section(agentdojo_runs: Sequence[Path]) -> list[str]:
    lines = ["## 7. Replay evidence", "", REPLAY_NOTE, ""]
    for run in agentdojo_runs:
        manifest = _load(run / "manifest.json")
        path = run / "replay.json"
        if not path.exists():
            lines.append(f"- AgentDojo run `{manifest['run_id']}`: replay was not run for this copy of the run.")
            continue
        result = _load(path)
        missing = len(result.get("missing_journals", []))
        lines.append(f"- AgentDojo run `{manifest['run_id']}`: {result['exact']} of {result['episodes']} episodes replayed exactly on CPU; "
                     f"{len(result['mismatches'])} mismatches; {len(result['final_environment_mismatches'])} final-environment mismatches"
                     + (f"; {missing} case(s) not replayed because a journal was missing." if missing else "."))
    if not agentdojo_runs:
        lines.append("- No AgentDojo runs were found, so no AgentDojo replay is recorded. The Steadyhand and Bulkhead test suites replay their recorded episodes on every test run.")
    return lines + [""]


def build_report(*, attack_runs: Sequence[Path] = (), overhead_runs: Sequence[Path] = (), usability_runs: Sequence[Path] = (), agentdojo_runs: Sequence[Path] = ()) -> str:
    lines = [
        "# Bulkhead results", "",
        "Generated by `python -m bulkhead.eval.report` from the run records named below; every number is read from them.", "",
        "## 1. Threat model and architecture", "", ARCHITECTURE, "",
    ]
    lines += environment_section(attack_runs, overhead_runs, usability_runs, agentdojo_runs)
    lines += attack_section(attack_runs)
    lines += escapes_section(attack_runs)
    lines += agentdojo_section(agentdojo_runs)
    lines += ["## 5. Usability and overhead", ""]
    lines += usability_section(usability_runs)
    lines += overhead_section(overhead_runs)
    lines += taxonomy_section(attack_runs, usability_runs, agentdojo_runs)
    lines += replay_section(agentdojo_runs)
    lines += negatives_section(attack_runs, usability_runs, overhead_runs, agentdojo_runs)
    return "\n".join(lines).rstrip() + "\n"


def write_report(root: Path = RESULTS_ROOT, *, attack_runs: Sequence[Path] | None = None, overhead_runs: Sequence[Path] | None = None, usability_runs: Sequence[Path] | None = None,
                 agentdojo_runs: Sequence[Path] | None = None, output: Path | None = None) -> Path:
    """Write ``RESULTS.md`` from the given runs (default: every attack and AgentDojo run, the latest overhead and usability runs)."""
    attack_runs = latest_runs("attacks", root) if attack_runs is None else attack_runs
    overhead_runs = latest_runs("overhead", root)[-1:] if overhead_runs is None else overhead_runs
    usability_runs = latest_runs("usability", root)[-1:] if usability_runs is None else usability_runs
    agentdojo_runs = latest_runs("agentdojo", root) if agentdojo_runs is None else agentdojo_runs
    output = output or root / "RESULTS.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(attack_runs=attack_runs, overhead_runs=overhead_runs, usability_runs=usability_runs, agentdojo_runs=agentdojo_runs), encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bulkhead.eval.report", description="Write results/bulkhead/RESULTS.md from recorded runs.")
    parser.add_argument("--root", default=str(RESULTS_ROOT))
    parser.add_argument("--attacks", nargs="*", help="attack run directories, merged in order (default: all)")
    parser.add_argument("--overhead", nargs="*", help="overhead run directories (default: the latest)")
    parser.add_argument("--usability", nargs="*", help="usability run directories (default: the latest)")
    parser.add_argument("--agentdojo", nargs="*", help="AgentDojo run directories (default: all)")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    path = write_report(
        Path(args.root),
        attack_runs=[Path(p) for p in args.attacks] if args.attacks is not None else None,
        overhead_runs=[Path(p) for p in args.overhead] if args.overhead is not None else None,
        usability_runs=[Path(p) for p in args.usability] if args.usability is not None else None,
        agentdojo_runs=[Path(p) for p in args.agentdojo] if args.agentdojo is not None else None,
        output=Path(args.output) if args.output else None,
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_report", "latest_runs", "merge_attack_runs", "write_report"]

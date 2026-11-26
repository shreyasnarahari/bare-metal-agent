# Tidemark Handoff

This is the canonical operational handoff for a fresh implementation session. The eight numbered files in this directory are the plan; this file is current state, not a ninth plan part. Replace it atomically after every completed phase. Earlier states remain available in Git history.

## Status

- **Current state:** `pushed_acceptance_pending`. Part 1 is implemented, CPU-verified, and on `origin/main`. The user asked for that push on 2026-11-26 before the plan's required live acceptance. That acceptance (tokenizer parity, cache probe, utilization smoke, and baseline on a real Colab T4 **and** L4) has **not** run. Part 1 is therefore not complete, and no commit or record claims it is.
- **Last completed implementation phase:** none (part 1 awaits its live acceptance).
- **Next action:** run part 1's live notebook gate on a T4 and on an L4 (see *Next-session bootstrap*). Do not start part 2 until both pass and this Handoff records them.
- **Branch / remote:** `main` / `origin`.
- **Base commit:** `30a9ef1e988b1d24ad5c041eed4bfc5c5da7acda`.
- **Implementation commit:** `54612db611a0c7ed313d87b2f7241bad595bd359`, "Tidemark: part 1 baseline and context contract (T4/L4 acceptance pending)". It holds 44 paths. Its tree, `df519f12b5f523d6c5c291552b84bf17d2bff729`, is identical to the acceptance candidate that passed the full suite in a clean worktree (1898 passed, 54 skipped).
- **Other commit in this push:** `b29d0086f8d4eb3e62a6d0bdbf911834b52ec2b6`, the user's own changes, which the user had already run on GPU: `SO_REUSEADDR` in `LlamaCppServer`'s port probe, and a rerunnable Day 9 Qwen3.5 cell in `notebooks/01_toolsmith.ipynb`. It is not part of Tidemark.
- **Handoff commit:** the commit containing this file, after `b29d008`. Check its SHA with `git log -1 -- Tidemark/Handoff.md`.
- **Previous Handoff commit:** none (the planning Handoff was never committed).
- **Test branch:** `origin/tidemark/part-1-acceptance` at `f0e15df9dce089665a542aeea2783d0f38a633fe`, the same tree as the implementation commit. It is redundant now that `main` has part 1. Delete it only with the user's approval.
- **Written:** 2026-11-26T06:36Z (UTC).

## Delivered scope (on main, CPU-verified)

### Compatibility contract

- `run_episode(..., context=None)` behaves exactly as before. The committed Steadyhand goldens are byte-identical (SHA-256 pinned in `tests/test_steadyhand_context.py`). Re-recording their three scenarios with the current code reproduces every record apart from timestamps and timings, and no Tidemark event or manifest field appears. Installing `tidemark` enables nothing.
- With a `ContextSession`, every model-originated request goes through `steadyhand.context.ModelSeam`: agent turns, parse-error follow-ups, and the final paraphrase. The session chooses the exact messages; Toolsmith still templates, parses, validates, and repairs. The logical history stays in `EpisodeResult.messages` and the journal.
- `model_turn.messages_sha256` hashes the rendered messages actually sent. Replay re-renders on CPU with a fresh session whose `identity()` must equal the manifest's `context_config`.

### Public interfaces and changed paths

| Path | What changed |
|---|---|
| `src/steadyhand/context.py` (new) | `ContextSession` protocol, `RenderedContext`, `ModelSeam`, `context_config_sha256`, `CONTEXT_BUDGET_EXHAUSTED` |
| `src/steadyhand/loop.py` | `run_episode(context=...)`; every model call through the seam; tool messages through `shape_observation`; the controller's history-only-grows slot check is off when a session owns slot headroom |
| `src/steadyhand/journal.py` | Optional events `context_render` (before `model_turn`) and `backend_timings` (after it), value-checked on write and read; optional `episode_started.context`; optional `context_render.lint` for part 2 |
| `src/steadyhand/manifest.py` | Optional `context_config`, `context_config_sha256` and `tidemark_version` (identity fields), and `context_server` (resume-checked). All are omitted when absent, so legacy ids are unchanged |
| `src/steadyhand/replay.py` | `replay_episode(context=...)`, `run_claimed_episode(context=..., context_server=...)`; recorded turns rebuilt with metrics from `backend_timings` |
| `src/steadyhand/types.py` | `AttemptMetrics`, `TurnMetrics`; `ModelTurn.metrics` (default `None`) |
| `src/steadyhand/client.py` | `complete(id_slot=...)`, `render_prompt`, `request`; turns carry `TurnMetrics`; `settings()["generation_path"]` only when not the default |
| `src/toolsmith/api.py` | `ToolsmithService(generation_path="non_streaming"|"buffered_stream")`, `chat(request, id_slot=None)`, `render_prompt`; per-attempt evidence in `toolsmith.attempts` and `APIError.attempt_metrics` |
| `src/toolsmith/timings.py` (new) | Whitelisted, finite, non-negative backend field sanitizer; `AttemptTiming` |
| `src/toolsmith/backends/llama_cpp.py` | `complete/stream_complete(id_slot=...)`, sent only when given |
| `src/tidemark/` (new) | `types` (`ContextConfig`, `SegmentBudget`), `telemetry` (cache formulas, `derive_cache`, `backend_timing_fields`, repeated-prompt probe), `tokenizer` (protocol, `RegexTokenizer` fake, `HFTokenizer`, `ServerTokenizer`, parity corpus and probe, `CachedTokenizer`, `python -m tidemark.tokenizer`), `baseline` (`NaiveAppendTruncate`), `runtime` (`HardwareEnvelope`, `ResourceBudget`, `ResourceSampler`, `GpuSampler`, `summarize_utilization`), `scheduler` (`SlotPool`, `Scheduler`, `PriorityPool`, `SlotLeasedClient`, `bounded_prefetch`) |
| `src/tidemark/eval/` (new) | `baseline` (seed set and validator, scripted and forced-prefix clients, `EvidenceProbe`, scoring, resumable `run_baseline`, `step_rows`, `steps.csv`), `serving` (`ModelSpec`/`QWEN3_8B`, `CapacityPlan`, `plan_capacity`, `fit_capacity`, `autotune`, `FrozenConfig`, `ShapeGuard`, `CapacityCache`, `utilization_smoke`), `report` (summary, `prefill_by_step.svg`, `report.md`, `python -m tidemark.eval.report`) |
| `evaluation/data/tidemark_long_horizon_v1.jsonl` (new) | 12 tasks: file path, exact error code, and user constraint, each needed at N = 5, 10, 20, 40. Dataset SHA-256 `63ab909cbe9b1de55f59a98991d7158ef37ce3e7d091255c12d1751ad6528019` |
| `notebooks/04_tidemark.ipynb` | Part 1 cells (below); shared setup cells match Toolsmith; `steadyhand-runtime` and `cleanup` match Steadyhand byte for byte |
| Tests (new) | `tests/test_tidemark_{baseline,telemetry,tokenizer,runtime,scheduler,serving}.py`, `tests/test_steadyhand_context.py`, `tests/tidemark_support.py`, `tests/fixtures/tidemark/{backend_responses,parity_cases}.json` (synthetic, labeled as such) |
| Tests (changed) | `tests/test_notebook_04.py` (part 1 cells); `tests/test_steadyhand_security.py` (one assertion: a legacy manifest also omits the new context fields) |
| `README.md` | Tidemark part 1 description and status |

### Invariants and versions

- `JOURNAL_VERSION` stays 1: the new events are additive and optional; no existing required field changed.
- `ContextConfig.version` 1; `tidemark.__version__` 0.1.0; seed set `tidemark_long_horizon_v1`.
- `backend_timings`: counts, measurement path, slot id, and cache fields compare exactly in replay. `attempt_timings`, `ttft_ms`, `prompt_ms`, `predicted_ms`, and `response_elapsed_ms` are volatile. Missing values stay null.
- Cache metrics (`reused_tokens = max(0, rendered - evaluated)`, `cache_reuse_fraction`, `prefill_savings_fraction`) are null until `ContextConfig.cache_semantics.confirmed` is set from the live repeated-prompt probe, and null with `cache_status=unreconciled_counts` if the server's counts disagree with the rendered count.
- TTFT exists only on the `buffered_stream` path, timed at the first non-empty streamed content inside Toolsmith (same parser, validator, and at most one repair). Non-streaming TTFT is always null.
- The naive baseline drops only whole turns (an assistant call together with every result). It pins the system message, the current request (the last user message of the opening input), and the newest turn. It never lowers `max_tokens`; a turn that cannot fit is `context_budget_exhausted`, scored as `did_not_fit`.
- The session sends messages with sorted keys (the journal's form), so live and replay render the same structure. Tool schemas keep registry order.

## Verification evidence (this tree, CPU only)

Commands run on the final tree (WSL2, Python 3.13.16, `.venv`):

```bash
.venv/bin/python -m pytest tests/test_tidemark_baseline.py tests/test_tidemark_telemetry.py \
  tests/test_tidemark_tokenizer.py tests/test_steadyhand_context.py tests/test_steadyhand_replay.py
# 142 passed
.venv/bin/python -m pytest
# 1898 passed, 54 skipped (the skips are the existing opt-in Bulkhead integration and AgentDojo suites)
.venv/bin/python -m compileall -q src tests   # ok
git diff --check                              # ok (new untracked files also checked: no trailing whitespace, final newlines present)
```

- Before this work, the full suite on the base tree had 1,744 passed, 54 skipped.
- Notebook cells exercised by tests: `tidemark-cpu-gate` (executes for real: 3 golden replays plus seed-set validation) and `tidemark-settings` (executes). The live cells (`tidemark-capacity-probe`, `tidemark-tokenizer-probe`, `tidemark-utilization-smoke`, `tidemark-baseline`, `tidemark-baseline-report`) are only parsed and checked for their contract strings; they need a GPU.
- CPU, fake-backed evidence only: scripted episodes through the real Steadyhand loop and journal, a regex fake tokenizer, a Qwen-like fake template, fake streaming backends for TTFT, and fake T4/L4 hardware envelopes. A scripted CPU baseline with a small slot shows truncation dropping planted facts at large distances. This shows the mechanism works; it is not a model result.
- The chart was rendered with headless Chrome in light and dark themes and checked by eye; its palette passed the dataviz validator.

**Not exercised anywhere:** a real model, real tokenizer parity, the cache-counter probe, buffered-stream TTFT against llama-server, capacity fitting, the utilization smoke, the baseline, T4 or L4.

## Live/runtime state

None. No llama-server, background job, Drive mount, run id, manifest, journal, resource sample, or report from part 1 exists. Model/GGUF/tokenizer/llama.cpp revisions, capacity and scheduler hashes, and run ids will be produced by the notebook. Expected identities: Qwen3-8B Q4_K_M from `Qwen/Qwen3-8B-GGUF` on llama.cpp `b5401` (the shared Steadyhand provisioning cell with `STEADYHAND_PHASE=qwen3/wide`).

Every T4 and L4 cell (all 12 tasks per GPU) is **untested**.

Expected live behavior to check, not a result: the seed set's mock-store observations are small (about 300 characters each), so on the required 16K path even a 40-step forced prefix should fit (roughly 7K tokens), with no truncation steps. Truncation is exercised on CPU. If the live runs show no truncation rows, that is expected, not a failure; part 3's observation corpus adds the large payloads.

## Durable artifacts (once the notebook runs)

```text
local active state: /content/tidemark/
  baseline/<run_id>/                  spec <run_id>.json, <run_id>.scores.jsonl, journals/, resource_samples.jsonl, utilization.json
  smoke/<smoke_run_id>/               autotune candidates' runs
  report/<shape>/                     steps.csv, prefill_by_step.svg, summary.json, report.md
durable records: MyDrive/toolsmith/records/<gpu>/tidemark/
  cpu-gate.json
  capacity/<shape>-<plan sha>.json, capacity/plans.json (CapacityCache)
  parity/<run_id>.json, parity/<run_id>-cache-probe.json
  smoke/autotune-<gpu>-<stamp>.json
  baseline/<run_id>/                  mirror of the local run folder
  report/<shape>/
```

Default `run_id`: `baseline-v1-<t4|l4>-wide` (override with `TIDEMARK_BASELINE_RUN_ID`). Resume: rerun the notebook from the top. The baseline cell restores its run folder from Drive if the local copy is missing, the smoke cell confirms the recorded frozen scheduler instead of retuning, completed units are skipped, an interrupted episode resumes from its journal, and a changed spec (new fitted shape, model, context config, or dataset) is refused with `BaselineMismatch`. A deliberate change needs a new run id. No blob catalog, memory database, or slot file exists in part 1.

## Known issues and boundaries

- **Open gate:** T4 and L4 live acceptance (parity, cache probe, utilization smoke, baseline, report) has not run. Part 1 is on `main` at the user's request but is not complete. Any code fix a live run needs is a new commit on top of `54612db`.
- **Pre-existing, not caused by Tidemark:** the golden manifests' episode ids (`ep_8393620aa57df809`, `ep_b345550151d9160c`, `ep_07922e66f89c0aa9`) predate Steadyhand part 6, which added `recovery_policy` and `system_prompt_variant` to the identity hash. Unchanged base code already derives `ep_268aa8ce0b358631`, `ep_b4a01a2f9f338bd8`, and `ep_7499cf1f6182975d` for the same scenarios (checked in a clean worktree of the base commit). Replay never compares ids, so the goldens still replay. The test pins both sets of ids; regenerating the goldens is a separate, deliberate decision.
- The part 1 autotune varies only the scheduler's episode-worker count on the smoke workload (no server restart). Affinity stays `pinned` because the baseline measures cache reuse. KV/flash is decided by the fit ladder, and batch, thread, and compactor-slot grids are implemented and CPU-tested (`capacity_candidates`, `autotune`) but not run in the notebook.
- Forced-prefix live steps send the real request (real prefill and cache counts) and then substitute the scripted calls. Their completion tokens and decode time are those of the discarded generation, and the slot cache holds the model's own output rather than the scripted turn. Rows are labeled `forced` and `model`.
- The `tidemark-capacity-probe` cell defines `stop_steadyhand_runtime` so the shared cleanup cell stops the Tidemark server.
- `required_availability` values `blob_handle`, `memory`, and `any_restorable` reduce to inline checks in part 1: there is no blob store or memory yet.
- The port-probe `SO_REUSEADDR` change is now on `main` (`b29d008`), so `fit_capacity`'s same-port restart after an out-of-memory shrink can reuse a port left in TIME_WAIT. The test branch does not have it. The working tree has no unrelated changes left.
- `results/` holds local scratch (for example `results/tidemark-chart-check/`) and is gitignored.

## Next-session bootstrap

1. Read this file and `Tidemark/01-baseline-and-context-contract.md`. Check the state and that part 1 is in history:

   ```bash
   git status --short --branch --untracked-files=all
   git pull --ff-only origin main            # only with a clean worktree
   git merge-base --is-ancestor 54612db611a0c7ed313d87b2f7241bad595bd359 HEAD && echo part-1-present
   ```

   If the Handoff and the repository disagree, stop and report the mismatch.
2. Rerun the CPU starting gate:

   ```bash
   .venv/bin/python -m pytest tests/test_notebook_04.py tests/test_tidemark_baseline.py tests/test_steadyhand_context.py
   ```

3. On a fresh **T4** runtime, then a fresh **L4** runtime, run `notebooks/04_tidemark.ipynb` top to bottom. The `drive-and-secrets` cell pulls `main`, which now has part 1, so no branch checkout is needed. Cells, in order: `tidemark-cpu-gate`, `tidemark-settings`, `steadyhand-runtime`, `tidemark-capacity-probe`, `tidemark-tokenizer-probe` (must pass parity; it stops the notebook otherwise), `tidemark-utilization-smoke`, `tidemark-baseline`, `tidemark-baseline-report`, `cleanup`. Secrets by name only: `HF_TOKEN`, `LOCAL_KEY`.
4. Acceptance to confirm on each GPU: parity passed; cache semantics confirmed (or explicitly unconfirmed with evidence); a legal fitted shape; resource samples and the utilization summary with explanations for any miss; the baseline manifest and scores for the same 12 task ids; `steps.csv` and `prefill_by_step.svg` generated from records with cold, ordinary, and truncation steps distinguishable; the report showing both GPUs.
5. If a live run needs a code change: fix it with tests, rerun the phase tests, the full suite, `compileall`, and `git diff --check`, and commit the fix by its exact paths (never `git add -A` or `git add .`). Then rerun the affected live cells.
6. When both GPUs pass: rewrite this Handoff (status `complete_when_pushed`, the live evidence, run ids, model/GGUF/tokenizer/llama.cpp revisions, capacity and frozen-config hashes, and Drive paths), commit `Tidemark: hand off part 1` with only this file, run `git push origin main`, and verify that `git ls-remote origin refs/heads/main` equals `git rev-parse HEAD`. Only then is part 2 the next action.
7. If the push is rejected or authentication fails, keep the local commits and stop; do not rebase, merge, force-push, or retry indefinitely.

## Safety

Secrets are referred to by the names `HF_TOKEN` and `LOCAL_KEY` only. Journals redact both. The notebook passes them to clients and redaction and never prints or records them (a notebook test checks this). Records hold hashes and counts, not raw prompts or token ids. No API key, authorization header, environment dump, hidden reasoning, raw private prompt, or memory payload is in this file or the tree.

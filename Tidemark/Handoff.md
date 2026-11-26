# Tidemark Handoff

This is the canonical operational handoff for a fresh implementation session. The eight numbered files in this directory are the plan; this file is current state, not a ninth plan part. Replace it atomically after every completed phase. Earlier states remain available in Git history.

## Status

- **Current state:** parts 1 and 2 are both `pushed_acceptance_pending`. Both are on `origin/main`, pushed at the user's request before their live acceptance. Neither part's T4/L4 acceptance has run, so neither is complete and no commit or record claims it is. The user plans to run the notebook later and fix whatever it finds.
- **Last completed implementation phase:** none (neither part has its live acceptance).
- **Next action:** run `notebooks/04_tidemark.ipynb` (part 1's live cells, then `tidemark-prefix-smoke`) on a T4 and an L4. Fix any failures with new commits. Record the evidence here. Part 3 waits for that.
- **Branch / remote:** `main` / `origin`.
- **Part 2 commits:** implementation `1f43c2a4ad525b13ba70db2f4430cc8e5592cc36` (tree `bf81b71215c91878fb2b1a37eab98578473af334`, 27 paths, base `eb909584792a2385b5c20196837b9263e73e77b6`). The Handoff commit is the commit containing this file; check it with `git log -1 -- Tidemark/Handoff.md`.
- **Part 1 commits:** implementation `54612db611a0c7ed313d87b2f7241bad595bd359` (tree `df519f12b5f523d6c5c291552b84bf17d2bff729`), the user's own `b29d0086f8d4eb3e62a6d0bdbf911834b52ec2b6` (port probe and notebook 01), and Handoff `eb909584792a2385b5c20196837b9263e73e77b6`.
- **Previous Handoff commit:** `eb909584792a2385b5c20196837b9263e73e77b6`.
- **Test branch:** `tidemark/part-1-acceptance` was deleted, with the user's approval, after part 1 reached `main`. No test branch exists.
- **Written:** 2026-11-26 (UTC).

## Part 1 delivered scope (on main, CPU-verified)

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

## Part 2 delivered scope (on main, CPU-verified)

Paths (27): new `src/tidemark/{assembler,budgets,cache,prefix_lint,recitation,segments}.py`, `src/tidemark/eval/prefix.py`, `tests/test_tidemark_{assembler,budgets,prefix_lint,recitation}.py`; changed `src/steadyhand/{client,context,journal,loop,replay}.py`, `src/tidemark/{baseline,scheduler,types}.py`, `src/tidemark/eval/baseline.py`, `src/toolsmith/{api,prompting}.py`, `notebooks/04_tidemark.ipynb`, `tests/test_notebook_04.py`, `tests/test_steadyhand_context.py`, `tests/tidemark_support.py`, `README.md`.

- **Segments and state:** `ContextState` is immutable, with explicit transitions (`append`, `with_recitation`, `advance_epoch`). `ContextAssembler.render` is pure. Messages are legal: one leading system message with the frozen tools (canonical keys, declared order), a user-role pinned charter and memory block, a user-role canonical-JSON summary (absent in epoch 0), recent messages in order, and one user-role recitation tail last. The tail is never kept in the history.
- **Spans:** `toolsmith.prompting.render_with_token_spans` normalizes history once, renders boundary prefixes without the generation prompt, and renders the final prompt exactly as `ToolsmithService`. It requires each prefix to be a text and token prefix of the next, or returns `span_mapping_unavailable` with no spans. `ChatPromptRenderer.render` gained `add_generation_prompt` (default `True`). A test checks that `ToolsmithService.render_prompt` equals the linted text for both protocols.
- **Budgets:** the hard limit is `slot_context - max_tokens - safety_margin`; the soft limit defaults to 70% of it. Per-segment counts come from exact spans only. The decision order is exhausted, then compaction pending, then ok. The overflow policy is `exhaust` (part 2 has no compactor). `check_static` and `PrefixStableSession.validate` reject a frozen catalog that cannot fit.
- **Recitation:** a typed, bounded state with no clock or random values. Plan items are agent-authored and empty unless supplied; nothing is derived from tool output. Trimming is by priority, and the tail cap is checked on exact spans.
- **PrefixLint:** the cause comes from order-sensitive segment hashes. A token change with no logical change is `token_boundary_merge`. `FingerprintStore` keeps the previous ids as 4-byte binary, not in JSONL; resume re-derives fingerprints by replay.
- **Journal:** `context_render` gains optional fields `stable_prefix_tokens`, `expected_common_prefix_floor`, `observed_common_prefix_tokens`, `first_divergent_index`, `first_changed_segment`, `divergence_cause`, `lint_status`, `previous_prompt_sha256`, and `allowed_tool_names`. The new `context_transition` event (`start`/`append`/`truncate`/`epoch`, with journal seqs) is written before its `context_render`. The naive baseline now emits transitions too. Journal version stays 1, and part 1 records still read. Lint fields are not volatile.
- **Seam:** `ContextSession.render` also receives `seqs` (from `steadyhand.context.history_seqs`) and `budget_state` (the controller's counts). The paraphrase passes `unjournaled=1`. `allowed_tool_names` goes to `client.complete` only when set.
- **Tool masking:** `ToolsmithService.chat(allowed_tool_names=...)` keeps every tool in the prompt, compiles the grammar and validator from the subset, and rejects unknown or empty subsets (`invalid_allowed_tools`). `None` is byte-for-byte legacy. `SteadyhandClient.complete(allowed_tool_names=...)`; `ContextConfig.allowed_tool_names`.
- **Caches:** `ContentCache` (thread-safe LRU with hit counters) holds rendered prefixes, keyed by exact order-sensitive content, tools, protocol, thinking flag, and template revision. `ToolsmithService(grammar_cache=...)` keys grammars by tools, choice, protocol, and compiler version. Counters are overhead only. `run_baseline(prepare=...)` warms upcoming tasks on the scheduler's prefetch thread.
- **ContextConfig:** strategy `prefix_stable`; new fields `pinned_memory`, `allowed_tool_names`, `recitation`, and `overflow`. These are always in `as_dict`, so part 1 naive config hashes changed too; no live part 1 records exist.
- **Notebook:** `tidemark-prefix-gate` (CPU; the planted suite for both protocols, executed by the tests) after `tidemark-cpu-gate`, and `tidemark-prefix-smoke` (live) after `tidemark-baseline-report`.

Part 2 CPU verification on the exact committed tree: the plan's command (`tests/test_tidemark_assembler.py tests/test_tidemark_budgets.py tests/test_tidemark_prefix_lint.py tests/test_tidemark_recitation.py tests/test_api.py tests/test_steadyhand_replay.py`) gives 98 passed. The full suite gives 1940 passed, 54 skipped. `compileall` and `git diff --check` are clean. The planted suite passes on the CPU fake template for both Qwen protocols, with first-divergence indexes inside the named segments.

## Part 1 verification evidence (CPU only)

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

- **Part 2 boundaries:**
  - The planted breakers are generated in code (`tidemark/eval/prefix.py`), not stored under `tests/fixtures/tidemark_prompts/`.
  - The batch/ubatch and `--cache-reuse` autotune grid is not run in the notebook. The part 1 `capacity_candidates(batches=...)` and `autotune` mechanisms exist, but every scored part 2 run uses the frozen part 1 shape.
  - Prefetch uses the scheduler's prefetch thread, not the process pool, because rendered prefixes live in an in-process cache.
  - For Qwen3.5, re-serialized old arguments decode to an identical prompt. PrefixLint still reports the rewrite as a `violation`: old messages are immutable.
  - The live warm-reuse figure needs `cache_semantics.confirmed`. Without it, reuse is null and the smoke reports zero warm steps.
- **Open gate:** T4 and L4 live acceptance (parity, cache probe, utilization smoke, baseline, report) has not run. Part 1 is on `main` at the user's request but is not complete. Any code fix a live run needs is a new commit on top of `54612db`.
- **Pre-existing, not caused by Tidemark:** the golden manifests' episode ids (`ep_8393620aa57df809`, `ep_b345550151d9160c`, `ep_07922e66f89c0aa9`) predate Steadyhand part 6, which added `recovery_policy` and `system_prompt_variant` to the identity hash. Unchanged base code already derives `ep_268aa8ce0b358631`, `ep_b4a01a2f9f338bd8`, and `ep_7499cf1f6182975d` for the same scenarios (checked in a clean worktree of the base commit). Replay never compares ids, so the goldens still replay. The test pins both sets of ids; regenerating the goldens is a separate, deliberate decision.
- The part 1 autotune varies only the scheduler's episode-worker count on the smoke workload (no server restart). Affinity stays `pinned` because the baseline measures cache reuse. KV/flash is decided by the fit ladder, and batch, thread, and compactor-slot grids are implemented and CPU-tested (`capacity_candidates`, `autotune`) but not run in the notebook.
- Forced-prefix live steps send the real request (real prefill and cache counts) and then substitute the scripted calls. Their completion tokens and decode time are those of the discarded generation, and the slot cache holds the model's own output rather than the scripted turn. Rows are labeled `forced` and `model`.
- The `tidemark-capacity-probe` cell defines `stop_steadyhand_runtime` so the shared cleanup cell stops the Tidemark server.
- `required_availability` values `blob_handle`, `memory`, and `any_restorable` reduce to inline checks in part 1: there is no blob store or memory yet.
- The port-probe `SO_REUSEADDR` change is now on `main` (`b29d008`), so `fit_capacity`'s same-port restart after an out-of-memory shrink can reuse a port left in TIME_WAIT. The test branch does not have it. The working tree has no unrelated changes left.
- `results/` holds local scratch (for example `results/tidemark-chart-check/`) and is gitignored.

## Next-session bootstrap

1. Read this file and `Tidemark/01-baseline-and-context-contract.md` and `Tidemark/02-prefix-stable-context-and-budgets.md`. Check the state:

   ```bash
   git status --short --branch --untracked-files=all
   git pull --ff-only origin main            # only with a clean worktree
   git merge-base --is-ancestor 1f43c2a4ad525b13ba70db2f4430cc8e5592cc36 HEAD && echo part-2-present
   ```

   If the Handoff and the repository disagree, stop and report the mismatch.
2. Rerun the CPU gates: `.venv/bin/python -m pytest tests/test_notebook_04.py tests/test_tidemark_assembler.py tests/test_tidemark_prefix_lint.py tests/test_steadyhand_context.py`.
3. On a fresh **T4**, then a fresh **L4**, run `notebooks/04_tidemark.ipynb` top to bottom. The notebook pulls `main`, which has both parts. Part 1's acceptance is listed in its section above. `tidemark-prefix-smoke` must show:
   - the lint prompt equals the production prompt, with exact spans;
   - the live planted suite passes;
   - the planted timestamp forces a near-full same-slot prefill, while the legal append reuses the cache;
   - no live PrefixLint violation;
   - warm reuse is reported against the 90% target (a target, not a pass condition).
4. Fix any live failure with tests and a commit of exact paths (never `git add -A` or `git add .`), then rerun the affected cells.
5. When both GPUs pass: rewrite this Handoff with the live evidence for parts 1 and 2 (run ids; model, GGUF, tokenizer, template, schema, and grammar hashes; planted divergence and cache-probe numbers; capacity and frozen-config hashes; Drive paths), set both parts to `complete_when_pushed`, commit it alone, `git push origin main`, and verify that `git ls-remote origin refs/heads/main` equals `git rev-parse HEAD`. Only then is part 3 the next action.
6. If the push is rejected or authentication fails, keep the local commits and stop; do not rebase, merge, force-push, or retry indefinitely.

## Safety

Secrets are referred to by the names `HF_TOKEN` and `LOCAL_KEY` only. Journals redact both. The notebook passes them to clients and redaction and never prints or records them (a notebook test checks this). Records hold hashes and counts, not raw prompts or token ids. No API key, authorization header, environment dump, hidden reasoning, raw private prompt, or memory payload is in this file or the tree.

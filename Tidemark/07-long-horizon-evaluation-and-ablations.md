# Part 7 — Long-horizon evaluation, ablations, and benchmark adapters

**Track:** Tidemark, Day 4 of 10
**Milestone:** M6, evenings 10–11
**Depends on:** `06-secure-memory-tools-and-session-resume.md`
**Produces:** the full own-suite, a paired seven-mode ablation runner, observation/compaction/memory metrics, LongMemEval and context-rot adapters, and record-only reports

## Outcome

Measure whether each context mechanism pays for itself in task quality and compute. The main signal is the repository's long-horizon agent suite, not a synthetic token-count improvement. Every compared mode runs the same eligible task ids, model, generation path, tools, fault schedule, approval behavior, and server shape. Quality, evidence availability, evaluated prefill tokens, cache reuse, TTFT/prefill latency, compaction cost, and peak context remain separate metrics.

Public benchmarks add retrieval and long-context evidence; they do not replace the agent suite or turn a small subset into a leaderboard claim.

## What already exists

- Part 1's seed dataset defines planted facts at N=5/10/20/40 and separates task success from evidence availability. Part 3 adds 30–50 must-keep observations; part 4 adds supported/unsupported compaction cases; part 5 adds cross-session memory fixtures.
- `steadyhand.eval.suite` demonstrates final-store gold scoring, fault schedules, exclusive claims, completed manifests, retries, and infrastructure-excluded denominators.
- `toolsmith.evaluation.metrics` supplies `percentile`, `wilson_interval`, `paired_bootstrap`, and `deterministic_subset`; `artifacts.py` supplies mirrors/snapshots.
- Parts 1–6 provide all seven context configurations and their strict journal evidence. No additional behavior may be special-cased just for a benchmark row.
- `notebooks/04_tidemark.ipynb` owns the project CPU gates and future live sweeps. Core unit tests use recorded models/journals and fake clocks/embedders.

## What this part adds

```text
evaluation/data/tidemark_long_horizon_v1.jsonl       # expanded from part 1
evaluation/data/tidemark_observations_v1.jsonl
evaluation/data/tidemark_compaction_v1.jsonl
evaluation/data/tidemark_memory_v1.jsonl
src/tidemark/eval/datasets.py                        # Strict loaders and eligibility
src/tidemark/eval/runner.py                          # Paired modes, claims, resume, retries
src/tidemark/eval/metrics.py                         # Step/episode/task/cache/quality metrics
src/tidemark/eval/longmemeval.py                     # Pinned upstream adapter
src/tidemark/eval/context_rot.py                     # Increasing-length quality probe
src/tidemark/eval/report.py                          # Markdown/CSV/SVG from records only
tests/test_tidemark_eval_datasets.py
tests/test_tidemark_eval_runner.py
tests/test_tidemark_eval_metrics.py
tests/test_tidemark_longmemeval.py
tests/test_tidemark_context_rot.py
tests/test_tidemark_report.py
tests/fixtures/tidemark_eval/
```

Generated records stay under gitignored `results/tidemark/`; Colab mirrors them under the per-GPU `tidemark/` records tree. External datasets are evaluation artifacts pinned by upstream commit and content hash, not core runtime dependencies.

## Own long-horizon suite

Expand the seed set into balanced task families:

| Family | What is planted | Delay/difficulty | Required outcome |
|---|---|---|---|
| `path_recall` | Workspace-relative file path and operation constraint | N=5/10/20/40 plus similarly named distractors | Correct later tool/action; path available inline or by handle |
| `error_recall` | Exact exception/error code/failing test amid long logs | Large observations before and after the fact | Correct diagnosis/recovery without repeating known failed attempt |
| `user_constraint` | Amount, status, date, or prohibition from the task/user | Many intervening calls and paraphrased distractors | Final store/task diff honors constraint; no invented replacement |
| `blob_recovery` | Needed evidence omitted from a preview | Handle remains after several turns/epochs | Agent reads the right range and uses exact evidence |
| `compaction_state` | Decisions, failed attempts, unresolved questions | One or more compaction epochs before use | Supported current state survives; rejected invention is not acted on |
| `cross_session_memory` | User/profile/task fact from an earlier session | New episode, updated/superseded facts, undated and dated queries | Correct current retrieval and taint/policy behavior |
| `large_observation` | pytest/compiler/JSON/HTML payloads from the corpus | Mixed with dependent store operations | Task succeeds with bounded peak context and restorable evidence |

Each entry has a deterministic reference script, expected tools, final store/task gold, required evidence refs, availability class, distance, observation ids, memory initialization snapshot, fault schedule, and seed. Include negative controls where the planted-looking distractor is wrong and where no memory write should occur.

### Two scores, not one conflated score

1. **Task success:** final environment/store objective matches gold, with duplicate effects counted separately. Final prose is not the gold.
2. **Evidence availability:** at the first step needing each planted fact, the exact fact is present in the rendered context or callable through a visible valid handle/memory result. Record `inline`, `blob_restorable`, `memory_retrieved`, `missing`, and `wrong_version`.

Also record whether the model actually used the evidence. A correct guess with missing evidence is not a context success; an available fact with a wrong model action is a model/task failure, not a shaping retention failure.

### Fault and poisoning variants

Reuse Steadyhand's deterministic fault schedules for explicit, implicit, and lost-acknowledgment failures where meaningful. Add untrusted text instructing the agent to remember a false constraint, supersede a trusted fact, read another namespace, or exfiltrate memory. Measure ALLOW/DENY/ASK, approval mode, stored trust, duplicate effects, and whether the attacker objective occurred. No auto-approval is used to inflate utility.

## Seven-mode paired ablation

Use these exact cumulative rows:

| Mode | Configuration |
|---|---|
| `naive_truncate` | Append all; drop oldest complete turns at hard limit |
| `sliding_window` | Last K complete turns; K fixed in the manifest, no prefix-stability machinery |
| `prefix_stable` | Five-segment deterministic layout and recitation only |
| `shaped` | Prefix stable + observation reducers + blob handles |
| `compact_unchecked` | Shaped + compactor with structural citation validation only; no lexical/judge source check |
| `compact_checked` | Same compactor/candidates + full source checker |
| `memory` | Compact checked + long-term retrieval and governed memory tools |

`compact_unchecked` still enforces grammar and valid-looking event refs so its comparison isolates support checking rather than malformed JSON. It must not drop the source-check stage only for difficult candidates while `compact_checked` receives a different candidate. Where practical, feed both modes the same recorded compactor candidates for the support ablation; live end-to-end results are a separate labeled view.

All rows share:

- eligible ids/subset hash, model/GGUF revision, tokenizer/template, native protocol, seed/temperature, max tokens, buffered/non-streaming path;
- frozen tools and task policy except for the feature being measured, approval mode, fault schedule, recovery strategy, max steps;
- per-slot context, model-slot count, episode/CPU/IO worker counts, slot-affinity/compactor policy, KV type, cache flags, prefetch depth, and initial cache/session state;
- embedding model/revision/ranking for rows that use memory.

Changing one row's task subset, model, or slot shape is not a paired result. Schema- or context-ineligible ids remain listed with reasons and are not silently removed from only one row.

### Cache-state controls

Report the first cold step separately. For paired task units, start from the same explicit cold state or verified initial slot snapshot and balance execution order to reduce thermal/server drift. Never clear cache before the baseline only, or let `memory` inherit a warm prefix that `naive_truncate` did not get.

Use part 1's slot semaphore and one declared affinity policy (`pinned` or `work_conserving`) plus a declared compactor policy across every row. Keep model-slot count, episode/CPU/IO pools, prefetch, background jobs, and resource budgets identical for timing rows. Autotuning finishes before scoring; no worker/slot shape may adapt mid-row. Slot-save/resume is excluded from the primary matrix unless every row uses the same policy and a separate restore-ablation label.

## Metric contract

Compute from raw per-step and per-episode records:

### Quality and retention

- task success with raw numerator/denominator and Wilson interval; paired per-task difference with seeded bootstrap;
- duplicate side effects, false memory writes, attacker success, benign over-denial/ASK rate;
- evidence availability/use by distance N and category;
- observation must-keep inline/restorable rate and real-token reduction;
- unsupported summary entries before/after checking, false accepts/rejects on human labels, compaction-trigger count;
- memory Recall@5/@10, NDCG@5/@10, superseded/wrong-version retrieval, deletion/poisoning outcomes.

### Compute and latency

- exact rendered prompt tokens and actual prompt-evaluated tokens per step;
- reused tokens/cache fraction with cold, ordinary, compaction, and resumed steps separated;
- mean and p95 `ttft_ms` where the buffered-stream path supplies it; otherwise null, with `prompt_ms` mean/p95 always labeled as prefill rather than TTFT;
- response elapsed, decode tokens/time, and end-to-end episode time;
- peak rendered context and slot headroom;
- compactor input/output tokens, judge calls, compaction re-prefill, shaping time, blob I/O, embedding time, SQLite time, and memory retrieval tokens;
- infrastructure retries/aborts and `context_budget_exhausted` separately from semantic task failure.

Use per-step denominators for per-step latency and per-episode denominators for episode rates. Provide micro averages over all steps and macro averages per episode where their interpretation differs. Do not compute p95 from a hand-selected subset or average ratios without showing counts.

### Cost of compaction

For each compaction, retain enough raw data to compute:

```text
one_time_cost = compactor_prefill + compactor_decode + judge_cost + post_compaction_reprefill
future_savings = sum(baseline_evaluated_tokens - checked_evaluated_tokens on later paired steps)
net_savings = future_savings - one_time_cost
```

Use actual measured values when paired steps exist and label model-based estimates separately. A smaller prompt after compaction is not automatically lower total cost.

## Runner, claims, and resume

A run unit is `(task_id, variant, mode, repeat, dataset_hash, context_config_hash, model_identity, server_identity)`. Its manifest includes all shared controls, initial memory/blob/cache snapshots, code versions, approval mode, selected ids, and execution order.

- Use exclusive claims and one journal/result per unit. Resume only exact matching completed records; a changed implementation/config starts a new run id or stops.
- Write local JSONL and per-step CSV atomically and mirror whole-line snapshots. A forced mirror at unit end makes the unit eligible for durable resume.
- Bound infrastructure retries as in Steadyhand. Connection/timeout/server 5xx/OOM are not task failures, successful defenses, no-call answers, or evidence misses.
- Context/compaction failures are their own taxonomy and remain visible. Do not retry them as transient backend failures.
- Pair analysis joins only units scored in both modes and reports unpaired/ineligible/abort ids explicitly.
- Cap simultaneous model calls at the fixed slot count; episode workers may exceed it only for CPU/tool/memory work under the bounded scheduler. Timing/overhead microbenchmarks run serially on a quiet host with background jobs recorded.

The report reads result records and journals, never notebook variables or hardcoded tables. Unrun rows remain `untested`.

## Runtime optimization and utilization controls

Run the full eligible suite faster by removing idle time, never by deleting tasks or changing denominators:

- Use the part 1 scheduler with at most the fitted number of model calls and enough episode workers to keep those slots filled during tool/memory/approval waits. CPU preparation, source checks, blob shaping, FTS/vector indexing, and report generation run in the shared resource-budgeted pools.
- Prepare the next task's initial context/artifacts while the current task generates. Interleave paired modes in a deterministic balanced order to reduce server drift, but reset/restore each unit to the same declared initial cache state. Do not reuse a previous mode's warm prompt cache unless the experimental design explicitly calls for it.
- Reuse immutable CPU artifacts aggressively: tokenizer/grammar caches, shaped observation bytes, source-event bundles, document embeddings, and dataset conversions. Keys include every semantic configuration hash. Share compactor candidates only for the declared source-check ablation; all live task-success runs otherwise call the model normally.
- Precompute LongMemEval FTS/vector indexes on CPU during an eligible GPU phase and generate context-rot prompt/artifact bundles ahead of its separate server phase. Reports for completed phases can overlap the next phase if CPU headroom exists.
- Use `pinned` scheduling for cache-measurement experiments. Use `work_conserving` only as a separately named whole-matrix scheduler configuration, applied identically to every row, with migrations/evictions reported. Never combine a work-conserving latency result with a pinned-cache result as though one run achieved both.
- Run utilization autotuning on representative smoke units separately on T4 and L4, then freeze the complete shape. The core same-model quality suite runs on both GPUs. Timing comparisons stay within one GPU/shape; a T4 uses fewer/different workers but the same task ids and gold, not a reduced suite.
- Record model-ready slot occupancy, CPU-ready utilization, episode/model queue waits, CPU/IO pool saturation, slot migrations, RAM/VRAM/disk, and phase tasks/minute. Investigate any ready-but-idle window instead of hiding it in full-phase averages.

Timing/overhead microbenchmarks that require a quiet serial host remain serial and exclude background work. The production long-horizon sweep is deliberately overlapped; its manifest and resource samples describe that concurrency, and its timings are not compared with the quiet microbenchmark.

Tests execute a mixed model/tool/index workload with fake T4 and L4 budgets. They prove full model-slot occupancy under ready load, bounded CPU/RAM queues, identical results to serial execution, no invalid cross-mode artifact reuse, and no scheduler/config change during scoring.

## LongMemEval adapter

Before implementation, inspect and pin the exact [LongMemEval](https://github.com/xiaowu0162/LongMemEval) revision, file format, evidence labels, and license. Record upstream commit, selected split/id hash, and any conversion script version. Do not assume an API from the current README.

Support two local workflows:

1. **Index/retrieve/read:** ingest session rounds into the memory store, expand keys, run time-aware hybrid retrieval, and score returned memory ids against evidence labels with Recall@k/NDCG@k.
2. **QA:** give the agent/reader the same retrieved evidence and compare answers with exact/structured labels where possible and a separately pinned local judge where necessary.

Use `deterministic_subset` for a declared subset and publish the actual ids. A local judge is not GPT-4o; label the resulting QA metric as not directly comparable with the paper's official judge numbers. No OpenAI API call is needed.

The full ~115K-token `LongMemEval_S` full-history baseline may not fit next to the Q4 model on either a T4 or an L4, and is especially constrained on T4. Try only a declared subset/server configuration with q8_0 KV and YaRN if the actual model supports it. Otherwise record `did_not_fit` with VRAM/model/config evidence on that GPU; do not drop the baseline silently or compare a truncated baseline as though it were full history. The memory-based core suite still runs on both GPUs.

## Context-rot adapter

Inspect and pin [Chroma's context-rot harness](https://github.com/chroma-core/context-rot) or implement a small compatible local needle/question set with its own name rather than claiming official comparability. For fixed questions and evidence:

- vary total input length and needle position while holding question, model, decoding, and irrelevant-text source constant;
- repeat enough declared positions/seeds to separate length from placement;
- record exact correctness, prompt/evaluated tokens, prefill/TTFT, and memory use;
- predeclare the quality-knee rule before looking at results (for example, the first length below a chosen fraction of the short-context score with a confidence interval), then set a proposed compaction soft limit below it;
- validate the proposed limit on held-out long-horizon task ids rather than tuning and reporting on the same examples.

This informs configuration; it does not prove one universal optimal context length.

## Optional systems comparisons, outside core acceptance

- **C++ cache replay simulator:** replay journaled token-id streams through parent-chained block hashing and a radix tree with LRU leaf eviction under a fixed KV budget. Differential-test small traces against a Python reference. Report it as a simulator for concurrency/capacity questions, not measured llama.cpp/vLLM/SGLang performance.
- **vLLM comparison:** a separate pinned environment/model quantization/server manifest, using automatic prefix caching. Do not install it into the core project or compare different models/quantizations as a cache-only result.
- **Acon/AppWorld/OfficeBench:** a stretch adapter with its own environment and eligibility report, after the own suite and LongMemEval are complete.
- **SIMD log shaping:** only after CPU profiles identify the reducer as a hotspot; differential-test against the Python reducer and retain restorability tests.

None of these delays the core seven-mode matrix.

## Tasks

1. Complete and validate the own-suite families, negative controls, memory initialization snapshots, and exact evidence labels.
2. Implement seven configuration builders and assert all non-intended controls are identical across rows.
3. Implement per-unit manifests, claims/resume, cache-state controls, paired execution, and infrastructure/context failure accounting.
4. Implement all quality, retention, token/cache/latency, and compaction-cost metrics with hand-computed fixtures.
5. Implement record-only Markdown/CSV/SVG reports with untested/ineligible/did-not-fit cells and first-failure links.
6. Pin and inspect LongMemEval and context-rot upstream formats; implement small local fixtures before any downloads/runs.
7. Add notebook CPU evaluation gates and live `tidemark-ablations`, `tidemark-longmemeval`, and `tidemark-context-rot` cells with deterministic subsets and per-GPU mirrors.
8. Implement the utilization-aware but frozen-for-scoring scheduler, artifact reuse, balanced mode ordering, and resource telemetry. Autotune on T4 and L4 before the full same-id suite and include phase tasks/minute and ready-queue occupancy in the report.
9. Keep optional C++/vLLM/Acon/SIMD work outside core acceptance and outside the primary result table.
10. Before the GPU run, commit the CPU-tested evaluation implementation/datasets and pre-GPU Handoff, automatically push both and verify remote `main`. Part 8 may then start; the user reviews GPU results and fixes separately without another Handoff or agent bug-fix cycle.

## Tests

- Dataset loaders reject duplicate ids, invalid refs, impossible gold, missing evidence labels, wrong namespaces, and secret-like payloads.
- Configuration tests prove that only intended features differ between rows, including identical candidates for the source-check ablation.
- Hand-computed metrics cover simultaneous model failure + evidence present, correct guess + evidence missing, partial pairing, zero denominators, cold/compaction steps, and infrastructure abort.
- Cache-state tests reject an unpaired initial state, changed slot shape, or different background concurrency in a timing comparison.
- Resume refuses changed mode/config/dataset/model and never counts a torn mirrored result as complete.
- LongMemEval fixture checks evidence-id mapping, Recall/NDCG, undated queries, supersession, local-judge labeling, and subset identity.
- Context-rot fixture checks length/position controls and predeclared knee calculation without choosing a threshold from the observed best result.
- Report fixtures show partial runs, did-not-fit, unpaired ids, negative results, and raw numerator/denominator without invented numbers.
- All prior unit/golden replay/security tests remain green.

Run:

```bash
.venv/bin/python -m pytest tests/test_tidemark_eval_datasets.py tests/test_tidemark_eval_runner.py \
  tests/test_tidemark_eval_metrics.py tests/test_tidemark_longmemeval.py \
  tests/test_tidemark_context_rot.py tests/test_tidemark_report.py
.venv/bin/python -m pytest
```

## Phase delivery and handoff

After all CPU evaluation tests pass and **before the user's separate GPU run**, follow part 1's pre-GPU publication protocol:

1. Stage only part 7's datasets/adapters/runner/metrics/report changes, sanitized fixtures, tests, notebook, README, and deliberate shared changes. Exclude generated results, public benchmark copies, caches, and unrelated dirty paths.
2. Create `Tidemark: implement part 7 paired evaluation`.
3. Rewrite `Tidemark/Handoff.md` with status `pushed_acceptance_pending`, implementation SHA, dataset/subset/eligible-id hashes, seven configuration identities, frozen capacity plans, initial cache/session controls, exact T4/L4 ablation cells, and required paired/resource/run evidence. Include part 8 as blocked and all untested/did-not-fit cells.
4. Create `Tidemark: hand off part 7 for GPU acceptance`, push both commits to `origin/main`, and verify remote `main` before the user starts.

Once the pre-GPU Handoff is remote-verified, part 8 may start. The user reviews pairing, metrics, resource evidence, and any failures separately; no further Handoff or agent bug-fix cycle is part of delivery. Incomplete/non-paired CPU fixtures, red tests, or publication failure blocks publication of the current implementation. Never delete ids to improve a delivered metric or commit raw private benchmark payloads.

## Acceptance

- The own-suite covers N=5 through N=40, large observations, compaction, and cross-session updates with task and evidence scores kept separate.
- The seven-mode runner is paired, resumable, and exposes every ineligible/unpaired/infrastructure/context-failure id.
- Quality, token/cache, TTFT/prefill, peak context, and full compaction overhead are reported from raw records with appropriate intervals and denominators.
- LongMemEval and context-rot adapters pass local fixtures and state exact upstream/subset/judge comparability before live execution.
- No score improves by deleting difficult cases, weakening gold, hiding infrastructure, or silently changing cache/model/server controls.
- The full core suite completes on both T4 and L4 with the same task ids and correctness checks. Each GPU uses its measured resource plan and shows model/CPU ready-queue utilization, bounded memory, and no unexplained scheduler stalls.
- The CPU-tested implementation and pre-GPU Handoff are committed and automatically pushed before the user's run, with remote `main` verified and the part 8 bootstrap plus run/result/resume context included. Part 8 may proceed while the user independently reviews GPU results and fixes; no further Handoff or agent bug-fix cycle is required.

## Pitfalls

- Reporting token reduction while task success or evidence availability falls, without presenting both.
- Giving each ablation a different compactor candidate, task subset, initial cache state, or model/server shape.
- Counting a correct guess as memory recall or an available-but-unused fact as proof of model understanding.
- Omitting summary generation/judge/post-compaction re-prefill from compaction cost.
- Publishing a deterministic subset as full LongMemEval or comparing a local judge with the paper's GPT-4o judge without a caveat.
- Replacing a full-history baseline that did not fit with a truncated one under the same label.
- Tuning the soft limit on the same context-rot or task ids used for the final claim.
- Presenting a C++ cache simulation as a real GPU cache-hit measurement.

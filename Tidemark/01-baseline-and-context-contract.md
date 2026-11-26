# Part 1 — Baseline telemetry and the context contract

**Track:** [Tidemark, Day 4 of 10 (Nov 10, 2026)](https://agent-track-day-4-of-10-nov-10-2026-4gdn.pages.grok.bot/)
**Milestone:** M0, evenings 1–2
**Package:** `tidemark`, a sibling of `toolsmith`, `steadyhand`, and `bulkhead` under `src/`
**Depends on:** the implemented Toolsmith shim, Steadyhand loop/journal/replay, and Bulkhead policy adapter
**Produces:** a naive-context baseline, exact prompt/cache/latency telemetry, a versioned long-horizon task seed set, and the compatibility contract every later part must preserve

This file is part 1 of 8. Implement the parts in order and stop at each acceptance gate. A target in the project brief is not a result; no cache-hit, latency, recall, or task-success number may be published without its raw records and manifest.

| Part | File | Deliverable |
|---|---|---|
| 1 | `01-baseline-and-context-contract.md` | Naive baseline, telemetry, context/replay contract |
| 2 | `02-prefix-stable-context-and-budgets.md` | Five prompt segments, real-token budgets, PrefixLint |
| 3 | `03-observation-shaping-and-blob-store.md` | Typed reducers, content-addressed blobs, `read_blob` |
| 4 | `04-cited-compaction-and-replay.md` | Grammar-constrained, source-checked epochs and replay |
| 5 | `05-long-term-memory-and-retrieval.md` | SQLite/FTS5/vector store, supersession and deletion |
| 6 | `06-secure-memory-tools-and-session-resume.md` | Bulkhead-governed tools, persistent sessions, safe replay |
| 7 | `07-long-horizon-evaluation-and-ablations.md` | Own suite, seven-mode ablation, public benchmark adapters |
| 8 | `08-live-runs-results-and-handoff.md` | Colab runs, generated report, limitations and handoff |

## Outcome

Measure the existing agent before optimizing it. The baseline is **append every message, then truncate the oldest complete turns only when the next request cannot fit**. Record what was rendered, what llama-server actually evaluated, and whether the task succeeded. The first useful artifact is a per-step CSV and a prefill-by-step chart, not a new context algorithm.

This part also establishes the central compatibility rule: with `context=None`, Steadyhand behaves and replays exactly as it does now. With Tidemark enabled, every model request still goes through `ToolsmithService`; Tidemark chooses the messages and measures the rendered prompt, while Toolsmith remains responsible for model-owned templating, native tool-call parsing, grammars, validation, and repair.

## What already exists

| Existing component | Current responsibility | Tidemark boundary |
|---|---|---|
| `src/steadyhand/loop.py` | Appends OpenAI-shaped history and calls `client.complete` before every model turn | Add one context-render seam before the call and one observation-shaping seam after a result. Route the optional final paraphrase through the same seam. |
| `src/steadyhand/client.py` | Builds `ChatCompletionRequest`, calls `ToolsmithService.chat`, classifies `ModelTurn` | Extend telemetry without creating a second model protocol or executing tools in the shim. |
| `src/toolsmith/prompting.py` | `normalize_history` and `ChatPromptRenderer` apply the model's Jinja template | Reuse the same renderer for budgeting and PrefixLint. Do not concatenate a homemade Qwen prompt. |
| `src/toolsmith/api.py` | Renders the prompt, calls raw `/completion`, parses/validates, and performs at most one repair | Preserve that pipeline and its no-tool handler boundary. Expose backend timings as additive metadata. |
| `src/toolsmith/backends/llama_cpp.py` | Raw non-streaming and streaming llama.cpp calls | Preserve raw numeric timing fields. A true TTFT needs a first streamed token; a non-streaming elapsed time is not TTFT. |
| `src/steadyhand/types.py` | `ModelTurn` currently carries token usage and total elapsed time | Add optional typed metrics with defaults so old fixtures remain valid. |
| `src/steadyhand/journal.py` | Strict, contiguous JSONL events, redaction, and field-level replay comparison | Add Tidemark events without changing required fields on old events. Old journals need no migration when the feature is off. |
| `src/steadyhand/manifest.py` | Identity, claims, resume, and server-shape checks | Context configuration is an optional identity block. Server/cache properties that affect resume are checked separately. |
| `src/steadyhand/replay.py` | Serves recorded turns, verifies the exact request-message hash, and replays safe handlers | Hash the messages actually sent after context rendering, not an unrendered logical history. |
| `src/steadyhand/eval/suite.py` | Store-diff task scoring, claims, resume, and infrastructure accounting | Reuse these mechanics; do not score a long-horizon task by prose similarity. |
| `src/toolsmith/evaluation/metrics.py` | `percentile`, `wilson_interval`, `paired_bootstrap`, and deterministic subsets | Reuse rather than copy. |
| `notebooks/04_tidemark.ipynb` | New project notebook with shared setup and a CPU starting gate | Add the baseline cells here. Do not grow another project's notebook or claim the scaffold is a live result. |

The current `usage.prompt_tokens` is derived from llama.cpp's `prompt_n`/`tokens_evaluated`; it must not automatically be treated as the full rendered prompt length under cache reuse. Tidemark keeps **rendered tokens**, **newly evaluated tokens**, and **cached/reused tokens** as distinct measurements.

## What this part adds

```text
src/tidemark/__init__.py
src/tidemark/types.py                 # ContextConfig, TurnMetrics, SegmentBudget
src/tidemark/telemetry.py             # Sanitized backend fields and derived counters
src/tidemark/baseline.py              # Naive append/truncate-oldest strategy
src/tidemark/tokenizer.py             # Tokenizer parity probe and Tokenizer protocol
src/tidemark/runtime.py               # HardwareEnvelope, ResourceBudget, runtime sampler
src/tidemark/scheduler.py             # Shared CPU/I/O pools, slot leases, bounded prefetch
src/tidemark/eval/__init__.py
src/tidemark/eval/baseline.py         # Resumable baseline runner and CSV records
src/tidemark/eval/serving.py          # T4/L4 capacity plans and measured autotuning
src/tidemark/eval/report.py           # Initial SVG/CSV report from raw samples
src/steadyhand/context.py             # Optional loop adapter; legacy path delegates unchanged
evaluation/data/tidemark_long_horizon_v1.jsonl
tests/test_tidemark_baseline.py
tests/test_tidemark_telemetry.py
tests/test_tidemark_tokenizer.py
tests/test_tidemark_runtime.py
tests/test_tidemark_scheduler.py
tests/test_tidemark_serving.py
tests/test_steadyhand_context.py
tests/fixtures/tidemark/
```

Use only declared core dependencies for the runtime. CSV and the initial SVG chart can be written with the standard library; do not add pandas or matplotlib merely for the baseline. Colab-only model/tokenizer packages remain in the existing `colab` extra.

## Colab hardware and utilization contract

Tidemark must run the same core suite on **both Colab T4 and L4** without editing GPU-specific constants in notebook cells. “Maximum utilization” means maximum **useful throughput with no avoidable idle resource while work is ready**, not busy loops or a promised 100% on both processors at every instant. Sequential model/tool dependencies, server fitting, and durability drains have unavoidable idle phases and must be labeled as such.

### Detect an envelope, do not branch on a GPU name

`HardwareEnvelope` extends the existing `toolsmith.hardware` values with total/free VRAM, vCPUs, RAM/free RAM, local free disk, compute capability, and measured llama.cpp capabilities. Derive a `ResourceBudget` from those values and a short probe. GPU names select only starting candidates; actual memory and measurements decide what runs.

These are **starting candidates, not promises or benchmark shapes**:

| Resource | T4 candidate | L4 candidate |
|---|---|---|
| Typical Colab envelope used for test fixtures | 2 vCPUs, about 13 GB RAM, 14–16 GiB VRAM | 8 vCPUs, about 54 GB RAM, 22–24 GiB VRAM |
| Primary model | Qwen3-8B Q4_K_M | Same model, so quality comparisons do not change weights |
| Per-slot context | 16K first; 32K tall only after the memory probe | 16K wide or 32K tall after the same probe |
| Agent slots | 2 at 16K, shrink to 1 before reducing per-slot context | 4 at 16K or 2 at 32K, adjusted to measured VRAM |
| Compactor | Shared slot lease first; test a dedicated second slot | Test one dedicated slot versus a shared lease |
| Setup-only CPU pool | At most 2 workers, also bounded by RAM | At most 8 workers, also bounded by RAM |
| Live CPU work | Small shared pool after reserving llama.cpp/orchestration capacity | Larger shared pool; never one pool per episode |

Qwen3.5-9B is an optional separate profile and must pass the same fitting probe; it is not required for the claim that the core Qwen3 suite works on both GPU families. A tall task that does not fit after reducing to one slot is `did_not_fit`, never silently truncated. The required own-suite must have a proven 16K path on both GPUs; public full-history baselines that cannot fit remain explicit unsupported cells.

`CapacityPlan` records model/config-derived KV bytes per token, weights/runtime overhead, `per_slot_context`, total `-c`, agent/compactor slots, KV/flash settings, batch/ubatch, llama.cpp threads, CPU process workers, I/O threads, embedding workers/threads, RAM/disk ceilings, and prefetch depth. Keep the existing free-VRAM margin and a measured RAM margin; never trade a faster candidate for swapping or host OOM.

### Two phases: autotune, then freeze for measurement

1. **Unscored fitting/autotune:** try the KV/flash ladder, slot counts at fixed per-slot context, batch/ubatch candidates, llama.cpp thread counts, CPU pool partitions, compactor-slot policy, and prefetch depth on a small representative workload. Reuse `fit_server` for model fit and decode sanity, then compare end-to-end tasks/minute, prompt tokens/sec, decode tokens/sec, queue delay, and resource samples.
2. **Scored execution:** freeze the selected complete configuration in the manifest. No dynamic slot, worker, KV, cache, batch, or scheduler changes mid-comparison. A resource failure starts a new manifest segment and excludes cross-shape latency comparisons.

Cache candidate capacity plans by hardware/build/model/config hash, but rerun a short health/throughput confirmation each Colab session. T4 and L4 records remain in separate per-GPU trees. Compare latency only within the same GPU and identical frozen shape; never claim an L4-versus-T4 speedup from different task sets or tuning policies.

### Work-conserving scheduler

Use one scheduler per notebook run, shared by all episodes and phases:

- **Model-slot semaphore:** at most `usable_slots` agent + compactor + judge calls can enter llama-server. There is no unbounded HTTP/server queue. `id_slot` leases make the allocation explicit.
- **Episode workers:** may exceed slot count because they also execute tools, shape observations, and wait on memory. They prepare work ahead but acquire a slot only for a model call. Start with `slots + max(1, slots // 2)` and tune it; this is not permission to issue extra simultaneous completions.
- **Slot affinity policies:** `pinned` retains an episode's slot for cache-measurement runs; `work_conserving` allows another ready episode to use a slot during a long tool/I/O wait and records slot migration/cache eviction when the first returns. Both are valid for different goals, never mixed within a paired run.
- **Shared CPU pool:** setup gates can use all RAM-safe vCPU workers before server fitting. During live execution, use a smaller resource-budgeted process pool for shaping, lexical source checks, FTS/indexing, dataset preparation, and reports. Do not nest process pools or launch a pool per episode.
- **I/O pool:** bounded threads handle local blob reads, journal writes, and atomic Drive copies. Drive never blocks a model-ready worker. Required current-step shaping still completes before that step's next model call.
- **Embedding service:** one CPU-resident model per planned embedding worker, bounded batch/queue, and capped Torch threads. Avoid duplicating the model in every process or loading it on the GPU by default.
- **Bounded prefetch:** prepare initial contexts, tokenize cached segments, load required local artifacts, and index upcoming memory corpora ahead of the GPU. Bound ready work by RAM and preserve episode order where replay depends on it.

A model call always wins its slot before its episode's CPU result is needed. Conversely, CPU results already required by a running episode are not deprioritized merely to start more episodes. Backpressure must prevent memory growth and starvation as well as maximize occupancy.

### Resource telemetry and utilization gates

Write a volatile `resource_samples.jsonl` beside each run, separate from deterministic replay events. Sample total/per-process CPU, load average, RAM, local disk, GPU utilization/memory, llama.cpp slots busy, model/episode/CPU/I/O queues, active tool calls, and Drive-copy lag. Use `/proc` and `os` APIs plus one managed `nvidia-smi` sampler; do not spawn a fresh GPU query per model step or read process environments/command lines.

Report utilization only when the relevant queue has ready work:

```text
gpu_slot_occupancy = busy_slots / usable_slots               # model-ready windows
cpu_ready_utilization = nonidle allocated CPU time / capacity # CPU-ready windows
ready_but_idle = a free slot + ready model call, or idle CPU budget + ready CPU work
```

Autotune accepts the candidate with the best measured end-to-end throughput subject to fit/correctness limits. The smoke target is at least 95% median slot occupancy during model-ready windows and at least 85% allocated-CPU utilization during CPU-ready windows, with no unexplained sustained ready-but-idle window after warmup. These are tuning targets, not fabricated results; if a workload cannot reach them, preserve samples and explain whether the bottleneck is sequential dependencies, CPU starvation, GPU saturation, memory pressure, or I/O. Do not generate artificial work to hit a percentage.

## Context and model-call contract

### One optional adapter, no parallel agent loop

Introduce `ContextSession` in `steadyhand.context` with a small protocol implemented by Tidemark:

```python
rendered = session.render(history, step=step, budgets=controller_state)
turn = client.complete(rendered.messages, rendered.tools, tool_choice=choice)
```

`RenderedContext` carries the exact request messages, frozen tools, rendered prompt hash/token count, segment accounting, epoch, and lint evidence. The raw logical history remains available for journal reconstruction and `EpisodeResult.messages`; it is not assumed to equal the prompt after compaction.

All model-originated calls use this seam, including parse-error follow-ups and `_paraphrase`. Compaction uses a separate structured-output client in part 4, never a hidden extra tool-enabled turn inside `run_episode`.

When no session is supplied, the existing `history`, registry schemas, request hash, event order, controller behavior, golden episode ids, and manifests stay unchanged. Do not quietly enable Tidemark by installing its package.

### Tokenizer parity

Define a tokenizer protocol with `encode(text) -> tuple[int, ...]`. CPU tests use a deterministic fake. Live runs use the model's real tokenizer and verify it against the pinned llama-server's `/tokenize` behavior on a corpus containing:

- empty and non-empty system prompts, native tool schemas, and generation markers;
- non-ASCII text, combining characters, emoji, newlines, and invalid-surrogate rejection;
- strings adjacent to role/tool markup where token merges matter;
- prompts before and after a planted recitation tail.

Record model repository/revision, GGUF hash, tokenizer/template hashes, llama.cpp tag, and token counts. A parity mismatch blocks token-based acceptance; it is not permission to fall back to character counts. Cache repeated tokenization by content hash only after parity passes.

### Latency without false precision

Add `TurnMetrics` with optional fields rather than overloading `elapsed_s`:

- `rendered_prompt_tokens`, `prompt_evaluated_tokens`, `completion_tokens`;
- `prompt_ms`, `predicted_ms`, `response_elapsed_ms`;
- `ttft_ms`, nullable, and the measurement path (`buffered_stream` or `unavailable`);
- slot id when explicitly assigned, repair attempt count, and sanitized raw llama.cpp timing fields needed to audit derivations.

For the primary paired evaluation, add an opt-in **buffered-stream** path inside `ToolsmithService`: it uses the existing streaming backend, records the first non-empty upstream content timestamp, assembles the response, then uses the same parser, validator, and bounded repair policy. No deltas escape to Steadyhand, so the loop still receives one `ModelTurn`. Record every attempt separately. If this path is not enabled or a response has no content event, `ttft_ms` is null; report `prompt_ms` as prefill time, never rename total elapsed time TTFT.

The streaming/non-streaming generation path is a manifest identity field. All rows in a paired comparison use the same path. A live parity smoke must precede scored use; CPU equivalence with a scripted backend is not evidence of model-quality equivalence.

### Cache metric definitions

Keep these formulas in one module and test them with hand-computable examples:

```text
reused_tokens = max(0, rendered_prompt_tokens - prompt_evaluated_tokens)
cache_reuse_fraction = reused_tokens / rendered_prompt_tokens  # zero denominator -> null
prefill_savings_fraction = 1 - prompt_evaluated_tokens / naive_evaluated_tokens  # paired only
```

Confirm the exact meaning of `prompt_n`, `tokens_evaluated`, and any cache fields on the pinned llama.cpp build with a repeated-prompt A/B probe. Do not import semantics from llama.cpp HEAD documentation onto b5401/b8233. If full rendered length cannot be reconciled with server counts, mark the affected cache metric unavailable and retain the raw evidence.

Report first-step cold cache separately from later steps and compaction steps separately from ordinary steps. Do not average them into one flattering hit rate.

## Baseline behavior

`NaiveAppendTruncate` preserves the existing message order and content until the hard slot budget is exceeded. It then:

1. Preserves the initial system message and the current unanswered tool-call/result group.
2. Drops oldest **complete turns**, including an assistant's calls and every corresponding tool message. Never leave an orphan tool result or unanswered call.
3. Uses exact rendered prompt tokens plus `max_tokens` and a configured safety margin, not message characters.
4. Records every dropped message range and resulting token count.
5. Aborts with `context_budget_exhausted` if the required current turn itself cannot fit; it never deletes the user's current request or silently reduces `max_tokens` mid-run.

This deliberately lossy baseline is not the prefix-stable implementation. Sliding-window and all other modes are evaluation configurations in part 7.

## Journal and manifest additions

Add optional events only on Tidemark runs:

### `context_render`

Written immediately before the corresponding `model_turn`:

```text
step, epoch, strategy, context_config_sha256, messages_sha256,
prompt_sha256, prompt_token_ids_sha256, rendered_prompt_tokens,
segment_budgets, segment_tokens, max_tokens, slot_headroom,
epoch_start_seq, history_end_seq, recitation_sha256,
tool_schema_sha256, compaction_pending
```

Store hashes and counts, not megabytes of token ids or raw prompts. Raw prompts remain available only through explicitly redacted Toolsmith request records when needed for debugging.

### `backend_timings`

Written immediately after the corresponding `model_turn`:

```text
step, attempt_metrics, rendered_prompt_tokens,
prompt_evaluated_tokens, reused_tokens, cache_reuse_fraction,
ttft_ms, prompt_ms, predicted_ms, response_elapsed_ms,
slot_id, measurement_path
```

Timing values are volatile during replay. Recorded counts and measurement paths are reproduced from the recorded `ModelTurn`; missing fields remain null rather than becoming zero.

### Manifest identity and resume fields

When context is enabled, include the full canonical `context_config` and its hash in identity: strategy, segment caps, hard/soft limits, tokenizer/template versions, observation/compaction/memory configuration as later parts add them, and evaluation generation path. Include `tidemark_version`.

Model/GGUF, llama.cpp tag, slot context, KV type, cache flags, and slot-save configuration are resume-checked where relevant. A changed ablation or model gets a new run label/id; never mix records under one manifest.

Old manifests omit the block and retain their existing ids. Adding optional events does not require a journal version bump; changing an existing required field does.

## Long-horizon seed dataset

Create a small, strict v1 dataset that part 7 expands rather than waiting until evaluation to define success. Each task contains:

```text
task_id, version, category, goal, initial_memory,
scripted_steps, planted_fact, needed_at_step, distance_N,
distractors, observation_ids, expected_tools, gold_store_diff,
required_availability, possible, seed
```

Cover at least one file path, exact error code, and user constraint needed after N=5, 10, 20, and 40 steps. `required_availability` is `inline`, `blob_handle`, `memory`, or `any_restorable`. The scorer checks both final store/task objectives and whether the exact evidence was available when needed. A model can fail even with perfect context; report evidence availability separately from task success.

Validate unique ids, distances, known tools, valid source references, satisfiable gold diffs, and no secrets. Keep large observation payloads in the part 3 corpus and reference them by id rather than duplicating them in every task.

## Tasks

1. Add the package/types and the optional Steadyhand context seam with a legacy-no-op adapter. Route final paraphrase through it as well.
2. Add additive model/backend metrics and strict numeric sanitization. Preserve raw evidence without logging credentials or request headers.
3. Implement tokenizer parity fixtures and the live probe CLI; make parity failure explicit in the notebook and manifest.
4. Implement the buffered-stream TTFT path inside Toolsmith, with parser/validator/repair parity tests, or leave TTFT explicitly unavailable until it lands. Do not measure TTFT in Steadyhand by timing non-streaming `chat()`.
5. Implement naive complete-turn truncation and exact slot-headroom checks. Keep current controller behavior unchanged when context is disabled; when enabled, use actual rendered length rather than assuming history only grows.
6. Add the journal/manifest extensions and replay support for the rendered message hash and telemetry. Golden legacy journals must remain byte-identical.
7. Implement the baseline runner, per-step CSV, and simple SVG chart with claims, resume, infrastructure categories, and no hardcoded results.
8. Implement `HardwareEnvelope`, `CapacityPlan`, the shared scheduler/slot semaphore, resource sampler, and T4/L4 fake-hardware capacity tests. Add `tidemark-capacity-probe` and `tidemark-utilization-smoke` before any long baseline.
9. Add the seed dataset and validator. Add `tidemark-baseline` and `tidemark-baseline-report` cells to `notebooks/04_tidemark.ipynb`; retain its existing CPU starting gate.
10. After all CPU checks pass, automatically create the implementation commit and a `pushed_acceptance_pending` Handoff containing the exact T4/L4 review instructions and part 2 bootstrap, push both to `origin/main`, and verify the remote SHA. The user reviews GPU results and any failures separately; part 2 does not wait for another Handoff or an agent bug-fix cycle.

## Tests

- Legacy golden episodes and manifests replay unchanged, and no Tidemark event appears when context is off.
- A fake renderer proves the hash is over the exact messages sent to `complete`, including a replaced recitation tail rather than the raw logical history.
- Complete-turn truncation never produces invalid history for `ToolsmithService._validate_messages`; oversized current turns fail explicitly.
- Tokenizer parity fixtures cover Unicode, markup boundaries, template revisions, and mismatched token ids.
- A fake streaming backend verifies first-content TTFT, empty-chunk handling, truncation, native parse/schema validation, and at most one repair. Non-streaming TTFT remains null.
- Cache formulas handle full hits, full misses, partial reuse, zero tokens, repairs, and unknown server fields without negative counts or fabricated zeros.
- Journal tampering with rendered messages, prompt hashes, configuration, or measured token counts raises `ReplayMismatch` at the exact sequence/field; only declared timings are volatile.
- Baseline resume skips complete matching units and refuses a changed dataset/config. Infrastructure aborts stay visible and out of task-success denominators.
- The report reads raw CSV/JSONL samples and marks unrun cells as untested.
- Fake T4 and L4 envelopes produce legal capacity plans; slot shrink preserves per-slot context, and RAM/disk/VRAM ceilings reject unsafe candidates before launch.
- Scheduler fakes prove model calls never exceed slots, episode workers can prepare/run tools concurrently, queues stay bounded, every slot is filled when model work is ready, and pinned versus work-conserving affinity is recorded.
- Resource samples distinguish model-ready, CPU-ready, sequential, setup, and drain windows. Utilization reports do not count idle time with no ready work as a scheduler failure.
- Autotune freezes the selected configuration for scored runs; a mid-run shape change creates an incompatible segment rather than contaminating timing pairs.

Run:

```bash
.venv/bin/python -m pytest tests/test_tidemark_baseline.py tests/test_tidemark_telemetry.py \
  tests/test_tidemark_tokenizer.py tests/test_steadyhand_context.py tests/test_steadyhand_replay.py
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src tests
```

The user performs the live tokenizer probe and small model baseline separately from `notebooks/04_tidemark.ipynb`, following the pre-GPU Handoff. Imports, fake timings, or an agent's unverified claim of a GPU run do not satisfy that gate.

## Phase delivery and pre-GPU Handoff

Every part has **one implementation Handoff**, written and published after code/CPU verification and **before the user's separate GPU run**. GPU review and any findings are owned by the user as a separate activity; they do not create another Handoff, an agent bug-fix cycle, or a gate on the next numbered implementation phase.

### 1. Preflight and CPU verification before the GPU run

1. Start from `main`, read the current Handoff, and use `git pull --ff-only origin main` only with a clean worktree. Never overwrite or discard existing work.
2. Inspect `git status --short --untracked-files=all` and separate phase-owned files from unrelated/user changes. Never use `git add -A`, `git add .`, `git reset --hard`, `git clean`, or a force push.
3. On the exact candidate tree, run the phase-specific CPU tests, full `.venv/bin/python -m pytest`, `compileall`, `git diff --check`, and all notebook CPU gates. Record current counts/skips. Remove only test-generated files that are not deliverables and update stale comments/docs.
4. If a required CPU check is red or unexpected edits appear in phase-owned files, stop. Do not publish the phase as ready for GPU review. Keep generated results, models, blobs, DBs, slots, notebook outputs, secrets, and raw private prompts out of Git.

### 2. Implementation commit after CPU tests

Stage only explicit phase-owned source, test, dataset, notebook, README, and plan paths. Review `git diff --cached` and `git diff --cached --check`, including deliberate shared changes, and exclude unrelated dirty paths.

```bash
git add -- <exact-phase-paths>
git diff --cached --check
git commit -m "Tidemark: implement part N <short outcome>"
git rev-parse HEAD  # implementation_commit for the pre-GPU Handoff
```

The commit contains the code and tests that actually passed CPU verification. Any later implementation change before the GPU run requires another CPU test run and an updated implementation/pre-GPU Handoff.

### 3. Write and commit the pre-GPU Handoff

Atomically rewrite **`Tidemark/Handoff.md`** **before the GPU run**. Its status is `pushed_acceptance_pending`, which means live review is pending—not that implementation of the next part is blocked. Include:

- exact notebook path and ordered cell ids to run on fresh T4 and L4 runtimes;
- model/build revisions and recommended fitted/capacity settings, with any choices the user must make explicit;
- setup, authentication-by-name, resume, Drive-mirroring, and cleanup instructions;
- the T4/L4 review checklist and artifacts worth retaining: run ids, manifests, journals, resource samples, metrics, report paths, and known untested cells;
- current implementation limitations and optional features not requested;
- the next numbered phase's bootstrap, which becomes available as soon as this Handoff is remote-verified.

Commit it separately so it can name the implementation SHA without a self-reference:

```bash
git add -- Tidemark/Handoff.md
git diff --cached --check
git commit -m "Tidemark: hand off part N for GPU acceptance"
```

The file is operational state, not a ninth plan part. A fresh Colab session must be able to pull `main`, read it, and run without the preceding chat or an old notebook kernel.

### 4. Push the pre-GPU Handoff to GitHub

Push the implementation and pre-GPU Handoff commits in one authenticated push, then verify remote `main` equals the local Handoff commit:

```bash
git push origin main
test "$(git ls-remote origin refs/heads/main | cut -f1)" = "$(git rev-parse HEAD)"
git status --short --branch
```

Use existing authentication and never put a token in a command/file/log. If the push is rejected as non-fast-forward, authentication fails, or remote verification differs, keep the local commits and stop; do not automatically rebase, merge, force, or retry indefinitely. The user should not start from an unverified remote Handoff.

### 5. Independent user GPU review

The user follows the pre-GPU Handoff on T4 and L4 and reviews results and failures separately. The agent does not schedule another Handoff, verify returned evidence, or perform a follow-up bug-fix cycle as part of phase delivery. If the user later asks for a specific GPU issue to be addressed, that is a separate task with its own tests and commits.

Once the pre-GPU Handoff is remote-verified, the next implementation phase may start immediately. Live results remain unclaimed until they exist, and the Handoff continues to say GPU review is pending; it is not converted into a separate completion workflow.

### Required pre-GPU Handoff content

1. **Status:** `pushed_acceptance_pending`, base/implementation/previous Handoff commits, branch/remote, and UTC update time. Identify the commit containing the Handoff with `git log -1`, not by embedding its own SHA.
2. **Delivered scope:** behavior, interfaces, invariants, changed paths, schema/journal/manifest versions, and compatibility constraints.
3. **CPU verification:** exact commands, current counts/skips, compile/diff checks, notebook CPU cells, and fake-backed coverage. State explicitly that T4/L4 evidence is pending user review.
4. **Run guide:** exact cell order, setup/cleanup, model/GGUF/tokenizer/llama.cpp revisions and capacity/scheduler settings, review checklist, and suggested artifacts to retain.
5. **Durable paths:** local/per-GPU Drive locations, checkpoint/memory/catalog revisions when available, and exact restore/resume commands. Never assume a process, mount, slot, or background job survives.
6. **Known issues and boundaries:** residual CPU failures/skips, unsupported claims, unrelated dirty paths, and excluded optional work.
7. **Next implementation action:** the next plan's setup and first test/cell after remote verification. The user's independent GPU review does not block it.
8. **Safety:** secret names only, never values; no authorization headers, environment dumps, hidden reasoning, raw private prompts, or unredacted memory payloads.

### New-session read protocol

Before changing files, read `Tidemark/Handoff.md` and the relevant numbered plan, check branch/log/status, and verify the implementation commit with `git merge-base --is-ancestor <sha> HEAD`. `pushed_acceptance_pending` means the user still has a separate GPU review to perform; it does not require the next agent to perform that review or wait for it. Start the next implementation gate shown in the Handoff. If repository and Handoff disagree, stop and surface the mismatch instead of guessing.

## Acceptance

- A naive baseline produces per-step rendered/evaluated token counts, prefill or TTFT with an honest measurement path, cache reuse where supported, response latency, and task outcome.
- `steps.csv` and `prefill_by_step.svg` are generated from records; cold, ordinary, and truncation steps are distinguishable.
- Every Tidemark model request still passes through Toolsmith, and no handler moves into the shim.
- Existing episodes, policies, faults, recovery, security decisions, and golden replays are unchanged when Tidemark is off.
- No real-model baseline is claimed until the notebook run and its manifest exist.
- The baseline and utilization smoke run on both a real Colab T4 and L4 with the same primary model and eligible task ids. Each has a legal fitted shape, resource samples, and a throughput rationale for its selected scheduler.
- Model-ready slots and CPU-ready workers are kept occupied without exceeding their budgets; any sustained idle/oversubscription window is explained in the report rather than hidden by an aggregate utilization average.
- The CPU-tested implementation and pre-GPU Handoff are committed and automatically pushed before the user's run, with remote `main` verified and the part 2 bootstrap included. The user owns GPU review and any fixes separately; no additional Handoff or agent bug-fix cycle is required before part 2.

## Pitfalls

- Treating llama.cpp `prompt_n` as total prompt length without an A/B probe.
- Calling end-to-end non-streaming latency TTFT, or dropping repair to make streaming comparable when only one mode changed it.
- Hashing logical history while sending compacted/rendered messages, making replay pass the wrong contract.
- Truncating between an assistant tool call and its tool result.
- Enabling new behavior globally and invalidating all previous episode ids and golden files.
- Using character length as a fallback after tokenizer parity fails.
- Hardcoding L4's slot/worker counts for a T4, or lowering per-slot context until long tasks silently become different tasks.
- Confusing more HTTP requests with GPU utilization; only slot-leased calls may enter llama-server, while other episodes do CPU/tool work.
- Oversubscribing vCPUs with nested pytest, tokenization, Torch, llama.cpp, and process pools, or filling RAM with prefetched histories and duplicate embedding models.
- Changing resource settings during a scored comparison, or treating 100% sampled utilization as more important than completed correct tasks per minute.
- Committing untested code, staging unrelated changes with `git add -A`, publishing generated secrets/results, or calling a phase complete before remote push verification.
- Putting the implementation SHA in the same commit before it exists; use the separate Handoff commit defined above.
- Making the next session reconstruct state from an old chat, a live Colab kernel, or local-only files instead of the committed Handoff and durable artifacts.

# Part 8 — Live Colab runs, results, limitations, and handoff

**Track:** Tidemark, Day 4 of 10
**Milestone:** M7, evening 12 plus reserved live-model runtime; optional evenings 13–14
**Depends on:** `07-long-horizon-evaluation-and-ablations.md`
**Produces:** real self-hosted-model smoke/sweeps, per-GPU durable evidence, `results/tidemark/RESULTS.md`, a measured README section, and explicit next-project interfaces

## Outcome

Run the complete system end to end through the new Tidemark notebook and write only what the records support. The final question is not “did the prompt get smaller?” but “which mechanism improved task success or evidence retention, at what prefill/latency/compaction/memory cost, under which exact server configuration?”

Local CPU correctness and live model acceptance are separate. A full pytest run, valid notebook JSON, or scripted cache response is not a live cache-hit, recall, or task-success result.

## What already exists

- Parts 1–7 provide the implementation, strict journals/manifests, paired runners, public benchmark adapters, and fake-backed CPU gates.
- `notebooks/04_tidemark.ipynb` is the project scaffold created alongside the other project notebooks. Its setup cells are shared with Toolsmith and its cleanup with Steadyhand; later runtime/server copies must remain byte-identical to their canonical cells.
- `steadyhand.eval.serving` provides `fit_server`, KV/flash-attention ladders, slot-count shrink, decode-speed probes, and explicit `context_size / parallel` accounting. Tidemark adds long-context/cache requirements rather than duplicating server fitting.
- `toolsmith.provision`, `background`, `drive`, and `artifacts` already share llama.cpp/GGUF caches, bounded jobs, per-GPU records, Drive flush, and atomic mirroring.
- The root README separates implemented local checks from unrun live evaluations. Tidemark must follow the same evidence standard and retain negative/did-not-fit results.

## What this part adds

```text
src/tidemark/eval/live.py                  # T4/L4 phase orchestration, autotune, utilization gates
src/tidemark/eval/report.py                # Final combined report and utilization charts from raw records
results/tidemark/RESULTS.md                # Generated only when real records exist; gitignored
tests/test_tidemark_live.py
tests/test_notebook_04.py                  # Expanded with each implemented cell
README.md                                   # Status stub first, measured cells only after runs
```

No generated results, model weights, embeddings, SQLite databases, blobs, slot snapshots, raw private prompts, or notebook outputs are committed. Small sanitized failure excerpts may become regression fixtures only after review.

## Notebook structure and shared-cell discipline

Build the notebook toward this ordered flow; keep only implemented cells active and make pending stages explicit rather than importing nonexistent modules:

1. `introduction`
2. `environment`, `drive-and-secrets`, `install-project`, `hardware`
3. `model-runtime-notes`, `steadyhand-runtime`
4. `tidemark-cpu-gate` (dataset validation, unit subsets, golden replay, memory/blob/compaction fixtures)
5. Part-specific CPU gates from parts 1–7
6. `tidemark-live-notes`, `steadyhand-server`
7. `tidemark-tokenizer-probe`, `tidemark-capacity-probe`, `tidemark-autotune`
8. `tidemark-cache-probe`, `tidemark-utilization-smoke`, `tidemark-smoke`
9. `tidemark-baseline`, `tidemark-ablations`
10. `tidemark-longmemeval`, `tidemark-context-rot`
11. `tidemark-session-smoke` and optional `tidemark-slot-smoke`
12. `tidemark-report`
13. `steadyhand-server-stop`, `cleanup`

The four setup cells match `01_toolsmith.ipynb`. `steadyhand-runtime`, `steadyhand-server`, `steadyhand-server-stop`, and `cleanup` match `02_steadyhand.ipynb`, as they already do in the Bulkhead notebook. If a shared cell needs a Tidemark feature, change the canonical cell and every notebook copy plus their tests deliberately; do not let versions drift.

`tests/test_notebook_04.py` checks cell ids/order, no outputs/execution counts, Python syntax, shared-cell equality, no secrets/destructive commands, module-global declarations, CPU-runnable cells, and the live-cell invariants. Notebook tests execute real CPU gate code with fakes only at external boundaries; they do not merely search for function names.

On a fresh runtime or after a disconnect, run all cells from the top. Cached builds/models restore, completed matching units resume, and pending cells state exactly which implementation part is missing. Do not use keep-alive tricks.

## Live serving plan

### Hardware-adaptive model and context

Start with **Qwen3-8B Q4_K_M on both Colab T4 and L4** through the existing `STEADYHAND_PHASE`/runtime machinery. The same primary model and task ids make the core suite reproducible on both. Qwen3.5-9B is a separate optional comparison phase with its explicit profile/protocol and memory probe; it is not silently substituted and is not required for T4 acceptance. Only one GGUF is resident at a time.

Every cell reads the detected `HardwareEnvelope`; there are no cells that work only when the GPU is an L4. Use these starting candidates, then let measured memory/throughput decide:

| Setting | T4 starting candidate | L4 starting candidate |
|---|---|---|
| Typical fixture envelope | 2 vCPUs, ~13 GB RAM, 14–16 GiB VRAM | 8 vCPUs, ~54 GB RAM, 22–24 GiB VRAM |
| Wide/tall per-slot context | 16K wide; 32K tall if proven | 16K wide; 32K tall if proven |
| Agent slots | 2 wide, 1 tall | 4 wide, 2 tall |
| Compactor policy to probe | Shared two slots versus 1 agent + 1 compactor | Shared slots versus one dedicated compactor slot |
| Setup CPU workers | `min(2, RAM-safe maximum)` | `min(8, RAM-safe maximum)` |
| Live worker policy | Resource-budgeted pools, no nested pools; tune llama.cpp and CPU thread partition | Same policy with more CPU capacity, not a fixed multiple of T4 |
| Prefetch/embedding queues | Small and RAM-bounded | Larger only while throughput improves and RAM headroom remains |

Derive the final per-slot context from long-horizon p95 rendered prompt + `max_tokens` + safety margin. `context_size = per_slot_context * total_slots`; never quote total `-c` as one episode's window. Preserve per-slot context when shrinking slot count. The required own-suite must run at 16K on both GPUs; a required 32K tall configuration is attempted at one slot before being marked `did_not_fit`. Never reduce context until a task is silently truncated.

Compute KV memory from the actual pinned model `config.json`, not only the brief's example:

```text
layers × KV heads × head_dim × 2 (K,V) × bytes_per_element = bytes per token
36 × 8 × 128 × 2 × 2 = 147,456 bytes ≈ 144 KiB/token (fp16)
```

Record actual values and q8_0/f16 settings. The brief's approximate 32K=4.5 GiB fp16 and ~115K=16 GiB figures are planning context, not local measurements. Leave room for Q4 weights, CUDA/runtime overhead, compactor capacity, and the existing free-VRAM margin. q8_0 is tried first but is not assumed faster on T4; keep the first KV/flash candidate that fits and passes the measured throughput/quality gates. Full ~115K-token history may be `did_not_fit` on either GPU while the required memory-based pipeline still runs normally.

### Autotune for wall time, then freeze

Add `tidemark-capacity-probe` and `tidemark-autotune` before scored runs. On each GPU independently:

1. Verify real `/tokenize` parity, model health, free VRAM, and the pinned llama.cpp cache/slot/timing API.
2. Fit the KV/flash ladder and slot count at fixed per-slot context using `fit_server`, preserving at least the free-VRAM margin and the decode sanity gate.
3. Try a small declared grid of batch/ubatch, llama.cpp thread count, model-slot count, compactor shared/dedicated policy, slot-affinity policy, episode workers, CPU process workers, Torch/tokenizer threads, I/O depth, and prefetch depth. Keep the grid bounded so tuning itself does not consume the notebook budget.
4. Run representative short episodes with long prompts, large observations, compaction, memory retrieval, and tool waits. Select by **correct completed tasks/minute and phase wall time**, using prompt/decode tokens/sec, p95 latency, queue wait, RAM/VRAM pressure, and resource occupancy as diagnostics.
5. Save every candidate and the selected `CapacityPlan` with its hardware/build/model/config hash. Rerun a short confirmation on resume even when a previous candidate is cached.
6. Freeze the complete winner before scoring. Any OOM/resource-driven change starts a new manifest segment and a different timing cohort; it is not an invisible mid-run adaptation.

No latency comparison mixes candidates, GPU families, background concurrency, or scheduler policies. T4 can be slower with fewer slots while still running the complete core suite correctly.

### Cache behavior and slot scheduling

Probe the pinned llama.cpp build for `--cache-prompt`, `--cache-reuse`, explicit `id_slot`, `/slots`, raw timing fields, and optional slot save/restore. Enable prompt caching explicitly where supported and record the actual flags; do not rely on “on by default.”

- A model-slot semaphore caps agent + compactor + judge calls at `usable_slots`. Episode workers may exceed slots to perform tools/CPU work, but additional requests cannot queue silently inside HTTP or llama-server.
- `pinned` keeps episode-to-slot affinity for cache experiments. `work_conserving` lets another ready episode use a slot during long tool/I/O waits and records migration/eviction. Autotune may choose either for a whole throughput run; cache-target measurements use `pinned`. Never claim both policies' benefits from one run.
- Test shared versus dedicated compactor slots on both GPUs. A dedicated compactor can reduce eviction but removes an agent slot; accept it only when whole-phase wall time improves. Local support/compaction judgments use the same resident model and count in cost.
- Run the planted timestamp/unsorted-key cache breaker first, then the fixed assembler. Cache-hit acceptance joins PrefixLint to actual prompt-evaluated counts, not a server flag or CPU render cache counter.
- Preserve identical cache/generation/scheduler settings across ablation rows. Cache reset/snapshot and execution order follow part 7's pairing rules. Optional vLLM/KV comparisons have separate manifests.

### CPU/GPU overlap for shorter notebook wall time

| Phase | GPU work | CPU/I/O work overlapped where safe |
|---|---|---|
| Setup/provisioning | None initially; llama.cpp build may use GPU only for validation | GGUF/Drive restores, dataset validation, golden replay, observation corpus checks, FTS/vector preprocessing |
| Server fitting/autotune | Fit/probe one model | Quiet for controlled probes; prefetch public datasets/artifacts only when the probe is not timing-sensitive |
| Agent baseline/ablations | Agent, compactor, and judge calls fill leased slots | Initial-context/token preparation, shaping, lexical checks, tools, blob reads, embeddings/indexing, next-task preparation, completed-phase reports |
| LongMemEval | Declared QA phase only | Build FTS/vector indexes and compute/cache embeddings ahead of QA; score retrieval records |
| Context rot | Increasing-length prompts in a separate frozen server phase | Prebuild/tokenize prompt bundles, analyze earlier lengths without changing the server mid-cell |
| Reporting/cleanup | Stop as soon as the final model queue drains | Generate reports, verify snapshots, drain Drive queues, flush/unmount; do not leave the GPU allocated for CPU-only cleanup |

Use one shared process pool and one bounded I/O pool, not nested pools per cell/episode. Before server fitting, setup-only CPU work can use all RAM-safe vCPUs. During live execution, partition vCPUs among llama.cpp, shared CPU workers, tokenizer/BLAS/Torch threads, and orchestration using the autotuned budget. T4's two vCPUs require particularly small pools and one embedding model instance; L4 scales only while useful throughput improves.

The embedding model defaults to CPU and never loads beside the GGUF without a separate spare-VRAM/throughput probe. Local immutable artifacts stay on SSD. Drive copies and unrelated report generation cannot stall a model-ready worker, but the current episode's required CPU result still has priority over speculative prefetch.

### Utilization telemetry and acceptance targets

Write `resource_samples.jsonl` and per-phase `utilization.json` with total/per-process CPU, load, RAM/disk, GPU utilization/memory, llama.cpp slots busy, model/CPU/I/O queues, active tools, slot migrations, and Drive lag. Use `/proc`/`os` metrics and one managed `nvidia-smi` sampler, not a new subprocess for every model step.

Measure only windows with ready work:

```text
gpu_slot_occupancy = busy_slots / usable_slots
cpu_ready_utilization = nonidle allocated CPU time / allocated CPU capacity
ready_but_idle = free model slot + ready model call, or idle CPU budget + ready CPU work
```

The autotune/utilization smoke targets at least **95% median slot occupancy during model-ready windows** and **85% allocated-CPU utilization during CPU-ready windows**, with no unexplained sustained ready-but-idle window after warmup. These are targets to investigate, not numbers to hardcode into results. No artificial busy work is allowed. If a workload is intrinsically serial, report the dependency and compare candidates by completed work/wall time instead.

### Run phases

1. Start llama.cpp/GGUF/tokenizer provisioning and Drive restores in the background while the complete RAM-safe CPU gate and public-data/index preparation run.
2. Stop CPU-saturated test pools for controlled server fitting. Start one server, verify health, decode speed, free VRAM, slots, and tokenizer parity, then run the bounded autotune grid.
3. Run utilization/cache/structured-output smoke: repeated prompt, planted divergence, full slot load, mixed CPU/tool work, grammar-valid summary, large observation + `read_blob`, tainted memory ASK, and exact replay. Freeze the selected plan.
4. Run a small paired smoke across all seven modes. Inspect raw requests/events, resource occupancy, and intended configuration differences before starting the long sweep.
5. Run the own-suite paired ablation with slot-leased model calls and CPU/IO overlap. Checkpoint after every task; use different task ids to fill slots rather than temperature-zero duplicate repeats.
6. Precompute LongMemEval FTS/vector indexes on CPU during an eligible GPU phase, then run the declared QA subset in its own frozen server phase.
7. Run the context-rot length/position sweep and validate any proposed soft limit on held-out own-suite ids. Generate earlier-phase reports concurrently only when CPU headroom exists.
8. Run logical session pause/resume. Optional slot save/restore and cache-aware compaction run only after the core matrix is complete.
9. Repeat the core smoke and own-suite on the other GPU family with its own autotuned plan. T4 and L4 evidence remain separate; cross-GPU timing is never presented as a like-for-like shape comparison.
10. Generate final reports from records, stop the model as soon as its queue drains, force all mirrors, flush Drive through shared cleanup, and disconnect when queues are empty.

The brief's roughly 15–25 L4-hour / 25–50-compute-unit estimate is planning context from user-reported rates, not a guaranteed price. Check the live Colab resource panel and record actual elapsed/runtime metadata on both GPU families. Do not print a current rate as though Google publishes a fixed table.

## Durability and failure recovery

Use:

```text
local active state:  /content/tidemark/
durable records:     MyDrive/toolsmith/records/<gpu>/tidemark/
```

A run-owned checkpoint set includes manifest, journal, per-step records, accepted summaries/candidate artifacts, blob catalog/objects, SQLite online-backup snapshot, and optional slot snapshot. Use local writes and bounded background atomic copies, forced at unit/phase end. A unit is not marked durable-complete until its required set is copied and verified.

Manifest identity includes code/dataset/config hashes, model/GGUF/tokenizer/template revisions, llama.cpp tag, context/slot/KV/cache settings, the complete frozen `CapacityPlan` (batch/threads/model and episode workers/CPU/IO pools/prefetch/affinity/compactor policy), generation path, seed/temperature, tool/policy/approval hashes, compactor/judge prompts, embedding/ranking identities, initial memory/blob state, and run label. Resume refuses mismatches rather than merging records. Volatile resource samples describe a segment but are never used to excuse a semantic replay mismatch.

Infrastructure connection/timeout/5xx/OOM/disconnect follows bounded retry and remains visible. A case that exhausts retries is `infrastructure_retries_exhausted`, not a successful task, attack defense, evidence miss, or legitimate no-call. Context budget/compaction failures, schema ineligibility, `did_not_fit`, and unsupported slot features have separate labels.

Drive writes never block a GPU worker indefinitely. `sync_drive` runs at long-cell/cleanup boundaries; copy errors and lag are reported. Raw prompts/results stay local unless the declared redacted evidence policy permits their mirror. `HF_TOKEN`, `LOCAL_KEY`, authorization headers, memory namespaces containing private identifiers, and real secret values never enter committed cells or reports.

## Final generated report

`src/tidemark/eval/report.py` reads baseline, own-suite, observation, compaction, memory, public-benchmark, session, and environment records to produce `results/tidemark/RESULTS.md`. It contains:

1. **Scope and architecture:** five segments, frozen tools, shaping/handles, cited epochs, hybrid memory, Bulkhead boundary, and what replay proves.
2. **Environment coverage:** host/GPU, model/GGUF, llama.cpp, context/slots/KV/cache, the frozen `CapacityPlan`, tokenizer parity, embedding model, CPU/IO concurrency, and every unrun/unsupported configuration on T4 and L4.
3. **Baseline:** per-step prefill-by-step chart and cold/warm behavior, with CSV sample counts.
4. **Seven-mode paired results:** task success, evidence availability/use, duplicate effects, exact ids and intervals, including negative results.
5. **Compute/latency/utilization:** rendered/evaluated/reused tokens, cache fraction by step class, mean/p95 TTFT where measured, prefill/response latency, peak context, tasks/minute, model-ready slot occupancy, CPU-ready utilization, queue delays, migrations, and RAM/VRAM pressure.
6. **Observation results:** real-token reduction and inline/restorable must-keep outcomes, with the first lost/corrupt evidence case.
7. **Compaction results:** unsupported entries before/after checking, false accepts/rejects, epoch/re-prefill cost, future/net savings, and judge limitations.
8. **Memory results:** Recall@5/@10, NDCG, time/supersession/deletion behavior, poisoning/ASK/utility, and embedding/ranking configuration.
9. **Public benchmarks:** exact LongMemEval/context-rot revisions, ids, subsets, judge comparability, and did-not-fit cells.
10. **Session results:** logical cold resume correctness and optional measured slot restoration, clearly distinguished.
11. **Failure taxonomy:** model parse/semantic, evidence missing/not used, context budget, shaping/blob corruption, unsupported summary, memory poisoning/staleness, policy ASK/DENY, infrastructure, and durability failures.
12. **Real trajectories:** bounded journal excerpts for at least one important residual failure of each kind that occurred, using event ids rather than private chain of thought.
13. **Limitations and handoff:** unrun cells, small-sample/pairing caveats, local-judge comparability, coarse taint, cache/server specificity, and Day 5/6/9 interfaces.
14. **Context references:** external papers/posts in a separate section with their own harnesses; none of their percentages appears as a Tidemark result.

Update the root README initially with “implementation/local checks status; live results pending.” After real runs, replace only the relevant status cells with generated figures and links to manifests/raw records. Do not copy a target (90% cache reuse, 100% must-keep) into the measured-results table.

## Failure review and regression policy

Preserve the first failing task id, mode, epoch/step, event refs, configuration hash, and bounded raw evidence for every residual failure. Distinguish:

- evidence was dropped by shaping versus available but ignored by the model;
- prefix divergence versus cold/evicted slot versus server counter misunderstanding;
- unsupported summary accepted versus supported summary falsely rejected;
- stale/superseded memory returned versus current evidence not indexed;
- policy approval cost versus genuine model/task failure;
- context budget exceeded versus backend truncation/timeout;
- local state valid but Drive mirror lagging versus logical state corrupt.

Any correctness/security bug becomes a permanent regression. Never delete a difficult observation, unsupported summary, poisoning case, or long task to improve the table. If an intended behavior change requires new golden journals, regenerate them through an explicit reviewed process and keep old-journal compatibility tests.

## Stretch goals, outside core acceptance

- **Cache-aware compaction:** compare the fixed 70% trigger with a cost model that compacts only when expected future prefill savings exceed compactor + judge + post-epoch re-prefill costs. Use measured prefill/decode rates and label future-call estimates; evaluate on held-out tasks.
- **Compactor distillation:** distill to a 1.7B–4B-class local model and report support false accepts/rejects, downstream task loss, and actual latency. One resident model at a time.
- **Sub-agent isolation:** run a search subtask in a fresh context and return a short cited digest, with its own policy/taint/journal lineage.
- **KV quantization:** compare q8_0/q4_0/f16 long-context quality and memory under separate manifests; do not assume memory savings preserve accuracy.
- **Slot snapshots:** complete part 6's optional path and compare cold logical resume versus restored-KV resume.
- **vLLM / cache simulator / Acon / SIMD:** follow part 7's separate-evidence rules.

Stretch results never fill missing core cells or delay fixing a core evidence/security failure.

## Handoff contracts

- **Day 5 planning:** consume typed `RecitationState`, not raw model thoughts. Plan items stay distinct from cited state and expose budget/open-item fields for deterministic decomposition.
- **Day 6 evaluation harness:** reuse versioned datasets, unit manifests, claims, raw JSONL/CSV, paired metrics, failure taxonomy, and infrastructure denominators.
- **Day 7 sub-agents:** treat a child context as a separate session/namespace with explicit result digest and provenance; do not share mutable prompt history or memory connections.
- **Day 9 latency/observability:** consume `context_render`, `backend_timings`, `observation_shaped`, `compaction`, `memory_operation`, and session checkpoint records. Timing fields remain separate from deterministic replay fields.
- **Bulkhead:** memory tools and retrieved content continue through existing policy/taint/approval events. Tidemark does not introduce a policy exception for “internal” databases.

## Tasks

1. Complete notebook cells and their structural/CPU tests incrementally; keep shared cells identical and pending stages explicit. Add T4/L4 fake-envelope scheduler tests and real two-GPU acceptance sections.
2. Implement live serving/cache/tokenizer/capacity probes, the bounded autotune grid, resource sampler, shared CPU/IO/model-slot scheduler, and all-phase manifests/durability/retry/cleanup orchestration.
3. Run CPU gates and real smoke calls on T4 and L4; inspect raw journals and resource samples before any long run. Freeze each GPU's selected `CapacityPlan` before scoring.
4. Run the paired own-suite, public subsets, session tests, and declared optional experiments without changing controls mid-run. Measure tasks/minute and ready-queue utilization while overlapping safe CPU/GPU work.
5. Generate the combined report and review first failures/residual limitations, including any idle/oversubscription window. Update README with measured cells only.
6. Preserve regression evidence, drain mirrors, stop the server, flush Drive, and leave no claimed result without a retrievable manifest.
7. Before the GPU run, commit the CPU-tested live/report implementation and pre-GPU Handoff, automatically push both and verify remote `main`. The implementation is then delivered; the user reviews live results and fixes separately without another Handoff or agent bug-fix cycle.

## Final phase delivery and canonical Handoff

After all CPU/generated-report checks pass and **before the user's separate GPU run**, follow part 1's pre-GPU publication protocol:

1. Stage only part 8's live orchestration/report code, final tests, notebook, README, reviewed plan updates, and sanitized regression fixtures. Keep generated results, model/data caches, DBs, blobs, slots, raw private prompts/memory, and unrelated dirty paths out of Git.
2. Create `Tidemark: implement part 8 live evaluation and results`.
3. Rewrite `Tidemark/Handoff.md` with status `pushed_acceptance_pending`, the implementation SHA, exact T4/L4 smoke/sweep/report cells, model/capacity/scheduler settings, required evidence, and remaining limitations. The track remains incomplete and no part 9 is invented.
4. Create `Tidemark: hand off part 8 for GPU acceptance`, push both commits to `origin/main`, and verify remote `main` before the user starts.

Once the pre-GPU Handoff is remote-verified, the implementation is delivered and maintenance/Day 5 work may proceed. The user reviews notebook/report records and both-GPU acceptance separately; no further Handoff or agent bug-fix cycle is part of delivery. Live metrics remain unclaimed until the user's separate review, and no part 9 is invented.

This phase-level Handoff is separate from the conceptual Day 5/6/9 handoff contracts: it is the concrete operational file a new coding session reads before changing the repository.

## Verification and acceptance

CPU and documentation checks:

```bash
.venv/bin/python -m pytest tests/test_tidemark_*.py tests/test_steadyhand_*.py \
  tests/test_bulkhead_*.py tests/test_notebook_04.py
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src tests
git diff --check
```

Live acceptance additionally requires the user's separate notebook run to produce real tokenizer/server/cache/grammar/model calls, a seven-mode smoke, capacity/utilization probes, and generated raw records **on both a Colab T4 and a Colab L4**, following the pre-GPU Handoff. Imports, fake pipelines, valid notebook JSON, or success on only one GPU family are insufficient for the cross-GPU completion claim.

- Every published metric links to raw records and a complete manifest; unrun/unsupported/did-not-fit cells remain visible.
- Paired results share controls and exact eligible ids, and negative quality/latency tradeoffs are reported rather than hidden.
- Cache claims join PrefixLint to actual server evaluation counts; TTFT comes from first streamed content, and other timings are labeled correctly.
- Observation/compaction/memory claims include their retention, factual-support, recall, and overhead evidence—not just smaller prompts.
- Logical resume works without KV snapshots; optional slot restore has separate measured evidence.
- The README distinguishes local checks from live results and contains no invented cache, accuracy, latency, recall, security, utilization, or Colab-cost figure.
- Both GPUs complete the core same-model/same-id suite with legal context shapes and no silent task reduction. Each selected configuration is the best measured safe candidate from its bounded autotune grid and remains frozen during scoring.
- Resource records show slot occupancy and CPU utilization during their respective ready windows, bounded RAM/VRAM/queues, and improved or explained whole-phase wall time. Any unmet 95%/85% tuning target has a retained bottleneck explanation rather than a fabricated utilization claim.
- The CPU-tested final implementation and pre-GPU Handoff are committed and automatically pushed before the user's run, with remote `main` verified. Maintenance/Day 5 work may proceed while the user independently reviews live results and fixes; no further Handoff or agent bug-fix cycle is required.

## Pitfalls

- Calling a CPU scripted run live acceptance or writing pending performance numbers before the notebook sweep finishes.
- Loading the agent and embedding models together without a VRAM probe, or running compaction on the same slot without recording eviction.
- Comparing timings under different concurrency/background builds or silently changing KV/cache flags between rows.
- Treating a partial public subset/local judge as an official comparable leaderboard score.
- Losing the latest memory/blob/journal snapshot by declaring completion before Drive queues drain.
- Publishing only successful trajectories or paper percentages while omitting the first local failure and unrun cells.
- Letting stretch work postpone the core paired matrix or mask a core correctness/security regression.
- Treating L4 worker/slot counts as valid on T4, reducing per-slot context instead of slots, or declaring two-GPU support after only an L4 run.
- Oversubscribing T4's two vCPUs with llama.cpp, Torch, tokenizer, and nested process pools, or allowing prefetch/Drive queues to exhaust RAM/disk.
- Issuing more HTTP requests than slots and calling the resulting server queue utilization, or changing worker/KV/batch/affinity settings during scored timing runs.
- Optimizing sampled CPU/GPU percentages with artificial work rather than minimizing correct whole-notebook wall time, or combining pinned-cache and work-conserving results as if they came from one policy.

## Learning references for the write-up

Use these as design context, with their own claims kept outside the measured tables:

- [Manus: Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) and [Anthropic: Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).
- [vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/), [SGLang/RadixAttention](https://arxiv.org/abs/2312.07104), and the [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/HEAD/tools/server/README.md) (check the pinned build rather than assuming HEAD behavior).
- [LongMemEval](https://arxiv.org/abs/2410.10813), [Lost in the Middle](https://arxiv.org/abs/2307.03172), [MemGPT](https://arxiv.org/abs/2310.08560), and [Chroma Context Rot](https://www.trychroma.com/research/context-rot).
- [Acon](https://arxiv.org/abs/2510.00615), [Source-Judged Compaction](https://arxiv.org/abs/2610.20927), [StateComp](https://arxiv.org/abs/2609.27298), [TACO](https://arxiv.org/abs/2604.19572), and [SumOne](https://arxiv.org/abs/2611.06445).

The original [Tidemark project brief](https://agent-track-day-4-of-10-nov-10-2026-4gdn.pages.grok.bot/) is the source for the eight-to-fourteen-evening scope and optional evenings 13–14. Its targets and quoted research figures remain targets/external context until independently reproduced here.

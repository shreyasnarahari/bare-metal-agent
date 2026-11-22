# Part 7 — Evaluation and write-up

**Track:** Steadyhand, Day 2 of 10
**Milestone:** M6, evenings 10 to 11, and M7, evening 12
**Depends on:** `06-recovery-policies.md`
**Produces:** a measured comparison of recovery policies on the mock store, a scoped external-benchmark plan, and a results write-up that stays inside what was actually run

## Outcome

The loop from parts 1–6 is scored on tasks whose gold label is the final store, not the wording of the answer. Policy ablations share a fault schedule. The write-up states which policy moved which failure type, includes a real trajectory excerpt for each remaining failure, and labels partial external benchmarks as partial.

## What already exists

Toolsmith already has an evaluation stack. Steadyhand adds an episode runner beside it. It does not replace those scores or reuse their category names for a different task.

| Existing piece | Scope it already has | How Steadyhand should treat it |
|---|---|---|
| `evaluation/data/toolsmith_single_turn_v1.jsonl` | 200 single-turn exact-gold cases, 10 tools | A different dataset. Leave it on the Toolsmith path. |
| `src/toolsmith/evaluation/bfcl_adapter.py` | Single-turn categories `simple_python`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance` | Keep generating those scores only from this adapter. Multi-turn categories below are a new runner and a new label. |
| `src/toolsmith/evaluation/own_runner.py`, `metrics.py` | Parse/schema/semantic validity, Wilson intervals, paired bootstrap | Reuse `percentile`, `wilson_interval`, and `paired_bootstrap` for Steadyhand rates. Do not copy the functions. |
| `CheckpointRunner`, `build_manifest` | Resume when config and dataset hashes match | Same rule for sweeps: `evaluation/manifest.py` pattern, Steadyhand sweep manifests under `results/steadyhand/sweeps/`. |
| `JsonlRecorder` | Redacted append-only records | Sweep logs go through it. |
| README results section | Local grammar and unit tests are done. Live Qwen, BFCL, and JSONSchemaBench numbers are not claimed. | Steadyhand tables start empty of numbers until a journal exists. Do not fill them from the curriculum's paper citations. |

Paper figures in the Day 2 brief are context for why the loop matters. They come from other harnesses. The brief says to use them as context, not as baselines to match. The write-up may quote them only with the citation and the harness name, and it must not put them in the results table as Steadyhand scores.

Cited context, for the discussion section only:

- ReAct: uninformative search results were involved in 23% of HotpotQA error cases in that paper.
- Bench2Robust: 69 of 70 model/task pairs degraded under injected tool failures, by up to 46.7 points; Qwen3-4B fell from 56.1% to 15.8% on τ²-bench retail in that harness; a structured recovery context recovered 16.8 points there.
- UndoBench: naive retry produced duplicate external effects in 53.33% of lost-acknowledgment trials for the local models they ran.
- ToolMaze: recovery on implicit failures was on average 37 points lower than on explicit ones; a failure-aware prompt helped the models they tested by 1.5 to 20.8 points on their composite.

Colab unit rates in the brief are user measurements (about 1.27 units/hour for a T4 and 1.54 for an L4 in one September 2026 note; another table had the L4 at 1.71). Check the resource panel before a sweep. The brief's planning estimate is about 20 to 30 L4 hours plus 5 to 10 T4 hours, mostly for this part. M1–M5 stay on CPU replay.

## What this part adds

```text
evaluation/data/steadyhand_tasks_v1.jsonl
src/steadyhand/eval/suite.py
src/steadyhand/eval/external.py      # BFCL multi-turn and ToolMaze adapters
src/steadyhand/eval/report.py
tests/test_steadyhand_eval.py
```

The Steadyhand notebook (`notebooks/02_steadyhand.ipynb`) holds the sweep section: load secrets from Colab Secrets, start one llama-server for the selected model and shape, run the grid with one episode per slot, copy finished journals from `/content/steadyhand/` to Drive, and render `RESULTS.md` from those records. A crashed agent cell must not kill the server cell. The base URL comes from `TOOLSMITH_BASE_URL`. The GPU holds one GGUF at a time. Its server cell stops the previous Steadyhand server before loading another model onto a 16 GB T4. Steadyhand keeps that rule on both T4 and L4.

## Design

### Own suite

About 60 tasks over the mock store, in JSONL, each with a stable `task_id`.

| Category | What it checks | Gold |
|---|---|---|
| `single_step` | One read or one write | Final store hash, and the tool name |
| `multi_step` | Two or more dependent calls | Final store hash |
| `missing_information` | The user omitted an id or amount | No writes. Final status `done` with a question, or `escalated` only if the scripted user never answers. Part 7's own suite is single-shot: the user message is fixed, so the gold for missing information is a question and an unchanged store. |
| `fault_<mode>` | The same tasks under one pinned fault mode | Store hash plus fault-specific metrics below |

Each task runs clean and under each fault mode the task declares. Tasks that have no write do not get a lost-ack pin.

Gold is a declarative store diff, not a full snapshot copied by hand:

```json
{"refunds_for_order": {"ord_1001": [{"amount_inr": "120.00", "count": 1}]}, "tickets_created": 0}
```

`suite.py` applies the episode to a fresh fixture and checks the diff. Matching prose is optional and is not success. A correct paragraph with two refunds is a failure (`duplicate_side_effect`).

Metrics per task, then aggregated with the existing Wilson helpers for rates:

- `task_success`
- `steps`, `prompt_tokens`, `completion_tokens`
- `duplicate_side_effects` (count of extra refunds, tickets, or notes beyond the gold count)
- `escalation_precision`: escalations on tasks marked `possible: false` over all escalations. Also report the false-escalation count on `possible: true` tasks.
- `recovery_class`: for faulted tasks, whether the episode retried, switched, verified, or stopped, compared to the gold recovery label.

Report spread across repeats. Default sweep temperature is `0.0` and the fault seed is the task seed. When a non-zero temperature run is added, repeat it at least 5 times and publish the min, median, and max. A single noisy number is not a result.

### Ablation grid

Policies from part 6: `raw_error`, `typed_hint`, `retry_safe`, `idempotent_verify`, `switch`, `escalate`.

Prompt variants: `plain` and `failure_aware`.

Models, using existing profiles plus one smaller comparison profile if the weights are available:

| Profile | Role | Notes already in the repo |
|---|---|---|
| `QWEN35_PROFILE` | Main | `Qwen/Qwen3.5-9B`, protocol `qwen35`. Same Q4_K_M GGUF the notebook already selects from `unsloth/Qwen3.5-9B-GGUF`. Slot context is `context_size // parallel`, not the raw `-c` value. The wide and tall presets are in "Keeping a T4 or L4 busy" below. |
| `QWEN3_PROFILE` | Comparison | `Qwen/Qwen3-8B` Q4_K_M, the GGUF the notebook already caches. Same faults and tasks, so the table can separate loop changes from model changes. Loaded only after the 9B server has stopped. |
| Qwen3-4B-class | Cheap iteration | Add a profile only when the tokenizer and protocol are explicit. Do not infer the protocol from the name string. Bench2Robust's primary model in the brief was Qwen3-4B-Thinking-2507. Thinking-on runs use `use_grammars=False` and are a separate grid cell from the grammar path. |

Grammar-path cells set `enable_thinking=false`. Thinking cells set `use_grammars=False`. The manifest records both flags. Do not mix them in one request. The shim rejects that combination for fixed grammars.

The full grid is large. Correctness first, then one resident model, with slots filled:

1. CPU replay of policies (a)–(f) on the scripted-model fixtures, as a correctness gate. Run it while llama-server loads, and finish it before the smoke test.
2. One model, wide shape, `typed_hint` versus `escalate`, clean tasks plus lost-ack plus one explicit and one implicit fault, `parallel` episodes at a time. This is the first live table.
3. The rest of that model's wide phase, then its tall phase, then the next model. Resume from the sweep manifest. Do not reload a GGUF to switch policies.

Targets for the write-up, as goals rather than expected measured values:

- Zero duplicate side effects under `idempotent_verify` and under `escalate` on the lost-ack tasks.
- A higher recovery rate for `escalate` than for `raw_error` on explicit faults, with the interval reported.
- Clean-task success for `escalate` at least as high as `typed_hint` on the same model. If the guards lower clean success, that regression is a result, not something to hide by weakening the gold.
- A sentence for every remaining failure category, pointing at a journal `episode_id`.

### External benchmarks

These are optional live runs after the own suite, and they stay clearly scoped.

**BFCL multi-turn.** Categories `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context` (200 entries each in the leaderboard the brief points at). Run them through the shim with two loop settings: validation and guards on, and a bare loop that only executes calls. Reuse `convert_bfcl_tools` where the schema fits. Schemas the grammar compiler rejects are reported as ineligible with the JSON Pointer, the same way `constraint_support` already works. An ineligible id is not a failed call and not a success. Partial category runs are labeled with the id list and the hash. They are not full leaderboard scores.

The single-turn adapter's categories must not appear in the multi-turn table and the reverse. Publish both, in separate sections, if both were run.

**ToolMaze.** 400 base tasks, 270 simulated tools, expanded to 2,000 instances across four failure modes in that project. Steadyhand runs a subset of the simpler topologies. Record the subset id list in the manifest. Metrics to compute on that subset: task success rate, perturbation recovery rate (retry, switch, or stop as the task's gold recovery requires), and recovery cost (tool calls after the injected fault that did not change the outcome). Do not label a subset score as the full ToolMaze number.

**Lost-ack slice.** Already in the own suite. Report duplicate-effect rate for `raw_error` with a naive retry, for `retry_safe`, and for `idempotent_verify`. This is the UndoBench-shaped measurement on our store.

### Sweep mechanics

- One episode, one journal, flushed on local disk before the worker takes the next task. Drive copy is asynchronous, as in part 4.
- Sweep manifest: dataset hash, policy, prompt variant, profile, GPU name, `context_size`, `parallel`, `per_slot_context`, KV cache type, batch size, seed, task id list, code versions.
- Resume skips `task_id`s whose episode manifest is `completed: true` and whose hashes match. A mismatch stops the sweep. Claims are exclusive, so parallel workers do not double-run a task.
- Infrastructure failures (`LlamaCppConnectionError`, OOM, disconnect) are `aborted` and are not counted as task failures or as successful refusals. The README already requires this for irrelevance. The same rule applies to "the model correctly did nothing." An OOM that kills llama-server aborts the in-flight episodes, restarts the server one slot smaller, and resumes from the manifests. It does not retry those episodes inside the dead process.
- Disconnect the Colab runtime when the queue and the Drive-copy queue are both empty. Record wall-clock and the GPU name from `nvidia-smi`.

### Keeping a T4 or L4 busy

The GPU is the resource the Colab unit rate is paying for. Mock-tool episodes spend almost all of their time inside llama-server. High utilization means every slot is in a completion whenever the sweep has work left, one model is resident, and the CPUs are preparing the next prompts instead of recompiling grammars or blocking on Drive.

`LlamaCppServer` already accepts `parallel` and appends `--parallel` to the command. llama.cpp divides `-c` / `context_size` across those slots. The Day 10 cell uses 4096 tokens per slot (4 slots and `-c 16384` on a T4, 8 slots and `-c 32768` on an L4). Steadyhand episodes need a larger per-slot budget than that. Set the two flags together:

```text
per_slot_context = context_size // parallel
```

Confirm the division in the server log at startup (`slot` context or the equivalent line) and store both numbers in the sweep manifest. A prompt that does not fit the slot is `completion_truncated` and `aborted`, not a task failure and not a reason to drop the task from the log.

`extra_args` on `LlamaCppServer` carries the flags the constructor does not model yet:

- KV cache: `--cache-type-k q8_0 --cache-type-v q8_0` first. q8 KV roughly halves cache memory versus f16 and lets more slots fit. Whether a GPU runs it at full speed depends on the build (a T4 is sm_75), so the server cell tries a ladder: q8_0 with flash attention, f16 with flash attention, then f16 without it (a quantized V cache needs flash attention). The first rung that loads and passes the self-test is kept, and the manifest records the cache type and flash-attention setting actually used. An f16 run is a different configuration, not a silent upgrade.
- Flash attention: pass whichever of `--flash-attn on|off` or `-fa` the binary's `--help` documents. Where the binary's default may be `auto`, the no-flash-attention rung passes `--flash-attn off` explicitly.
- Prefill: `--batch-size 2048 --ubatch-size 512`, raised only if the smoke test shows prefill serialized while VRAM remains. These go through `extra_args`.
- CPU threads: `-t` and `--threads-batch` set to the detected vCPU count (`HARDWARE.cpu_count`), not higher. Extra threads on a 2-vCPU T4 only contend with the workers.

Startup probe, before any scored episode:

1. Query `nvidia-smi` for GPU name and total memory. Treat a T4 as 16 GB and an L4 as 24 GB only as a label. Trust the reported total. A card that is neither (an A100, a V100) starts from the L4 rows if it has at least 22 GiB, else the T4 rows if it has at least 14 GiB, and the probe corrects the guess.
2. Start the server at the preset below. Wait until `/health` succeeds.
3. Read used memory. If less than about 1 GB is free, stop and restart with `parallel` reduced by 1, keeping `per_slot_context` constant by shrinking `-c` in the same ratio. If the process runs out of memory during load, do the same. If it dies for another reason, move to the next KV rung instead: fewer slots cannot fix an unsupported kernel.
4. Self-test: one 64-token completion. Below 5 tokens/sec decode, an operation has fallen back to the CPU; stop and move to the next KV rung. `steadyhand.eval.serving.fit_server` implements steps 2 to 4 and records every attempt.
5. Run `parallel` short scripted episodes at once through `ThreadPoolExecutor`. Then read llama-server `GET /slots` (or the metrics endpoint the binary actually serves) and a few samples of `nvidia-smi --query-gpu=utilization.gpu,memory.used`.
6. Accept the shape only when every slot is busy during that smoke and the server stays up. Record single-stream tokens/sec and aggregate tokens/sec. The aggregate number is the utilization check. If slots sit idle, the client is not issuing concurrent `complete()` calls. If the GPU utilization is low while every slot is busy, the card is waiting on a small batch. Increase `parallel` only when free VRAM is still above about 1 GB and `per_slot_context` stays at least the largest prompt in that phase plus `max_tokens`.

Starting presets for the Q4_K_M GGUFs this notebook already downloads (`Qwen3-8B-Q4_K_M.gguf`, and the single Q4_K_M language file from `unsloth/Qwen3.5-9B-GGUF`). These are capacity guesses to confirm with the probe, not measured throughputs.

| GPU | Model | Shape | Total `-c` | `parallel` | Per-slot context | Which tasks |
|---|---|---|---|---|---|---|
| T4 | Qwen3-8B Q4_K_M | wide | 32768 | 4 | 8192 | Own suite, explicit and lost-ack faults |
| T4 | Qwen3-8B Q4_K_M | tall | 32768 | 2 | 16384 | Tasks whose rendered prompt exceeds 8192, including long multi-turn |
| T4 | Qwen3.5-9B Q4_K_M | wide | 24576 | 3 | 8192 | Same wide tasks, only if the probe stays under the memory ceiling |
| T4 | Qwen3.5-9B Q4_K_M | tall | 32768 | 2 | 16384 | Long-context multi-turn, only if the probe stays under the memory ceiling |
| L4 | Qwen3.5-9B Q4_K_M | wide | 65536 | 4 | 16384 | Own suite and fault grid |
| L4 | Qwen3.5-9B Q4_K_M | tall | 65536 | 2 | 32768 | Long-context multi-turn |
| L4 | Qwen3-8B Q4_K_M | wide | 65536 | 8 | 8192 | Comparison model, short tasks |

If the Qwen3.5-9B probe OOMs on a T4 even at one slot, skip that model on the T4 and run it on an L4. Do not drop to a smaller quant inside the sweep without a new manifest. A Qwen3-4B-class cell, if a profile is added later, uses the wide shape with more slots. It is a separate phase, loaded only after the 8B or 9B server has stopped.

Two shapes, not one server for the whole grid:

- **Wide.** Many short episodes. Maximize slots. Per-slot context is the p95 rendered prompt length of that task split, plus `max_tokens`, plus a margin of 512, rounded up to a power of two, then checked against the table.
- **Tall.** Fewer slots, longer context. Run it as its own phase after the wide phase for that same model, so the weights stay resident across the restart of llama-server only. Restarting the server to change `-c` is cheaper than reloading a different GGUF, and much cheaper than truncating long tasks.

Phase order, one resident model at a time:

1. While the runtime provisions in the background (GGUF copy or download, llama.cpp restore or build), run the CPU suite (dataset validator, scripted policies, golden replay) on a process pool with one worker per vCPU. Stop that pool before the smoke test so the vCPUs belong to prompt prep and sampling.
2. Wide grid for model A, all policies and both prompt variants, all tasks that fit the wide slot. Workers pull from one queue.
3. Tall grid for model A, only the tasks that did not fit.
4. Stop server A. Start model B. Repeat wide, then tall.
5. External BFCL multi-turn and ToolMaze subsets are later phases with their own slot size, still one model, still `workers == parallel` unless the smoke test says otherwise.

Worker pool:

- `ThreadPoolExecutor(max_workers=parallel)`. Threads block in `httpx`, so the GIL is not the limit. A process per episode would duplicate nothing useful and would fight the small CPU count.
- Each worker builds its own `ToolsmithService` on the shared `LlamaCppClient`, with `recorder=None`. Grammars for the mock-tool schema hash are compiled once per worker and reused. Part 2's `schemas()` hash is the key.
- In-flight episodes equal `parallel`. The executor's work queue should not stack extra episodes on top of llama-server's own queue. Extra queued prompts sit in RAM and do not raise GPU utilization.
- Raise `max_workers` to `parallel + 2` only if the smoke test shows slots idle while workers are inside verify or backoff. Cap the extra at 2.
- Within an episode, the part 6 read pool stays at 4 and does not multiply by the sweep width.

What not to parallelize on this GPU:

- Two GGUFs at once. An 8B Q4 and a 9B Q4 together do not leave a useful KV cache on a T4, and on an L4 they steal the slots the sweep needs.
- The Day 10 HF Qwen3-0.6B path in the same runtime as llama-server. The notebook already stops the 8B server first. Steadyhand does not add a second resident model for judging.
- `pytest` during the live sweep.
- Per-event writes to My Drive. Local append, background copy of finished episodes, drain the copy queue before disconnect.
- Temperature-zero repeats of the same task on several slots in the hope of a tighter interval. At temperature 0 the repeats are identical and only burn slots. Use the slots for different `task_id`s. Non-zero temperature repeats, when added, are separate tasks in the same queue so they still fill slots.

Resuming. Records are per GPU (`MyDrive/toolsmith/records/<gpu>/steadyhand/`). A sweep mirrors its manifest and scores to Drive as it runs and restores them on a fresh runtime, so a disconnect costs only the episodes in flight. The resume check ignores `parallel` and the total `-c`, because the probe may size a resumed run differently; `per_slot_context`, the KV setting, the GPU, and the model must still match. Each start appends a `sweep_segment` event with its slot count.

Report, next to the accuracy table, the smoke-test lines: GPU name, preset, single-stream tokens/sec, aggregate tokens/sec, slot count busy, and peak memory. Those figures describe the run. They are not a leaderboard score.

### Write-up

`results/steadyhand/RESULTS.md` is generated by `eval/report.py` from journals. A committed copy is only updated when real journals exist; until then the generator's fixture test writes a temp file. The narrative sections:

1. Setup: profiles, flags, task hash, policy list.
2. Results table: success, duplicate-effect rate, escalation precision, tokens, steps. Empty cells stay empty.
3. Failure taxonomy with excerpts: model-side (`parse_error`, `unknown_tool`, `invalid_arguments`, `repeated_call`), tool-side explicit, tool-side implicit, lost-ack, budget, abort. Each excerpt is a real `episode_id` and a few journal lines.
4. Which policy addressed which class, including negative results.
5. External runs, each with subset or full id list.
6. Context citations, separated from the table.

Also update the root `README.md` with a short Steadyhand section that points at this directory and at `RESULTS.md`, and that repeats the boundary: handlers run in Steadyhand; the shim does not execute them. Do that update in the same change as the first real results, or as a stub that says the sweep has not been run. A stub is better than invented numbers.

### Stretch goals

Not required for M6 acceptance. Record them so they are not mixed into the core table by accident.

- Recovery memory: per-tool, per-error-type success counts from past episodes, summarized into the system prompt. Evaluate on held-out task ids only.
- τ²-bench airline or retail with a second self-hosted model as the user simulator, so the run stays API-free. That score is not comparable to the Bench2Robust citation without saying the harness differs.
- Cancellation of parallel reads when one result makes the others unnecessary. Part 6 already orders parallel reads and serializes writes.
- A short note, after reading mini-swe-agent, listing design choices that package makes and Steadyhand does not. The curriculum points at that repo as a reference for how small a loop can be. The note is prose in the write-up, not a dependency.

## Tasks

1. Write the 60-task JSONL and a validator that checks unique ids, known tool names, and gold diffs the fixture can satisfy. Unit-test the validator the way `tests/test_evaluation_dataset.py` checks the Toolsmith gold set.
2. Implement `run_suite` on the CPU scripted model first. Assert the gold diffs and the duplicate-effect counter with the recovery tests' pinned faults.
3. Implement sweep manifests and resume. Test a two-task fake sweep that resumes after the first journal is marked complete and refuses when the policy name changes.
4. Add the report generator. Test it on a tiny fixture journal with a known duplicate refund and a known escalation, and check the markdown contains those episode ids and does not contain a hardcoded paper percentage in the results table.
5. Add the notebook section: one server cell per phase, a sweep cell with `ThreadPoolExecutor` sized to `parallel`, local journals under `/content/steadyhand/`, background copy to `MyDrive/toolsmith/records/<gpu>/steadyhand/`, secrets from the environment. The cell runs the startup probe and prints single-stream versus aggregate tokens/sec before the scored queue.
6. Implement the BFCL multi-turn and ToolMaze adapters as far as the pinned data formats allow offline (id parsing, ineligible-schema reporting, metric definitions). Live downloads stay in the notebook and are skipped when the data is absent. Tests use a three-row fixture, not the full leaderboard files.
7. After the first live sweep, generate `RESULTS.md` and update the root README stub with the measured cells only.

## Tests

- Dataset validator: duplicate id fails, unknown tool fails, missing-information task with a write in the gold diff fails.
- Scripted sweep: `idempotent_verify` on a lost-ack refund task has `duplicate_side_effects == 0` and `task_success == 1`. `raw_error` plus a scripted blind retry has `duplicate_side_effects == 1`.
- Implicit payment lie: `escalate` with the sanity checker does not count a clean success.
- Resume and hash mismatch.
- Report fixture: infrastructure `aborted` is excluded from the success denominator and listed separately.
- Ineligible BFCL schema fixture: counted in `ineligible`, absent from success and failure.
- Existing Toolsmith tests, including `tests/test_bfcl_evaluation.py` and `tests/test_evaluation_core.py`, still pass. The new multi-turn code must not change `BFCL_CATEGORIES`.

## Acceptance

- CPU suite is green before any scored GPU episode starts. It may run while the server is loading.
- A wide-shape smoke test shows all configured slots busy, and the manifest records `parallel`, total context, and per-slot context.
- Live results, once they exist, trace back to a manifest, a task-id list, seeds, and journals.
- Duplicate side effects under the idempotent policies are measured, and the lost-ack case is the one that defines them.
- Partial BFCL or ToolMaze runs are labeled with the ids that were actually scored.
- The README does not gain unsupported accuracy or throughput claims.

## Pitfalls

- Scoring the final sentence with a substring match and ignoring a double refund.
- Publishing the curriculum's paper percentages as this repo's measurements.
- Folding multi-turn BFCL into the existing single-turn adapter and calling the old category list a full agent benchmark.
- Dropping aborted infrastructure runs from the log so the success rate rises.
- Changing a gold diff after a bad run to make a policy look better. Change gold only when the task specification was wrong, and say so in the write-up.
- Running the sweep at temperature above zero once and reporting that single rate.
- Letting an idle Colab runtime sit between episodes or between model loads. Queue a whole phase, keep every slot fed, copy journals in the background, and disconnect when the copy queue is empty.
- Treating Day 10's `-c` (16384 on a T4, 32768 on an L4) as the tokens per episode. Day 10 runs 4096 tokens per slot. Size `-c` and `--parallel` as a pair.
- Starting more workers than slots "for throughput." The extra requests wait inside llama-server and the GPU does not get busier.
- Loading Qwen3-8B and Qwen3.5-9B in one runtime. Switch phases. One resident GGUF.

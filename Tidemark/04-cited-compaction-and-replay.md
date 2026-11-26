# Part 4 — Grammar-constrained compaction with citations and replay

**Track:** Tidemark, Day 4 of 10
**Milestone:** M3, evenings 6–7
**Depends on:** `03-observation-shaping-and-blob-store.md`
**Produces:** deterministic compaction epochs, a closed summary schema, citation/source checks, rejected-entry evidence, and CPU replay without a compactor model

## Outcome

At a soft token limit, replace the previous summary and compacted raw turns with a shorter, grammar-valid summary whose factual entries can be traced to journal events. Drop uncited or unsupported entries instead of allowing a fluent summary to become new ground truth. Keep plans/open items visibly separate from environment facts.

A compaction is an explicit epoch transition and an expected prefix-cache break. It never rewrites the append-only journal, and replay never needs a GPU to reproduce it.

## What already exists

- Part 2 defines segments 0–4, `compaction_pending`, safe step boundaries, epoch transitions, and exact budget/lint events.
- `ToolsmithService` supports `response_format={"type": "json_schema", ...}` only with no tools and requires `enable_thinking=false` on the grammar path. Its compiler supports the closed objects, arrays, strings, and enums needed here.
- `src/steadyhand/journal.py` gives every event an episode id and contiguous `seq`; it does not need a new `event_id` field to derive stable citations.
- `src/steadyhand/replay.py` serves recorded agent turns and compares the exact request messages. Compaction requires a separate recorded structured-output stream, not another call to `ReplayClient`'s ordinary turn list.
- Part 3 provides immutable blob handles and provenance. A summary may reference a handle but must not invent one.
- The brief's illustrative schema has uncited string `goal`/`open_items`. This implementation closes that ambiguity by giving factual goal/open-item/handle entries source refs and labeling open items as plan state rather than observed facts.

## What this part adds

```text
src/tidemark/summary_schema.py          # Versioned closed JSON Schema and Python values
src/tidemark/compaction.py              # Trigger, candidate generation, acceptance, epoch transition
src/tidemark/source_check.py            # Event refs, lexical checks, judged support, verdicts
src/tidemark/epochs.py                  # Lineage, carried entries, compacted ranges
src/tidemark/replay.py                  # RecordedCompactor and read-only source-check replay
src/steadyhand/context.py               # Structured compactor integration at safe boundaries
evaluation/data/tidemark_compaction_v1.jsonl
tests/test_tidemark_compaction.py
tests/test_tidemark_source_check.py
tests/test_tidemark_epochs.py
tests/test_tidemark_compaction_replay.py
tests/fixtures/tidemark_compaction/
```

The compaction benchmark contains small journaled episodes with human-labeled supported/unsupported entries. It is separate from the long-horizon task suite and the observation corpus.

## Event references and compacted ranges

Derive a canonical event ref as `<episode_id>:<seq>`; `evt-123` in the brief is a display example, not a reason to rewrite every existing journal line. A ref is valid only when:

- its episode id matches the authorized journal lineage;
- its sequence exists and is not a torn/redacted record;
- it lies in the full source range available to this compactor;
- the cited event type can carry evidence (`observation`, `tool_result`, `model_turn`, `observation_shaped`, security/recovery events, or a prior `compaction` with lineage), not merely a timing/budget row.

Each epoch records `source_start_seq`, `compacted_through_seq`, `previous_epoch`, and `parent_compaction_seq`. Carried entries retain their original source refs. If a prior compaction event is cited, the checker expands its lineage and verifies the underlying accepted entries; one accepted summary cannot launder an unsupported claim into the next epoch.

Raw journal events remain immutable and available for audit even though segment 3 resets. Segment 2 contains only the accepted canonical summary. `history_end_seq` in the next render makes the boundary unambiguous.

## Closed summary schema

Use `SUMMARY_SCHEMA_VERSION = 1` and a schema that stays inside Toolsmith's supported subset. Every object is closed and required fields precede optional fields in the grammar IR. Conceptually:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "goal": {"type": "object", "additionalProperties": false,
      "properties": {"text": {"type": "string"}, "src": {"type": "array", "items": {"type": "string"}}},
      "required": ["text", "src"]},
    "state": {"type": "array", "items": {"type": "object"}},
    "decisions": {"type": "array", "items": {"type": "object"}},
    "failed_attempts": {"type": "array", "items": {"type": "object"}},
    "open_items": {"type": "array", "items": {"type": "object"}},
    "handles": {"type": "array", "items": {"type": "object"}}
  },
  "required": ["goal", "state", "decisions", "failed_attempts", "open_items", "handles"]
}
```

The complete nested item schemas are code, not left open as the illustrative `{"type": "object"}` above:

- `state`: `key`, `value`, `src[]`.
- `decisions`: `what`, `why`, `src[]`.
- `failed_attempts`: `action`, `error`, `src[]`.
- `open_items`: `item`, `src[]`, `kind` (`plan` or `question`).
- `handles`: `blob`, `what`, `src[]`, and optional `range_start`/`range_end` integers when a specific range matters.

No free-form nested object, NaN, arbitrary source URI, or uncited factual entry is valid. Limit entry counts, source-ref counts, and string lengths in both semantic validation and `ContextConfig`; JSON Schema grammar support is not a resource limit by itself.

Canonical accepted summaries use sorted object keys, stable entry order from the candidate, finite numbers only, and no generated timestamp/random epoch id. `epoch_id` is derived from `(previous_epoch, accepted_summary, compacted_through_seq, config hash)`.

## Compactor model boundary

The compactor is a separate structured-output request through Toolsmith:

- no tools and no native tool choice;
- fixed `response_format` with the closed summary schema and `strict: true`;
- `enable_thinking=false` when grammars are enabled;
- fixed model/profile, seed, temperature, maximum tokens, and compactor prompt version in the manifest;
- only the selected source events, prior accepted summary, blob previews, and explicit compaction instructions are supplied;
- model output is summary JSON, not hidden reasoning, and is retained in a bounded/redacted artifact or blob for audit.

A compactor call gets its own usage/timing metrics and `purpose=compaction`. Its token/latency cost is included in total episode cost and shown separately from agent turns. It enters part 1's model-slot semaphore like any agent call: `shared` policy leases any eligible slot, while `dedicated` policy reserves a declared compactor slot after the capacity probe shows that this is faster overall. On T4, compare shared use of two slots with one agent + one compactor slot; on L4, compare shared use with one dedicated slot out of the fitted total. One-slot operation records the unavoidable eviction rather than hiding it. The selected policy is frozen for each scored run.

The compactor does not execute tools, write memory, approve a policy request, or mutate the episode. It returns a candidate only. Deterministic code decides acceptance.

## Source-check pipeline

Apply these stages in order and retain per-entry verdicts:

### 1. Structural and citation checks

Reject the whole candidate on JSON/schema failure after the existing bounded Toolsmith repair policy. At entry level, drop:

- missing/empty/duplicate/malformed refs;
- refs outside the authorized range or to unrelated episodes;
- duplicate/contradictory entries that cannot coexist in one canonical summary;
- unknown blob handles, handles outside the episode catalog, or ranges outside the object;
- entries exceeding count/length limits;
- a goal not grounded in the initial task charter.

Never “repair” an entry by inventing a citation. A candidate with no usable entries can produce an empty state plus a supported goal/open question; it cannot cause old raw turns to be replaced by a fabricated summary.

### 2. Deterministic lexical support

Extract source text and structured values from exactly the cited events. Classify critical literals: ids, paths, amounts, counts, dates, statuses, error codes, failing test names, blob digests, and quoted user constraints.

An entry is lexically supported only when every critical literal and the claimed relation are present in its cited evidence. Paraphrased non-critical wording may proceed to the judge. A wrong status, amount, path, or negation cannot be accepted merely because its strings occur somewhere in the source.

Record `matched_literals`, `missing_literals`, and a stable reason code. This checker is deliberately conservative and its false rejections are measured on the labeled corpus.

### 3. Optional local support judge

Entries that are cited but cannot be settled lexically may receive one short yes/no judgment from the same local model. The judge sees only one entry and its cited source excerpts, not an instruction to preserve the summary at all costs.

Constrain the response to a closed schema such as `{"support": "yes"|"no"}` with a fixed prompt version/seed. `no`, malformed output, unavailable judge, timeout, or truncation rejects the entry; none becomes an environment fact. The judge's prose is not a citation and no chain of thought is requested or stored.

The judge is a second filter, not proof of truth. The part 7 corpus measures its false-accept and false-reject rates independently of the compactor.

### 4. Accepted summary and resource invariant

Drop all `unsupported`, `uncited`, `invalid_ref`, `judge_unavailable`, and `contradictory` entries from the model-facing summary. Journal counts and bounded rejected-entry payloads/reasons, including the before/after source-check totals.

If the accepted summary still exceeds segment 2's cap, apply deterministic entry priority: supported goal, unresolved user constraints, current state, failed attempts, decisions, open plan items, oldest history. Every retained handle must remain restorable. If even the minimum supported summary cannot fit, fail `compaction_budget_exhausted`; do not silently fall back to naive truncation.

## Throughput scheduling without semantic drift

Compaction has a hard dependency on the episode's safe boundary, so utilization work must preserve its exact trigger and commit order:

- Put ready compactor/judge requests from different episodes into a bounded scheduler queue. Fill every available model slot with independent structured-output calls rather than running one judge at a time while other slots idle.
- Keep per-episode FIFO and candidate/source hashes in every queue record. Batching is concurrent execution of independent requests, not concatenating several entries into a changed judge prompt.
- Run lexical checks, source-event extraction, canonicalization, and artifact preparation in part 1's shared CPU pool while other episodes use the GPU. Preserve candidate entry order in the final verdict list regardless of worker completion order.
- A due hard-limit compaction has priority and the episode waits; never postpone or skip it to improve occupancy. Soft-threshold timing remains the deterministic configuration under test, not a GPU-load-dependent decision. A separate cache-aware trigger is only a part 8 stretch ablation.
- Autotune shared versus dedicated compactor slots and judge concurrency on T4 and L4 using end-to-end tasks/minute and net prefill cost. Freeze the winner for scored runs and include compactor queue/slot occupancy in resource samples.
- Reuse cached candidate outputs only for the explicitly declared unchecked-versus-checked support ablation when model, prompt, source events, schema, and candidate hash all match. Other live modes generate their own candidates; semantic model-output caching must not turn a task-success run into replay.
- Cache compiled summary/judge grammars by schema/protocol/compiler identity in each worker. Precompute source-event artifacts and blob previews on CPU without changing their citation range.

Record compactor queue delay, slot occupancy, lexical-check CPU time, judge fan-out, and GPU idle gaps. Tests run independent episodes with forced slow judges and T4/L4 slot budgets, proving full slot usage, bounded queues, unchanged accepted summaries/event order, no starvation of hard-limit compactions, and exact replay.

## Epoch commit and failure behavior

Compaction proceeds as one atomic logical transition:

1. Part 2 marks `compaction_pending` at a soft/hard threshold.
2. Reach a safe boundary with no unanswered tool calls or in-flight parallel reads.
3. Generate and check a candidate without changing `ContextState`.
4. Write all required candidate blobs/artifacts locally.
5. Append the `compaction` journal event and flush it. This is the commit point.
6. Replace segment 2, increment epoch, reset segment 3 to post-`compacted_through_seq` events, rebuild segment 4, and render.
7. Mirror the journal, summary, and required artifacts to Drive through the existing bounded background-copy pattern.

A compactor infrastructure failure leaves the old state and epoch untouched. Below the hard limit, the episode may continue with `compaction_pending` and a recorded reason. At the hard limit, it ends with a context/compaction failure instead of issuing an overflowing request. Evaluation keeps this separate from model task failure and backend abort.

A resumed run first replays the recorded `compaction` event and verifies accepted/rejected entries. Only after that does it restore the post-compaction segment state. A crash between journal commit and in-memory transition therefore resumes safely; a crash before commit produces no epoch.

## `compaction` journal event

Use one strict event with bounded fields:

```text
step, epoch_before, epoch_after, parent_compaction_seq,
source_start_seq, compacted_through_seq, trigger,
compactor_prompt_version, summary_schema_version,
candidate_sha256, candidate_artifact_handle,
accepted_summary, accepted_sha256,
entry_results [{entry_path, verdict, reason, src, missing_literals}],
unsupported_before, unsupported_after,
dropped_entry_counts, judge_metrics, usage, timings,
context_config_sha256, state_sha256
```

`accepted_summary`, refs, verdicts, range boundaries, and hashes replay exactly. Timings are volatile. Candidate artifacts are bounded and redacted; do not dump private source events or model reasoning into the event.

The next `context_render` names the new epoch and is expected to diverge at segment 2. `backend_timings` shows its actual re-prefill cost. This makes the tradeoff between saved future tokens and one-time compaction cost measurable.

## Replay and resume

`RecordedCompactor` serves candidate responses and judge verdicts from the journal/artifact stream in order. It never calls Toolsmith or a local model. The real source checker, epoch transition, assembler, and PrefixLint run again and must reproduce:

- candidate and accepted-summary hashes;
- every entry verdict and citation;
- compacted range and epoch id;
- shaped/raw recent messages after the reset;
- next exact request-message hash and lint status.

The ordinary `ReplayClient` continues serving agent turns. The two recorded streams have distinct purposes and cannot consume each other's rows. During resume, recorded compactions are replayed before any live compactor call.

Replay does not write summaries, blobs, SQLite, checkpoints, or Drive artifacts. Required recorded artifacts are opened read-only and hash-verified. An explicit live compaction replay may call a model into a separate report but never overwrites the original journal.

## Compaction evaluation corpus

Create replayed journal cases containing:

- supported state, decisions, failed attempts, goal, and handles;
- an uncited invention, a citation to the wrong event, and a citation to an unrelated episode;
- a fluent path/error/status contradiction with high word overlap but wrong relation;
- a supported paraphrase that needs the judge and a lexical false rejection;
- a carried entry across two epochs and an attempt to launder a previously rejected entry;
- an oversized summary, an unavailable judge, truncation, and a crash immediately before/after commit.

Human labels identify supported entries and the exact supporting refs. Report unsupported entries **before** and **after** the checker, false accepts, false rejects, judge abstention/failure, summary tokens, and downstream fact availability. Do not treat schema validity as factual support.

## Tasks

1. Implement event refs, complete closed schema, canonical summary values, and resource limits.
2. Implement the structured compactor through Toolsmith with separate cost/slot accounting and no tool access.
3. Implement structural/citation, lexical, optional judge, and handle checks with stable per-entry verdicts.
4. Implement epoch lineage, carried-entry validation, atomic commit, hard-limit failure, and next-render lint behavior.
5. Add the strict journal event, manifest configuration, `RecordedCompactor`, and read-only replay/resume.
6. Build the labeled replay corpus and before/after support metrics.
7. Add `tidemark-compaction-gate` and `tidemark-compaction-smoke` notebook cells. The live smoke inspects rejected entries and the next step's actual prefill before any sweep.
8. Implement bounded compactor/judge scheduling and parallel source checks. Run the shared/dedicated-slot autotune on both T4 and L4, then freeze the selected policy and verify full-slot execution with independent episodes.
9. Run the phase-delivery protocol: commit the tested compactor/checker/replay implementation, write the canonical Handoff for part 5, commit it, automatically push both commits, and verify remote `main`.

## Tests

- Every accepted summary parses against the closed schema and every retained factual entry has valid source refs.
- Uncited, wrong-ref, wrong-episode, forged-handle, contradictory, and judge-unavailable entries are dropped with exact reason codes.
- Critical literal/negation/relation tests prevent high-overlap contradictions from passing.
- Carried entries preserve original evidence; rejected entries cannot return through a later summary citation.
- Candidate failure leaves state/messages/epoch byte-identical. Commit-boundary crash fixtures resume to exactly one epoch transition.
- Segment 2 is replaced, segment 3 resets only at the recorded range, and PrefixLint calls it `expected_epoch` rather than `expected_append`.
- Replay needs no GPU/model and performs no filesystem/database/network side effects beyond read-only artifact verification.
- Schema-valid but unsupported candidates are counted separately from parse/schema failures and infrastructure aborts.
- Existing Toolsmith structured-output, Steadyhand recovery/fault, and Bulkhead security replays remain green.

Run:

```bash
.venv/bin/python -m pytest tests/test_tidemark_compaction.py tests/test_tidemark_source_check.py \
  tests/test_tidemark_epochs.py tests/test_tidemark_compaction_replay.py \
  tests/test_steadyhand_replay.py tests/test_api.py
.venv/bin/python -m pytest
```

Live acceptance requires real constrained summary and judge calls plus inspection of the resulting journal; a fake valid summary is not factual-support evidence.

## Phase delivery and handoff

After all CPU replay/source-check tests and real constrained compactor/judge acceptance on both T4 and L4 pass, follow part 1's mandatory two-commit/one-push protocol automatically:

1. Stage only part 4's schema/compactor/checker/epoch/replay code, labeled corpus, tests, notebook, README, and deliberate shared changes. Exclude generated candidates containing sensitive source text and all unrelated dirty paths.
2. Create the implementation commit `Tidemark: complete part 4 cited compaction`, update `Tidemark/Handoff.md`, then create `Tidemark: hand off part 4`.
3. Record the implementation SHA, summary/schema/prompt versions, citation and epoch invariants, compaction-corpus hash, before/after support results, compactor-slot policy, replay/resume status, per-GPU run ids, and durable candidate/summary/journal artifact paths.
4. Point the next session to part 5 and the initial memory-store fixture/setup command. Push both commits to `origin/main` and verify remote `main` equals the Handoff commit before memory implementation.

Unsupported entries incorrectly accepted, a replay mismatch, missing two-GPU evidence, or publication failure blocks completion. Do not commit private candidate/source payloads or silently convert an unrun judge into a passed gate.

## Acceptance

- Soft-limit compaction produces a bounded, grammar-valid summary whose retained entries pass citation and source checks.
- Unsupported-entry counts before and after filtering, false accepts/rejects on the labeled corpus, and summary/compaction costs are generated from records.
- Journal, raw evidence, epoch lineage, and replay reconstruct identical prompts without rerunning a model.
- Failure before the hard limit never silently switches to naive truncation, and failure after it cannot send an over-limit prompt.
- The compaction cache break and later prefill savings are visible as separate measurements.
- T4 and L4 compaction smokes fill available slots with independent compactor/judge calls and overlap source checks with other GPU work, while accepted summaries and event order remain identical to serial execution.
- The tested implementation and canonical part 4 Handoff are committed and automatically pushed, remote `main` is verified, and part 5 can start in a fresh session with exact epoch/schema/replay/artifact context.

## Pitfalls

- Treating citations as decorative strings or letting a prior summary certify itself recursively.
- Accepting an entry because its keywords occur in a cited event even when status, amount, path, or negation is wrong.
- Retrying a judge until it says yes, or treating judge agreement as proof of environment truth.
- Mutating state before the journal commit, then resuming into an epoch the journal does not contain.
- Including raw recent turns after `compacted_through_seq` accidentally and paying for both summary and full history.
- Measuring token savings while omitting compactor generation and re-prefill costs.
- Letting open plans become sourced environment facts on the next epoch.

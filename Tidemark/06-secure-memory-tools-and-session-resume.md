# Part 6 — Governed memory tools, session persistence, and safe replay

**Track:** Tidemark, Day 4 of 10
**Milestone:** M5, evening 9, with optional evenings 13–14 for llama.cpp slot snapshots
**Depends on:** `05-long-term-memory-and-retrieval.md`
**Produces:** `memory_write`/`memory_search`/`memory_forget` under Bulkhead, recorded memory effects, resumable context/session checkpoints, and an optional slot save/restore path

## Outcome

Expose long-term memory as ordinary native tools without allowing ordinary model output to bypass policy, provenance, namespace, or replay rules. Tainted writes ask by default and remain tainted after approval. Retrieval is explicit and traceable. Persistent sessions resume from journaled context plus verified memory/blob state; a saved llama.cpp slot is only a performance optimization, never the source of truth.

Default replay performs no memory mutation and no slot restore. It reconstructs decisions and exact prompts from recorded operations and immutable artifacts.

## What already exists

- `SteadyhandRegistry` freezes model schemas, policies, result schemas, and `SecuritySpec` side tables. Part 2 freezes the complete session tool list and can mask allowed names without changing prompt segment 0.
- Every model and recovery call passes validate -> guard -> `SecurityGate.authorize` -> injector -> handler. `SecuredExecutor` is the chokepoint; no memory handler gets a special bypass.
- Bulkhead supports episode taint, exact one-shot approvals, linked approved episodes, and policy/taint audit events. An approval never rewrites the denied journal.
- Part 3's `ToolExecutionContext` passes server-only episode/journal/blob resources to handlers while hiding them from model schemas. Part 5 supplies the actual memory store and revisioned mutation log.
- Existing replay treats only fresh in-memory store handlers as safe to reexecute. A persistent SQLite database is an external effect and needs its own recorded backend, just like exec/network.
- `run_claimed_episode` supports exclusive claims, manifest matching, torn-tail repair, recorded-prefix replay, then live continuation. `JournalCopier`/`MirroredFile` support bounded asynchronous Drive copies.
- `LlamaCppServer.extra_args` can carry cache flags, but the current client does not explicitly assign `id_slot`, save slots, or restore them. Actual b5401/b8233 capabilities must be probed, not assumed from current HEAD documentation.

## What this part adds

```text
src/steadyhand/tools_memory.py           # Native handlers, schemas, policies, security specs
src/tidemark/memory_tools.py             # Validation and server-side operation construction
src/tidemark/memory_replay.py            # RecordedMemoryBackend and resume recovery
src/tidemark/session.py                  # Safe-boundary checkpoints and logical resume
src/tidemark/slots.py                    # Optional llama.cpp slot probe/save/restore adapter
src/toolsmith/backends/llama_cpp.py      # Explicit cache/slot request and endpoint support
src/steadyhand/client.py                 # Per-episode slot identity and cache metadata
src/steadyhand/replay.py                 # Recorded-memory/resume integration
tests/test_tidemark_memory_tools.py
tests/test_tidemark_memory_replay.py
tests/test_tidemark_session.py
tests/test_tidemark_slots.py
tests/test_steadyhand_memory_security.py
tests/fixtures/tidemark_sessions/
```

No hosted memory service or new core dependency is introduced. Slot save/restore is optional and does not block the core eight-part completion when the pinned server cannot support it safely.

## Frozen native tool catalog

A Tidemark session includes `read_blob`, `memory_write`, `memory_search`, and `memory_forget` in its frozen registry from the beginning, even if the current task policy later masks or denies one. Never add `read_blob` only after an observation overflows or memory tools only after retrieval; doing so would change segment 0.

### `memory_write`

Closed model arguments:

```text
key, value, aliases?, valid_from?, valid_to?, source_refs[], supersedes?
```

- `Effect.IDEMPOTENT_WRITE`; deterministic exact-call idempotency plus the part 5 memory id. A blind retry after an unknown commit is not allowed: verify the receipt/read first.
- `SecuritySpec(output_trust=TRUSTED, sinks=(EXTERNAL_EFFECT,))`; the receipt contains ids/revisions/status only, not the stored fact text. Trust of the receipt does not mean the fact is trusted.
- Task policy includes `memory_write` in `taint_sensitive_tools`, so an untrusted episode gets ASK even when the basic tool rule otherwise allows the write. An explicit DENY remains DENY.
- `supersedes` widening or a cross-family update is denied unless the task explicitly grants it. Approval of a tainted write does not change the stored trust to trusted.

The handler receives trusted execution context. It validates every source ref against the journal/authorized lineage, checks that the cited evidence supports the required source contract, derives trust from the episode's pre-call taint, assigns namespace/time/embedding identity server-side, and only then executes the part 5 transaction.

### `memory_search`

Closed model arguments:

```text
query, k?, as_of?, valid_from?, valid_to?, key_filter?
```

- `Effect.READ`, idempotent; no blind retry on malformed input.
- `SecuritySpec(output_trust=UNTRUSTED, sinks=())`; stored text may carry injection even when the storage service is trusted.
- Server caps `k`, fixes namespace/ranking/embedding configuration, and rejects raw SQL, FTS operators used as syntax, arbitrary namespaces, model names, paths, and unbounded time ranges.
- Results include memory ids, values, key variants, source refs, validity, trust, revision, and bounded score components. The model cannot ask for deleted/superseded history unless a separate privileged audit tool is explicitly enabled.

### `memory_forget`

Closed model arguments: `memory_id`, `reason_code`, `source_refs[]`.

- `Effect.IDEMPOTENT_WRITE`; exact repeated deletion returns the tombstone receipt without a new effect.
- `SecuritySpec(output_trust=TRUSTED, sinks=(EXTERNAL_EFFECT,))`; the receipt has no deleted content.
- Default task policy uses `ask_tools=("memory_forget",)` and includes it in taint-sensitive tools. Forget is not a convenient way to hide evidence before evaluation.
- Namespace purge, arbitrary key/value deletion, and physical deletion of journal/blob evidence are not exposed.

### `read_blob`

Keeps part 3's read/idempotency/untrusted-output policy and catalog checks. Memory and blob tools use the same validation, guard, authorization, fault, recovery, and output ordering as every other tool. A recovery alternative cannot turn a denied memory operation into unrestricted SQL/filesystem access.

## Execution context and provenance

Extend `ToolExecutionContext` with:

```text
episode_id, step, read-only journal view, namespace,
blob_catalog, memory_store, memory_config,
taint_before, authorization_seq, logical_clock, operation_recorder
```

The context is server-built per episode and is never copied into model JSON or the manifest as a connection object. Parallel pure reads may each receive an independent read connection/view; writes serialize according to the existing effect/sink rules. Store locks are not held across model calls or embedding computation.

Embedding computation for a write may be expensive. Authorize and validate first, compute outside the SQLite transaction, then open the short write transaction with the expected revision/precondition. If authorization or the predecessor changes before commit, revalidate rather than committing stale state. Timeouts after an unknown commit follow verify-before-retry semantics.

Source refs for cross-session memories are authorized only when the session's namespace/lineage grants access to the referenced journal. A model cannot cite an arbitrary episode id to attach provenance or read another tenant's history.

## Automatic retrieval and taint

Automatic retrieval happens only at session creation or a declared compaction epoch, using a trusted `RetrievalPlan` from task/session state. It is not a hidden tool call inserted among native calls.

Write `memory_retrieval` before the affected `context_render`:

```text
step, purpose, query, query_sha256, ranking_config_sha256,
result_block, result_block_sha256, memory_ids, revisions,
score_components, strongest_untrusted_source, catalog_revision,
taint_before, taint_after
```

`result_block` is the exact bounded/redacted canonical content placed in segment 1 or handed to the compactor. It is needed for replay; a list of ids alone is insufficient if the database later changes.

If any retrieved fact is untrusted, the episode becomes untrusted before the next tool authorization and journals the appropriate taint transition/source. A trusted database or a trusted retrieval algorithm does not sanitize the content. Pinned memory changes only at the declared boundary and preserves the part 2 prefix contract.

## Memory operation audit and replay

### `memory_operation`

Write between the call's `policy_decision` and `tool_result` (or at the equivalent trusted-handler point when Bulkhead is off but Tidemark memory is explicitly enabled):

```text
step, tool_call_id, via, operation, request_sha256,
namespace, memory_id, family_id, revision_before, revision_after,
source_refs, content_trust, supersedes, result_sha256,
store_config_sha256, embedding_identity, mutation_sequence,
approval_grant_id, receipt
```

Do not duplicate the fact value in this event; the journaled tool arguments/results follow existing redaction rules. Operation, request/result hashes, source refs, trust, revision, and receipt replay exactly. Database timings and embedding timings are separate volatile metrics.

### Default replay: recorded effects

`RecordedMemoryBackend` reconstructs each expected request from validated arguments and execution context, checks `request_sha256`, and returns the recorded typed receipt/search result without opening a writable DB connection. Automatic retrieval uses `memory_retrieval.result_block`, not a new query against the current database.

- In-memory mock-store handlers still reexecute normally.
- Blob artifacts are read-only/hash-verified as in part 3.
- Memory writes, deletes, and searches do not touch the user's current SQLite state.
- Every policy, approval, taint, operation, tool result, context render, and model message must match at its original sequence.

A separate `live_memory_replay_report` may run against a restored **throwaway/run-owned** DB and fresh local embedder. It writes a comparison report only and never overwrites the original journal or production namespace. Control-plane replay does not prove SQLite integrity or embedding quality; those have their own tests.

### Live resume recovery versus replay

Ordinary replay is read-only. Resume after a disconnect is different: it must reconstruct a run-owned live store before new operations. Use the latest verified snapshot plus the database mutation log/journal to reach the exact recorded revision, verifying every operation hash and resulting revision. Then replay the recorded prefix read-only from the agent loop's perspective and hand subsequent calls to the restored live backend.

If another writer advanced a shared namespace, do not roll it back or overwrite it. Resume only when the configured run-owned namespace/revision matches, or stop for an explicit reconciliation/new session. Evaluation uses isolated run-owned databases so this rule is deterministic.

## Logical session checkpoint

At a safe step boundary, write a versioned `SessionCheckpoint` locally and mirror it atomically:

```text
episode_id, journal_revision/last_seq, context_config_sha256,
epoch, accepted_summary/hash, compacted_through_seq,
logical_history_hash, recent_state, recitation_state,
previous_prompt_token_artifact, previous_stable_prefix_length,
blob_catalog_revision/required_digests,
memory_snapshot_manifest/revision, next_step,
model/tokenizer/template/server identity, slot_identity optional
```

The previous prompt token ids may be stored as a bounded binary artifact for PrefixLint; JSONL retains only their hash. The checkpoint contains no API keys, environment dump, raw hidden reasoning, or SQLite connection.

Resume validates the manifest, journal tail, context state, every required blob, memory snapshot/revision, tokenizer/template/model identity, and server slot context/KV/cache settings. It reconstructs logical context first. Missing optional KV state causes a declared cold prefill; missing logical state is a correctness failure and stops resume.

Slot assignment for ordinary live calls is explicit (`id_slot` and `cache_prompt` when supported) through part 1's slot semaphore. Two declared policies serve different goals: `pinned` retains the episode's slot across steps for cache measurement, while `work_conserving` retains only affinity metadata, allows another ready episode to use the slot during a long tool/I/O wait, and records migration/eviction when the episode returns. Extra episode workers may prepare/run tools but can never exceed the model-call slot limit. The compactor follows the same lease policy and uses a dedicated slot only when the fitted/autotuned plan selects one. Record slot id, affinity policy, migrations, and cache flags, but never include them as text in prompt segment 0. Never change policy mid-run; cache comparisons normally use `pinned`, while throughput autotuning may select `work_conserving` for every compared mode.

## Optional llama.cpp slot save/restore

First probe the **pinned** server build and `--help`/endpoints for `--cache-prompt`, `--cache-reuse`, `--slot-save-path`, explicit slot selection, and save/restore operations. Record the exact API and limitations. Unsupported behavior produces `slot_resume_unavailable`, not a fake restored flag.

When supported:

1. Start with `--slot-save-path` on local SSD and the existing cache settings/KV type.
2. At the same safe boundary as the logical checkpoint, quiesce the episode's assigned slot and save it under a deterministic checkpoint/slot name.
3. Verify the slot file is complete and belongs to the current model/build/context/KV/prompt fingerprint; atomically mirror it and its manifest to Drive.
4. On resume, restore only to the same compatible server/slot configuration, then issue a short no-generation or metadata validation if the API provides one.
5. Compare the next prompt's PrefixLint floor and actual `prompt_evaluated_tokens` with a cold-resume control. A restored file without measured reuse is not evidence of a warm resume.
6. Delete superseded local/Drive slot snapshots under the declared retention policy only after newer logical checkpoints and artifacts are durable.

Never point llama-server directly at a Drive FUSE path for active save/restore I/O. Never restore a slot from a different GGUF, tokenizer/template, context shape, KV type, or prompt epoch. If validation fails, discard only the performance snapshot and continue from logical state with `cache_restored=false` and measured full prefill.

This optional path is not used in the primary ablation unless every compared mode has the same slot-restore policy. Core resume correctness must pass with slot snapshots disabled.

## Utilization, ordering, and checkpoint overlap

- Keep Bulkhead validation/policy checks synchronous with their call, but execute independent memory searches, blob reads, embedding requests, and checkpoint copies through part 1's bounded pools. More episode workers are useful only while model calls remain slot-leased.
- Serialize policy/taint/tool-result events exactly as the current loop requires. Parallel reads still finish and journal in call order, and same-turn taint-sensitive writes remain sequential. A faster queue must not create a new authorization path.
- Use the work-conserving policy only with explicit slot-migration/cache-eviction telemetry. A slot optimized for throughput is not simultaneously claimed to provide pinned-cache reuse; run the 90% cache-target smoke under `pinned` on both GPU families.
- Prepare the next checkpoint's canonical metadata while the GPU works, then perform only the short required commit at the safe boundary. SQLite online backup, blob mirroring, and Drive upload happen off the model critical path after that commit.
- Bound in-flight snapshots and do not start a new snapshot until local memory/disk ceilings allow it. On T4, preserve RAM/disk headroom before adding prefetch; on L4, scale queues only when throughput improves. Resume still waits for the required checkpoint set, never for an optional Drive copy whose local artifact is already complete.
- Autotune `pinned` versus `work_conserving`, I/O depth, and snapshot batching using whole-phase wall time and correctness checks, then freeze the selected policy for scored execution. Record policy/security wait separately from actual tool and GPU time.

Tests include slow memory reads, approvals, concurrent episodes, slot migration, and interrupted snapshots under T4/L4 budgets. Assert zero security-order changes, bounded resource use, no model-slot overcommit, no persistent writes during replay, and identical resumed prompts after cold recovery.

## Notebook integration

Add cells to `notebooks/04_tidemark.ipynb` for:

- `tidemark-security-gate`: scripted clean/tainted write, ASK/decline, approved linked retry, supersede, forget, retrieval poisoning, and no-effect replay;
- `tidemark-session-gate`: interrupted journal/memory/blob recovery with slot snapshots disabled;
- `tidemark-session-smoke`: real model pause/resume and cold-prefill baseline;
- `tidemark-slot-smoke`: optional, only after the pinned-server capability probe passes.

Records use `/content/tidemark/` locally and `MyDrive/toolsmith/records/<gpu>/tidemark/` for durable snapshots. Journals, memory DB snapshots, blob catalogs/objects, summaries, and optional slot files are copied as related checkpoint sets. Drain the copy queue before declaring a checkpoint durable or disconnecting.

## Tasks

1. Register frozen memory tools and server-only execution resources; implement provenance/namespace/time validation and idempotent handler behavior.
2. Add task policies/security specs and same-turn taint tests. Wire approved retries without changing the original episode or stored trust.
3. Implement automatic boundary retrieval, exact result-block journaling, and retrieval-driven taint.
4. Add `memory_operation`, `RecordedMemoryBackend`, read-only replay, throwaway live comparison, and run-owned resume recovery.
5. Implement logical checkpoints, artifact-set durability, previous-prompt lint restoration, and cold logical resume.
6. Add explicit slot ids/cache flags and probe the pinned llama.cpp API. Implement optional slot snapshots only behind that probe.
7. Add notebook security/session gates and real smoke cells with local/Drive checkpoint sets and bounded infrastructure handling.
8. Implement pinned/work-conserving slot leases and bounded security/I/O/checkpoint pipelines. Autotune them separately on T4 and L4, then freeze the selected policy and record migration, utilization, and durability evidence.
9. Run the phase-delivery protocol: commit the tested security/resume implementation, write the canonical Handoff for part 7, commit it, automatically push both commits, and verify remote `main`.

## Tests

- Tainted web/exec/memory content followed by `memory_write` asks; DENY stays DENY; approval produces one exact linked grant and stored trust remains untrusted.
- Same-turn untrusted read -> memory write cannot run in parallel ahead of taint. Recovery retry/verify/switch receives fresh authorization.
- A model cannot select another namespace, trust label, database path, embedding model, arbitrary FTS query, or unauthorized source episode.
- Search results taint the episode; automatic untrusted retrieval taints before the next sensitive sink and leaves segment 0 unchanged.
- Repeated exact writes/deletes have one effect; unknown-commit recovery verifies rather than duplicating supersession.
- Default replay opens no writable memory store, restores no slot, spawns no process, and reproduces every request/message/event. Editing trust, refs, revision, result block, or receipt fails exactly.
- Checkpoint resume after crashes at each commit boundary reconstructs identical state; missing/corrupt logical artifacts stop, while missing optional slot state yields an explicit cold prefill.
- Cross-writer revision mismatch stops rather than rolling back another session; tombstones survive snapshot/recovery.
- Slot fixtures cover unsupported flags/endpoints, wrong model/KV/context/prompt, partial files, and save/restore timing without calling an unverified restoration a hit.
- Existing Bulkhead approval/taint/replay and all legacy golden tests remain green.

Run:

```bash
.venv/bin/python -m pytest tests/test_tidemark_memory_tools.py tests/test_tidemark_memory_replay.py \
  tests/test_tidemark_session.py tests/test_tidemark_slots.py \
  tests/test_steadyhand_memory_security.py tests/test_bulkhead_replay.py \
  tests/test_steadyhand_replay.py
.venv/bin/python -m pytest
```

## Phase delivery and handoff

After all security/replay/session tests and required T4/L4 security/session smokes pass, follow part 1's mandatory two-commit/one-push protocol automatically. Optional slot restoration remains explicitly optional and cannot be labeled supported without its own probe evidence.

1. Stage only part 6's memory-tool/security/replay/session/slot changes, sanitized fixtures, tests, notebook, README, and deliberate shared changes. Exclude DBs, blobs, slot files, checkpoint payloads, credentials, and unrelated dirty paths.
2. Create the implementation commit `Tidemark: complete part 6 secure memory and resume`, update `Tidemark/Handoff.md`, then create `Tidemark: hand off part 6`.
3. Record the implementation SHA, tool schemas/policy/spec hashes, taint/approval and no-effect replay invariants, session/checkpoint format, memory/blob revisions, cold-resume evidence, affinity/compactor policy, optional slot capability status, per-GPU run ids, and exact artifact restore/next-session commands.
4. Point the next session to part 7 and its first dataset/ablation runner gate. Push both commits to `origin/main` and verify remote `main` equals the Handoff commit before evaluation.

Any security/taint/replay/resume mismatch, unsupported optional feature presented as working, missing required two-GPU evidence, or failed publication blocks completion. A fresh session must never be told to rely on a surviving Colab kernel or slot process.

## Acceptance

- All memory/blob tools pass the ordinary Bulkhead path, and tainted writes/deletes require the declared approval without trust laundering.
- Default replay reproduces exact prompts and operations with zero persistent memory effects; live resume reconstructs the same run-owned state before continuing.
- A session resumes correctly from journal + context + memory + blobs with slot snapshots disabled. Optional KV restore is separately probed and measured.
- Retrieval updates occur only at explicit boundaries and preserve deterministic pinned context; untrusted retrieval taints before sensitive actions.
- Durable checkpoints include consistent related artifacts and never contain credentials or claim Drive durability before copies drain.
- Security/session smokes run on both T4 and L4 without model-slot overcommit or CPU/IO pool oversubscription; throughput and pinned-cache claims use their correct, explicitly separate scheduler policies.
- The tested implementation and canonical part 6 Handoff are committed and automatically pushed, remote `main` is verified, and part 7 can start in a fresh session with exact security/checkpoint/artifact context.

## Pitfalls

- Replaying a memory write against the user's current DB, or querying a changed DB and expecting old messages to match.
- Treating human approval or SQLite storage as proof that untrusted content is now trusted.
- Adding tools to the registry only when needed and silently invalidating segment 0.
- Holding the episode store lock or SQLite transaction while computing embeddings or waiting on a model.
- Resuming into a changed shared namespace by rolling back someone else's writes.
- Calling a slot file the source of truth, restoring it across incompatible server/model settings, or claiming a cache hit without timings.
- Writing active SQLite or slot files directly to Drive and ignoring torn/FUSE-stalled I/O.

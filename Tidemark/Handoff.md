# Tidemark Handoff

This is the canonical operational handoff for a fresh implementation session. The eight numbered files in this directory are the plan; this file is current state, not a ninth plan part. Update it atomically **before the user's separate GPU run** with status `pushed_acceptance_pending`. The implementation and pre-GPU Handoff commits are pushed after CPU tests so a fresh Colab session can pull and follow them. The user reviews GPU results and any fixes independently; there is no later Handoff or agent bug-fix cycle in this workflow. Earlier states remain available in Git history.

## Status

- **Current state:** parts 1, 2, and 3 are `pushed_acceptance_pending`. All three implementation packages are on `origin/main`. Their T4/L4 acceptance has not run, so no live result is claimed. The user will review GPU results and any fixes independently; no additional Handoff or agent bug-fix cycle is scheduled.
- **Last delivered implementation phase:** part 3, restorable observations (CPU-verified; live review pending).
- **Next implementation action:** part 4 (`Tidemark/04-cited-compaction-and-replay.md`). It is blocked only until this Handoff commit is verified on `origin/main`. After that it may start; it does not wait for the user's GPU review of parts 1–3.
- **Branch / remote:** `main` / `origin`.
- **Part 3 commits:** implementation `eaee4ed04fefaea1d3c984ba66f2019ea049f931` (tree `5268e4c8602a6768bfb30da2eacb0a501e6f4424`, 76 paths, base `dd0430c87d6cae0d7e60fe6a55f487e28b47ffaa`). The Handoff commit is the commit containing this file; check it with `git log -1 -- Tidemark/Handoff.md`.
- **Part 2 commits:** implementation `1f43c2a4ad525b13ba70db2f4430cc8e5592cc36` (tree `bf81b71215c91878fb2b1a37eab98578473af334`), workflow revision `06003b6d03d773424ffe6f940291860d51d40809`, parts 1–2 pre-GPU Handoff `dd0430c87d6cae0d7e60fe6a55f487e28b47ffaa`.
- **Part 1 commits:** implementation `54612db611a0c7ed313d87b2f7241bad595bd359` (tree `df519f12b5f523d6c5c291552b84bf17d2bff729`), the user's own `b29d0086f8d4eb3e62a6d0bdbf911834b52ec2b6` (port probe and notebook 01), and Handoff `eb909584792a2385b5c20196837b9263e73e77b6`.
- **Previous Handoff commit:** `dd0430c87d6cae0d7e60fe6a55f487e28b47ffaa`.
- **Test branch:** none.
- **Written:** 2026-11-26T12:16Z (UTC).

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

## Part 3 delivered scope (on main, CPU-verified)

### Compatibility

- `run_episode(..., context=None)` is byte-for-byte the old path: the history append and `tool_result` keep their old order, and no new event, field, or manifest key appears. The Steadyhand goldens still replay, and the pinned golden hashes in `tests/test_steadyhand_context.py` pass.
- Parts 1 and 2 sessions without a shaper behave as before: no `observation_shaped` event, and the same messages. `ContextConfig.as_dict()` adds `observations` only when shaping is on, so part 1 and 2 configuration hashes (and their episode ids) are unchanged.
- `build_registry()` is unchanged. `read_blob` is opt-in (`build_registry(context_tools=True)`) and registered from session creation. `registry_for` also finds the context-tool registries.
- `JOURNAL_VERSION` stays 1: `observation_shaped` is a new optional event. `tidemark.__version__` stays 0.1.0.

### The observation path

With a session, `steadyhand.loop._finish` runs, in order: journal the raw `tool_result` (its seq is the source), `ModelSeam.shape_result` (the session shapes once, persists any blob, and the seam journals `observation_shaped` right after the `tool_result`), append the shaped message, then Bulkhead taint from the raw result and its seq. The next call is decided only after that. Typed errors (validation, guard, denial, ASK, timeout, parse, handler error) are never shaped. Results within the cap are left alone, and so is a reduction that would not shrink the message (`no_gain`). Audit mode (`ObservationPolicy(audit=True)`) records `reducer="none"` events for untouched results.

- **Envelope:** `{"ok": true, "value": <shaped>, "context": {"shape", "omitted", "preview_tokens", "blob", "bytes", "media_type", "restore"}}`. Exec results use `"blobs": {stream: {blob, bytes, media_type}}` instead of a single blob. Projections add `omitted_fields` and `cut` (pointers and counts, never values). An offloaded string becomes a typed object `{blob, media_type, original_kind, preview, truncated}` (pages also carry `title`). Inline markers read `[tidemark: omitted N lines (a-b), bytes [x, y); read_blob offset x]`. `context` is provenance, not a security label.
- **Policy** (`tidemark.observations.DEFAULT_POLICY`, `ObservationPolicy`/`ReducerRule`, version 1, sha256 `ec01bef050f86b58f2fad0361e2be79723ab4609bfe8f603ceae526b7e012f70`): `run_shell`, `run_python`, and the corpus exec tools use `exec_streams_v1` (1,200 tokens); `read_log` uses `log_head_tail_v1` (1,200); `read_text` uses `text_head_tail_v1`; `fetch_page` uses `html_text_v1` (1,000, body at `/body`); `get_order`, `get_ticket`, `list_payments`, and `search_orders` use `json_projection_v1` (768). The projections leave out free text that outside parties write, and `check_projections` checks every pointer against the registry's result schemas. `read_blob` is never shaped. Unknown tools get `json_bounded_v1` (768). The cap is real tokens on the final message content.
- **Reducers** (`tidemark.reducers`, `REDUCER_VERSION = "tidemark-reducers-v1"`): a reducer plans the complete payload(s) and ordered preview candidates, and `choose` binary-searches for the first candidate that fits the cap. A candidate that is still over the cap at the most aggressive level is reported (`over_cap`). Log: tier 1 is errors, exceptions, assertions, failures, compiler errors, linker errors, and kills; tier 2 is warnings and notes, paths with a location, and test ids. Only consecutive exact duplicates are coalesced. Under pressure, the head and tail shrink first, then tier 2 goes, then only the earliest and latest critical lines remain. Long lines are clipped around their first critical match. Exec streams are reduced separately and keep the gate's `[BEGIN/END UNTRUSTED OUTPUT id]` lines, with one blob per stream (its complete gated text). JSON bounding takes scalar members first and limits depth without changing types; the blob is canonical JSON (sorted keys, finite numbers, two-space indent). HTML: visible text, `[text](href)` links, table cells joined by `|`, verbatim `pre` blocks; script, style, template, and `nav` content stay out of the preview but are in the blob (the original page).

### Blob store, catalog, and namespace invariants (`tidemark.blobs`, `BLOB_FORMAT_VERSION = 1`)

- Objects live at `<root>/<namespace>/<aa>/<full sha256>`. Each is written to a temporary file in the same directory, fsynced, then hard-linked into place, and the directory is fsynced. An existing digest is always verified before reuse; different bytes raise `BlobIntegrityError` and are never overwritten. Invalid UTF-8 is refused.
- The catalog is `<root>/_catalog/catalog.jsonl`: a header (`catalog_id`, version), then one entry per `(namespace, sha256)` with `catalog_revision` = previous + 1. An entry is appended under `flock` and fsynced only after its object is complete. A torn tail is ignored and cut before the next append; a gap, duplicate, or bad field is refused. A crash can leave an uncataloged object, never a dangling entry.
- Reads (`BlobStore.read`/`NamespacedBlobs.read`) resolve only a handle cataloged in the caller's namespace; a known digest from another namespace is `BlobNotFound`. Handles must be `blob://sha256:` plus 64 lowercase hex characters. Offsets and limits are non-bool integers, at most `max_read_bytes` (16,384). Symlinked or non-regular path components are refused, and the size is checked on every read. The digest is checked on full reads and on `verify`. Text is returned only at UTF-8 boundaries: an offset inside a character is refused, and a range end inside one moves back (`boundary_adjusted`).
- The namespace is server-chosen and part of `ContextConfig.observations` (`ObservationConfig`: policy, namespace, `catalog_id`, `max_read_bytes`, blob format, reducer version), so it is an identity field. The catalog id comes from the store's header, so resume needs the same (Drive-restored) catalog. The benchmark uses one namespace per episode (`<run label>.<observation id>`).
- Replay opens the store `read_only=True`. The shaper then calls `verify_put` (the entry must exist and the bytes must match) and never writes.
- **Checkpoint:** the journal is the checkpoint. Each `observation_shaped` names its blob parts and catalog revision, and `tidemark.blobs.checkpoint(records)` gives `{catalog_revision, digests}`. A session's `check_journal` (called by `replay_episode` and by `run_claimed_episode` before resuming) verifies every referenced blob (entry, revision, size, digest) and raises `ReplayMismatch` (`field="blob_handle"`) before any model call. A changed reducer policy or namespace changes `context_config` and is refused as a manifest mismatch; nothing is reshaped.
- **Maintenance** is explicit: `python -m tidemark.blobs verify <root>` and `python -m tidemark.blobs gc <root> [--journals ...] [--delete] [--min-age-s 3600]`. GC removes only old uncataloged objects and stale temporary files, and only with `--delete`. Cataloged objects are never deleted (`retained_until` is recorded as null).
- **Durability:** `BlobMirror` copies new objects (`copy_file_atomic`) and catalog snapshots to Drive on a background thread; failures are recorded, not raised. `restore(drive, local)` rebuilds a local store and verifies every object.

### `read_blob` and trusted execution resources

- `steadyhand.tools_context.read_blob(handle, offset=None, max_bytes=None)`: a closed schema inside the grammar subset (`anyOf [integer, null]`). Handle syntax and ceilings are semantic checks. `READ_BLOB_POLICY` is a read, idempotent, retried only on timeout, rate limit, or server error, with no verify tool and no alternatives (so recovery cannot switch to another file reader). `READ_BLOB_SECURITY` is no sandbox, no sinks, `UNTRUSTED` output, so a read taints. With Bulkhead the task policy must list it (`store_task_policy(available_tools=("read_blob",))`); unlisted means ASK, and a missing spec is still denied.
- Results: `handle, sha256, media_type, encoding, offset, end_offset, next_offset, total_bytes, truncated, boundary_adjusted, content`. An unknown or other-namespace handle is `not_found` (field `/handle`). An offset past the end is `invalid_arguments` (new `steadyhand.store.ArgumentRejected`, mapped by the executor). Without a store it is a handler error.
- `SequentialExecutor.run(execution_context=...)` injects the context only into handlers whose signature names it, like `idempotency_key`; `SERVER_PARAMETERS` keeps it out of every schema. `SecuredExecutor` forwards it only after ALLOW, and `RecoveryContext` passes it to retries. The loop builds a `steadyhand.context.CallEnvironment` (episode id, step, tool, call id, read-only `JournalView`, taint) per allowed call in call order. The session returns `tidemark.resources.ToolExecutionContext` with the episode's `NamespacedBlobs`. Part 6 extends it with memory resources.

### Journal: `observation_shaped`

Fields: `step, source_seq, tool_call_id, tool, reducer, raw_content_sha256, shaped_content_sha256, raw_bytes, shaped_bytes, raw_tokens, shaped_tokens, omitted_ranges, blob_handle, blob_bytes, blob_parts, catalog_revision, preview_metadata`. `blob_parts` (`{part, handle, bytes}`) is an addition to the plan's list, because exec results have one blob per stream; `blob_handle`/`blob_bytes` name the first part. Values are checked on write and on read: `source_seq` must be the record just before (the write path also checks that it is that `tool_result`), at most 64 omitted ranges, metadata under 4 KiB, full handles, and a revision whenever there is a blob. Every field compares exactly in replay; timings stay in volatile shaper metrics.

### Throughput pipeline

- `ShapingPool(workers, max_inflight_bytes)` runs reductions on threads shared by every episode. Admission is FIFO and bounded by in-flight input bytes; an oversized input runs alone. Its counters feed `ResourceSampler.add_counts` (new). `ReducerCache` is byte-bounded and keyed by raw content, rule, reducer version, and tokenizer; a hit still writes that namespace's catalog entry. Local writes are synchronous and atomic; Drive copies stay off the critical path.
- `tidemark.eval.observations`: `shaping_candidates` (from vCPUs, llama threads, and the RAM budget, never the GPU name), `autotune_shaping` (the fastest profile whose journals replay cleanly; ties keep fewer workers), `run_observation_benchmark` (claimed, resumable episodes through the scheduler), `ForcedFirstCallClient`, `replay_benchmark`, and `CorpusScriptedModel` for CPU.

### Observation corpus

- `evaluation/data/tidemark_observations_v1.jsonl` plus `evaluation/data/tidemark_observations_v1/` (42 payload files, 2,838,946 bytes). It is built deterministically by `python -m tidemark.eval.observation_corpus`, and `--check` verifies the committed files. Categories: pytest 7, compiler 7, log 8, text 1, exec 4, JSON 8, HTML 7. There are 90 must-keep labels (69 `inline`, 21 `blob_handle`). Every payload is synthetic in real tool formats, with no third-party or private content. Exec values went through Bulkhead's real `sanitize`/`render` with its 64 KiB cap (one entry is deliberately capped).
- Coverage: a fatal line in the middle of a 4,200-line log; non-adjacent identical errors; 1,800 consecutive duplicates; a 30 KB minified line; ANSI and carriage returns, NUL/BEL/OSC, invalid UTF-8, a bidi override, and a forged wrapper line; 420 refunds; a 60 KB note field; 650 orders; deep nested flags; 261 tsc errors where only the earliest and latest fit (the middle error is restorable); and HTML with an href-only link, a script-held fake key, tables, `pre`, and entities.
- Corpus sha256 (`corpus_sha256`, canonical index): `66a7bf0625dd7cbd4f5e833c7e6e66d86b1beb197b44a96272fd114ffdd7fd07`. Index file sha256: `927cd8a33f9812a79768d4e0025aa12b574d3b9256851513f96f001db4c11a9c`. Reducer goldens: `tests/fixtures/tidemark_observations/reducer_goldens.json`.
- CPU gate result (fake tokenizer `regex-fake-v1`, so not a real-token claim): 42/42 shaped, 1,109,472 → 33,787 tokens (0.9695 reduction; median per entry 0.9596), 90/90 labels pass (72 inline, 18 restorable only; 69/69 inline-required met), no forbidden text in previews, no over-cap entry, deterministic, read-only re-verification identical, and range round trips equal.

### Paths

| Path | What changed |
|---|---|
| `src/tidemark/blobs.py` (new) | `BlobStore`, `BlobCatalog`, `BlobRecord`, `NamespacedBlobs`, errors, `parse_handle`, `checkpoint`, `journal_references`, `verify_references`, `collect_orphans`, `verify_all`, `BlobMirror`, `restore`, `python -m tidemark.blobs verify|gc` |
| `src/tidemark/reducers.py` (new) | Reduction plans and candidates, `choose`, line documents and selections, log/text/exec/JSON/HTML planners, page-text extraction, pointer helpers, `REDUCER_VERSION` |
| `src/tidemark/observations.py` (new) | `ReducerRule`, `ObservationPolicy`, `DEFAULT_POLICY`, `check_projections`, `ObservationConfig`, `ObservationShaper`, `ReducerCache`, `ShapingPool`, `ShapingSupport` (session mixin), `envelope` |
| `src/tidemark/resources.py` (new) | `ToolExecutionContext` |
| `src/tidemark/eval/observation_corpus.py` (new) | Deterministic corpus builder and `--check` |
| `src/tidemark/eval/observations.py` (new) | Corpus loading and validation, `score_entry`, `summarize_scores`, `observation_gate`, `round_trip`, corpus registry, `ForcedFirstCallClient`, `CorpusScriptedModel`, benchmark runner, replay check, shaping autotune |
| `src/steadyhand/tools_context.py` (new) | `read_blob`, its semantic check, policy, security spec, result schema, `register_context_tools` |
| `src/steadyhand/context.py` | `JournalView`, `CallEnvironment`, `ObservationSource`, `ShapedObservation`, `ModelSeam.shape_result`/`execution_context`, `check_recorded_journal` |
| `src/steadyhand/loop.py` | Raw `tool_result`, then shape, then append, then taint (with a session); per-call execution context built in `_plan` and passed to the first attempt and to recovery |
| `src/steadyhand/journal.py` | `observation_shaped` event, value checks (`check_observation_values`), adjacency check on read and write |
| `src/steadyhand/executor.py`, `registry.py`, `security.py`, `recovery.py` | Server-only `execution_context` injection (`SERVER_PARAMETERS`), forwarded by `SecuredExecutor` after ALLOW and by `RecoveryContext`; `ArgumentRejected` -> `invalid_arguments` |
| `src/steadyhand/store.py`, `validate.py` | `ArgumentRejected`; `NotFoundError(field=, hint=)`; `rejected_argument_observation` |
| `src/steadyhand/tools.py`, `replay.py` | `build_registry(context_tools=)`; `registry_for` tries context-tool registries; replay and resume call the session's `check_journal` first |
| `src/tidemark/types.py` | Optional `ContextConfig.observations` (in `as_dict` only when set) |
| `src/tidemark/assembler.py`, `baseline.py` | Both sessions take `observations=` (`ShapingSupport`); the recitation lists open blob handles |
| `src/tidemark/eval/baseline.py` | `fact_available(..., blobs=)` blob path; `EvidenceProbe` passes the part 3 hooks through |
| `src/tidemark/runtime.py` | `ResourceSampler.add_counts` (shaping-pool counts in samples) |
| `src/tidemark/__init__.py` | Package docstring |
| `evaluation/data/tidemark_observations_v1.jsonl`, `evaluation/data/tidemark_observations_v1/` (new) | The 42-entry corpus |
| `notebooks/04_tidemark.ipynb` | `tidemark-observation-gate` (CPU, after `tidemark-prefix-gate`), `tidemark-observation-benchmark` (live, after `tidemark-prefix-smoke`); intro and runtime notes |
| Tests (new) | `tests/test_tidemark_{observations,reducers,blobs}.py`, `tests/test_steadyhand_context_tools.py`, `tests/fixtures/tidemark_observations/reducer_goldens.json` |
| Tests (changed) | `tests/test_notebook_04.py` (new cells; the gate cell executes; the benchmark cell runs end to end on CPU fakes, including a resume from Drive), `tests/tidemark_support.py` (`shaping_session`, `tool_registry`, `big_log`) |
| `README.md` | Part 3 description and status |

### Part 3 verification evidence (CPU only)

Commands run on the committed implementation tree (WSL2, Python 3.13.16, `.venv`):

```bash
.venv/bin/python -m pytest tests/test_tidemark_observations.py tests/test_tidemark_reducers.py \
  tests/test_tidemark_blobs.py tests/test_steadyhand_context_tools.py \
  tests/test_bulkhead_output.py tests/test_steadyhand_replay.py
# 161 passed
.venv/bin/python -m pytest
# 2053 passed, 54 skipped (the skips are the existing opt-in Bulkhead integration and AgentDojo suites)
.venv/bin/python -m compileall -q src tests                    # ok
git diff --check                                               # ok (new files checked for trailing whitespace and final newlines)
.venv/bin/python -m tidemark.eval.observation_corpus --check   # corpus matches the builder
```

- On the base tree (`dd0430c`), the full suite had 1,940 passed and 54 skipped.
- Tests (CPU, fake-backed) cover:
  - typed errors and small results byte-identical to a run without a session;
  - `tool_result` → `observation_shaped` → `taint_transition` order and seqs;
  - shaping once, with `expected_append` lint on later steps;
  - 1,000-run byte determinism;
  - replay identical with zero filesystem changes, and replay refusing an edited or deleted blob;
  - a changed policy refused as a `context_config` mismatch;
  - shape evidence edits caught at their event;
  - resume that verifies blobs and continues live, and resume refusing a missing blob with no model call;
  - `read_blob` through validate → guard → Bulkhead → handler, with taint;
  - unlisted (ASK), denied, and missing-spec cases;
  - other-namespace handles not found and no recovery switch;
  - the full blob read contract (UTF-8 boundaries, empty objects, final partial ranges, duplicate and concurrent writes, torn catalog tails, torn, edited, and symlinked objects, traversal, over-limit requests);
  - FIFO byte-bounded pool admission;
  - reducer-cache reuse across namespaces;
  - a Drive mirror off the shaping path;
  - concurrent scheduler episodes that replay exactly;
  - the reducer goldens.
- Notebook cells exercised by tests:
  - `tidemark-observation-gate` executes for real.
  - `tidemark-observation-benchmark` executes end to end with fake model clients: real shaper, blob store, scheduler, sampler, mirror, autotune, and replay. A second run after deleting the local folder restores from the "Drive" copy and reruns nothing.
  - Only the model, the real tokenizer, and the GPU are missing there.

**Not exercised anywhere:** real-tokenizer corpus counts, a real model reading previews or calling `read_blob`, Drive FUSE latency, the T4/L4 shaping autotune, and shaping of live `bwrap` exec output.

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

None. No llama-server, background job, Drive mount, run id, manifest, journal, resource sample, blob store, or report from parts 1–3 exists. Model/GGUF/tokenizer/llama.cpp revisions, capacity and scheduler hashes, shaping profiles, catalog ids, and run ids will be produced by the notebook. Expected identities: Qwen3-8B Q4_K_M from `Qwen/Qwen3-8B-GGUF` on llama.cpp `b5401` (the shared Steadyhand provisioning cell with `STEADYHAND_PHASE=qwen3/wide`).

Every T4 and L4 cell (parts 1–3) is **untested**.

Expected live behavior to check, not a result: the seed set's mock-store observations are small (about 300 characters each), so on the required 16K path even a 40-step forced prefix should fit (roughly 7K tokens), with no truncation steps and no shaping. The corpus payloads are 5K–143K fake tokens raw, and the real tokenizer will count differently, so inline labels near a budget edge could become restorable-only or fail under the live gate. The live cell reports that rather than hiding it.

## Durable artifacts (once the user runs the notebook)

```text
local active state: /content/tidemark/
  baseline/<run_id>/                  spec <run_id>.json, <run_id>.scores.jsonl, journals/, resource_samples.jsonl, utilization.json
  smoke/<smoke_run_id>/               autotune candidates' runs
  report/<shape>/                     steps.csv, prefill_by_step.svg, summary.json, report.md
  observations/<obs_run_id>/          blobs/ (_catalog/catalog.jsonl + <namespace>/<aa>/<sha256>), journals/,
                                      tune/w<workers>-m<MiB>/ (unscored autotune runs), gate-blobs/, observations-summary.json
durable records: MyDrive/toolsmith/records/<gpu>/tidemark/
  cpu-gate.json
  capacity/<shape>-<plan sha>.json, capacity/plans.json (CapacityCache)
  parity/<run_id>.json, parity/<run_id>-cache-probe.json
  smoke/autotune-<gpu>-<stamp>.json
  baseline/<run_id>/                  mirror of the local run folder
  report/<shape>/
  prefix-gate.json                    part 2 CPU PrefixLint gate (both protocols)
  prefix/prefix-v1-<gpu>-<shape>/     part 2 prefix-stable run: spec, scores, journals, prefix-summary.json
  observation-gate.json               part 3 CPU corpus gate (fake tokenizer)
  observations/<obs_run_id>/          observation-gate-real-tokenizer.json, shaping-profile.json (frozen profile + measurements),
                                      observations-summary.json, journals/, blobs/ (BlobMirror: objects + catalog snapshot)
```

Default `obs_run_id`: `observations-v1-<t4|l4>-<shape>` (override with `TIDEMARK_OBSERVATION_RUN_ID`). Resume: rerun the notebook from the top. If the local blob catalog is missing and Drive has one, the benchmark cell restores the blob store (every object verified) and the journals first. It reuses the frozen `shaping-profile.json` instead of retuning, completed episodes are skipped, and an interrupted episode resumes from its journal after its referenced blobs are verified. A missing or corrupt blob stops the resume with `ReplayMismatch` before any model call. Keep the whole `observations/<obs_run_id>/` folder (journals and blobs together); journals without their catalog cannot be replayed. Never delete blobs because an episode finished or Drive is behind; use `python -m tidemark.blobs gc` only after scanning the journals, and only with `--delete` deliberately.

The baseline and prefix runs resume as before (`baseline-v1-<t4|l4>-wide`, override `TIDEMARK_BASELINE_RUN_ID`; a changed spec is refused with `BaselineMismatch`).

## Known issues and boundaries

- **Part 3 boundaries:**
  - CPU token counts use the regex fake tokenizer. Real-token reduction and real-token inline survival come only from the live `observation_gate(tokenizer=TIDEMARK_TOKENIZER)` in the benchmark cell.
  - The corpus is synthetic and real-shaped, not captured from real tools. `tests/fixtures/tidemark_observations/` holds only the reducer goldens; the payloads live in `evaluation/data/`.
  - In the benchmark, exec-shaped corpus entries are served by read tools (`run_tests`, `build_project`, `run_command`) in a corpus registry, not by the sandbox. Exec shaping of real `run_shell`/`run_python` results is covered by unit tests on gated values. The forced first turn sends the real request and discards its generation, as part 1 does. Answers that quote the evidence are reported, not gated.
  - Shaping uses threads, not the scheduler's process pool, because the tokenizer, blob store, and cache are in-process (HF fast tokenizers release the GIL; the CPU fake does not). The reducer cache is in-process only.
  - HTML omitted-range byte offsets are exact at extracted-line starts and approximate inside a clipped line. `nav` menus are left out of page previews.
  - `blob_parts` was added to the plan's `observation_shaped` field list (exec results have one blob per stream).
  - With Bulkhead, a tool missing from the task policy is ASK (Bulkhead's existing rule), so secured runs must list `read_blob`.
  - The catalog id is in the context configuration, so a fresh catalog gives new episode ids. That is deliberate: episodes are bound to the catalog that holds their blobs.

- **Part 2 boundaries:**
  - The planted breakers are generated in code (`tidemark/eval/prefix.py`), not stored under `tests/fixtures/tidemark_prompts/`.
  - The batch/ubatch and `--cache-reuse` autotune grid is not run in the notebook. The part 1 `capacity_candidates(batches=...)` and `autotune` mechanisms exist, but every scored part 2 run uses the frozen part 1 shape.
  - Prefetch uses the scheduler's prefetch thread, not the process pool, because rendered prefixes live in an in-process cache.
  - For Qwen3.5, re-serialized old arguments decode to an identical prompt. PrefixLint still reports the rewrite as a `violation`: old messages are immutable.
  - The live warm-reuse figure needs `cache_semantics.confirmed`. Without it, reuse is null and the smoke reports zero warm steps.
- **GPU review pending (user-owned):** no T4 or L4 run of parts 1, 2, or 3 has happened, so no live cache, latency, utilization, parity, or task-success result exists or is claimed. The user runs and reviews these separately; any fix they request is a separate task with its own tests and commits. It does not gate part 3 or part 4.
- **Pre-existing, not caused by Tidemark:** the golden manifests' episode ids (`ep_8393620aa57df809`, `ep_b345550151d9160c`, `ep_07922e66f89c0aa9`) predate Steadyhand part 6, which added `recovery_policy` and `system_prompt_variant` to the identity hash. Unchanged base code already derives `ep_268aa8ce0b358631`, `ep_b4a01a2f9f338bd8`, and `ep_7499cf1f6182975d` for the same scenarios (checked in a clean worktree of the base commit). Replay never compares ids, so the goldens still replay. The test pins both sets of ids; regenerating the goldens is a separate, deliberate decision.
- The part 1 autotune varies only the scheduler's episode-worker count on the smoke workload (no server restart). Affinity stays `pinned` because the baseline measures cache reuse. KV/flash is decided by the fit ladder, and batch, thread, and compactor-slot grids are implemented and CPU-tested (`capacity_candidates`, `autotune`) but not run in the notebook.
- Forced-prefix live steps send the real request (real prefill and cache counts) and then substitute the scripted calls. Their completion tokens and decode time are those of the discarded generation, and the slot cache holds the model's own output rather than the scripted turn. Rows are labeled `forced` and `model`.
- The `tidemark-capacity-probe` cell defines `stop_steadyhand_runtime` so the shared cleanup cell stops the Tidemark server.
- `required_availability`: `blob_handle` and `any_restorable` now also accept a blob the rendered request names and whose bytes contain the fact (`fact_available(..., blobs=...)`, through the session's namespaced reader). `memory` still reduces to inline until part 5. The part 1 seed set's observations are small, so they are never shaped and every live check there stays inline.
- The port-probe `SO_REUSEADDR` change is now on `main` (`b29d008`), so `fit_capacity`'s same-port restart after an out-of-memory shrink can reuse a port left in TIME_WAIT. The test branch does not have it. The working tree has no unrelated changes left.
- `results/` holds local scratch (for example `results/tidemark-chart-check/`) and is gitignored.

## GPU run guide (the user's independent review)

Runtime: a fresh Colab **T4**, then a fresh **L4** (records stay in separate per-GPU folders). Secrets are referred to by name only: `HF_TOKEN` and `LOCAL_KEY`, in `MyDrive/toolsmith/.env` or Colab Secrets. Model: Qwen3-8B Q4_K_M from `Qwen/Qwen3-8B-GGUF` on llama.cpp `b5401` with the `Qwen/Qwen3-8B` tokenizer (the shared provisioning cell under `STEADYHAND_PHASE=qwen3/wide`). Capacity is fitted at runtime: T4 starts at 2 slots of 16K, L4 at 4 slots of 16K, and the KV ladder is q8_0 with flash attention, then f16 with it, then f16 without. Slots shrink on OOM; per-slot context does not.

Run `notebooks/04_tidemark.ipynb` top to bottom. The `drive-and-secrets` cell pulls `main`. Cell order:

1. `environment`, `drive-and-secrets`, `install-project`, `hardware` (shared setup);
2. `tidemark-cpu-gate` (golden replays and seed-set validation), `tidemark-prefix-gate` (planted PrefixLint suite on CPU), then `tidemark-observation-gate` (corpus check against its builder, then must-keep scoring, determinism, read-only re-verification, and range round trips on CPU);
3. `tidemark-settings`, `steadyhand-runtime` (provisioning in the background);
4. `tidemark-capacity-probe`: envelope, capacity decisions, fitted server, scheduler, resource sampler;
5. `tidemark-tokenizer-probe`: parity against `/tokenize` (stops the notebook on a mismatch) and the repeated-prompt cache probe;
6. `tidemark-utilization-smoke`: episode-worker autotune on the N ≤ 10 tasks, then freeze;
7. `tidemark-baseline` and `tidemark-baseline-report`: the naive baseline on all 12 seed tasks, then `steps.csv`, `prefill_by_step.svg`, `summary.json`, `report.md`;
8. `tidemark-prefix-smoke`: part 2 live checks and the prefix-stable run;
9. `tidemark-observation-benchmark`: part 3. It re-scores the corpus with the real tokenizer, autotunes the shaping pool on every fourth entry (unscored; a profile counts only if its journals replay), then runs all 42 entries as episodes with the frozen profile. It replays every journal on CPU with a read-only store and drains the Drive mirror;
10. `cleanup`: stops the server, drains background copies, flushes Drive.

Optional settings (environment variables, set before `tidemark-settings`):
- `STEADYHAND_PHASE=qwen3/tall` for the 32K shape;
- `TIDEMARK_GENERATION_PATH` (default `buffered_stream`);
- `TIDEMARK_MAX_TOKENS` (512);
- `TIDEMARK_AFFINITY` (`pinned`);
- `TIDEMARK_BASELINE_RUN_ID`;
- `TIDEMARK_OBSERVATION_RUN_ID`;
- `TIDEMARK_AUTOTUNE` (`1`; `0` takes the first shaping candidate for the observation benchmark too);
- `TIDEMARK_SMOKE_MAX_N` (10).

After a disconnect, run everything again from the top. Completed units are skipped, and an interrupted episode resumes from its journal.

Review checklist (not a gate on the next part):
- **Part 1:**
  - parity passed;
  - the cache semantics probe is confirmed, or unconfirmed with evidence;
  - a legal fitted shape with recorded attempts;
  - the utilization summary, with explanations for any miss of the 95% slot or 85% CPU targets;
  - all 12 tasks scored, with infrastructure and did-not-fit outcomes kept visible;
  - `prefill_by_step.svg` distinguishes cold and ordinary steps (truncation rows are not expected on 16K; see *Live/runtime state*).
- **Part 2:**
  - the lint prompt equals the production prompt, with exact spans;
  - the live planted suite passes;
  - in the same-slot cache probe, the planted timestamp forces a near-full prefill while the legal append reuses the cache;
  - no live PrefixLint violation;
  - warm reuse is reported against the 90% target (a target, not a pass condition).
- **Part 3** (`observations-summary.json`, `observation-gate-real-tokenizer.json`):
  - the real-tokenizer gate passes: `survival_or_restorability` 1.0, `inline_required_met == inline_required`, no forbidden text, no blob failure, no over-cap entry;
  - real `token_reduction` and `median_entry_reduction` are reported alongside omitted ranges and restorability (no "nothing lost" claim);
  - autotune measurements per profile and the frozen profile;
  - `replay.passed` for every scored and tuning episode (no event-order divergence);
  - `drive_copies.failures` is empty;
  - pool `peak_inflight_bytes` ≤ the bound (or `oversized` counted), plus `min_ram_available_mib`;
  - shaper `reducer_s`, `queue_wait_s`, and `io_s`, and the `observations-scored` utilization summary (shaping overlapping slot occupancy);
  - `answers_with_evidence` and `read_blob_calls` (reported, not gated).

Artifacts worth keeping: everything under `MyDrive/toolsmith/records/<gpu>/tidemark/` listed in *Durable artifacts*. In particular: the run specs, scores, journals, `resource_samples.jsonl`, `utilization.json`, `prefix-summary.json`, the parity and cache-probe records, the report folder, and the whole `observations/<obs_run_id>/` folder (journals and blobs together).

## Next-session bootstrap (part 4)

1. Read this file and `Tidemark/04-cited-compaction-and-replay.md`. Check the repository:

   ```bash
   git status --short --branch --untracked-files=all
   git pull --ff-only origin main            # only with a clean worktree
   git merge-base --is-ancestor eaee4ed04fefaea1d3c984ba66f2019ea049f931 HEAD && echo part-3-present
   ```

   If the Handoff and the repository disagree, stop and report the mismatch.
2. Rerun the starting CPU gate:

   ```bash
   .venv/bin/python -m pytest tests/test_notebook_04.py tests/test_tidemark_assembler.py \
     tests/test_tidemark_observations.py tests/test_tidemark_blobs.py tests/test_steadyhand_context.py
   .venv/bin/python -m tidemark.eval.observation_corpus --check
   ```

3. Begin part 4. It builds on these interfaces:
   - `ContextState.advance_epoch(summary, keep_from)` and `ContextConfig.overflow` (only `exhaust` today; part 4 adds `compact`), with `compaction_pending` already on `context_render`.
   - Shaped observations are immutable history: compaction may drop whole recent turns, but must never re-shape an old observation. Cite evidence by journal seq (`tool_result`/`observation_shaped` `source_seq`) and blob handle; a cited handle must stay restorable (`check_journal` / `tidemark.blobs.verify_references`).
   - `RecitationState.open_blob_handles` is now filled from the session's shaped observations (newest kept).
   - `read_blob`, `ToolExecutionContext`, and `CallEnvironment` are the pattern for part 6's memory tools; `steadyhand.store.ArgumentRejected` maps handler-side argument checks to `invalid_arguments`.
   - Paired ablations can turn shaping on or off per mode: both `NaiveAppendTruncate` and `PrefixStableSession` take `observations=`.
4. Deliver part 4 with the plan's pre-GPU protocol: CPU tests, implementation commit, `pushed_acceptance_pending` Handoff with its GPU run guide, one push, and remote verification. If a push is rejected or authentication fails, keep the local commits and stop; do not rebase, merge, force-push, or retry indefinitely.

## Safety

Secrets are referred to by the names `HF_TOKEN` and `LOCAL_KEY` only. Journals redact both. The notebook passes them to clients and redaction and never prints or records them (a notebook test checks every live cell, the observation benchmark included). Records hold hashes and counts, not raw prompts or token ids; `observation_shaped` holds hashes, counts, ranges, and handles, never payloads. The corpus's script-held key (`sk_test_FAKE_...`) is a labeled fake used to prove script content stays out of previews. No API key, authorization header, environment dump, hidden reasoning, raw private prompt, or memory payload is in this file or the tree.

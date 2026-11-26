# Part 3 — Observation shaping and the restorable blob store

**Track:** Tidemark, Day 4 of 10
**Milestone:** M2, evenings 4–5
**Depends on:** `02-prefix-stable-context-and-budgets.md`
**Produces:** typed tool-output reducers, immutable content-addressed blobs, bounded `read_blob`, and a 30–50-output must-keep corpus

## Outcome

Shrink large observations **once, when they arrive**, without losing the exact path, error, failing test, or other evidence a later step needs. Compression is restorable: anything omitted from the model's preview remains available through a handle and a governed range-read tool.

This is not generic oldest-first truncation. Typed errors from Steadyhand stay byte-for-byte raw. Old shaped observations are never recomputed to fit a later budget; the prefix remains stable. Compaction in part 4 is the only mechanism that removes whole recent turns from the rendered context.

## What already exists

- `ToolResult.observation()` preserves `{"ok": true, "value": ...}` and the typed error channel. `observation_message` serializes it with sorted JSON keys.
- `loop._finish` currently appends the model message, journals the raw `tool_result`, then applies Bulkhead taint. This is the single shaping chokepoint for model and recovery calls alike.
- Bulkhead's output gate already caps and sanitizes sandbox stdout/stderr before the model or journal sees it. Tidemark performs further context compression; it must not weaken or relabel that boundary.
- `SecurityGate.observe` uses the raw result and the raw `tool_result` sequence to decide taint. A short preview must not make untrusted content look trusted.
- Part 2 requires immutable recent messages and exact real-token caps. Shaping at render time would rewrite the prefix and is therefore excluded.
- `src/toolsmith/artifacts.py` provides atomic copies/snapshots; `JsonlRecorder` provides configured-secret redaction. Neither is a content-addressed object catalog.
- The executor supports server-only `idempotency_key` arguments hidden from model schemas. The same pattern can carry trusted execution resources without letting the model choose paths or stores.

## What this part adds

```text
src/tidemark/observations.py           # ObservationShaper and shaped message envelopes
src/tidemark/reducers.py               # Log, JSON, HTML/text, and exec-stream reducers
src/tidemark/blobs.py                  # BlobStore, BlobCatalog, ranged reads
src/tidemark/resources.py              # Trusted ToolExecutionContext protocol/types
src/steadyhand/tools_context.py        # read_blob registration and handler
src/steadyhand/executor.py             # Optional server-only execution-context injection
evaluation/data/tidemark_observations_v1.jsonl
evaluation/data/tidemark_observations_v1/
tests/test_tidemark_observations.py
tests/test_tidemark_reducers.py
tests/test_tidemark_blobs.py
tests/test_steadyhand_context_tools.py
tests/fixtures/tidemark_observations/
```

Generated blobs live under the gitignored run directory, for example `results/tidemark/blobs/<namespace>/`. Colab uses local SSD for active files and atomic Drive copies for durability. Only small sanitized fixtures belong in `evaluation/data/` and `tests/fixtures/`.

## Raw result versus model observation

Keep two representations:

1. **Raw journaled result:** the existing `ToolResult`, after any Bulkhead output gate, with its existing value/error and content. It remains the replay and taint source of truth.
2. **Shaped model observation:** a deterministic message derived from that raw result and the registered reducer policy. This is what enters segment 3 and the request-message hash.

The ordered path becomes:

```text
execute/recover -> raw ToolResult -> journal raw tool_result (obtain source seq)
-> shape once and persist any blob -> journal observation_shaped
-> append shaped tool message -> apply taint from the raw result/source seq
-> authorize the next call
```

A denied/asked/guard/validation/parse error takes the existing typed path and is not shaped. A handler error also remains raw even if verbose; apply existing upstream output caps rather than inventing a second error schema. No `observation_shaped` event is needed for an untouched observation unless the run's explicit audit mode records `strategy=none`.

With Tidemark disabled, the order and bytes remain the current ones. With it enabled, replay recomputes the shaped message from the recorded raw result and verifies the exact event and request hash.

### Envelope

Preserve the ordinary `ok`/`value` or typed error contract. A shaped success may add a bounded top-level `context` member:

```json
{
  "ok": true,
  "value": {"preview": "...", "truncated": true},
  "context": {
    "shape": "log_head_tail_v1",
    "blob": "blob://sha256:<64-lowercase-hex>",
    "bytes": 183420,
    "preview_tokens": 612,
    "omitted": true
  }
}
```

For projected JSON, keep the projected value rather than replacing every result with an arbitrary string. For a fully offloaded value, use a typed value object with `blob`, `media_type`, `preview`, `original_kind`, and `truncated`. Do not change `ToolResult.value` itself merely to carry prompt metadata.

`context` is provenance for the model, not a security label. Trust continues to come from Bulkhead's `SecuritySpec`, and delimiter text is never treated as enforcement.

## Reducer policy

Reducer selection is a frozen `ObservationPolicy` side table keyed by tool name and, for exec results, stream kind. It is part of `context_config` and the schema/policy evidence, not a model argument. Unknown tools default to deterministic bounded JSON with offload; they do not get a custom lossy transformation by guessing from their names.

### Logs and test/compiler output

`log_head_tail_v1`:

1. Normalize only the encoding form already permitted by the raw result; do not strip spaces or rewrite paths within retained lines.
2. Split with line endings preserved.
3. Mark every line matching configured error, exception, assertion, failing-test, compiler diagnostic, and path patterns. Include tool-specific must-keep selectors.
4. Coalesce only **consecutive exact duplicate** lines and record their repeat count. Never dedup two separated failures that happen to share text.
5. Select a bounded head and tail plus every marked line, in original order, with explicit omitted line/byte ranges.
6. Tokenize the candidate with the parity-checked tokenizer. If marked lines make it too large, offload the complete normalized observation and build a smaller preview containing the handle and the earliest/latest critical lines.

Exit code, signal, error type, stderr, failing test names, and workspace-relative paths stay associated with their stream/call. Do not flatten stdout and stderr into an ambiguous combined string.

### Structured JSON

`json_projection_v1` uses a per-tool allowlist of JSON Pointers and required passthrough fields. Preserve types, array order, `ok`, ids, statuses, counts, and errors. Record omitted pointers/counts, not their values. Canonical full JSON is the blob payload when offload is required, using strict finite-number and duplicate-key rules.

Projection rules must be reviewed against each tool's result schema. A must-keep field outside the preview is valid only when the complete value is reachable through a handle and the task's availability label permits `blob_handle`.

### HTML and long text

Use a standard-library HTML parser or an already declared dependency; do not add a browser/framework. Preserve visible text order, meaningful link targets, headings, table cell boundaries, and code/pre blocks. Remove script/style content from the **preview**, but retain the original gated payload in the blob when restorability is required. HTML extraction is not sanitization or proof that prompt injection is safe.

Plain text receives deterministic head/tail selection with omission ranges. Invalid UTF-8 is rejected or handled by the existing upstream gate; do not silently replace bytes and then hash the result as though it were the original.

### Offload threshold

A result over its preview token cap after typed reduction is written to the blob store. The preview states what was omitted, total bytes/items, media type, digest handle, and how to ask for a range. The cap is in real prompt tokens; byte limits are separate storage/read ceilings and cannot substitute for token accounting.

## Blob store and catalog

### Immutable objects

For each object record:

```text
handle, sha256, media_type, byte_length, source_episode_id,
source_event_seq, source_tool, created_seq, namespace,
encoding, reducer_version, retained_until, catalog_revision
```

The handle contains the full lowercase SHA-256 digest, for example `blob://sha256:<64 hex>`. Short display prefixes are labels only and cannot be used for lookup.

Store bytes at `<root>/<namespace>/<first-two-hex>/<full-digest>` using a temporary file, fsync where appropriate, and atomic link/rename. Verify existing content before reusing a digest. Namespace and catalog membership are checked server-side; a digest known from another tenant/session is not authorization to read it.

The catalog is a strict JSONL or SQLite-local index with atomic snapshots and monotonically increasing revisions. It is updated only after the object is complete. A crash may leave an unreferenced immutable object, which retention can collect; it must never leave a catalog entry pointing at partial bytes.

### Read contract

`BlobStore.read(handle, offset, max_bytes)`:

- resolves only a cataloged handle in the caller's namespace;
- rejects malformed handles, negative/out-of-range offsets, excessive ranges, non-integer values, path separators, URL authorities, and symlinked object paths;
- reads at most the configured ceiling and returns `handle`, `sha256`, `offset`, `next_offset`, `total_bytes`, `truncated`, `media_type`, and `content`;
- uses byte offsets explicitly and returns text only at valid UTF-8 boundaries, with boundary metadata; never silently splits a code point;
- verifies the digest on full reads and on a full-file verification command, not merely on the first range request;
- never accepts a filesystem path, SQL, URL, glob, or `..` from the model.

The server supplies the namespace and catalog through execution context. The model cannot select another root or tenant.

## `read_blob` as a governed tool

Register `read_blob` in the frozen session tool catalog from session creation; do not add it only after the first offload. Its closed schema contains only `handle`, optional `offset`, and optional `max_bytes` supported by Toolsmith's grammar subset. Numeric ceilings and handle syntax are semantic checks because arbitrary regex/range schemas are outside the compiler contract.

Suggested policy:

- `Effect.READ`, idempotent, no retry on malformed/not-found input;
- `SecuritySpec(output_trust=UNTRUSTED, sinks=())`, with no sandbox dispatch merely because the trusted library reads its own catalog;
- task policy lists the tool explicitly; a missing security spec remains a Bulkhead denial;
- returned content is untrusted and applies the ordinary taint transition;
- recovery cannot switch it to an arbitrary file-read tool to bypass the catalog.

Introduce optional `ToolExecutionContext` injection in `SequentialExecutor`, modeled on `idempotency_key`: the registry hides the parameter from model schemas, and only the server supplies episode id, step, journal view, blob store/catalog, namespace, and taint metadata. Existing handlers do not accept it and see no behavior change. `SecuredExecutor` forwards the same trusted context after authorization; part 6 extends it with memory resources.

Default replay may read immutable blobs referenced by the journal or preload them from verified artifacts. It must not create new blobs or reuse a mutable host path silently. Blob writes during replay are replaced by digest/content verification against the recorded raw result and catalog.

## Parallel shaping and I/O pipeline

Run shaping through part 1's resource-budgeted pools so large observations from one episode do not leave other GPU slots idle:

- Execute reducers concurrently **across different tool calls/episodes**, never concurrently for two records whose journal order depends on each other. Each result still commits its raw result, shape event, model message, and taint in call order.
- Keep the reducer required by the current episode on its critical path. Work for other ready episodes may fill CPU/GPU gaps, but a later observation cannot delay an earlier needed model turn merely to improve a batch.
- Bound the shaping queue by estimated input bytes and output tokens. Apply backpressure before admitting more large tool outputs on the 2-vCPU/limited-RAM T4 profile; L4 can use a larger measured queue, not an unbounded one.
- Split CPU reducer work from blob/Drive I/O. Local immutable writes and catalog appends are fast and atomic; background threads mirror verified objects to Drive. A required local blob read completes before its model call, while Drive copies and unrelated future reads proceed asynchronously.
- Preload evaluation payloads and required blobs to local SSD before timing. Batch independent catalog/range reads where this reduces overhead without changing the model's individual `read_blob` results.
- Cache reducer output only by raw-content hash + reducer/policy + tokenizer identity. Cache hits skip duplicate CPU work but still emit the correct episode/source event; they do not reuse a blob across namespaces without adding that namespace's authorized catalog entry.

Profile reducer time, queue wait, local/Drive I/O, CPU occupancy, and model-slot wait on both GPU families. On T4, autotune reducer workers against llama.cpp thread counts so the two vCPUs are useful rather than thrashing; on L4, scale only while end-to-end tasks/minute improves. If profiling shows shaping is not a bottleneck, keep the simple Python path rather than adding SIMD.

Tests use slow reducers, several concurrent model-ready episodes, large blobs, and T4/L4 resource budgets. They prove bounded memory, exact event order, no duplicate persistent writes, no Drive call on the model critical path, and forward progress when every GPU slot has work.

## Journal and resume records

Add `observation_shaped` immediately after its raw `tool_result` and before taint:

```text
step, source_seq, tool_call_id, tool, reducer,
raw_content_sha256, shaped_content_sha256, raw_bytes,
shaped_bytes, raw_tokens, shaped_tokens, omitted_ranges,
blob_handle, blob_bytes, catalog_revision, preview_metadata
```

Do not duplicate a large raw or preview payload in this event; the raw result and shaped request history already have their respective content. Store bounded counts/ranges and hashes. Reducer decisions, hashes, handle, and byte ranges replay exactly; wall time belongs in volatile metrics.

The episode/run manifest includes reducer policy, blob format version, token limits, namespace, and catalog root identity. A checkpoint records catalog revision and every referenced blob digest. Resume refuses missing/corrupt required blobs or a changed reducer config; it does not regenerate a different preview and continue as if history were stable.

Orphan collection and retention run only from an explicit maintenance command after catalog/checkpoint references are scanned. Never delete blobs merely because their episode finished or because Drive synchronization is behind.

## Observation corpus

Build 30–50 sanitized real-shaped outputs: pytest failures, compiler/linker errors, long shell logs, nested store JSON, exec stdout/stderr, and long HTML pages. Each entry contains:

```text
observation_id, version, tool, payload_file, media_type,
reducer, must_keep [{exact, pointer_or_line, availability}],
optional_forbidden_preview, source_license_or_origin, sha256
```

Include the critical line in the middle of a multi-thousand-line log, repeated errors that are not adjacent, invalid/control-character cases already handled by Bulkhead, a huge JSON field, and an HTML page whose visible text omits a needed link target unless it is preserved.

`availability` is `inline` or `blob_handle`. Every labeled string must either appear exactly in the preview or be returned exactly by the referenced blob range. The corpus reports both, rather than claiming 100% retention when the evidence merely exists on an unrelated host path.

## Tasks

1. Implement raw/shaped separation at `_finish`, preserving taint, recovery order, and legacy behavior.
2. Add reducer policies and deterministic log/JSON/HTML/text implementations with exact token measurement.
3. Implement the content-addressed store, catalog, atomic persistence, namespace checks, ranged reads, and corruption detection.
4. Add server-only execution resources and the governed `read_blob` tool without changing the frozen tool list mid-session.
5. Add journal/manifest/replay support, read-only replay blob access, resume catalog checks, and retention safeguards.
6. Create and validate the 30–50-output corpus with explicit must-keep labels and availability.
7. Add `tidemark-observation-gate` and `tidemark-observation-benchmark` notebook cells. Report token reduction and restorability separately.
8. Implement bounded parallel shaping, content-keyed reducer reuse, and asynchronous local/Drive blob pipelines. Autotune their worker/queue limits on both T4 and L4 and record reducer/queue/I/O utilization.
9. Run the phase-delivery protocol: commit the tested implementation/corpus, write the canonical Handoff for part 4, commit it, automatically push both commits, and verify remote `main`.

## Tests

- Typed validation, guard, denial, ASK, timeout, and parse errors remain byte-identical; sandbox output remains gated and untrusted.
- A critical error in the middle survives inline or through an exact blob read; non-adjacent duplicate errors are not collapsed.
- Projection preserves required JSON pointers and types; HTML preserves labeled visible text and links.
- The same raw result/reducer produces the same shaped content and digest across 1,000 runs; shaping occurs once and never mutates an earlier message.
- Blob full/range round trips cover UTF-8 boundaries, zero-length objects, final partial ranges, duplicate writes, torn files, wrong digest, unknown namespace, symlink/path traversal, and over-limit requests.
- A handle from another namespace cannot be read even when its digest is known.
- `read_blob` passes validate -> guard -> Bulkhead authorize -> handler, taints on outside content, and cannot be replaced by an unrestricted path read during recovery.
- Replay produces identical shaped messages without writing blobs or repeating external effects; deleting/editing a required object fails explicitly.
- Corpus scoring reports token reduction and 100% labeled must-keep survival **or restorability**; a broken handle is a failure, not a compression win.
- Existing Steadyhand, Bulkhead output-gate, and golden replay tests stay green.

Run:

```bash
.venv/bin/python -m pytest tests/test_tidemark_observations.py tests/test_tidemark_reducers.py \
  tests/test_tidemark_blobs.py tests/test_steadyhand_context_tools.py \
  tests/test_bulkhead_output.py tests/test_steadyhand_replay.py
.venv/bin/python -m pytest
```

## Phase delivery and handoff

After every CPU test, corpus gate, and required T4/L4 observation-pipeline acceptance passes, follow part 1's mandatory two-commit/one-push protocol automatically:

1. Stage only part 3's reducers/blob/tool/executor changes, observation corpus, tests, notebook, README, and deliberate shared changes; exclude generated blobs, raw private outputs, and unrelated dirty paths.
2. Create the implementation commit `Tidemark: complete part 3 restorable observations`, update `Tidemark/Handoff.md`, then create `Tidemark: hand off part 3`.
3. Record the implementation SHA, reducer/schema versions, observation-corpus hash, must-keep/token-reduction results, blob/catalog format, namespace/access invariants, per-GPU run ids, durable blob/catalog/checkpoint paths, and restore commands in the Handoff.
4. Point the next session to part 4 and its first compaction replay fixture. Push both commits to `origin/main` and verify remote `main` equals the Handoff commit before compaction work starts.

Missing must-keep evidence, corrupted/inaccessible required blobs, red tests, or failed publication blocks completion. Never commit raw generated observations or hide unrelated user changes in the phase commits.

## Acceptance

- The observation corpus has at least 30 labeled outputs, and every must-keep string is inline or demonstrably reachable through its handle.
- Token reduction is reported from real-token counts alongside omitted ranges and restorability, with no unsupported “no information lost” claim.
- Raw results remain the taint/replay source; model messages are shaped once and remain prefix-stable.
- Blob reads are bounded, content-verified, namespace-scoped, and governed by Bulkhead; replay performs no new persistent writes.
- Resume detects missing or corrupted referenced objects before issuing another model call.
- The observation benchmark runs with the fitted T4 and L4 worker/queue profiles and shows that large outputs overlap with other GPU work without unbounded RAM, Drive stalls, or event-order divergence.
- The tested implementation and canonical part 3 Handoff are committed and automatically pushed, remote `main` is verified, and part 4 can start in a fresh session with exact corpus/blob/checkpoint context.

## Pitfalls

- Re-running reducers over all history on every render, causing a cache break from the first changed observation.
- Truncating typed error observations or stripping the one path/test name needed later.
- Calling a short digest, a preview delimiter, or a model-supplied handle an access-control boundary.
- Writing directly to Drive and letting FUSE latency or a torn object block the agent; use local immutable objects and atomic snapshots.
- Counting evidence stored only in a raw journal as “available in context” when neither preview nor `read_blob` can retrieve it.
- Treating HTML text extraction as prompt-injection sanitization.
- Re-executing a persistent write during ordinary replay to reconstruct a blob.

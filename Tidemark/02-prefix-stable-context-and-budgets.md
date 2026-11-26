# Part 2 — Prefix-stable context, real-token budgets, and PrefixLint

**Track:** Tidemark, Day 4 of 10
**Milestone:** M1, evening 3
**Depends on:** `01-baseline-and-context-contract.md`
**Produces:** deterministic five-segment rendering, per-segment token accounting, a recitation tail, and exact prefix-divergence evidence

## Outcome

Replace accidental history concatenation with a pure `ContextAssembler.render(state)` whose ordinary step changes only by appending to recent history and rewriting a tail. Make a cache breaker observable as the first divergent token id and a logical segment/cause, then confirm actual reuse with llama-server timings from part 1.

The brief's **90% or more cached prompt tokens on steps without compaction** is a target for a same-slot warm run, not an expected local result. The first step, a slot change, an explicit cache reset, and a compaction epoch are reported separately.

## What already exists

- Part 1 supplies the optional `ContextSession`, `RenderedContext`, tokenizer parity, telemetry, manifest identity, and naive baseline.
- `ChatPromptRenderer` owns model-specific Jinja rendering; `normalize_history` owns Qwen3.5's historical argument conversion.
- `SteadyhandRegistry.schemas()` returns model-facing tools in deterministic registration order and hashes them. It does not currently provide Tidemark's frozen segment wrapper or request-local tool-name masking.
- `ToolsmithService._validate_messages` requires one optional leading system message, closed native call/result groups, and string content. Pinned memory, summaries, and recitation therefore cannot be inserted as later system messages.
- `EpisodeController` currently checks cumulative budgets and uses the previous response's prompt usage for legacy slot-overflow checks. Tidemark must use the actual candidate render before each call.
- `src/steadyhand/messages.py` owns fixed system/goal text and deterministic controller give-up text. The recitation tail must contain short actionable state, not private chain of thought.

## What this part adds

```text
src/tidemark/segments.py              # Segment ids, canonical values, immutable transitions
src/tidemark/assembler.py             # ContextState and ContextAssembler
src/tidemark/budgets.py               # Exact total budget and per-segment accounting
src/tidemark/prefix_lint.py           # Token diff, expected/violating divergence, cause
src/tidemark/recitation.py            # Bounded plan/state/budget tail
src/toolsmith/prompting.py            # Additive render-with-token-spans support
tests/test_tidemark_assembler.py
tests/test_tidemark_budgets.py
tests/test_tidemark_prefix_lint.py
tests/test_tidemark_recitation.py
tests/fixtures/tidemark_prompts/
```

No C++ optimization is needed here. Correct deterministic rendering and evidence come first.

## Logical layout

```text
[0] frozen system + tool schemas       fixed at session creation
[1] pinned memory + task charter       fixed except at a declared epoch/session boundary
[2] compaction summary, epoch k        empty in epoch 0; replaced only by part 4
[3] raw recent turns                   append-only between epochs
[4] recitation tail                    replaced on every render, at the end
```

Map those logical segments to legal OpenAI-shaped messages:

- Segment 0 is the one leading system message plus the separate `tools` argument passed to the same renderer. It contains no timestamp, request id, random episode id, changing approval count, or runtime load.
- Segment 1 is a bounded user-role reference block containing the original task charter and selected memory. It is labeled as data, not new instructions. It does not silently retrieve different memories every step.
- Segment 2 is a user-role, canonical JSON compaction block with an epoch and citations. Empty epoch 0 contributes no message. Its replacement is an expected cache invalidation, not a PrefixLint bug.
- Segment 3 contains assistant/user/tool messages in original order. Once appended, a message's content and serialization are immutable. Corrections are new messages; old observations are not rewritten or redacted in place.
- Segment 4 is one final user-role message containing a short current plan/open-items list and numeric budget line. The previous tail is not retained in raw history. The next assistant turn and its tool results are committed to segment 3, not to the old tail.

`EpisodeResult.messages` remains the logical history, not a lossy list of old recitation tails. Each `context_render`/`model_turn` pair preserves the hash of the exact tail-bearing request that produced that turn.

## Pure state and transitions

Use immutable values for `ContextState`:

```text
session_config, frozen_system, frozen_tools, pinned_memory,
task_charter, epoch, summary, recent, recitation,
previous_render_fingerprint, previous_stable_prefix_length
```

`ContextAssembler.render(state)` performs no model calls, filesystem access, clock reads, random id generation, SQLite queries, or environment lookup. Retrieval and compaction are explicit transitions whose outputs become state inputs. The same state and renderer produce byte-identical prompt text and token ids.

Ordinary transition:

```text
append complete assistant/tool events to [3] -> rebuild [4] -> render
```

Epoch transition, implemented fully in part 4:

```text
persist accepted compaction event -> replace [2] -> advance epoch
-> reset [3] to events after compacted_through_seq -> rebuild [4] -> render
```

Compaction occurs only at a safe step boundary after every native call has exactly one tool response. It never splits parallel tool planning/execution or changes a previous `model_turn` hash.

## Frozen tools and request-local restriction

Freeze the complete tool list at session creation, including `read_blob` and memory tools when those features are enabled. Canonicalize object keys without depending on dictionary insertion accidents, preserve a declared tool ordering, and record the schema hash in the manifest and every render. The tool list never changes to save prompt tokens mid-session.

If a task permits only a subset, keep the same rendered schemas and add a **server-internal** `allowed_tool_names` option through `SteadyhandClient`/`ToolsmithService`:

- prompt rendering still receives all frozen tools;
- GBNF compilation and post-validation permit only the named subset for that call;
- the existing public OpenAI `tool_choice` shapes retain their behavior;
- the subset is recorded in `context_render`, and a model call outside it fails validation rather than reaching Bulkhead;
- the default `None` means all frozen tools and preserves legacy requests byte-for-byte.

Use the existing named `tool_choice` when exactly one function is required. Do not claim the current public API already masks an arbitrary subset. This additive internal path is the implementation of that requirement.

## Renderer token spans

Extend Toolsmith with an additive `render_with_token_spans` helper rather than duplicating Jinja logic in Tidemark. It must:

1. Apply `normalize_history` with the session's explicit `ModelProfile.protocol`.
2. Render incrementally at logical segment boundaries with identical tools and `enable_thinking` settings, without adding the generation prompt to intermediate renders.
3. Render the final tail-bearing prompt with `add_generation_prompt=True` exactly as `ToolsmithService.chat` does.
4. Tokenize the final string once with the parity-checked tokenizer and return text, ids, hashes, and segment spans.

Require each intermediate token sequence to be a prefix of the next. If the model template rewrites earlier content, merges across a boundary, or otherwise makes spans ambiguous, fail the live parity check and record `span_mapping_unavailable`; do not invent token offsets. Total prompt tokens always come from the final render, never from summing independently tokenized messages.

A differential test compares `render_with_token_spans(...).text` with the existing `renderer.render(...)` on the same inputs. The production request and the linter must not disagree about which template or thinking flag was used.

## Token budgeter

Define a versioned `SegmentBudget` for each segment plus:

```text
slot_context, max_tokens, safety_margin,
soft_limit (default 0.70 of usable prompt budget),
hard_limit = slot_context - max_tokens - safety_margin
```

The exact final render controls admission. Per-segment caps use exact spans when the parity contract supports them. If spans are unavailable, the run cannot claim per-segment token compliance; only a separately declared diagnostic accounting mode may run, and it cannot satisfy this part's live acceptance.

Before every model call, journal a table with used/cap/remaining tokens for all five segments, exact total, generation reserve, and hard headroom. Apply this deterministic order:

1. Shape a newly arrived observation once through part 3; never reshape old observations to make room.
2. Rebuild and bound the recitation tail with deterministic field priorities. This is allowed to change every step.
3. If the exact prompt exceeds the soft limit, mark `compaction_pending` for the next safe boundary.
4. Compact before the model call when the hard limit would be exceeded. If compaction is disabled, apply only the evaluation mode's declared strategy; never silently use a different algorithm.
5. If a legal render still exceeds the hard limit, stop with `context_budget_exhausted` before generation. This is distinct from llama.cpp `completion_truncated` and from an infrastructure timeout.

Segment 0 and the current task's minimum legal history cannot be sacrificed by an emergency truncator. A huge frozen tool catalog must fail configuration validation rather than repeatedly compacting around an impossible prompt.

## Recitation tail

`RecitationState` is typed and bounded:

```text
goal_reference, current_plan_items, next_action,
completed_step_count, prompt_tokens_used, remaining_step_budget,
open_blob_handles, unresolved_approval
```

Render a compact deterministic text or canonical JSON form. Plan items are agent-authored hypotheses/actions, not environment facts and not hidden reasoning. Keep at most a configured number of short items; evict completed/oldest lower-priority items by rule and record the tail hash. Exact source-backed facts belong in segment 2 or memory, not in an uncited tail.

The tail is the Day 5 handoff. Do not put timestamps, random anchors, raw model thoughts, or whole summaries in it.

## PrefixLint

For each render, store the full previous prompt's token ids in the local checkpoint and the length of its stable prefix before the old recitation tail. Do not write the full ids to JSONL.

Compute:

- `first_divergent_index`: first position where previous and current ids differ, or the first length mismatch;
- `observed_common_prefix_tokens`;
- `expected_common_prefix_floor`: previous stable-prefix length on an ordinary append, previous frozen/pinned prefix on an epoch transition;
- `first_changed_segment` and `cause` from the logical transition and segment hashes;
- `status`: `expected_append`, `expected_tail_only`, `expected_epoch`, `cold_start`, or `violation`.

On an ordinary step, every previous token before the old tail must match at the same index. New recent content may extend the common prefix beyond that floor; boundary tokenization changes are examined rather than assumed harmless. On an epoch transition, only segments 0–1 are required to remain stable. A slot change does not excuse a textual divergence; it explains missing server reuse independently.

Planted regression fixtures must include:

- a timestamp inserted into segment 0;
- unsorted tool-schema keys and reordered tool definitions;
- whitespace changed in an old tool observation;
- an old assistant argument re-serialized differently;
- pinned memory replaced mid-epoch;
- an old summary edited without advancing the epoch;
- a legal recent append, tail rewrite, and epoch transition.

`PrefixLint` diagnoses prompt identity; it does not prove llama.cpp reused KV. Join its event to `backend_timings` to distinguish a stable prompt with a cold/evicted slot from a broken prefix on a warm slot.

## CPU/GPU execution, caching, and utilization

Use part 1's shared scheduler and `ResourceBudget`; the assembler does not create private threads or process pools.

- Cache immutable canonical segment serializations, rendered fragments, token ids, and compiled grammars by a key containing content hash, model/tokenizer/template revision, protocol, thinking flag, full frozen schema hash, tool-choice/allowed-name set, and compiler version. A cache hit must be byte/token identical, not merely semantically similar.
- Tokenize only new/changed segments and the final candidate required for exact headroom/PrefixLint. After live parity passes, use the local tokenizer rather than an extra `/tokenize` HTTP request per step. Store the previous token ids in a compact local binary artifact, not repeated JSON arrays.
- Prepare upcoming episodes' segment 0–1 and initial segment 3 on CPU while the GPU serves current episodes. A running episode's next turn cannot be speculatively assembled before its model/tool result exists; do not waste CPU or invent prefetch work.
- Use the shared RAM-bounded process pool for parallel rendering/tokenization across episodes and cap tokenizer/BLAS thread settings from the measured CPU partition. Preserve per-episode transition order regardless of completion order.
- Keep exact prompt admission synchronous with the model-slot lease. A stale cached count cannot authorize a call after the tail or allowed-tool set changes.
- Probe llama.cpp batch/ubatch and cache-reuse settings on both T4 and L4. Larger batches are useful only when measured prefill throughput improves without violating VRAM or starving CPU prompt preparation.

Record `segment_cache_hits`, `grammar_cache_hits`, render/tokenize/compile microseconds, ready-queue depth, and CPU worker occupancy in volatile performance records. These counters measure overhead; they are not KV-cache hits and must not be folded into `cache_reuse_fraction`.

Tests include T4- and L4-shaped fake workloads with cold/warm caches, concurrent episodes, changing allowed tools, and a full CPU queue. Assert no nested pools, no stale token/grammar reuse, deterministic output, bounded RAM, and no idle model slot while a prepared request is waiting.

## Journal additions

Extend `context_render` with:

```text
stable_prefix_tokens, expected_common_prefix_floor,
observed_common_prefix_tokens, first_divergent_index,
first_changed_segment, divergence_cause, lint_status,
previous_prompt_sha256, allowed_tool_names
```

Hashes, statuses, segment causes, and deterministic budget decisions replay exactly. Do not mark lint violations volatile. A live timing difference remains in `backend_timings` only.

Add `context_transition` when messages are appended or an epoch changes:

```text
step, transition, epoch_before, epoch_after,
history_start_seq, history_end_seq, appended_seqs,
dropped_seqs, reason, state_sha256
```

Naive baseline drops and future compaction resets use the same explicit evidence. No silent history rewrite is permitted.

## Tasks

1. Implement canonical segments, immutable state transitions, and legal message mapping. Keep legacy no-op behavior outside the adapter.
2. Add renderer/token-span parity support to Toolsmith and the span-unavailable failure path.
3. Implement exact total budgets, per-segment accounting, deterministic recitation trimming, and pre-generation overflow handling.
4. Implement PrefixLint and persist the previous fingerprint in resumable local state without bloating journals.
5. Freeze tool schemas and implement optional request-local tool-name masking in grammar compilation and post-validation without altering prompt segment 0.
6. Wire all model calls, including final paraphrase, through the assembler. Update controller slot checks to use candidate renders on Tidemark runs.
7. Add journal/manifest support and replay for transitions, lint, budgets, and allowed tool subsets.
8. Add `tidemark-prefix-gate` and `tidemark-prefix-smoke` notebook cells. The live smoke deliberately plants a timestamp and unsorted JSON, observes the divergence/cache effect, fixes them, and only then runs the warm same-slot target measurement.
9. Implement content-keyed render/token/grammar caches and shared-pool prefetch, then autotune batch/cache-reuse settings independently on T4 and L4 without changing a scored run's configuration.
10. Run the phase-delivery protocol: commit the tested implementation, write the canonical Handoff for part 3, commit it, automatically push both commits, and verify remote `main`.

## Tests

- Render the same state 1,000 times and assert identical text, ids, hashes, spans, and message lists for both Qwen protocols with fake tokenizers.
- Every planted cache breaker produces the exact first token divergence and segment/cause; legal append/tail/epoch cases are not violations.
- Changing request-local allowed tools leaves segment 0 token ids unchanged while the grammar and validator reject a disallowed name.
- Qwen3.5 historical arguments are normalized exactly once, and the production prompt equals the lint prompt.
- Per-segment and total counts use real-token fixtures; hard headroom includes `max_tokens` and the safety margin.
- The assembler never emits an unanswered call, orphan tool response, non-leading system message, or tail trapped inside recent history.
- Recitation truncation follows declared priorities and contains no clock/random values.
- Replay reproduces every render/transition/hash/lint decision; editing an old observation or summary fails at its event.
- Existing golden episodes and Tool/API tests remain green.

Run:

```bash
.venv/bin/python -m pytest tests/test_tidemark_assembler.py tests/test_tidemark_budgets.py \
  tests/test_tidemark_prefix_lint.py tests/test_tidemark_recitation.py \
  tests/test_api.py tests/test_steadyhand_replay.py
.venv/bin/python -m pytest
```

Live acceptance additionally runs the planted cache-breaker smoke and repeated same-slot prompts on the pinned llama-server on both T4 and L4. The same smoke also records render/tokenizer/grammar-cache overhead and model-ready slot occupancy on each GPU; the target is measured independently per runtime.

## Phase delivery and handoff

After every CPU test and the planted PrefixLint/cache/utilization acceptance on both T4 and L4 passes, follow part 1's mandatory two-commit/one-push protocol automatically:

1. Stage only part 2's new files and its deliberate shared changes to Toolsmith, Steadyhand, notebook, README, and tests; never include unrelated dirty paths or generated prompts/artifacts.
2. Create the implementation commit `Tidemark: complete part 2 prefix-stable context`, then rewrite `Tidemark/Handoff.md` with the implementation SHA and create `Tidemark: hand off part 2`.
3. Handoff fields specific to this phase include renderer/tokenizer/template/schema/grammar hashes, segment/budget configuration, planted divergence evidence, PrefixLint replay status, capacity/scheduler settings, and per-GPU run/artifact paths.
4. Set the next action to part 3 with exact first tests/notebook cell. Push both commits to `origin/main` and verify remote `main` equals the Handoff commit before any part 3 work.

A red test, missing T4/L4 evidence, unexpected phase-file change, or failed/non-fast-forward push blocks delivery and the next phase. No force push, automatic rebase, broad staging, or secret/result inclusion is permitted.

## Acceptance

- Ordinary renders change only at the append point and tail; any earlier change is a precise lint violation.
- Exact real-token budgets are journaled before every call, and no over-hard-limit request reaches llama-server.
- Frozen tool schemas can coexist with request-local grammar/validation restriction.
- The warm no-compaction cache target is measured separately from cold/compaction steps; a miss caused by slot eviction is not mislabeled a rendering bug.
- CPU and live replay reproduce the exact message lists and lint evidence.
- The tested implementation and canonical part 2 Handoff are committed and automatically pushed, remote `main` is verified, and a fresh session can start part 3 solely from the Handoff plus repository artifacts.

## Pitfalls

- Summing independently tokenized segments and treating boundary merges as impossible.
- Putting a second system message in the middle of history, or allowing the template to inject today's date.
- Auto-retrieving new memory into the pinned block every step and destroying the cache.
- Editing or re-shaping an old observation instead of appending a correction.
- Treating PrefixLint's string/token comparison as proof of server KV reuse.
- Hiding a tool from the prompt but leaving it callable, or changing the prompt tool list to restrict it.
- Filling the recitation tail with hidden chain of thought or uncited factual claims.

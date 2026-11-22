# Days 6–8: Vocabulary trie, automaton, and token-masking sampler

## Context

Days 4–5 added `GBNFCompiler` in `src/toolsmith/grammar.py:215`. It now produces the GBNF text used by llama.cpp, and the same 12 grammars/90 valid-invalid cases have been checked with llama.cpp's real validator. That compiler currently stores rendered rule strings, so the custom sampler has no machine-readable grammar to execute. Parsing our own GBNF back into another representation would add a second parser and another place for the two decoding paths to drift apart.

The outcome for Days 6–8 is a pure-Python constrained-decoding core plus a thin Hugging Face integration:

- One shared, immutable grammar intermediate representation (IR) used for both GBNF rendering and custom execution.
- A nondeterministic pushdown automaton (PDA) for nested JSON/tool calls, with explicit stacks and persistent branching states.
- Incremental UTF-8 decoding over raw token bytes, including tokens ending in partial code points.
- A Qwen byte-level-BPE vocabulary trie and a trie-guided allowed-token mask engine.
- A batch-one Transformers `LogitsProcessor` and generation wrapper supporting greedy and sampled generation.
- Safe trigger-activated constraints for thinking mode, including split markers and tokens containing both a marker and following argument bytes.
- Correctness, cache-equivalence, latency, and real Qwen3-0.6B evidence in the existing canonical notebook.

The existing llama.cpp Qwen3-8B server and FastAPI endpoint remain the main serving path. The custom Transformers sampler is a standalone experimental backend in this milestone; rewriting `ToolsmithService` to support a second model backend is deferred. A full fresh-notebook run will also close the still-pending real Colab acceptance for the earlier streaming and GBNF cells.

## Approach

Use one implementation path, in three dependency layers:

1. **Grammar semantics:** refactor the existing rule builder to construct IR first; render GBNF from that IR without changing existing grammar text.
2. **Pure-Python execution:** compile the IR to a codepoint PDA, decode token bytes incrementally as UTF-8, build the vocabulary trie, and compute/test masks without importing PyTorch.
3. **Model integration:** add a lazy-imported Hugging Face adapter, trigger controller, and canonical-notebook evaluation on Qwen3-0.6B.

This keeps almost all correctness tests runnable in the current CPU-only development environment, where `torch`, `transformers`, `tokenizers`, and `numpy` are not installed. Only real model integration requires Colab.

### Evening 6 — shared IR and pushdown automaton (CPU)

1. Introduce immutable grammar expressions and ordered grammar programs.
2. Refactor `GBNFCompiler` so its existing schema/tool logic emits IR; keep its public API and GBNF output stable.
3. Compile IR to PDA instructions with explicit return stacks, epsilon closure, and tail-call handling.
4. Add incremental UTF-8 input and byte-prefix acceptance tests.

**Exit gate:** all existing tests remain green; generated GBNF is byte-identical for the existing corpus and tool-choice modes; all 90 corpus cases and every prefix of every valid case behave correctly in our automaton; long plain text does not grow the PDA stack.

### Evening 7 — vocabulary trie and mask engine (CPU, tokenizer download optional)

1. Extract exact Qwen token bytes without lossy `decode()` calls.
2. Build and audit a byte trie, including duplicate byte sequences and non-special added tokens.
3. Traverse trie and automaton states together to produce allowed-token masks.
4. Add state-keyed caching and compare optimized masks against a brute-force oracle.

**Exit gate:** synthetic-vocabulary masks match brute force at every tested prefix; split UTF-8, cross-boundary JSON tokens, EOS, special tokens, and padded IDs pass; a real Qwen tokenizer audit reconstructs encoded text exactly and reports full vocabulary coverage.

### Evening 8 — Hugging Face sampler, thinking trigger, and benchmarks (L4)

1. Add the batch-one/no-beams `LogitsProcessor` and high-level generation result/finalization checks.
2. Add atomic trigger-aware masking for think-then-call generation.
3. Extend `notebooks/01_toolsmith.ipynb`; load Qwen3-0.6B alongside the already tested llama.cpp path.
4. Verify fixed and triggered generation, cache equivalence, truncation, and unconstrained baselines; record p50/p95/p99 mask latency and end-to-end throughput.

**Exit gate:** all accepted constrained Qwen generations parse and schema-validate; incomplete calls fail explicitly; cache-on/cache-off greedy token IDs match exactly; trigger tests preserve reasoning and constrain calls; all evidence files are written and cleanup remains last.

## Key changes

### 1. Shared grammar IR; no second GBNF parser

Add `src/toolsmith/sampling/ir.py` with frozen expression nodes:

- `Empty`, `Literal`, `CharSet`, `Sequence`, `Choice`, `Repeat`, and `RuleRef`.
- Character sets represented as explicit codepoint ranges plus negation, not Python regular expressions.
- An immutable `GrammarProgram` containing an ordered rule mapping and `root` entry point.
- A GBNF renderer that owns literal/range escaping and precedence/parenthesization.

Refactor `_RuleBuilder` and the construction methods in `src/toolsmith/grammar.py` to store these nodes rather than expression strings. `CompiledGrammar.text`, `mode`, `tool_names`, `sha256`, and `rule_count` retain their current behavior. Add its immutable `program` as a non-comparison/non-repr field so existing equality and hash-based logs remain stable. Automaton construction requires this IR; arbitrary hand-written GBNF strings are not silently accepted by the custom sampler.

Preserve the existing schema subset and all unsupported-schema errors. Do not broaden `$ref`, unions, patterns, or numeric/string limits while refactoring.

Add `tests/test_grammar_ir.py` and update existing grammar tests to assert:

- Existing GBNF fixtures and all four tool-choice outputs are byte-identical before/after the refactor.
- IR is immutable, deterministic, and has no accidental mutable default/shared rule state.
- GBNF precedence and escaping round-trip through the IR renderer.
- `CompiledGrammar` objects built manually without IR fail clearly if passed to the custom automaton compiler.

### 2. Codepoint pushdown automaton with raw-byte UTF-8 input

Add `src/toolsmith/sampling/automaton.py`.

Compile the program into a small instruction set such as `Match`, `Call`, `Return`, `Fork`, `Jump`, and `Accept`. A configuration contains an instruction pointer and a persistent return-stack frame. Use immutable linked frames so nondeterministic branches share stack tails rather than copying whole stacks. A complete `AutomatonState` contains a deduplicated configuration set plus incremental UTF-8 decoder state.

Required execution behavior:

- Compute epsilon closure over empty expressions, choices, jumps, nullable calls/returns, and bounded-repeat exits, with visited-state detection.
- Consume one decoded Unicode codepoint across all viable configurations and discard branches with no match.
- Use explicit stack-depth, configuration-count, and epsilon-step limits; exceeding one raises a diagnostic error rather than falling back to unconstrained output.
- Reject unbounded repetition of a nullable body at automaton-compile time. Bounded JSON whitespace is unrolled by count.
- Optimize tail-position rule references into jumps. The existing right-recursive plain-text rules must run with bounded stack usage; nested objects/arrays use stack space proportional to actual nesting.
- `can_finish()` is true only when an accepting configuration is reachable and no UTF-8 codepoint is incomplete.
- `advance_bytes()` accepts any byte prefix of valid UTF-8, rejects impossible encodings, and permits a token boundary between a leading byte and its continuation bytes. EOS is not permitted while a codepoint is pending.
- Expose safe debugging information: byte offset, current rules/configuration count, stack depth, and a short expected-input description. Do not expose secrets or dump an entire prompt in exceptions.

Add `tests/test_automaton.py` and reusable byte-level cases in `tests/fixtures/automaton_cases.json`:

- Run every existing schema/tool valid and invalid case through the PDA, one byte at a time and with random chunk boundaries.
- Assert every prefix of every valid UTF-8 input remains viable and the complete value accepts; invalid complete inputs never accept.
- Exercise nested objects/arrays, optional-property branches, escaped strings, literal tag-like text in strings, parallel calls, and all `tool_choice` roots.
- Exhaustively compare short integer/number strings over a small alphabet against an independent JSON-syntax predicate.
- Test valid/invalid UTF-8 lead and continuation bytes, overlong encodings, surrogate codepoints, values above U+10FFFF, and token/EOS boundaries inside a codepoint.
- Assert long plain text and long strings keep bounded stacks, deep nesting hits the configured limit explicitly, and nullable-repeat/epsilon-cycle safeguards work.

### 3. Exact Qwen vocabulary bytes and trie

Add `src/toolsmith/sampling/vocabulary.py` with:

- `VocabularyEntry(token_id, raw_bytes, is_eos)` and `Vocabulary` metadata.
- A `ByteTrie` whose edges are integer bytes and whose terminal nodes hold all token IDs sharing that byte sequence.
- Build statistics: vocabulary size, usable tokens, trie nodes, duplicate-byte tokens, excluded special/padded/empty tokens, build time, and approximate memory.
- A Qwen codec adapter and a clear `UnsupportedTokenizerError` for other tokenizer families.

For the initial Qwen3 path, load the slow `Qwen2Tokenizer` explicitly (`use_fast=False`) and use its public `byte_decoder`/byte-level-BPE representation. Do **not** use `tokenizer.decode([token_id])` or `convert_tokens_to_string()` as the source of truth: both decode UTF-8 with replacement behavior and can corrupt a token containing only part of a multi-byte codepoint.

Handle vocabulary categories separately:

- Base byte-level-BPE tokens: invert the byte-encoder mapping to recover the original 0–255 bytes, including Qwen's leading-space representation.
- Non-special added tokens such as `<tool_call>`, `</tool_call>`, `<think>`, and `</think>`: store their literal UTF-8 bytes. Discover their IDs from the tokenizer, never hardcode IDs.
- Special/control tokens: exclude from ordinary trie terminals. Treat all configured EOS IDs as control actions allowed only when the current controller can finish.
- Padded model-vocabulary IDs absent from the tokenizer and empty normal tokens: always mask them.

Add `tests/test_vocabulary.py` with a synthetic tokenizer and hex-encoded byte fixtures. Cover shared prefixes, multiple IDs with identical bytes, empty tokens, leading spaces, ASCII/non-ASCII added tokens, single tokens split across UTF-8 codepoints, and special/padded IDs. In Colab, audit the real Qwen vocabulary by asserting that concatenated raw bytes for representative encoded strings equal the tokenizer's normalized input UTF-8 exactly, with no replacement characters and no silent token loss.

### 4. Trie-guided token-mask engine and cache

Add `src/toolsmith/sampling/mask.py`.

For a fixed automaton state, perform a joint depth-first traversal:

1. Start at the trie root with the current `AutomatonState`.
2. For each byte edge, advance UTF-8/PDA state; prune that trie branch if no configuration survives.
3. At each reached terminal, mark every terminal token ID allowed if its **entire** byte sequence remains a viable grammar prefix. A token need not complete the JSON value.
4. Add configured EOS IDs separately only if `can_finish()` is true.
5. Return a deterministic `MaskResult` with sorted allowed IDs, visited trie/automaton counts, and cache diagnostics.

A token may close a string, emit a comma and next key, close a whole tool call, or contain several grammar transitions; always test the whole token. Never mask one character at a time and never assume token and grammar boundaries align.

Add a bounded LRU mask cache keyed by the complete immutable automaton state, including return stacks and pending UTF-8 bytes. Do not cache only by rule name: the same rule can have different legal continuations depending on its stack. Include vocabulary/grammar fingerprints in engine construction and expose hit/miss/eviction counters. An empty allowed set is a hard `DeadEndError` with state diagnostics, never permission to sample from all-negative-infinity logits.

Add `tests/test_token_mask.py`:

- Compare trie masks with a brute-force oracle that independently advances every vocabulary token from the same state.
- Check every prefix of synthetic valid outputs and random PDA states; include tokens spanning `","`, `":"`, brackets, `</tool_call>`, and multiple fields.
- Verify split UTF-8 tokens are allowed only when their combined bytes are viable and that EOS is blocked mid-codepoint or mid-call.
- Assert special, unknown, padded, and empty tokens are excluded.
- Verify cache hits return identical results, different stacks/UTF-8 states cannot collide, eviction works, and cache-on/cache-off masks are identical.
- In the real-tokenizer notebook check, compare trie and brute-force masks for a bounded set of representative states; use the full synthetic suite for exhaustive local testing.

### 5. Batch-one Hugging Face generation adapter

Add `src/toolsmith/sampling/controller.py` for pure-Python generation-state synchronization and `src/toolsmith/sampling/hf.py` for the thin optional adapter. Keep torch/transformers imports out of the package's top-level import.

The controller owns the current fixed/trigger state, prompt length, processed generated-token count, EOS IDs, and per-step metrics. The Hugging Face `LogitsProcessor`:

- Accepts only batch size one, one new token per call, `num_beams=1`, and `num_return_sequences=1`; reject beam expansion, rewinds, speculative/assistant decoding, and unexpected sequence changes explicitly.
- Synchronizes the previously sampled token before computing the next mask.
- Applies `-inf` to disallowed logits before Transformers sampling/warping and verifies at least one finite allowed logit remains.
- Supports greedy and ordinary temperature sampling. Use greedy generation for deterministic correctness comparisons; seeded sampling is an additional check, not the primary oracle.

The high-level `generate_constrained()` wrapper:

- Tokenizes the already rendered prompt, calls `model.generate()`, slices prompt IDs from the result, and returns raw generated bytes/text, token IDs, stop reason, and metrics in a `GenerationResult`.
- Uses strict raw-byte reconstruction for non-special generated tokens; no lossy per-token decoding and no cleanup of tokenization spaces.
- Distinguishes EOS from `max_new_tokens`. If a fixed grammar is incomplete, or a trigger has activated and its call is incomplete, raise/record an explicit `IncompleteConstrainedGeneration` containing the partial output and automaton summary.
- Returns an unconstrained baseline through the same measurement path so timing comparisons use equivalent prompt/token accounting.

Add `torch>=2.4,<3` to the existing `colab` optional dependencies; `transformers>=4.51,<5` is already declared. Colab's preinstalled compatible torch should satisfy this without replacement. Do not add `accelerate`: load with the declared Transformers-compatible dtype argument and `.to('cuda')`. Core `dev` tests remain torch-free. If the optional adapter is imported without its runtime dependencies, raise an informative installation/runtime error.

Add `tests/test_sampler_controller.py` for token synchronization, EOS/length finalization, invalid batch/beam/rewind calls, all-masked logits decisions, metrics, and strict byte reconstruction using fake token/logits sources. Real `model.generate()` behavior is verified in Colab rather than mocked into a false local integration pass.

### 6. Safe trigger-activated thinking mode

Implement `TriggeredMaskController` in `src/toolsmith/sampling/controller.py`; do not merely activate a grammar **after** seeing a marker. Such a design is unsafe when one token contains `<tool_call>` plus invalid or unconstrained following bytes.

Use a byte-level trigger scanner for `<tool_call>` with two modes:

- **Free mode:** reasoning/normal text and EOS remain available. Track the longest generated suffix that is a proper prefix of the trigger.
- **Active mode:** once a complete trigger is consumed, the required tool-call PDA constrains every remaining byte; EOS is allowed only after a complete permitted call sequence.

Precompute trigger-bearing vocabulary metadata from the raw-byte trie. In free mode, ordinary tokens that cannot complete/start a trigger remain unrestricted. Atomically evaluate candidate tokens that complete/start a marker, including any bytes after the marker, through the combined trigger/PDA state:

- If the whole candidate leaves a viable free or active state, allow it and commit that state only if selected.
- If a candidate completes a marker but its following bytes violate the tool grammar, mask that candidate before emission.
- Handle markers split at every byte/token boundary, a marker in the middle of a token, marker-plus-argument tokens, and a false prefix such as `<tool_cat>` returning to free mode.
- Once active, use the normal fixed trie mask. No unconstrained trailing prose is allowed after the constrained call sequence; the existing parser still supports such text for prompt-only baselines.

This milestone supports trigger mode for `tool_choice='auto'`: the model may finish ordinary/reasoning text without a call, but any opened call is constrained. Required/named choices continue using fixed grammars with thinking disabled. A literal marker in reasoning is treated as a trigger by design and documented as such.

Add `tests/test_trigger.py` for every split point, atomic marker-plus-valid/invalid-suffix tokens, false prefixes, marker-like text inside arguments, parallel calls, free-text EOS, active truncation, and cache equivalence. Include Qwen's standalone non-special added marker tokens without relying on their numeric IDs.

### 7. Canonical notebook and evaluation

Extend only `notebooks/01_toolsmith.ipynb`, before the existing `cleanup` cell:

1. **`sampler-notes`** — explain IR, PDA versus regular expressions, persistent stacks, raw token bytes, trie pruning, EOS, and why trigger activation must be atomic.
2. **`sampler-offline-differential`** — execute the full existing grammar corpus through our PDA, all valid prefixes, synthetic vocab masks, and brute-force comparisons. Save failure inputs and configuration summaries.
3. **`sampler-vocabulary`** — load Qwen/Qwen3-0.6B's slow tokenizer, build/audit the trie, verify marker tokens and representative raw-byte round trips, and record build size/time.
4. **`sampler-model-setup`** — check CUDA/free memory, load Qwen3-0.6B in FP16 on the L4 without `accelerate`, and record model revision/dtype/device. The existing Qwen3-8B GGUF server is expected to leave sufficient room; do not silently fall back to CPU or silently stop it.
5. **`sampler-fixed-generation`** — compare unconstrained and fixed-mask greedy generation at identical prompt/settings for simple, nested/array/enum, parallel, and named-tool cases. Parse with `Qwen3ToolCallParser`, validate with `ToolCallValidator`, and record semantic tool choice separately. Repeat selected constrained cases with cache off and require identical generated token IDs. Include a short token-limit case that must report explicit truncation.
6. **`sampler-trigger-generation`** — render with thinking enabled; verify a reasoning-then-tool-call response and a no-tool-needed response. Preserve the raw output, reasoning/content split, trigger activation step, and validated calls. Synthetic tests, not model luck, must cover every marker split.
7. **`sampler-benchmarks`** — collect at least 200 warmed-up mask measurements across representative states; report p50/p95/p99/max CPU mask time, cache hit rate, trie nodes visited, configuration counts, trie build time/memory, and constrained/unconstrained end-to-end tokens/sec. Synchronize CUDA around end-to-end timings but report CPU mask time separately. Do not set or claim the C++ 50 µs stretch target for this Python implementation.
8. Extend the existing cleanup cell to release HF model/tokenizer references and GPU cache before closing the OpenAI client and stopping the managed llama-server. Keep cleanup as the only final cell.

Write redacted JSONL evidence:

```text
MyDrive/toolsmith/records/days-06-08-automaton.jsonl
MyDrive/toolsmith/records/days-06-08-vocabulary.jsonl
MyDrive/toolsmith/records/days-06-08-fixed-sampler.jsonl
MyDrive/toolsmith/records/days-06-08-trigger-sampler.jsonl
MyDrive/toolsmith/records/days-06-08-benchmarks.jsonl
```

Update `tests/test_notebook_01.py` to execute the torch-free offline sampler cell against temporary directories, validate new IDs/Python syntax/no stale outputs, and ensure no model token or local API key appears in records. Preserve all existing server/cache/download safeguards and previous milestone cells.

## Verification

Run locally after each evening and again on the final deliverable:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src tests
git diff --check
```

Also rebuild the pinned llama.cpp `test-gbnf-validator` in a temporary project-local directory, rerun all 90 existing GBNF cases, and remove the temporary build afterward. This proves the IR refactor did not change the llama.cpp path. Check `git status --short --branch` before editing and after verification; preserve user changes and do not commit/push unless separately requested.

Real Colab acceptance on a fresh L4:

1. Make the current working tree available in `/content/bare-metal-agent`; GitHub `main` does not contain the current uncommitted work unless it is published first.
2. Run the entire canonical notebook from top to bottom. Earlier health, parser, streaming, SDK, and 90-case GBNF checks must still pass.
3. Our PDA accepts every valid corpus input and rejects every invalid one, including arbitrary byte chunking and partial UTF-8 token boundaries. No valid prefix may reach a dead state.
4. Trie masks equal brute-force masks for the complete synthetic suite and selected real-vocabulary states. Special/padded IDs and invalid EOS are never allowed.
5. Every accepted fixed-mask Qwen generation parses and validates against the selected tool. Named/required/parallel postconditions hold; semantic selection is reported separately rather than conflated with grammar correctness.
6. Cache-on/cache-off greedy runs produce identical token IDs and masks. An invalid highest-logit token must never be selected. Short-budget/incomplete streams produce explicit truncation errors, not apparently successful calls.
7. Triggered generation preserves free reasoning before the marker and validates every emitted call; no-trigger text can finish normally. Atomic marker/suffix tests all pass independently of model output.
8. Benchmark records include p50/p95/p99 and raw counts, not just averages. Slow Python results are reported honestly and do not block correctness completion by an invented latency threshold.
9. All five evidence files exist, contain no secrets, and cleanup releases both model runtimes.

If a check fails, preserve the first failing cell output, raw token IDs/bytes, grammar hash, automaton summary, and relevant logs. Add a minimal regression fixture, fix the specific layer, and rerun the full suite plus real validator/notebook checks. Never resolve a mismatch by deleting invalid cases, allowing fallback unconstrained sampling, or weakening truncation/trigger tests.

## Assumptions and boundaries

- Initial tokenizer support is deliberately Qwen2/Qwen3 byte-level BPE using the slow Qwen2 tokenizer. Other vocabularies require a separately tested codec and fail loudly for now.
- Initial model execution is batch one, no beams, no speculative decoding, and no asynchronous serving. Immutable states make future rollback possible, but those execution modes are not claimed yet.
- Qwen3-0.6B is the correctness/performance debug model; Qwen3-8B remains exercised through llama.cpp. Small-model semantic mistakes are not automaton failures.
- The FastAPI service continues using llama.cpp. Integrating the HF sampler as a second service backend, `response_format` endpoint support, repair retries, and agent execution are outside this milestone.
- No vLLM/SGLang integration, C++/pybind11 port, AVX2 bitsets, full MaskBench/JSONSchemaBench/BFCL run, Qwen3.5 XML format, or new schema features are included.
- Python mask latency is measured as a baseline; the original under-50-µs p99 aspiration applies only to a later optimized/C++ implementation.
- Existing uncommitted work and the single-canonical-notebook structure are preserved.

References:

- [Qwen3-0.6B tokenizer configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json), including byte-level BPE and standalone non-special tool/think tokens.
- [Transformers Qwen2 tokenizer](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/tokenization_qwen2.py), including `bytes_to_unicode`, `byte_decoder`, and replacement-based string decoding that the raw-byte codec must avoid.
- [Transformers generation logits processors](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/generation/logits_process.py).
- [Outlines: Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702).
- [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100).

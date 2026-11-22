# Day 10: Toolsmith test set, BFCL V4 single-turn, and JSONSchemaBench

## Context

Days 1–9 built the two model protocols, GBNF/PDA/mask engines, OpenAI-compatible service, structured responses, validation, and one-retry repair. The local baseline is 285 passing tests and real b5401/b8233 grammar validation, but the complete real-model Colab acceptance is still pending. Day 10 must produce reproducible evidence, not convert CPU fixtures into benchmark claims.

The deliverable is a repeatable evaluation package plus one results report covering:

1. A reviewed **200-case Toolsmith test set** over 8–10 tools, with exact gold calls and four decoding ablations.
2. BFCL V4 **`simple_python`, `multiple`, `parallel`, `parallel_multiple`, and `irrelevance`**, using the official AST evaluator and explicit full/subset denominators.
3. JSONSchemaBench **generation quality**, all-schema **compile coverage**, and MaskBench **valid/invalid replay plus TTFM/TBM timings** against peer engines.
4. Failure taxonomy, latency/token costs, model/runtime/dependency manifests, and resumable raw evidence.

Two constraints shape the design:

- BFCL's current category name is `simple_python`, not `simple`. Java/JavaScript and all live/multi-turn/agentic categories are outside this milestone.
- BFCL function schemas use Python-oriented types such as `dict`, `tuple`, and `any`; many schemas are outside Toolsmith's deliberately strict compiler subset. These cases must be surfaced as **constraint-unsupported**, never silently removed or weakened to make constrained scores look better.

## Approach

Build a small `src/toolsmith/evaluation/` package and extend only `notebooks/01_toolsmith.ipynb`. Use official harnesses for their scoring logic, not for hidden model-specific generation behavior:

- **Toolsmith-owned generation** through the existing OpenAI-compatible service, with pinned prompts/tools/options and raw responses logged.
- **BFCL-owned AST evaluation**, using a thin registered `BaseHandler.decode_ast()` for already normalized OpenAI tool calls and BFCL's result files/eval runner.
- **JSONSchemaBench-owned generation/evaluation**, through a custom `core.engine.Engine` that calls `response_format=json_schema`.
- **MaskBench-owned replay/timing**, through a custom `maskbench.engine.Engine` that drives our PDA and trie mask engine.

This avoids maintaining an upstream BFCL model fork or accidentally testing a vendor handler's prompt format. Pin every harness commit and dataset/model revision; do not install unpinned `main` at run time.

### Evaluation modes

Use stable mode IDs across all datasets:

| Mode | Grammar | Repair | Thinking | Purpose |
|---|---|---:|---|---|
| `prompt` | none | 0 | off | Native prompting/parser baseline |
| `grammar` | own GBNF | 0 | off | Constraint benefit without repair |
| `repair` | own GBNF | at most 1 | off | Recovery from model-output failures |
| `thinking` | none | 0 | on | Think-then-parse comparison |

For constrained tools and structured responses, fixed grammar continues to require `enable_thinking=false`. `thinking` uses the service's `use_grammars=False` path. The standalone HF trigger sampler is reported separately; it is not presented as a FastAPI backend.

For `repair`, reuse a `grammar` result when it needs no repair: its first attempt is identical under the same seed/settings. Generate only cases with a repair-eligible parse/schema/empty-output failure. Truncation and transport errors are not repair-eligible. Record reused versus newly generated cases explicitly, including repair-trigger count, recovery/exhaustion, and added tokens/latency.

### Execution order and budget control

1. Validate all adapters and gold data locally.
2. Run a model smoke pilot (own 12 cases, BFCL 2 IDs/category, JSONSchemaBench 2/split).
3. Complete the own 200-case ablations.
4. Run BFCL full prompt/thinking baselines and constrained-eligible cases for both models.
5. Run all-schema compile coverage, then JSONSchemaBench generation subsets.
6. Stop model runtimes before running CPU-only MaskBench peers and analysis.

Before long runs, estimate time from the pilot's p50/p95 prefill/decode rates and record the Colab unit estimate. Full means every ID in the five pinned BFCL categories; do not silently replace it with a sample. JSONSchemaBench explicitly uses documented subsets for generation/masking and all schemas for compile coverage.

## Key changes

### 1. Evaluation core, manifests, and resumable artifacts

Add these modules:

- `src/toolsmith/evaluation/models.py` — immutable `RunConfig`, `CaseResult`, `AttemptRecord`, status/mode enums, and JSON serialization.
- `src/toolsmith/evaluation/metrics.py` — parse/schema/semantic/repair/performance aggregates and failure taxonomy.
- `src/toolsmith/evaluation/manifest.py` — environment and reproducibility metadata.
- `src/toolsmith/evaluation/runner.py` — sequential request execution, attempt capture, checkpointing, and resume.
- `src/toolsmith/evaluation/report.py` — JSON/Markdown tables and per-mode/per-category summaries.

Every case result contains: dataset/case ID, model/profile, mode, seed/options, prompt and grammar hashes, attempts, raw native response(s), parsed calls, validation outcome, semantic outcome, usage/latency, and error details. Keep tool handlers unexecuted.

Manifest fields:

- Repository commit plus dirty-state marker and a patch/hash for the actual working tree used.
- Toolsmith/harness versions and immutable commits, dependency lock data, Python/CUDA/GPU information.
- Model/tokenizer/GGUF repository revisions, exact files/sizes, llama.cpp tags, model profile, vocabulary fingerprint.
- Dataset revision, dataset hash, selected IDs hash, tools hash, seed, `max_tokens`, temperature, mode flags, and command arguments.
- No HF token, shim key, authorization header, or `.env` contents.

Checkpoint after every case. A resume key is `(run_id, dataset, model, mode, case_id)`. Resume only when the entire config/dataset manifest hash matches; otherwise start a new run directory rather than mixing evidence. Use append-only JSONL for attempts and atomic replacement for final result JSON. Treat stale MaskBench `pending_file` placeholders as incomplete only when they belong to the current owned run; do not delete unrelated outputs.

Run model requests sequentially. Use one CPU thread per MaskBench engine and run engines separately so contention does not distort timings.

### 2. Build and lock the 200-case Toolsmith test set

Add `evaluation/data/toolsmith_single_turn_v1.jsonl` and `evaluation/data/toolsmith_tools_v1.json`. Use 8–10 hand-written tools whose schemas fit the supported subset, for example:

- `get_weather`, `search_docs`, `create_profile`, `schedule_meeting`, `convert_temperature`
- `send_message`, `translate_text`, `sum_numbers`, `stock_quote`, `manage_reminder`

Do not use model-generated gold arguments without review. Gold data contains `case_id`, category, messages, `tool_choice`, ordered expected calls and exact argument objects, shared tool-registry version, and semantic-comparison rule. Parallel comparisons use order-independent matching that still handles duplicate calls; sequential comparisons preserve order.

Use exactly 200 unique cases:

| Category | Count | Main assertion |
|---|---:|---|
| Simple scalar calls | 30 | Tool and exact required arguments |
| Optional properties/defaults | 20 | Omitted versus explicitly supplied optional fields |
| Enums and arrays | 20 | Enum values, empty/nonempty arrays, item types |
| Nested objects | 20 | Nested required fields and exact numbers/booleans/null |
| Parallel calls | 20 | Correct two-call multiset |
| Mixed multiple/parallel-multiple calls | 15 | Correct names, count, and ordering rule |
| No-tool/irrelevance | 25 | No false call; nonempty useful text where appropriate |
| Named tool choice | 15 | Exactly one call to the selected function |
| `required`/`none` choices | 15 | Choice postconditions |
| Unicode/escaping/boundaries | 20 | Unicode, quotes, newlines, marker-like strings, long-but-bounded values |
| **Total** | **200** | |

Add `tests/test_evaluation_dataset.py` to assert count/categories/unique IDs, schema and gold-call validity, all referenced tools exist, no contradictory choice/gold pairs, deterministic ordering, and no secrets. Review and lock the dataset hash before generation.

Compute per mode and category:

- Request, backend-error, timeout, truncation, and empty-response counts.
- Protocol parse rate and schema-valid rate.
- Tool-selection accuracy, exact argument match, exact full-response match.
- False-call rate on no-tool cases.
- Repair trigger/recovery/exhaustion and added attempts/tokens/latency.
- Prompt/completion tokens, tokens/sec, p50/p95 latency, grammar compile time.
- Wilson 95% intervals for proportions and paired case-level bootstrap intervals for mode deltas where meaningful.

A valid call to the wrong tool or with wrong arguments is a semantic failure, not a parser failure.

### 3. Add explicit evaluation policies without weakening production validation

Add a narrowly scoped validator factory hook to `ToolsmithService` for benchmark baselines, and make `use_grammars=False` skip grammar compilation rather than enforcing grammar-subset compatibility on prompt-only requests.

The default service behavior remains unchanged: normal callers still compile and post-validate schemas. The evaluation-only prompt baseline may defer argument semantics to BFCL's official checker when Toolsmith's schema subset cannot represent a BFCL type. Log when this policy is active; never use it for constrained generation or to report schema validity from Toolsmith itself.

Add `src/toolsmith/evaluation/policies.py` with tested configurations for the four modes. API/service errors remain distinct from legitimate no-call text, especially for irrelevance: transport/truncation failures must never be encoded as plain text and accidentally counted as a correct no-call result.

### 4. BFCL adapter and official AST scoring

Add:

- `src/toolsmith/evaluation/bfcl_adapter.py`
- `src/toolsmith/evaluation/bfcl_handler.py`
- `src/toolsmith/evaluation/bfcl_runner.py`

At setup, clone the pinned Gorilla commit into the evaluation environment and install only BFCL's normal evaluation dependencies, not vLLM/SGLang extras. Record `bfcl test-categories` and the actual per-category counts from the pinned data.

Generation:

1. Load each test entry without changing its messages.
2. Convert function docs with BFCL's own `convert_to_tool(..., GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)` so type conversion matches the official FC path. Preserve a mapping between original names and OpenAI-safe underscore names.
3. Send the converted tools through Toolsmith with `tool_choice=auto`, fixed `temperature=0`, pinned `seed`, and a common explicit token budget for all compared modes. Non-streaming is the scoring path; streaming equivalence remains covered by earlier tests.
4. Convert returned OpenAI tool calls to BFCL's standard `[{"function_name": {decoded arguments}}]` form and write `{id, result, latency, input_token_count, output_token_count}` records. Plain text remains plain text for irrelevance.
5. Attempt compilation before each constrained request. If the converted schema is outside Toolsmith's subset, mark `constraint_unsupported` with keyword/JSON Pointer and make no model call. Do not put an invented call into BFCL results.

Scoring:

- Register a `ModelConfig`/`BaseHandler` in the same pinned evaluator process. `decode_ast()` parses the standardized result and converts JSON argument strings to objects; `decode_execute()` uses BFCL's standard conversion helpers. Set `underscore_to_dot=True` so BFCL applies the same name mapping as its OpenAI-compatible models.
- Use `BaseHandler.write(..., update_mode=True)` semantics to produce sorted, duplicate-free `BFCL_v4_<category>_result.json` files at the expected non-live paths.
- Invoke `bfcl_eval.eval_checker.eval_runner.main()` with the registered handler, never an unrelated OpenAI model's handler.
- Produce both:
  - **Full prompt/thinking baseline scores** for every ID in all five categories.
  - **Constrained-eligible scores** using `--partial-eval`, with exact eligible IDs and denominator per category/mode.
- For apples-to-apples ablations, also compute the **eligible intersection** across compared modes. Never call a partial constrained result a full BFCL V4 leaderboard score.

Run all IDs for `prompt` and `thinking`. Run all IDs for `grammar`/`repair`, but unsupported cases are coverage outcomes and only eligible IDs enter constrained scores. Use repair-on-failure reuse as described above. Infrastructure failures block full-score claims; partial pilot scores are always labeled partial.

If the pinned BFCL data has the historical 1,240 IDs across these five categories, verify and report that count; otherwise use and explain the actual pinned counts.

### 5. Compare with llama.cpp's native tool path where supported

Add a `native` comparison mode using the same OpenAI SDK requests against llama-server's own `/v1/chat/completions`, bypassing the Toolsmith parser/grammar/repair. Keep prompts, converted tools, seed, token budget, and model revision matched.

Probe and log `/props`, server tag, selected chat format, and whether native tool calling/grammar is actually supported. Generic fallback is not mislabeled as native Qwen3.5 support. Where unsupported, record `not_supported` rather than a zero accuracy score.

Compare Toolsmith's BFCL eligible-intersection accuracy against this engine baseline. The original “no regression versus llama.cpp built-in parsing” goal is a comparison target, not permission to discard regressions.

### 6. JSONSchemaBench generation quality

Add `src/toolsmith/evaluation/jsonschema_engine.py` implementing the official `core.engine.Engine` API and `src/toolsmith/evaluation/jsonschema_runner.py`.

The engine should:

- Accept the official messages/schema without model-generated examples or custom gold values.
- Call the existing shim with `response_format={"type":"json_schema", "json_schema":{"name":"jsonschemabench", "schema":schema, "strict":True}}`, `enable_thinking=false` for fixed grammar, and the shared seed/token budget.
- Map unsupported schemas to `CompileStatusCode.UNSUPPORTED_SCHEMA`, preserve pointer/message, and make no generation call.
- Map prompt-too-long, timeout, backend, and decode failures to the official status enums.
- Put the assistant JSON text in `output.generation`, actual usage in `token_usage`, and official generation/evaluator metadata in the output. Do not fabricate TTFT/GCT from a non-streaming response; report total generation time and separate compile/mask benchmarks instead.
- Use the profile tokenizer for token counting/context checks and `max_context_length=16384` for the current server configuration. Close clients in `close()`.

Run `core.bench.bench()` and its evaluator, saving raw outputs. Use deterministic IDs by sorting `unique_id`, then seeded stratified selection:

- **Smoke:** 2 schemas per split.
- **Generation:** 25 schemas per split, for each model/mode, across the ten public splits (`Github_easy`, `Github_hard`, `Github_medium`, `Github_trivial`, `Github_ultra`, `Glaiveai2K`, `JsonSchemaStore`, `Kubernetes`, `Snowplow`, `WashingtonPost`; excluding the auxiliary `default` split).

This is 250 generation schemas per model/mode, explicitly a subset rather than a claim of full 9,558-schema coverage. Include prompt/grammar/repair/thinking comparisons. Run the same 250 schemas with Qwen3-0.6B HF unconstrained and fixed-mask generation, labeled separately from the two llama.cpp models.

### 7. All-schema compile coverage and MaskBench correctness

Add `src/toolsmith/evaluation/schema_coverage.py`.

For every schema in the ten public splits:

1. Record schema ID/revision and attempt `GBNFCompiler.compile_schema()` and PDA compilation with bounded rule/depth/configuration limits.
2. Classify `compiled`, `explicit_unsupported` (keyword/pointer), `invalid_schema`, `limit_exceeded`, `compile_error`, `timeout`, or `crash`.
3. Compare with llama.cpp's pinned `json_schema_to_grammar` converter, recording warnings and skipped keywords. Do not equate successful text emission with enforcement.
4. Run each schema in an isolated worker with CPU time, address-space, and wall-clock limits. Kill and replace timed-out workers and checkpoint the outcome. Do not run untrusted schemas unbounded in the notebook kernel.

For the deterministic **100 schemas per split (1,000 total)** MaskBench subset, replay every supplied valid/invalid instance and classify:

- `validation_error`: a valid instance is rejected.
- `invalidation_error` / **under-constrained**: an invalid instance is accepted.
- `passing`: every supplied instance receives the expected verdict.
- Compile errors, timeouts, and crashes remain separate from under-constraint.

The 13 documented unsatisfiable GlaiveAI schemas are reported with the upstream caveat, not quietly removed.

### 8. MaskBench engine and fair peer comparison

Add:

- `src/toolsmith/evaluation/maskbench_engine.py`
- `src/toolsmith/evaluation/maskbench_runner.py`
- A tested Llama 3 vocabulary adapter in `src/toolsmith/sampling/vocabulary.py`.

The custom engine implements `init`, `compile_grammar`, `reset`, `compute_mask`, `commit_token`, `get_id`, `get_name`, `get_module`, `get_version`, and `get_status_extras`:

- `init()` builds the vocabulary/trie once; per-schema `compile_grammar()` builds IR, PDA, and mask engine with cache diagnostics.
- `reset()` restores the initial automaton state while preserving only valid grammar-state caches.
- `compute_mask()` uses the normal `TokenMaskEngine.mask()` and records the returned allowed IDs/timing.
- `commit_token()` returns false without advancing for disallowed IDs, advances atomically for allowed IDs, and never falls back to unconstrained acceptance.
- Expose grammar hash, rule count, vocabulary fingerprint, trie/automaton counts, and cache hit/eviction counts.

MaskBench's reference engines use the Llama 3 tokenizer. Add a separate raw-byte codec for that tokenizer rather than pretending it is Qwen2:

- Recover byte-fallback tokens and ordinary SentencePiece/token pieces without lossy per-token decoding.
- Exclude special/control IDs, handle added tokens and padded IDs explicitly, and discover EOS IDs.
- Prove raw-byte round trips on representative ASCII, whitespace, Unicode, split-codepoint, nested JSON, and marker strings before running any benchmark.
- Keep the Qwen codec unchanged. Use the **same tokenizer and input token IDs for every peer engine** so acceptance and timing are comparable.

Install MaskBench's pinned peer requirements in an isolated evaluation environment: llama.cpp Python bindings, LLGuidance, XGrammar, Outlines Core, plus the official runner. Check `pip check` and record exact versions; do not upgrade the serving environment to satisfy peers.

Run the official `maskbench.runner.process_file()` loop on the same 1,000 schemas and valid/invalid token streams, with warmups excluded and engines isolated. TTFM includes per-schema compilation; TBM is measured per token, including cache hits. Report p25/p50/p75/p90/p95/p99/p99.9/max, compile timeout counts, token counts, and validation/invalidation outcomes. Timeouts count at the configured timeout in TTFM, and timing-only subsets never hide coverage failures.

Use `maskbench_results.main(folder)` per engine and combine outputs with `evaluation/report.py`. Do not edit upstream result tables or run the README auto-aggregation that only knows peer engine IDs.

The Python 50 µs p99 C++ stretch target does not apply. Report actual Python performance honestly and include cache-on/cache-off correctness checks at representative real-vocabulary states.

### 9. Canonical notebook orchestration

Extend only `notebooks/01_toolsmith.ipynb`, before the shared `cleanup` cell:

1. **`day10-notes`** — datasets, modes, denominator definitions, and why BFCL constrained subsets are not full leaderboard scores.
2. **`day10-offline-validation`** — validate all 200 gold cases, schema conversions, fake BFCL results, evaluator registration, resume logic, and report generation.
3. **`day10-eval-setup`** — clone/install pinned harnesses and dependencies, record commits/versions/subset IDs, verify no serving packages were changed unexpectedly.
4. **`day10-runtime-helpers`** — reusable safe phase switching: close SDK/shim/client/server, then restore the already validated Qwen3/b5401 or Qwen3.5/b8233 model without rebuilding/downloading again. Create mode-specific service variants on one port and use them sequentially.
5. **`day10-own-dataset`** — print dataset hash/category counts and run the 12-case smoke check.
6. **`day10-own-qwen3`** and **`day10-own-qwen35`** — all 200 cases and all four modes, with repair reuse and checkpoints.
7. **`day10-bfcl-pilot`** — 2 IDs/category for both models/modes and official smoke scoring.
8. **`day10-bfcl-qwen3`** and **`day10-bfcl-qwen35`** — full prompt/thinking categories, constrained-eligible runs, native-path probe/comparison, and repair reuse.
9. **`day10-bfcl-score`** — register handlers and run official full/partial evaluators, copying raw result/score trees to Drive.
10. **`day10-schema-coverage`** — all-schema compile checks and the 1,000-schema valid/invalid replay subset.
11. **`day10-jsonschema-generation`** — official engine/bench runs for both models/modes and the separately labeled HF debug model.
12. **`day10-maskbench`** — stop active model runtimes, validate Llama 3 raw-byte vocabulary, run custom and peer engines separately, and aggregate metrics.
13. **`day10-report`** — produce final tables/failure taxonomy/manifests and print evidence paths.

Keep cleanup last and extend it for evaluation clients, worker pools, shim variants, and both model runtimes. Resume cells must not force-download models or discard prior evidence.

### 10. Evidence and report

Write under `MyDrive/toolsmith/records/day-10/`:

```text
manifest.json
own-results.jsonl
own-summary.json
bfcl-attempts.jsonl
bfcl-eligible-ids.json
bfcl-result/...
bfcl-score/...
schema-coverage.jsonl
jsonschema-generation/...
maskbench/<engine>/...
failure-examples.jsonl
summary.json
RESULTS.md
```

The reviewed `RESULTS.md` should contain:

- Model/mode/category tables for the own 200 cases and BFCL full/eligible/intersection scores.
- JSONSchemaBench declared/empirical coverage and generation compliance by split.
- MaskBench passing/validation/invalidation counts and TTFM/TBM percentiles for every engine.
- Repair recovery and added-cost table, thinking on/off comparison, and constrained/unconstrained throughput.
- A failure taxonomy: request schema, prompt rendering, backend/transport, truncation, native parse, invalid JSON, unknown tool, missing/extra/wrong-typed/wrong-valued arguments, wrong tool/count, false calls, under-constraint, mask dead ends, timeout, and OOM.
- Exact limitations and links/paths to raw examples and reproducibility metadata. No invented accuracy, performance, or “100%” result.

Update the project README only from these reviewed results; do not hardcode unexecuted benchmark numbers.

## Verification

### Local and fake-harness gates

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src tests
git diff --check
```

Add tests for:

- Dataset integrity and exact semantic comparison.
- BFCL type/name conversion, argument decoding, irrelevance behavior, result paths, handler registration, and no fake-success on infrastructure failures.
- Unsupported-schema pointers and eligible-ID/intersection accounting.
- Repair reuse eligibility and exact usage accounting.
- Deterministic subset selection, resume/hash mismatches, pending-file handling, and per-worker limits.
- JSONSchemaBench status mapping and no-TTFT-fabrication behavior.
- Llama 3 vocabulary raw bytes with synthetic tokens plus real-tokenizer Colab checks.
- Mask commit/reset/cache behavior and disallowed-token non-advancement.
- Notebook cell IDs, syntax, cleanup position, and execution of torch/harness-free Day 10 cells.

Keep the earlier 285 tests and 12 original grammar hashes green. Rebuild real b5401/b8233 validators only if grammar semantics change; adapter/report changes alone should not trigger that work.

### Real Colab acceptance

1. First complete the still-pending Days 1–9 canonical-notebook acceptance. Do not benchmark around a broken serving/parser/sampler gate. Publish or copy the actual working tree to `/content/bare-metal-agent`; GitHub `main` does not contain uncommitted work.
2. Smoke every adapter before long runs: official BFCL smoke scores, one full prompt case per model, one constrained-eligible case, one unsupported-schema case, JSONSchemaBench status outputs, and Llama 3 vocabulary round trips.
3. Own set: all 200 cases have complete results for every mode. Constrained structural failures are investigated rather than relabeled semantic failures.
4. BFCL: full prompt/thinking IDs match the pinned dataset counts exactly, official scores are generated, constrained denominators and eligible intersections are explicit, and no infrastructure error becomes an irrelevance success.
5. JSONSchemaBench: all-schema coverage records exist; generation runs use exactly the selected 250 IDs per model/mode; unsupported schemas make no generation call; evaluator outputs agree with independent `jsonschema` checks.
6. MaskBench: all selected schemas/instances are replayed with the same tokenizer, valid rejection and invalid acceptance are distinguished, peer versions match the lock, cache-on/off results agree, and timing files contain raw per-schema/per-token data.
7. Manifest/evidence files are complete and secret-free, final tables match raw records, and cleanup releases every runtime/worker.

If a check fails, preserve case ID, raw prompt/output/token IDs and bytes, model/runtime revisions, schema and grammar hashes, parser/automaton state, and first harness error. Add a minimal regression fixture and rerun the affected layer plus the full local suite. Never improve a metric by deleting invalid cases, weakening gold answers, hiding under-constraint, or mixing partial results with full BFCL scores.

## Assumptions and boundaries

- Only BFCL V4 Python non-live single-turn AST categories listed above are included. Java/JavaScript, live, multi-turn, memory, web-search, executable, and format-sensitivity categories are deferred.
- Full BFCL scores refer to prompt/thinking baselines over all category IDs. Constrained results use explicitly labeled eligible subsets and intersections because schema rejection is part of Toolsmith's contract. No full constrained leaderboard claim is made without full eligible coverage.
- JSONSchemaBench generation and MaskBench use deterministic subsets (250 and 1,000 respectively per model/engine); compile coverage uses all public splits. These are not presented as full-benchmark generation or timing runs.
- Qwen3-8B and Qwen3.5-9B use llama.cpp. Qwen3-0.6B remains the HF custom-sampler debug model; no Qwen3.5 HF-serving or vLLM integration is claimed.
- No tool execution, multi-turn agent loop, C++ mask port, AVX2 optimization, or C++ 50 µs target is part of Day 10.
- Evaluation dependencies are isolated from the serving environment where practical. The canonical-notebook, Drive-cache, secret-redaction, safe-process, and final-cleanup rules remain unchanged.
- Existing uncommitted work is preserved. Do not commit/push or publish benchmark claims without a separate explicit request and reviewed real evidence.

References:

- [BFCL installation, generation, evaluation, result paths, and partial evaluation](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [BFCL category definitions](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/TEST_CATEGORIES.md)
- [BFCL custom handler requirements](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/CONTRIBUTING.md)
- [BFCL OpenAI-compatible conversion and decoding](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/openai_completion.py)
- [JSONSchemaBench dataset and methodology](https://github.com/guidance-ai/jsonschemabench)
- [JSONSchemaBench custom engine API](https://github.com/epfl-dlab/jsonschemabench/blob/main/docs/custom_engine.md)
- [MaskBench protocol, metrics, and peer versions](https://github.com/guidance-ai/jsonschemabench/tree/main/maskbench)
- [JSONSchemaBench paper](https://arxiv.org/abs/2501.10868)

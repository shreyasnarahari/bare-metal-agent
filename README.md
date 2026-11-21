# Bare Metal Agent: Toolsmith

Toolsmith is a from-first-principles function-calling and constrained-decoding library. It serves open-weight models through raw llama.cpp completions and implements the layers normally hidden by an agent framework: prompt rendering, streaming tool-call parsers, JSON Schema-to-GBNF compilation, a pushdown automaton and token-mask engine, validation and repair, and a small OpenAI-compatible FastAPI shim.

**Status:** local implementation and CPU/grammar verification are complete. Real Qwen/Colab acceptance and BFCL/JSONSchemaBench model benchmarks have not been run. This README intentionally contains no inferred accuracy or performance scores.

## Run and verify

On Colab, clone or `git pull` this repository into `/content/bare-metal-agent`, then run `notebooks/01_toolsmith.ipynb` from the synced tree.

Local checks:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src tests
git diff --check
```

The single canonical Toolsmith notebook is `notebooks/01_toolsmith.ipynb`. On a fresh Colab L4, run the entire notebook top-to-bottom. Do not create per-day notebooks. Put `HF_TOKEN` and `LOCAL_KEY` in Colab Secrets or the protected Drive secrets file, never in committed cells or result records.

## Architecture

| Layer | Implementation | Responsibility |
|---|---|---|
| Model runtime | `src/toolsmith/backends/` | Managed llama-server lifecycle and raw `/completion` streaming/non-streaming calls |
| Model profiles | `src/toolsmith/protocols.py` | Explicit Qwen3 and Qwen3.5 model/protocol selection; no model-name substring guessing |
| Tool registry | `src/toolsmith/registry.py` | Python functions/OpenAI definitions to JSON Schemas; handlers are never executed by the shim |
| Prompt renderer | `src/toolsmith/prompting.py` | Model-owned Jinja templates and model-specific assistant/tool-result history |
| Qwen3 parser | `src/toolsmith/streaming.py` | Incremental JSON-in-`<tool_call>` parsing, reasoning/content separation, parallel calls, truncation errors |
| Qwen3.5 parser | `src/toolsmith/qwen35.py` | Incremental `<function=...><parameter=...>` parsing and schema-directed argument conversion |
| Grammar compiler | `src/toolsmith/grammar.py` | Strict Draft 2020-12 subset to shared immutable grammar IR and GBNF |
| Custom sampler | `src/toolsmith/sampling/` | Pushdown automaton, incremental UTF-8, vocabulary byte trie, allowed-token masks, fixed/atomic-trigger controllers |
| Validation | `src/toolsmith/validation.py` | Strict JSON, registered names, tool-choice postconditions, JSON Schema checks, deterministic JSON Pointers |
| Repair | `src/toolsmith/repair.py` | One bounded non-streaming retry for eligible model-output failures |
| OpenAI shim | `src/toolsmith/api.py` | Authenticated chat completions, SSE, structured responses, usage/repair metadata, health and model routes |
| Evaluation | `src/toolsmith/evaluation/` | Gold data, BFCL adapters, JSONSchemaBench coverage/generation, MaskBench, metrics and resumable records |

The llama.cpp GBNF path and the custom sampler use the same grammar IR. The sampler does not parse its own GBNF back into a second representation.

## Supported API behavior

`/v1/chat/completions` supports ordinary text, streaming and non-streaming tool calls, and these `tool_choice` values:

- `auto`, `none`, `required`
- `{"type": "function", "function": {"name": "..."}}`

Structured responses use:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "answer",
    "schema": {"type": "object", "properties": {}, "additionalProperties": false},
    "strict": true
  }
}
```

Important boundaries:

- `json_object` is unsupported. `text` and an absent `response_format` retain ordinary text behavior.
- `response_format=json_schema` cannot currently be combined with a non-empty `tools` list.
- Fixed tool/schema grammars require `chat_template_kwargs.enable_thinking=false`.
- Prompt-only thinking is supported through the service's explicit `use_grammars=False` configuration. The standalone HF trigger controller is separate from FastAPI.
- Repair is limited to one non-streaming attempt for parse, schema, unknown-tool, choice, or empty-output failures. It does not retry request errors, backend failures, cancellation, or truncation.
- Streaming never replays a second response after deltas have been emitted. A final parse/validation failure produces an SSE error, not a success finish.
- Unsupported generation fields are rejected rather than silently ignored. Multimodal content, arbitrary JSON Schema keywords, beam search, batched HF generation, and speculative decoding are outside the current contract.
- Authentication is enabled with `create_app(..., api_key=...)`. The notebook binds services only to localhost.

## Schema and protocol guarantees

The compiler supports closed objects, required and optional properties, strings, integers, numbers, booleans, null, a single schema or null, scalar enums, homogeneous arrays, and inline nesting. Objects must explicitly set `additionalProperties: false`. Required properties precede optional properties, and declaration order is preserved within each group. Whitespace inside JSON is bounded. Plain text must contain a visible character, so a whitespace-only completion is not a successful constrained response.

Unsupported keywords, unions other than one schema or null, `$ref`, patterns, ranges, and other constraints fail with a schema JSON Pointer. Empty enums are rejected. The implementation does not silently discard them. Post-validation remains enabled even with a grammar.

Qwen3.5 uses raw parameter text rather than JSON strings for string values. Reserved tool markup cannot be represented inside these raw strings and is reported as a protocol limitation. Names and enum values that cannot be represented unambiguously are rejected.

Three distinct outcomes are reported separately:

1. **Protocol validity:** the native output can be parsed.
2. **Schema validity:** parsed arguments or structured content satisfy the selected JSON Schema.
3. **Semantic correctness:** the model selected the expected tool and argument values. A valid call can still be wrong.

## Verified local results

| Check | Verified result |
|---|---|
| Full Python test suite | **310 passed**; only the existing Starlette deprecation warning |
| Notebook cache/replay/offline cells | **8 tests passed** as part of the suite |
| Original Qwen3 grammar compatibility | **12/12 grammars byte-identical** to the pre-Day-9 compiler |
| Original grammar corpus on llama.cpp **b5401** | **90/90 cases** accepted/rejected as expected |
| Original grammar corpus on llama.cpp **b8233** | **90/90 cases** accepted/rejected as expected |
| Qwen3.5 grammar cases on **b5401** | **6 valid and 7 invalid cases** behaved as expected |
| Qwen3.5 grammar cases on **b8233** | **6 valid and 7 invalid cases** behaved as expected |
| Gold evaluation dataset | **200 cases**, **10 tools**, **10 categories**, unique IDs, valid gold calls |
| Python compilation and whitespace checks | `compileall` and `git diff --check` passed |

Temporary llama.cpp validator builds were removed after verification. These results exercise code, parsers, grammars, API fakes, and synthetic vocabularies; they are not model accuracy or throughput measurements.

## Performance: current evidence and required measurements

No real Qwen generation, token-mask latency, BFCL, JSONSchemaBench generation, or peer-engine timing result is available yet. The following table is a collection plan, not a results table.

| Metric | Current result | Canonical collection cell |
|---|---|---|
| Fixed/trigger mask correctness and cache-on/off token equivalence | Colab acceptance pending | `sampler-fixed-generation`, `sampler-trigger-generation` |
| Python mask p50/p95/p99/max and cache behavior | Not measured on a real Qwen vocabulary | `sampler-benchmarks` |
| Constrained versus unconstrained generation tokens/sec | Not measured | `sampler-benchmarks`, `day10-jsonschema-generation` |
| Grammar compile time / TTFM | Not measured across JSONSchemaBench | `day10-schema-coverage`, `day10-maskbench` |
| Per-token mask time / TBM across engines | Not measured | `day10-maskbench` |
| Own 200-case semantic and repair ablations | Not run | `day10-own-qwen3`, `day10-own-qwen35` |
| BFCL V4 single-turn AST scores | Not run | `day10-bfcl-*`, `day10-bfcl-score` |
| JSONSchemaBench declared/empirical coverage and generation compliance | Not run | `day10-schema-coverage`, `day10-jsonschema-generation` |
| Qwen3-0.6B HF prompt/fixed-mask generation | Not run | `day10-hf-jsonschema` |

The under-50-µs p99 aspiration belongs only to a future optimized C++ mask engine. It is not a target or claim for the current Python implementation.

## Evaluation scope and reporting rules

The evaluation package contains the workflows and fake-backed unit tests, but not their real benchmark outputs.

- **Own test set:** 200 exact-gold cases, measuring parse/schema validity, tool selection, exact arguments, false calls, repair recovery, and cost.
- **BFCL V4:** only `simple_python`, `multiple`, `parallel`, `parallel_multiple`, and `irrelevance` in the non-live single-turn Python AST track.
- **Full BFCL claims:** prompt/thinking baselines must include every ID in the pinned categories. Constrained results use exact eligible subsets and eligible intersections because unsupported schemas are rejected, not silently omitted.
- **JSONSchemaBench generation:** deterministic 25-schema-per-split subsets, 250 per model/mode. These are not full-benchmark generation runs.
- **Compile coverage:** attempted across all public splits.
- **MaskBench:** deterministic 100-schema-per-split subsets, 1,000 per engine, with valid rejection and invalid acceptance reported separately.
- **Peer timings:** llama.cpp, LLGuidance, XGrammar, Outlines Core, GRID, and Toolsmith use the same tokenizer and input token IDs. Real Llama tokenizer round trips remain an acceptance gate; only synthetic codec tests have run locally.

Infrastructure failures are not legitimate no-call responses, particularly for irrelevance. Partial BFCL results are never labeled full leaderboard scores.

## Failure taxonomy

Reports should retain the first failing case and enough raw evidence to reproduce it:

- **Request/schema:** invalid tools/history, unsupported keywords, ambiguous protocol values, excessive depth/rules.
- **Prompt/rendering:** wrong template, argument-mapping mismatch, reasoning left in history, multimodal input.
- **Backend:** connection, timeout, HTTP error, invalid backend payload, process startup or shutdown failure.
- **Truncation:** token limit during reasoning, markup, JSON, UTF-8, or tool arguments.
- **Native parse:** missing/stray/nested tags, duplicate parameters, malformed envelopes, incomplete XML.
- **JSON/schema:** duplicate keys, non-finite numbers, invalid JSON, missing/extra properties, wrong types or enum values.
- **Semantics:** wrong tool, wrong call count, wrong argument value, or a false call where no tool was needed.
- **Constraint engine:** dead mask, stack/configuration limit, valid-prefix rejection, or invalid-instance acceptance (under-constraint).
- **Repair:** retry triggered, recovered, exhausted, or deliberately not attempted.
- **Resources:** GPU OOM, worker memory/CPU limit, stale checkpoint, or manifest mismatch.

A semantic error must not be relabeled as a parser defect, and an unsupported schema must not be converted into an apparently successful unconstrained generation.

## Reproducibility and evidence

The notebook preserves model/build caches, writes redacted JSONL records, and keeps one shared final `cleanup` cell. Day 10 evidence is configured under:

```text
MyDrive/toolsmith/records/day-10/
```

Expected artifacts include the manifest, own-set results, BFCL result/score trees, schema coverage, JSONSchemaBench generation, per-engine MaskBench results, failure examples, and `RESULTS.md`. A run resumes only when its complete configuration and dataset hashes match. No result should be published without its model/runtime revisions, harness commits, selected IDs, seed/options, and raw evidence.

## Next acceptance steps

1. Publish or copy the exact working tree to Colab and run the canonical notebook from a fresh L4 runtime.
2. Complete all earlier serving, parser, sampler, Qwen3.5, and cleanup checks before benchmarking.
3. Run the small adapter smoke tests before long BFCL or JSONSchemaBench jobs.
4. Generate results from recorded evidence, then replace the pending performance section with measured values and explicit limitations.
5. Preserve failing raw outputs as regression fixtures. Never improve a metric by deleting invalid cases, weakening gold answers, or hiding under-constraint.

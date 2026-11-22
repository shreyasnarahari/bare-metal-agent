# Bare Metal Agent: Toolsmith

Toolsmith is a from-first-principles function-calling and constrained-decoding library. It serves open-weight models through raw llama.cpp completions and implements the layers normally hidden by an agent framework: prompt rendering, streaming tool-call parsers, JSON Schema-to-GBNF compilation, a pushdown automaton and token-mask engine, validation and repair, and a small OpenAI-compatible FastAPI shim.

**Status:** local implementation and CPU/grammar verification are complete. Real Qwen/Colab acceptance and BFCL/JSONSchemaBench model benchmarks have not been run. This README intentionally contains no inferred accuracy or performance scores.

## Run and verify

On Colab, clone or `git pull` this repository into `/content/bare-metal-agent`, then run `notebooks/01_toolsmith.ipynb` (Toolsmith) or `notebooks/02_steadyhand.ipynb` (Steadyhand) from the synced tree.

Local checks:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src tests
git diff --check
```

There are two notebooks: `notebooks/01_toolsmith.ipynb` for the Toolsmith track and `notebooks/02_steadyhand.ipynb` for the Steadyhand agent loop. Both provision llama.cpp and the GGUFs through `toolsmith.provision`, which shares one set of Drive caches. On a fresh Colab T4 or L4, run a notebook top to bottom. Do not create per-day notebooks. Put `HF_TOKEN` and `LOCAL_KEY` in Colab Secrets or the protected Drive secrets file, never in committed cells or result records.

### Running on a T4 or L4

Each notebook's shared `hardware` setup cell detects the GPU, vCPUs, RAM, and free disk once (`toolsmith.hardware`). Every setting below comes from those numbers, not from the GPU's name. Without a GPU, with less than 14 GiB of VRAM, or with compute capability below 7.0, the cell stops and says how to switch runtimes.

| Setting | Rule | T4 (2 vCPUs, about 13 GB RAM) | L4 (8 vCPUs) |
|---|---|---|---|
| llama.cpp build jobs | one per vCPU, at most one per 2 GiB of RAM | 2 | 8 |
| Day 10 server | 4096 tokens per slot; more VRAM means more slots, not longer ones | 4 slots, `-c 16384` | 8 slots, `-c 32768` |
| Steadyhand server | preset per GPU and shape, confirmed by the probe | per preset | per preset |
| CPU pools (CPU gate, schema coverage) | one per vCPU, at most 8 | 2 | 8 |
| HF Qwen3-0.6B generation | one fp16 model per process | 2 processes | 6 processes |
| MaskBench | the Toolsmith engine in the kernel, plus single-threaded peers | 1 peer at a time | 5 peers at once |

Each Day 10 request pool sends one request per server slot. Any other GPU uses the T4 or L4 rows that match its VRAM.

- **Overlapped setup.** Downloads, builds, Drive restores, and Drive write-backs run as background jobs (`toolsmith.background`). Each job writes its own log under `/content/background-logs/`.
  - In the Toolsmith notebook, both GGUFs download while b5401 builds.
  - The b8233 build for Qwen3.5 starts after b5401 and runs through Days 2–8 at low priority.
  - The MaskBench peer environment installs during the Day 10 generation runs.
  - In the Steadyhand notebook, provisioning runs during the CPU gate.
  - On a cached runtime, these jobs are short copies from Drive.
- **Drive layout.** These folders under `MyDrive/toolsmith/` are shared by every GPU:
  - `builds/`, holding llama.cpp builds. Their folder names already include the CUDA compute capability.
  - `models/`, holding the GGUFs and the HF datasets cache.

  Records go to `records/<gpu>/` (`t4`, `l4`, or the GPU's name), so a T4 run and an L4 run never resume, overwrite, or skip each other's files.
- **Stamps and manifests.**
  - Every Toolsmith record carries a `runtime` stamp: the GPU, vCPUs, load average, and running background jobs.
  - Day 10 checkpoint manifests also include the compute capability, the llama.cpp tag, and the server shape. If those change, the run stops and lists every field that differs.
- **Timing under concurrency.** Timing cells are not serialized. The MaskBench peers run side by side, and the Days 6–8 benchmarks may overlap a background build. Each timing record states the concurrency it ran under, in the stamp or in `run.json`. Compare timings only between runs with the same concurrency.
- **Steadyhand serving.**
  - The server cell tries KV cache and flash-attention settings in this order: q8_0 with flash attention, f16 with it, then f16 without it.
  - It keeps the first setting that loads, leaves 1 GB of VRAM free, and passes a decode-speed self-test.
  - Running out of VRAM drops one slot.
  - Sweeps mirror their scores to Drive and resume from that copy after a disconnect.

## Steadyhand: the agent loop

`src/steadyhand/` is the agent loop that runs on top of the Toolsmith shim. The design notes are in `Steadyhand/`. It sends a conversation and tool schemas through `ToolsmithService.chat`, validates each native tool call, and runs it against an in-memory mock store. On a tool failure it applies one of six recovery policies, and it writes one append-only journal per episode. CPU replay of a journal reproduces the episode exactly.

The boundary is unchanged: **handlers run in Steadyhand, after the shim returns. The shim and its FastAPI routes never execute a tool.** Steadyhand runs on Colab from `notebooks/02_steadyhand.ipynb`.

**Status:** the 60-task suite, the fault injector, the policies, replay, and the report generator are implemented. They are checked on CPU with a scripted model and golden journals. **The live sweep has not been run.** No Steadyhand accuracy, duplicate-effect, or throughput number is claimed. Once a sweep runs, `results/steadyhand/RESULTS.md` is generated from its journals by `steadyhand.eval.report`. This section will cite only the measured cells, with their manifests.

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

The notebook preserves model/build caches, writes redacted JSONL records, and keeps one shared final `cleanup` cell. Day 10 evidence is configured under (`<gpu>` is `t4`, `l4`, or the GPU's name):

```text
MyDrive/toolsmith/records/<gpu>/day-10/
```

Expected artifacts include the manifest, own-set results, BFCL result/score trees, schema coverage, JSONSchemaBench generation, per-engine MaskBench results, failure examples, and `RESULTS.md`. A run resumes only when its complete configuration, dataset, and runtime (compute capability, llama.cpp tag, server shape) hashes match. No result should be published without its model/runtime revisions, harness commits, selected IDs, seed/options, and raw evidence.

## Next acceptance steps

1. Publish or copy the exact working tree to Colab and run the Toolsmith notebook from a fresh T4 or L4 runtime. Results are kept per GPU.
2. Complete all earlier serving, parser, sampler, Qwen3.5, and cleanup checks before benchmarking.
3. Run the small adapter smoke tests before long BFCL or JSONSchemaBench jobs.
4. Generate results from recorded evidence, then replace the pending performance section with measured values and explicit limitations.
5. Preserve failing raw outputs as regression fixtures. Never improve a metric by deleting invalid cases, weakening gold answers, or hiding under-constraint.

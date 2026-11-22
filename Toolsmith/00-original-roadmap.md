# Toolsmith: day-by-day implementation plan

## Context

The repository is currently empty except for its Git metadata. The goal is a Python library and OpenAI-compatible service that implements tool prompting, streaming tool-call parsing, JSON Schema-to-grammar compilation, constrained sampling, validation, and one repair retry without delegating those features to llama.cpp's built-in chat/tool parser.

We will follow the supplied canvas over 11 core evenings, with Evening 12 reserved for spillover. Work will proceed one evening at a time. After approval, implement **Evening 1 only**, stop at its verification checkpoint, and use the recorded outputs as the starting point for Evening 2.

Primary scope:
- Qwen3-8B GGUF Q4_K_M on an L4 through llama.cpp's raw `/completion` API.
- Qwen3-0.6B or 1.7B through Hugging Face Transformers for custom-sampler development.
- Qwen3.5-9B after the Qwen3 baseline works.
- BFCL V4 single-turn AST categories, including irrelevance. Multi-turn and agentic evaluation belong to Day 2, as the canvas specifies.
- JSONSchemaBench coverage plus MaskBench-style correctness and timing measurements.

## Approach

Use a Python 3.10 `src/` package with small, independently testable components and Colab notebooks for GPU work:

- `toolsmith.registry`: Python/Pydantic functions to named tool definitions and JSON Schemas.
- `toolsmith.prompting`: model-specific chat-template rendering with the tokenizer's own `apply_chat_template(..., tools=...)`; send the resulting string to raw completion, never to llama.cpp's tool-aware chat endpoint.
- `toolsmith.backends`: a common generation interface, with an HTTP llama.cpp backend and a Transformers backend for the custom sampler.
- `toolsmith.parsing`: non-streaming and incremental protocol parsers producing normal text, reasoning text, tool-call deltas, terminal events, and explicit truncation/malformed-output errors.
- `toolsmith.grammar`: one shared schema/grammar intermediate representation, emitted as GBNF for llama.cpp and consumed by our own automaton for masking. Unsupported semantics must raise a structured error rather than be ignored.
- `toolsmith.masking`: a byte-level pushdown automaton, decoded-vocabulary trie, whole-token acceptance checks, EOS/EOT handling, and mask caching. A `LogitsProcessor` sets invalid token logits to `-inf` before normal sampling.
- `toolsmith.validation`: `jsonschema` validation after parsing, including semantic checks the grammar does not express, and at most one error-containing repair request.
- `toolsmith.api`: FastAPI `/v1/chat/completions`, streaming SSE and non-streaming responses, OpenAI-shaped errors, `tools`, all requested `tool_choice` modes, `parallel_tool_calls`, and schema-based `response_format`.

Keep llama.cpp's schema converter and other engines as test/benchmark references only, never as the implementation of our compiler or masker. Record raw prompts, responses, token information where available, versions, seeds, sampling settings, and timings as JSONL so parser and grammar work can continue on CPU without using Colab units.

## Key changes and nightly checkpoints

### Evening 1 — M0: reproducible Colab serving

Create the initial project scaffolding:
- `pyproject.toml`, `.gitignore`, `src/toolsmith/`, and the requested test layout.
- `notebooks/01_colab_server.ipynb` with pinned llama.cpp and model revisions, CUDA build or pinned prebuilt binary, Drive caching, and GGUF copying to local Colab storage before loading.
- A small `toolsmith.backends.llama_cpp` client for health and raw completion requests, API-key headers, timeouts, response errors, and JSONL capture.
- `tests/` for request construction, error handling, and capture redaction using a mocked HTTP transport; no GPU required.

The notebook will start Qwen3-8B Q4_K_M using `--jinja -ngl 99 -c 16384`, an explicit port, and an API key read from Colab Secrets. It will poll `/health`, call `/completion`, record GPU/build/model metadata and actual resource usage, and cleanly stop the server/runtime. Model weights, binaries, secrets, logs, and benchmark outputs stay out of Git.

**Checkpoint:** health succeeds and a real raw completion returns from the Colab L4; its sanitized JSONL record can be used offline. The notebook must also be valid JSON and all CPU tests must pass locally.

### Evening 2 — M1: expose the raw protocols

Implement the tool registry and model-aware prompt renderer. Add `notebooks/02_raw_protocol.ipynb` to:
- Render identical conversations and tools for Qwen3 and a Hermes-style model.
- Diff rendered prompts against each model's native Jinja template.
- Send raw prompts to `/completion`, inspect raw text and tokens, and locate thinking blocks, tool tags, parallel calls, and end-of-turn tokens.
- Save representative valid and malformed outputs as versioned fixtures under `tests/fixtures/`.

**Checkpoint:** recorded Qwen3 and Hermes examples reproduce through the raw-completion path, with no manual prompt reformatting or built-in tool parsing.

### Evenings 3–4 — M2: parser

Build the non-streaming parser first, then the incremental state machine and OpenAI delta adapter. Model format adapters share the same event interface. Cover:
- Thinking, ordinary content, and calls in the same output.
- Markers split across arbitrary chunk boundaries and multiple calls in sequence.
- Partial JSON argument deltas versus final parsed arguments.
- Missing closing tags, malformed/truncated JSON, trailing text, incomplete UTF-8, and explicit end-of-input behavior.
- Equivalence between aggregated streaming events and the non-streaming result.

**Checkpoint:** fixtures pass with one-token chunks, every possible split point for small inputs, and varied larger chunkings. Malformed and truncated calls become explicit errors, never apparently successful empty calls.

### Evenings 4–5 — M3: schema-to-GBNF compiler

Implement the canvas's defined subset: objects, required/optional properties, strings, integers, numbers, booleans, enums, arrays, nesting, and `additionalProperties: false`. Preserve JSON escaping and bound generated whitespace. Compile the whole tool list into normal-reply/tool-call alternatives and generate variants for:
- `auto`, `none`, and `required`.
- A named function.
- Single versus parallel calls.

Reject unsupported or silently ignored keywords with schema paths and reasons. Compare generated grammars and accepted examples against llama.cpp's converter as a differential reference, then send our GBNF through the raw request's `grammar` field.

**Checkpoint:** valid examples parse and validate, invalid examples fail, every `tool_choice` variant behaves correctly, and constrained real-model calls no longer produce malformed syntax for the supported test set.

### Evenings 6–8 — M4: custom token-masking sampler

Use the shared grammar representation to build a byte-oriented pushdown automaton and vocabulary trie. Implement:
- Whole-token transitions, including tokens crossing JSON/marker boundaries and partial UTF-8 sequences.
- Push/pop behavior for nested objects and arrays, alternatives, repetition, accepting states, and allowed EOS/EOT only when valid.
- Stateful logits processing and rollback/advance behavior required by generation.
- State-keyed mask caching and efficient NumPy bitsets/logits updates before ordinary temperature/top-k/top-p sampling.
- Trigger-activated constraint mode so thinking remains unconstrained until a tool-call opening is detected.

Test with Qwen3-0.6B/1.7B before attempting the larger model. Use a simple brute-force masker on tiny vocabularies as a correctness oracle, then test the real tokenizer. Measure compile time, p50/p99 mask latency, memory, and tokens/second with and without constraints. Optimize only after differential tests pass.

**Checkpoint:** generated outputs satisfy the supported grammars in real small-model runs; the masker agrees with the oracle on adversarial token sequences; no empty-mask or invalid-EOS case is hidden; latency and memory results are recorded.

### Evening 9 — shim, repair, and Qwen3.5 support

Finish the FastAPI endpoint over the common backend interface, including OpenAI-compatible streaming and non-streaming tool-call responses, finish reasons, usage, and errors. Add post-validation and exactly one repair retry for validation failures. Truncation remains an explicit failure.

Add the Qwen3.5 protocol adapter from real recorded outputs, then run its quantized GGUF with a compatible pinned llama.cpp build. Disable thinking initially for parser debugging, then enable it for ablations. Tool execution itself remains outside this service and is deferred to Day 2's agent loop.

**Checkpoint:** the official OpenAI Python client can make real text, tool, named-tool, required-tool, parallel-tool, schema-response, and streaming requests to our endpoint. Contract tests also run against a fake backend locally.

### Evening 10 — evaluation and ablations

Add reproducible evaluation scripts and pinned evaluation dependencies:
- A roughly 200-case local suite over 8–10 tools measuring parse rate, schema validity, exact argument match, tool selection, false-call rate, truncation, and throughput.
- A BFCL V4 adapter/configuration targeting our endpoint. Smoke-test selected IDs, then run the single-turn simple, multiple, parallel, parallel-multiple, and irrelevance categories using the official evaluator. Label partial results clearly.
- JSONSchemaBench compile/reject/under-constraint reporting across all schemas, valid/invalid instance checks where supplied, and end-to-end generation on a documented representative subset.
- MaskBench-style compile and mask timings alongside pinned llama.cpp GBNF, XGrammar, and LLGuidance references under the same resource limits.

Run prompt-only, our grammar, grammar plus one repair, and thinking on/off ablations on Qwen3-8B and Qwen3.5-9B using the same fixed evaluation subset. Run the full selected BFCL single-turn categories for the best configuration per model; do not present subset ablation scores as leaderboard scores.

**Checkpoint:** all raw results and environment metadata are saved outside Git, concise result summaries are reproducible, and every failure is classified as unsupported schema, malformed output, invalid arguments, wrong tool/arguments, false call, truncation, timeout, or engine error.

### Evening 11 (and 12 if needed) — M6: write-up

Complete the canvas-requested README with setup, architecture, exact API support, schema support matrix, benchmark commands, results, throughput, failure taxonomy, and guarantees versus limitations. Include reproduction instructions and the distinction between syntactic validity, schema validity, and semantic correctness. Keep C++/AVX2, vLLM batching, and a third model family as stretch work only after the core checkpoints pass.

## Verification

At each evening's checkpoint:
- Install the project with its declared development dependencies and run `pytest`; parser, grammar, masker, and API tests must pass before moving on.
- Exercise real model calls for GPU milestones, not just imports or mocks. Save raw evidence and report any Colab-only step that still needs user execution.
- Test streaming consumers end to end and compare streamed aggregation with non-streaming responses.
- Use official BFCL evaluation commands for scores and `jsonschema` for validation; never treat grammar acceptance alone as semantic correctness.
- Record model/quantization, llama.cpp commit, tokenizer/template revision, dependency versions, GPU, seed/sampling settings, and actual Colab resource usage with every benchmark table.
- Keep generated datasets, model files, logs, caches, and large raw results ignored; leave unrelated files untouched and check `git status` after test/build runs.

## Assumptions

- We implement one evening per working session, beginning with Evening 1 after this plan is approved, rather than generating the entire project in one pass.
- You will run the supplied Colab notebooks in your own L4/T4 runtime and provide any failure output; this local environment cannot directly verify Colab GPU execution. CPU-side implementation and tests will be completed locally before each handoff.
- Colab Secrets will supply Hugging Face/API credentials, and Drive will supply optional caching. No secret values will be written into notebooks, source, logs, or Git.
- Costs and runtimes are estimates, not guarantees. Actual Colab unit burn is checked before long runs; recorded fixtures and CPU tests are preferred for parser/compiler development.
- BFCL V4 scope for this project is the canvas's single-turn AST set. Multi-turn, web-search, and memory categories require Day 2 and, for some categories, external services.
- The custom sampler's main path is Python/NumPy with Transformers, exactly as in the canvas. The C++ AVX2 port and vLLM integration are optional stretch goals and are not required for Day 1 completion.
- Parse-rate, validity, accuracy, and latency figures in the canvas are targets to measure, not results we can promise before benchmarking.

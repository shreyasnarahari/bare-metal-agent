# Day 9: FastAPI shim, validation, repair, and Qwen3.5

## Context

The repository already contains more of this milestone than the original outline suggests:

- `ToolsmithService` and `/v1/chat/completions` in `src/toolsmith/api.py:85` already support non-streaming and OpenAI-style SSE responses, all four `tool_choice` modes, raw llama.cpp completions, and JSONL recording.
- `ToolCallValidator` in `src/toolsmith/validation.py:31` already checks JSON arguments, registered tool names, Draft 2020-12 schemas, and tool-choice postconditions. It does not execute tools.
- `Qwen3IncrementalParser` in `src/toolsmith/streaming.py:265` and `GBNFCompiler` in `src/toolsmith/grammar.py:266` implement the Qwen3 JSON-in-`<tool_call>` protocol.
- The shared grammar IR/PDA/mask engine from Days 6–8 can execute any grammar built from the existing IR; it must not gain a second GBNF or XML parser.
- `response_format` is explicitly rejected at `src/toolsmith/api.py:434`, and a parse/validation failure currently ends the request without a retry.

Day 9 should therefore finish and harden the existing shim, not create a parallel service. The deliverable is:

1. A validated OpenAI-compatible endpoint with schema-constrained `response_format`, clear errors, bounded non-streaming repair, and complete attempt/usage records.
2. A separately tested Qwen3.5 profile using its native `<function=...><parameter=...>` format for prompt rendering, non-streaming/streaming parsing, GBNF, and the existing PDA/mask engine.
3. Real Qwen3-8B and Qwen3.5-9B checks in the one canonical notebook, `notebooks/01_toolsmith.ipynb`.
4. Evidence that distinguishes protocol validity, schema validity, repair recovery, and semantic tool choice.

The last recorded local baseline is 224 passing tests plus 90/90 real llama.cpp GBNF cases, but no complete real Qwen/Colab run has yet been performed. Day 9 completion must not be inferred from CPU tests alone.

## Approach

Use one request pipeline for both model families, with an explicit per-service model profile:

```text
validate request -> normalize tools/history -> render model-owned Jinja template
-> compile the profile's grammar -> raw llama.cpp completion
-> profile parser -> tool/structured-output validation
-> optional single repair -> OpenAI response + attempt records
```

Keep the default profile and GBNF byte output for Qwen3 unchanged. Qwen3.5 is an explicit profile, never selected by guessing from a model-name substring. The service remains backed by raw llama.cpp `/completion`; integrating the experimental Hugging Face sampler into FastAPI or adding a vLLM dependency is not needed for this milestone.

Work in three checkpoints, which may be done in one long evening or two sittings:

- **Checkpoint A, CPU:** request/response contracts, structured-output validation, repair state machine, and deterministic tests.
- **Checkpoint B, CPU:** Qwen3.5 protocol, grammar, prompt-history handling, arbitrary-chunk streaming tests, and real llama.cpp grammar validation.
- **Checkpoint C, L4:** extend the canonical notebook, rerun all earlier gates, then launch a separate Qwen3.5-capable runtime and record live evidence.

## Key changes

### 1. Freeze the existing API contract, then make unsupported behavior explicit

Refactor `src/toolsmith/api.py` around a small prepared-request/output pipeline rather than adding more branches to `chat()`:

- `_PreparedRequest` should carry the model profile, tools, normalized choice, rendered prompt, optional grammar, optional response schema, and the appropriate output validator.
- A non-streaming attempt returns the raw backend response, parsed response, usage, and a structured failure suitable for either repair or an API error. A streaming attempt uses the same final parse/validation code but never enters repair.
- Keep successful response shapes compatible with the official OpenAI SDK: assistant `content`, optional `reasoning_content`, indexed `tool_calls`, `finish_reason`, and token usage. Do not execute a tool or turn a tool result into an assistant call automatically.
- Validate that `request.model` matches the model exposed by the service profile so a Qwen3 process cannot claim to be Qwen3.5.

Harden request handling before generation:

- Replace silently ignored `extra="allow"` request fields with explicit supported fields and a 400 error for unsupported generation options. In particular, do not accept `n`, beams, penalties, `top_p`, `top_k`, `stop`, or penalties and then silently drop them. Adding forwarding for these is a separate task.
- Retain the existing checks for `stream_options`, `chat_template_kwargs`, tool definitions, named choices, and schema pointers. Add clear role/content/tool-call-history checks, including JSON-object tool arguments and matching tool-result IDs where required by the profile.
- Reject multimodal content for this text-only milestone instead of dropping image/video parts.
- Add a local `/health` endpoint and optionally `/v1/models` reporting the configured profile. Let `create_app(..., api_key=...)` enforce a bearer key with constant-time comparison; the notebook passes `LOCAL_KEY`, binds only to `127.0.0.1`, and does not expose a public tunnel.
- Preserve 400 for caller/request/schema errors, 502 for backend failures, and explicit terminal errors for malformed model output and truncation. Never convert an empty/dead mask, backend error, or truncated request into unconstrained generation.

Add/adjust API tests to prove invalid requests make zero backend calls, authentication fails before generation, existing Qwen3 requests retain their prompts/grammars, and official SDK models still validate every non-streaming chunk and response.

### 2. Add shared strict JSON diagnostics and `response_format=json_schema`

Extend `src/toolsmith/validation.py` without mixing structured responses into tool calls:

- Add a reusable strict JSON loader that rejects trailing non-whitespace, `NaN`/infinities, and duplicate object keys with useful locations. Use it for tool arguments and structured responses.
- Add `StructuredOutputValidator` alongside `ToolCallValidator`. It returns the decoded value on success and raises deterministic, bounded diagnostics containing a JSON Pointer on failure.
- Sort validation errors by serialized pointer rather than raw path components; the current mixed string/integer path sorting can itself fail on nested object/array errors. Aggregate a bounded number of errors across calls so the repair prompt is useful, while retaining stable error codes such as `unknown_tool`, `invalid_tool_call_json`, `schema_validation_failed`, and `tool_choice_violation`.
- Keep full post-validation even when a grammar was supplied. A grammar proves syntax/schema subset compliance, not that the selected tool or argument meaning is correct.

Implement the existing `response_format` request field for:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "weather_answer",
    "schema": {"type": "object", "...": "..."},
    "strict": true
  }
}
```

Behavior:

- Accept an optional description but require a non-empty name, a schema object, and `strict: true`. Compile with the existing `GBNFCompiler.compile_schema()` and the same deliberately supported Draft 2020-12 subset; unsupported keywords still fail with their JSON Pointer.
- Allow any root type supported by our compiler, while documenting that this is the Toolsmith subset rather than a promise to reproduce every OpenAI Structured Outputs rule.
- Initially reject `json_schema` combined with a non-empty `tools` list, avoiding two competing output roots. `json_object` remains unsupported and fails loudly; absent/`text` format retains ordinary text behavior.
- Add the schema and an instruction to return only JSON to a copied system message in `src/toolsmith/prompting.py`, then render through the model's own chat template. A grammar alone does not teach the model what the requested schema means. Do not mutate the caller's message dictionaries.
- Fixed schema generation requires `enable_thinking=false`, just like fixed tool grammars. Prompt-only schema requests can retain thinking for ablation, but validation sees only the final response and incomplete/thinking-only output fails explicitly.
- Return the validated JSON text in `message.content`, with no synthetic tool call and `finish_reason="stop"`. Streaming emits ordinary content deltas and validates the complete aggregate before sending a success finish. Truncation or schema failure emits an SSE error and `[DONE]`, never a successful finish.

Add `tests/test_structured_output.py` and API coverage for valid/invalid JSON, nested pointers, duplicate keys, unsupported formats/keywords, tools-plus-schema conflicts, truncation, and streamed aggregation. Use the official OpenAI SDK for both a normal request and a Pydantic `parse()` request whose generated schema lies in our subset.

### 3. Implement one bounded, observable repair retry

Add `src/toolsmith/repair.py` and configure `ToolsmithService` with a policy such as `RepairPolicy(max_repairs=1, max_output_chars=..., max_error_chars=...)` plus an explicit `use_grammars=True` switch for prompt-only ablations.

The retry rule is deliberately narrow:

- Retry at most once, only for non-streaming model-output failures: malformed native calls, invalid structured JSON, schema/choice violations, unknown tools, and empty output.
- Do **not** retry request errors, authentication, backend connection/HTTP failures, client cancellation, dead masks, or token-limit truncation. Each attempt retains the original `max_tokens`; no silent budget increase.
- Rebuild a complete ephemeral conversation through `ChatPromptRenderer`, including the original user request, tools/response schema, the bounded invalid model output in an explicit untrusted-data block, and bounded validation diagnostics. Ask for a complete corrected response, not a patch and not a tool result. Never insert the invalid output as a valid assistant `tool_calls` history entry.
- Reuse the same profile and grammar on the retry. Repair is not permission to remove constraints, call another tool, execute the call, or coerce an invalid value until it passes.
- For Qwen3.5, schema-directed conversion failures are included in the diagnostics. A literal string `"null"` for a string field is not converted to null, and invalid booleans are not coerced to false.
- Keep normal streaming incremental. Once deltas have been sent, replaying a repair would duplicate content/arguments and violate OpenAI stream semantics. Streaming therefore performs final validation and returns a terminal error without repair. Buffered stream repair is explicitly out of scope.

Return only the final successful assistant message. Aggregate prompt/completion/total token usage across all attempts, and record each attempt separately with prompt, raw response, grammar hash, parser/validator error, elapsed time, and usage. A successful repaired response is logged as `repaired`, not confused with a first-attempt success. Exhaustion returns a terminal server error preserving the final failure code and bounded diagnostics, with `repair_attempts=1`; raw model output remains in the redacted JSONL record rather than bloating the HTTP error.

Add `tests/test_repair.py` for first-attempt success, success on attempt two, two failures, all non-retryable categories, identical grammar forwarding, bounded prompts, original-message immutability, summed usage, redaction, and proof that streaming invokes the backend only once. Include both Qwen3 and Qwen3.5 malformed-output fixtures.

### 4. Introduce explicit model/protocol profiles

Add `src/toolsmith/protocols.py` with immutable profiles for at least:

- `qwen3`: model/tokenizer `Qwen/Qwen3-8B`, JSON envelope tool protocol, existing parser/compiler behavior.
- `qwen35`: model/tokenizer `Qwen/Qwen3.5-9B`, Qwen3 Coder-style XML parameter protocol, thinking enabled by default.

A profile selects the parser factory, tool-grammar envelope, and history-normalization hook. `ToolsmithService` defaults to the Qwen3 profile for compatibility; a separate service instance is constructed for Qwen3.5. The model profile and protocol are recorded in every JSONL attempt.

Update `ChatPromptRenderer`/`src/toolsmith/prompting.py` so Qwen3.5 assistant history is converted on a deep copy before `apply_chat_template`:

- OpenAI responses expose `function.arguments` as a JSON string, while the official Qwen3.5 template iterates argument mappings to render `<parameter>` blocks. Strictly decode those strings for Qwen3.5 only and reject malformed history before generation.
- Leave Qwen3 history rendering byte-for-byte unchanged.
- Verify assistant tool call -> tool result -> next assistant response for both profiles. Tool results remain `role="tool"` externally even though both Qwen templates render them as user-side `<tool_response>` blocks internally.
- Snapshot representative prompts with tools, thinking on/off, parallel calls, and tool results. These snapshots test our use of the official Jinja template; they do not replace it with a handwritten prompt.

The existing fixed-grammar rule remains: constrained tool/schema requests use `chat_template_kwargs.enable_thinking=false`. A service configured with `use_grammars=False` may run thinking-plus-tools through prompt, parser, validation, and non-streaming repair. This keeps Qwen3.5's default thinking mode testable without pretending that the fixed GBNF root permits free reasoning. The standalone HF trigger controller remains separate from the FastAPI service.

### 5. Add a strict Qwen3.5 XML parameter parser

Add `src/toolsmith/qwen35.py` with `Qwen35IncrementalParser` and a non-streaming `Qwen35ToolCallParser` facade. Reuse the existing `ParsedResponse`, `ParsedToolCall`, and parser-event/OpenAI delta types. Extract only genuinely shared reasoning/text buffering helpers from `streaming.py`; do not copy the entire Qwen3 parser or change its established behavior.

Parse exactly the trained form documented by the official template:

```xml
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
<parameter=unit>
c
</parameter>
</function>
</tool_call>
```

Required behavior:

- A handwritten incremental state machine handles `tool_call`, `function`, and `parameter` boundaries at arbitrary chunk splits. Do not use regex-only partial parsing, a general XML parser, `eval`, or `ast.literal_eval`.
- Support ordinary/reasoning text, omitted opening `<think>` caused by the generation prompt, explicit `</think>`, and Qwen3.5's tool call directly from reasoning. Reject nested calls, stray closers, duplicate function/parameter blocks, trailing malformed content, and incomplete input with stable error codes.
- Buffer function/parameter names only until `>`, and buffer a parameter value until its exact closing marker. Preserve value whitespace except for removing at most one template-framing newline at each end; do not `.strip()` meaningful whitespace or XML-unescape ordinary ampersands.
- Convert values using the selected tool's property schema: strings remain raw strings; integer/number/boolean/null use strict JSON conversion; arrays/objects use strict `json.loads`; string enums compare as raw strings and non-string enums as JSON. Invalid conversions remain diagnosable schema failures rather than being silently coerced. Unknown tools/parameters remain representable so the validator can report them.
- Detect duplicate parameter names instead of silently taking the first or last. Reserved closing markup inside a raw string is a limitation of the native format and must produce an explicit parse error in prompt-only mode.
- Emit `ToolCallStart` once the name is known. Build OpenAI argument deltas as valid compact JSON fragments: opening `{`, one encoded key/value fragment per completed parameter with commas, then `}`. It is correct to hold an incomplete parameter rather than emit a delta that cannot be aggregated safely. Empty arguments become `{}`.
- Generate stable unique call IDs and preserve parallel-call indices. Non-streaming parse and arbitrary one-character/random-chunk streaming aggregation must produce identical final messages.

Add `tests/test_qwen35_parser.py` and `tests/fixtures/qwen35_cases.json` covering all scalar types, nested arrays/objects, enums, Unicode, escaped JSON content, multiline/leading/trailing-whitespace strings, literal `&lt;`, empty arguments, parallel calls, reasoning/no-call responses, unknown tools, duplicate parameters, wrong typed values, every tag split point, and truncation at every state. Include recorded real model output after the Colab run without making tests depend on network/model availability.

### 6. Compile Qwen3.5 envelopes through the same grammar IR

Extend `GBNFCompiler` with a profile/protocol option defaulting to Qwen3. Reuse `_compile_node()`, all schema checks, the immutable IR, GBNF renderer, and PDA. Only the tool-call envelope and schema-to-XML parameter encoding differ.

For Qwen3.5:

- Emit the exact template newlines for `<tool_call>`, `<function=name>`, each `<parameter=key>`, and the corresponding closers. Preserve existing `auto`, `none`, `required`, named-tool, and parallel-call root semantics.
- Reuse ordered required/optional property compilation. String parameters use raw text, not JSON quotes; non-string scalars use their JSON spelling; object/array parameters embed the existing JSON rules.
- Add raw-string automaton rules that hold possible reserved-tag prefixes and reject actual `</parameter>`, `</function>`, or `</tool_call>` sequences inside a value. Record this conservative protocol limitation in grammar metadata/logs. Reject unrepresentable property names or enum literals with a schema JSON Pointer instead of generating ambiguous markup.
- Keep `compile_schema()` independent of tool protocol. Existing Qwen3 tool GBNF texts and SHA-256 hashes must remain byte-identical; add new Qwen3.5 fixtures rather than rewriting the existing 90-case corpus.
- Exercise new grammars through `PushdownAutomaton.from_compiled()` and trie/brute-force mask tests. `TriggeredMaskController` can continue using the same atomic `<tool_call>` trigger with a Qwen3.5 required grammar; no trigger redesign is needed.
- Audit the real slow Qwen3.5 `Qwen2Tokenizer` with `Vocabulary.from_qwen_tokenizer()` and its 248,320-entry padded model vocabulary. Discover token IDs; never hardcode them. This verifies vocabulary/mask compatibility without claiming that the 9B HF model itself has been run locally.

Add Qwen3.5 valid/invalid cases to the grammar/PDA differential suite and run them with llama.cpp's real validator. Include invalid raw newlines, missing tags, wrong parameter order, duplicate/unknown parameters, malformed nested JSON, reserved markup in strings, and all tool-choice modes.

### 7. Extend only the canonical notebook

Add Day 9 cells to `notebooks/01_toolsmith.ipynb` before the shared `cleanup` cell. Do not create another notebook or remove any serving/cache/parser/GBNF/sampler safeguards.

1. **`day09-notes`** — explain the three guarantees (protocol parse, schema validation, semantic correctness), why streaming cannot transparently repair, and why Qwen3.5 needs a different parser/grammar despite sharing `<tool_call>`.
2. **`day09-offline-replay`** — run Qwen3/Qwen3.5 non-streaming and arbitrary-chunk fixtures, structured-output validation, repair fakes, and PDA/mask differentials. Save sanitized failures and assert old Qwen3 hashes are unchanged.
3. **`day09-qwen3-api`** — exercise the real existing Qwen3-8B service through authenticated OpenAI SDK calls: health, plain text, all tool choices, streaming, tool-result history, `response_format`, structured streaming, and explicit errors. Run a prompt-only/grammar/grammar-plus-repair comparison on the small existing corpus. For a deterministic live repair check, wrap the real client so only its first completion returns a recorded invalid response and its retry delegates to the actual server; label this injected first failure honestly.
4. **`day09-qwen35-tokenizer`** — load the official slow tokenizer, snapshot rendered XML prompts/history, audit raw vocabulary bytes, and validate the new native grammars with the real validator before downloading the 9B weights.
5. **`day09-qwen35-runtime`** — use a separate cached llama.cpp build pinned to **b8233** (whose release notes add Qwen3.5 dense support), leaving the known b5401 Qwen3 runtime/cache intact. Resolve `unsloth/Qwen3.5-9B-GGUF` file metadata with `HfApi`, select exactly one Q4_K_M GGUF or fail if ambiguous, pin its repository revision and exact byte size, and reuse the existing download/atomic-copy/GGUF-magic checks. Do not guess a filename, silently change quantization, or install vLLM/nightly Transformers.
6. **`day09-qwen35-api`** — explicitly finish the HF sampler phase and stop/close the Qwen3 shim, SDK client, and llama-server before starting the second runtime; print this phase transition. Start the Qwen3.5 server and a new service with the Qwen3.5 profile/renderer, using a short bounded context for these non-thinking tests and documenting that this is not a long-context quality evaluation. Check health and run raw completion, native XML replay, all tool choices, parallel/nested calls, streaming SDK aggregation, tool-result follow-up, schema responses, injected-first-failure repair, and prompt-only thinking/parser behavior.
7. **`day09-results`** — write a compact per-model/per-mode table: attempts, parse rate, schema-valid rate, repair recovery, truncation, semantic tool selection, false-call rate, tokens/sec, and grammar compile time. Report semantic failures separately; do not weaken a structural assertion because a model chose the wrong tool.
8. Extend **`cleanup`** to close both SDK clients/shims and both llama.cpp clients/servers and to release the sampler model. Cleanup remains the only final cell and repeated cleanup must be safe.

Expected Drive evidence:

```text
MyDrive/toolsmith/records/day-09-offline.jsonl
MyDrive/toolsmith/records/day-09-qwen3-api.jsonl
MyDrive/toolsmith/records/day-09-qwen3-repair.jsonl
MyDrive/toolsmith/records/day-09-qwen35-vocabulary.jsonl
MyDrive/toolsmith/records/day-09-qwen35-api.jsonl
MyDrive/toolsmith/records/day-09-qwen35-repair.jsonl
MyDrive/toolsmith/records/day-09-results.jsonl
```

Update `tests/test_notebook_01.py` to execute the CPU-only Day 9 replay cells against temporary directories/fake clients, validate all new cell IDs and Python syntax, assert cleanup is last, and ensure no secret or stale error output is stored. Preserve the existing b5401 cache tests and add separate b8233 cache identity coverage rather than overwriting the old baseline.

## Verification

### Local gates after each checkpoint

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src tests
git diff --check
```

Also:

- Rebuild the pinned b5401 `test-gbnf-validator` in a temporary project-local directory and rerun the original 12 grammars/90 cases. Require every original Qwen3 grammar hash to match its pre-Day-9 baseline.
- Build the b8233 validator and run every original plus new Qwen3.5 grammar case. Remove temporary validation trees afterward.
- Verify that every Qwen3.5 valid prefix survives the PDA with one-byte and arbitrary chunk input, every complete invalid case is rejected, and optimized masks match the brute-force oracle at representative states.
- Run API tests through real FastAPI `TestClient` requests and official OpenAI SDK response/chunk validation, not only direct function calls.
- Check `git status --short --branch` before and after implementation. Remove only generated verification artifacts; preserve the current uncommitted work and do not commit/push without a separate explicit request.

### Required real Colab acceptance

Publish or copy the current working tree to `/content/bare-metal-agent`; cloning GitHub `main` is insufficient while this work remains uncommitted. On a fresh L4, run the entire canonical notebook top-to-bottom:

1. All earlier health, raw completion, parser, streaming, SDK, original GBNF, PDA, vocabulary, fixed/triggered sampler, cache-equivalence, invalid-EOS, seeded-sampling, truncation, and benchmark gates still pass. This closes the still-pending Days 3–8 live acceptance rather than assuming it.
2. Qwen3-8B structured responses parse and validate in non-streaming and streaming modes. Tool calls and tool-result history still work with unchanged Qwen3 prompt/grammar fixtures.
3. Injected invalid first attempts lead to exactly one corrected real generation, valid final output, two recorded attempts, and summed usage. Non-retryable errors and streaming failures invoke no retry.
4. Qwen3.5 grammar validator and PDA/mask suites pass before model loading. Real tokenizer raw-byte round trips succeed without replacement characters or hardcoded IDs.
5. The actual Qwen3.5 server reports healthy and its raw output is saved. Accepted non-streaming/streaming native XML calls produce identical aggregated OpenAI calls and pass schema/choice validation; a follow-up tool-result request renders and completes successfully.
6. Qwen3.5 constrained required/named/parallel calls, no-tool text, structured responses, repair, and a thinking prompt-only example are recorded. Semantic tool correctness is evaluated against the prompt but never conflated with parser success.
7. Every expected evidence file exists, has model/runtime revisions and grammar hashes, and contains no HF token, API key, or authorization header. Final cleanup releases both model runtimes and the HF sampler.

If live output exposes a parser mismatch, preserve the first failing cell, raw text/token bytes, rendered prompt, profile, model/server revision, grammar hash, parser state, and backend log tail. Add the minimal fixture, fix only that layer, rerun the entire local suite and relevant real validator, then restart the notebook from a fresh runtime. Do not fix model quirks by accepting unterminated tags, using unsafe Python evaluation, dropping invalid cases, or weakening repair/truncation assertions.

## Assumptions and boundaries

- The existing llama.cpp raw-completion backend remains the only FastAPI backend in Day 9. The custom HF sampler is evaluated separately; its service integration, vLLM/SGLang adapters, batching, and speculative decoding remain future work.
- Qwen3.5 support means text-only serving, official Jinja rendering, native XML parsing/streaming, GBNF/PDA/mask support, and a real Q4_K_M llama.cpp run. It does not claim vision/audio support, BF16 operation, full 128K/262K context quality, or parity with vLLM's `qwen3_coder` implementation.
- b8233 is the initial pinned Qwen3.5-capable checkpoint suggested by the project brief and confirmed release notes. If that exact build/GGUF combination fails, record the failure and choose a newer immutable llama.cpp tag only after rerunning both Qwen3 and Qwen3.5 validation; do not switch to an unpinned `master`.
- Fixed grammars still require thinking disabled. Thinking-plus-tools uses prompt/validation/repair in the service or the standalone atomic trigger sampler, not a fixed whole-response grammar.
- Non-streaming requests get at most one model-output repair. Streaming remains incremental and non-repairing. Tool handlers are never executed by this library milestone.
- The supported JSON Schema subset is not broadened in Day 9. Unsupported `pattern`, ranges, unions, `$ref`, and similar keywords still fail loudly; post-validation remains mandatory. No BFCL V4 or JSONSchemaBench score is claimed until the separate benchmark milestone runs those harnesses.
- The one-canonical-notebook rule, Drive caching, secret handling, safe process ownership, and final cleanup are preserved. Existing user changes and all uncommitted earlier milestones remain untouched except for the planned edits.

References:

- [Official Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B): default thinking, non-thinking configuration, serving recommendations, and benchmark reporting.
- [Official Qwen3.5 tokenizer/chat template](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/tokenizer_config.json): exact XML tool-call rendering and argument-mapping history format.
- [llama.cpp b8233 release](https://github.com/ggml-org/llama.cpp/releases/tag/b8233): Qwen3.5 dense/Gated DeltaNet support.
- [vLLM Qwen3 parser source](https://github.com/vllm-project/vllm/blob/main/vllm/parser/qwen3.py): protocol reference and known newline/streaming edge cases; our implementation remains independent and strict.
- [SGLang Qwen3 Coder detector](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/qwen3_coder_detector.py): schema-directed parameter conversion reference; unsafe/coercing fallback behavior is intentionally not copied.
- [OpenAI Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs): `json_schema` response-format shape and the distinction between structured responses and function calling.

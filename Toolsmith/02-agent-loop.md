# Days 3–4: non-streaming and streaming tool-call parsers

## Context

The current `Qwen3ToolCallParser` scans complete strings and handles basic thinking blocks and JSON-in-`<tool_call>` tags. It cannot consume chunks, and the current API rejects `stream=True`. The raw llama.cpp client only supports non-streaming requests.

The pinned llama.cpp `b5401` backend provides a useful reference: streaming `/completion` returns JSON SSE events, non-final events carry `stop: false`, and a final event carries `stop: true`, token timings, and `truncated`. Its streaming and non-streaming text should match for the same request. The current non-streaming client checks `stopped_limit` and `stop_type`, but must also recognize the pinned backend's `truncated: true` field.

The goal is one protocol implementation that can parse a whole response or arbitrary chunks without semantic differences, plus thin adapters for llama.cpp streaming input and OpenAI streaming output. This milestone is about parsing, not grammars, tool execution, repair, or the agent loop.

## Approach

Use a shared incremental state machine as the single source of truth:

```text
raw llama.cpp SSE bytes
  -> SSE/JSON decoder
  -> Qwen3/Hermes incremental parser
  -> reasoning/text/tool-call events
  -> non-streaming response assembler OR OpenAI chunk adapter
```

Refactor the existing non-streaming `Qwen3ToolCallParser.parse()` into a facade that feeds one string through the same incremental core used by streaming. This prevents the two modes from drifting.

Support the JSON-in-tags protocol shared by Qwen3 and Hermes-style models, with Qwen3-specific thinking behavior. Qwen3.5's XML-like parameter format remains a later adapter.

Keep synchronous iterators in this milestone. Async streaming, parallel tool execution, and other model formats are out of scope.

## Key changes

### Day 3, M1 — Freeze the protocol and build a fixture corpus

Define the accepted protocol before changing implementation:

- Plain text with no markers.
- `<think>...</think>` reasoning, including empty thinking blocks.
- Qwen3 output with reasoning before `</think>` but no opening tag.
- Zero, one, or multiple `<tool_call>{...}</tool_call>` blocks in sequence.
- Text before, between, and after calls.
- Tool envelopes with string `name` and object `arguments`; support nested arguments and arbitrary JSON whitespace.
- Explicit errors for malformed JSON, missing names/arguments, non-object envelopes/arguments, duplicate IDs, unexpected closing tags, nested opening tags, unclosed thinking/calls, and backend truncation.

Document normalization rules:

- Emit reasoning and content text without protocol tags.
- Trim only outer response text at final assembly, not every incoming chunk.
- Preserve raw argument text in streaming deltas so their concatenation is valid JSON. At final assembly, compare arguments semantically and optionally canonicalize them for non-streaming responses.
- Assign tool-call indices `0..N-1` in emission order and create each ID exactly once.
- Treat a partial marker at end-of-input according to state: an unterminated call/thinking block is an error; an incomplete literal marker prefix in ordinary text is emitted as text.

Expand `tests/fixtures/` with small sanitized Qwen3/Hermes examples, including recorded real outputs from the canonical notebook and hand-written adversarial cases. Keep large/raw model artifacts out of Git.

### Day 3, M2 — Implement the incremental protocol core

Add `src/toolsmith/streaming.py` with typed events:

- `ReasoningDelta(text)`
- `ContentDelta(text)`
- `ToolCallStart(index, id, name)`
- `ToolCallArgumentsDelta(index, text)`
- `ToolCallEnd(index)`
- `ParseCompleted(calls, finish_reason)`

Use a parser interface such as:

```python
parser.feed(text: str) -> Iterator[ParserEvent]
parser.finish() -> ParsedResponse
```

`feed()` may return buffered events; `finish()` validates end-of-input, flushes literal text, and returns the same semantic result as non-streaming parsing. Feeding after completion or after an error is invalid. Error behavior must be deterministic and carry the existing `ToolCallParseError.code` taxonomy.

Implement explicit states for initial/reasoning detection, reasoning, ordinary content, and a tool call. Marker detection must buffer the longest suffix that could be a prefix of `<think>`, `</think>`, `<tool_call>`, or `</tool_call>` so markers split across any chunks are recognized exactly once.

For Qwen3 reasoning modes:

- `expect_thinking=False`: treat all non-tool text as content.
- `expect_thinking=True`: support both explicit `<think>` and Qwen's omitted-opening-tag form.
- `expect_thinking=None` (auto): preserve compatibility with the existing non-streaming parser; buffer only the initial ambiguous prefix rather than delaying known ordinary content. The API should pass the request's actual `enable_thinking` value to avoid unnecessary ambiguity.

Do not locate closing tags with a regex or naive substring search inside the JSON payload. Add a small JSON lexical scanner tracking string state, escapes, and object/array depth so tag-like text inside JSON strings is not treated as protocol markup.

For incremental tool calls:

- Decode the outer JSON envelope incrementally; do not require the entire call before emitting events.
- Emit `ToolCallStart` once the function name is known, then raw argument deltas as the argument value arrives.
- Track nested objects/arrays, quoted strings, escaped quotes/backslashes, and Unicode without splitting decoded text.
- Buffer a call's early argument bytes if `arguments` appears before `name`; flush them after the name becomes available. This preserves JSON object-order independence without emitting a nameless call.
- Validate the complete envelope and arguments again at `ToolCallEnd`; streaming is not a substitute for final validation.

Refactor `src/toolsmith/parsing.py` so the existing dataclasses and public non-streaming API remain compatible, but delegate to this core and a shared response assembler.

### Day 3, M3 — Add raw llama.cpp streaming transport

Extend `src/toolsmith/backends/llama_cpp.py`:

- Factor generation-payload construction so `complete()` and `stream_complete()` cannot disagree on `prompt`, `n_predict`, temperature, grammar, seed, or stream settings.
- Add `stream_complete(...) -> Iterator[dict[str, Any]]` using `httpx.Client.stream` and a context manager that always closes the response.
- Add a small SSE decoder that handles CRLF/LF, blank event boundaries, comments, multi-line `data:` fields, optional `[DONE]`, and JSON object validation.
- Decode bytes with an incremental strict UTF-8 decoder so multi-byte characters split across network reads are neither corrupted nor silently replaced.
- Yield validated backend events containing string `content` and boolean `stop`. Preserve final `timings`, `tokens`, `truncated`, and `stop_type` for the consumer.
- Require a terminal event; a connection ending before `stop: true` is an explicit incomplete-response error, not a successful finish.
- Raise `CompletionTruncatedError` for `truncated: true`, `stopped_limit: true`, or `stop_type == "limit"`, carrying the final/accumulated response metadata.
- Apply the same `truncated` check to non-streaming `complete()` to cover the pinned backend behavior.

Closing the returned iterator early must close the backend response and release the server slot. Record enough final metadata for replay without requiring a GPU.

### Day 4, M4 — Add the OpenAI delta adapter

Add `src/toolsmith/deltas.py` to convert parser events into OpenAI `chat.completion.chunk` objects:

- Initial assistant role chunk with stable completion ID, timestamp, model, and choice index.
- `reasoning_content` deltas as an explicitly documented extension.
- Normal `content` deltas.
- First tool-call delta containing `index`, `id`, `type: "function"`, function `name`, and an initial argument string.
- Subsequent deltas containing the same `index` and raw `function.arguments` fragments; no new ID per fragment.
- One final chunk with an empty delta and `finish_reason: "tool_calls"` when calls exist, otherwise `"stop"`.
- A final usage chunk with empty `choices` only when `stream_options.include_usage` is requested, followed by `data: [DONE]`.

Provide an aggregator for tests that converts these chunks back into one assistant message and verifies text, reasoning, call IDs/indices, names, arguments, finish reason, and usage against the non-streaming response.

### Day 4, M5 — Enable streaming in the existing shim without changing its scope

Update `ToolsmithService` and the FastAPI route:

- Keep the existing non-streaming path intact.
- Route `stream=True` to `chat_stream()` and return `StreamingResponse` with `text/event-stream` rather than rejecting it.
- Reuse the same request validation, native renderer, tool schemas, parser core, recorder, and completion ID/metadata helpers.
- Continue supporting only the existing `tool_choice=auto` prerequisite scope; do not add grammar, named-tool choices, response formats, repair, or tool execution in this milestone.
- Support `stream_options.include_usage`; reject unknown stream options rather than silently ignoring them.
- Validate request errors before opening the backend stream, so these remain normal HTTP 400 responses.
- Once streaming has started, report backend/parser failures as an OpenAI-shaped SSE error event followed by `[DONE]`, with no false successful finish chunk. Preserve the original exception in server logs.
- Aggregate raw events only for the JSONL record; do not buffer the whole response merely to produce client deltas.
- Close upstream iterators in `finally` blocks on completion, parser failure, or client disconnect.

### Day 4, M6 — Extend only the canonical notebook

Append a `Days 3–4: streaming parser` section to `notebooks/01_toolsmith.ipynb`; do not create another notebook.

Reuse the existing setup and server to:

1. Record plain-text, single-call, parallel-call, and thinking/no-thinking outputs.
2. Replay those outputs with one-character chunks, token-like chunks, and randomized chunks, printing the event timeline.
3. Compare raw llama.cpp stream/non-stream text with the same seed and settings.
4. Consume our shim with `OpenAI(...).chat.completions.create(stream=True)` and aggregate text and tool arguments with the official client.
5. Save small sanitized fixtures and timing results to Drive for CPU replay.

Keep malformed/truncated cases in offline fixtures; do not rely on forcing nondeterministic model mistakes. Retain one shared cleanup cell.

## Verification

### Parser correctness and chunk invariance

For every valid fixture, compare the whole-string result with streaming under:

- One complete chunk.
- Every possible single split point for short fixtures.
- One-character chunks.
- Fixed chunk sizes covering several token-like lengths.
- Seeded randomized partitions for larger fixtures.
- Network-byte partitions cutting through multi-byte UTF-8 characters and every protocol marker.

Aggregate semantics must match: reasoning, content, call order, names, argument JSON, indices, and finish reason. Raw argument strings may differ in whitespace only if the final assembler canonicalizes them.

Adversarial cases must include:

- Empty and omitted-opening thinking blocks.
- Multiple calls with text between/after them.
- Escaped quotes/backslashes and literal `<tool_call>`/`</tool_call>` text inside JSON strings.
- Nested objects/arrays, Unicode, booleans/null/numbers, and `arguments` before `name` in the envelope.
- Split markers such as `<`, `tool`, `_call`, `>` and `</tool`, `_call>`.
- Malformed JSON, duplicate/unexpected tags, missing fields, non-object arguments, and every truncation point inside thinking/call regions.
- Duplicate ID-factory outputs, feed-after-finish, finish with no terminal backend event, and early iterator cancellation.

No truncated call may be returned as a valid empty or partial tool call.

### Transport and API tests

Use mocked HTTP streams and a real local subprocess SSE server to verify:

- Payload parity and stream flags.
- SSE comments, CRLF, split data lines, invalid JSON, non-object events, and missing terminal events.
- Strict UTF-8 decoding and exact stream/non-stream text aggregation.
- `truncated`, `stopped_limit`, and `stop_type` handling in both modes.
- Response/iterator closure on normal completion, errors, and early close.
- OpenAI SDK model validation for non-streaming responses and SDK consumption of streaming chunks.
- Correct role/content/reasoning/tool deltas, stable IDs, argument reconstruction, finish chunks, optional usage, and terminal `[DONE]`.
- Pre-stream HTTP errors versus mid-stream SSE errors.
- Client disconnect cleanup and JSONL records without credentials.

Run the entire suite with `.venv/bin/python -m pytest`; all existing tests must remain green. Validate notebook JSON/code cells, run `compileall`, check `git diff --check`, and inspect `git status` for generated artifacts.

### Real Colab acceptance

On Qwen3-8B, complete real raw and OpenAI-SDK streaming calls for text, one tool, and multiple tools. Confirm:

- Streamed text matches the corresponding non-streaming response under the same settings.
- Tool arguments reconstruct into the expected JSON values.
- Thinking and content are separated correctly.
- Truncation and disconnects do not leave a busy llama.cpp slot.
- Sanitized JSONL evidence is saved for offline replay.

## Assumptions

- Qwen3/Hermes JSON-in-tags is the Day 3–4 protocol; Qwen3.5's XML parameter format is a later adapter.
- The same incremental core backs both parsing modes; no independent regex-based streaming parser.
- Streaming output is synchronous. Async servers/consumers can wrap the same events later.
- Raw argument deltas preserve JSON text; semantic equality, not byte-for-byte whitespace equality, is required between streaming and canonical non-streaming arguments.
- API work is limited to exposing the parser through the existing `tool_choice=auto` shim. Grammar compilation, repair, tool execution, and agent behavior are not part of this milestone.
- Real Colab execution is a user-run checkpoint. CPU tests provide correctness evidence but not model-quality claims.
- Existing uncommitted work and the canonical-notebook structure will be preserved; no automatic commit, push, or destructive notebook synchronization.

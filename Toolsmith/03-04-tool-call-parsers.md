# Day 2: minimal tool-using agent loop

## Context

Day 2 should turn the Day 1 model service into an agent that can repeatedly:

1. Ask the model for an assistant response or structured tool calls.
2. Validate and execute those calls against registered Python functions.
3. Return one tool-result message per call to the model.
4. Stop with a final answer or an explicit, bounded failure.

The current repository contains only Evening 1 infrastructure: raw `/completion`, llama.cpp process management, seeded requests, and redacted JSONL recording. The Day 1 registry, native prompt renderer, tool-call parser, and OpenAI-compatible chat endpoint do not yet exist. Day 2 therefore needs an explicit integration milestone for the minimal Day 1 surfaces it consumes. Grammar compilation and the custom sampler can remain separate Day 1 work; the agent must not depend on them.

No separate Day 2 canvas was supplied. This plan uses the Day 1 canvas's stated scope: a minimal ReAct-style loop, tool registry, argument validation, error recovery, step limits, and reusable JSONL logs. It also includes the BFCL V4 multi-turn categories that Day 1 deferred.

## Approach

Build a synchronous, deterministic agent runtime around native OpenAI-style tool calling, not stop-word-based text ReAct. Qwen's official function-calling guidance cautions against stop-word ReAct templates for reasoning models, so `tool_calls` and `tool_call_id` provide the action/observation structure instead.

Separate the runtime into four contracts:

- **Chat backend:** turns messages and tool schemas into one validated `AssistantTurn`; production uses the Day 1 `/v1/chat/completions`, tests use scripted responses.
- **Tool registry:** owns names, schemas, handlers, JSON argument validation, and result serialization.
- **Agent state machine:** owns transcript transitions, termination, budgets, and recovery; it does not perform HTTP or know about llama.cpp.
- **Run recorder/replay:** emits structured run events through `JsonlRecorder` and turns recorded assistant turns into offline test fixtures.

Keep model calls sequential and execute tool calls in model-emitted order. Do not automatically re-execute a tool after an error. The model may deliberately request another call after receiving a structured error, which avoids hidden duplicate side effects.

## Key changes

### M0 — Establish the minimal Day 1 integration contract

Before implementing the agent, add only the Day 1 functionality required for a real agent round trip:

- `src/toolsmith/registry.py`
  - A `Tool` definition with name, description, JSON Schema, and Python handler.
  - A registry with duplicate/name/schema validation and deterministic OpenAI `tools` serialization.
  - Schema generation from typed Python functions or Pydantic models where practical; always allow an explicit schema.
- `src/toolsmith/prompting.py`
  - Render system/user/assistant/tool history with the model tokenizer's native `apply_chat_template(..., tools=..., add_generation_prompt=True)`.
  - Preserve assistant `tool_calls`, tool result `tool_call_id`, and supported reasoning fields. Add focused template fixtures rather than manually recreating Qwen's Jinja.
- `src/toolsmith/parsing.py`
  - Non-streaming Qwen3 `<tool_call>` parsing into OpenAI-shaped calls with stable IDs.
  - Return ordinary text, multiple calls, malformed JSON, missing closing tags, and truncation distinctly. Streaming parser work remains outside this milestone.
- `src/toolsmith/api.py`
  - A small non-streaming FastAPI `/v1/chat/completions` route that sends the rendered prompt through the existing `LlamaCppClient.complete()` and returns text or `tool_calls` with correct finish reasons and usage.
  - Support the `tool_choice=auto` path required by the agent; leave a clear contract for the remaining Day 1 choices and streaming rather than silently claiming them.
  - Return OpenAI-shaped errors and never execute tools inside the model endpoint.

Declare the new dependencies explicitly: JSON Schema validation, Pydantic, FastAPI/Uvicorn, the OpenAI client, and tokenizer/chat-template support in the appropriate runtime or Colab extras.

**Gate:** with real Qwen3 in Colab, a no-tool request returns text and a tool request returns a parsed call; a subsequent request containing the assistant call and its `role=tool` result returns a final answer. Save both raw JSONL exchanges as fixtures. Do not start M1 until this works.

### M1 — Define the agent-facing types and backend boundary

Add a separate track-level namespace, keeping `toolsmith` focused on model/tool plumbing:

- `src/bare_metal_agent/models.py`
  - `ToolCall`: ID, function name, and raw JSON argument string.
  - `AssistantTurn`: optional content, tool calls, finish reason, raw message data needed for history, and usage.
  - `AgentConfig`: maximum model steps, total tool calls, recoverable tool errors, transient model retries, request token/sampling settings, and optional result-size limit.
  - `AgentResult`: final content, stop reason, complete transcript, executed calls, errors, model steps, and aggregate usage.
  - Structured stop/error enums such as `final_answer`, `max_steps`, `max_tool_calls`, `tool_error_limit`, `model_error`, `invalid_model_response`, and `cancelled`.
- `src/bare_metal_agent/backend.py`
  - A small `ChatBackend` protocol accepting a copied message list, tool definitions, and generation settings.
  - An OpenAI SDK implementation targeting the Day 1 endpoint.
  - A scripted implementation for deterministic tests, with assertions on the messages/tools received at every round.

Validate backend responses at the boundary. Empty content with no calls, duplicate/missing call IDs, malformed function objects, and truncation must become explicit errors rather than a successful empty final answer.

### M2 — Implement safe tool validation and execution

Extend the shared registry with an execution path used by the agent:

1. Resolve the exact registered function name.
2. Decode the argument string as a JSON object.
3. Validate with `jsonschema`, including useful schema paths in errors.
4. Invoke the handler exactly once with keyword arguments.
5. Serialize a consistent success envelope, for example `{"ok": true, "result": ...}`.
6. Return one `role=tool` message with the matching `tool_call_id`, function name where supported, and JSON string content.

Failures produce structured envelopes such as:

```json
{"ok": false, "error": {"type": "invalid_arguments", "message": "...", "path": "city"}}
```

Use distinct types for `unknown_tool`, `invalid_json`, `invalid_arguments`, `tool_exception`, `non_serializable_result`, and `result_too_large`. Unknown or invalid calls must never invoke a handler. Handler exceptions become bounded error results without traceback exposure to the model; debugging details remain local and pass through configured redaction.

Execute parallel calls sequentially in emitted order during Day 2. This keeps side effects, call counts, logs, and tests deterministic. Concurrent execution is later work.

### M3 — Implement the bounded agent state machine

Create `src/bare_metal_agent/agent.py` with a synchronous `Agent.run(...)`:

1. Copy the caller's initial messages so runs cannot mutate caller-owned history.
2. Send the current transcript and serialized registry to the backend.
3. Append the exact assistant message required for valid tool history.
4. If there are no tool calls, return a final answer only after validating that the response is non-empty and not truncated.
5. Otherwise enforce call/error budgets, execute each call once, append all corresponding tool messages, and begin the next model step.
6. Return an `AgentResult` for normal budget terminations; reserve exceptions for programmer/configuration errors or unrecoverable backend failures.

Define budgets precisely:

- A **model step** is one backend completion request.
- `max_steps` includes the final completion request, so a one-tool run normally needs at least two steps.
- `max_tool_calls` counts requested calls before execution and prevents any calls beyond the remaining budget.
- Recoverable tool errors count only unknown tools, invalid arguments/JSON, and handler/result failures, not successful executions.
- Stop before an additional backend request if the next step would exceed the configured limit.

Include a repeated-call/loop diagnostic in run metadata, but do not deduplicate or cache tool execution automatically: identical calls can be legitimate, and replaying side effects is unsafe. Optional wall-clock cancellation may be checked between operations; do not claim it can interrupt a blocking Python handler.

### M4 — Add bounded recovery without hidden side effects

Implement recovery at two layers:

- **Model request failures:** retry only connection/timeout and explicitly retryable 429/5xx responses, with a small configured limit and injectable backoff/sleep for tests. Never retry authentication, invalid requests, or parser/truncation errors blindly.
- **Tool failures:** return the structured tool error to the model and allow another model step while budgets remain. Do not automatically rerun the failed handler. Stop with `tool_error_limit` if the model cannot recover.

Preserve call IDs and transcript ordering during recovery. A single assistant message containing several calls must receive exactly one result message for every call, including failed validation. If the run stops before executing requested calls, record those calls as skipped with a reason rather than fabricating results.

### M5 — Add run events, JSONL evidence, and replay fixtures

Reuse `toolsmith.recording.JsonlRecorder`; do not create another logging format.

Add `src/bare_metal_agent/replay.py` and an event sink that records:

- Run ID, configuration, model/metadata, and stop reason.
- Model-step request/response references, latency, usage, and retry events.
- Tool call ID/name, validation outcome, execution duration, result/error type, and skipped calls.
- Sanitized transcripts or fixture references suitable for later debugging.

Avoid making raw chain-of-thought a required output. Preserve model history fields needed by the native template, but expose concise structured action/result events to callers.

Implement replay by feeding recorded assistant turns through `ScriptedBackend` while using fake handlers. Replay tests must verify message ordering and outcomes without GPU access or real tool side effects. Keep large/raw outputs ignored by Git; commit only small, sanitized fixtures.

### M6 — Build the Colab Day 2 notebook and real scenarios

Add `notebooks/02_agent_loop.ipynb`, reusing the existing server/client lifecycle and Drive record locations. Use deterministic local tools such as arithmetic and a fixed fact/weather lookup; no `eval`, shell commands, arbitrary file access, or external network tools.

Exercise these real-model scenarios:

1. A question that needs no tool and finishes in one model step.
2. One tool followed by a final answer.
3. Two dependent rounds where the second model request consumes the first tool result.
4. Multiple/parallel calls with result IDs matched correctly.
5. Thinking disabled and enabled, recording accuracy/latency differences without requiring reasoning text in the UI.

Use scripted backends in the same notebook for guaranteed unknown-tool, invalid-argument, handler-exception, and limit-exhaustion demonstrations. Those failure cases should not depend on coaxing nondeterministic model behavior.

The notebook must cleanly stop the server and record actual Colab usage separately; GPU execution remains a user-run checkpoint.

### M7 — BFCL V4 multi-turn integration and evaluation

After local and real-model state-machine tests pass:

- Add a thin BFCL model-handler/configuration adapter for the Day 1 endpoint/agent runtime rather than modifying benchmark internals.
- Smoke-test selected IDs with the official generation and `--partial-eval` evaluation workflow.
- Run `multi_turn_base` first, then `multi_turn_miss_func`, `multi_turn_miss_param`, and `multi_turn_long_context` as budget permits. Clearly distinguish partial smoke results from full-category scores.
- Report task accuracy, average model steps, tool calls, validation failures, recovery success, stop reasons, token usage, and latency.
- Compare recovery enabled versus fail-fast on the same fixed subset. Do not present local scenario success as BFCL accuracy.

BFCL V4 memory, web-search, and other agentic categories remain outside Day 2 because they require additional backends/services and capabilities beyond this minimal loop.

## Verification

### CPU test suite

Add focused tests under `tests/agent/` plus integration-contract tests for the new Day 1 surfaces:

- Registry serialization, duplicate rejection, and schema validation paths.
- Native prompt fixtures for assistant calls and `role=tool` history.
- No-tool, one-tool, dependent multi-round, and multi-call transcripts.
- Caller message lists remain unchanged.
- Exact tool execution counts and model-step counts.
- Unknown tools, malformed JSON, invalid arguments, handler exceptions, serialization errors, and recovery after each.
- No handler invocation on validation failure and no automatic handler retry.
- Matching `tool_call_id` for every success/error result.
- Maximum steps/calls/errors stop before one extra request or execution.
- Empty, duplicate-ID, malformed, and truncated model responses.
- Transient model retry policy and immediate fatal-error behavior.
- JSONL redaction and offline replay equivalence.

Run the complete existing and new suite with `.venv/bin/python -m pytest`; all existing Evening 1 tests must remain green. Run syntax/notebook validation and `git diff --check`, then inspect `git status` for generated artifacts.

### Real end-to-end acceptance

On Colab with Qwen3-8B:

- Use the official OpenAI client against our Day 1 endpoint, not llama.cpp's built-in tool parser.
- Complete a real two-step tool run and verify the final answer incorporates the deterministic tool value.
- Inspect raw requests to confirm the second request contains the exact assistant tool call and matching tool result.
- Demonstrate bounded termination and preserve JSONL evidence.
- Run official BFCL evaluation commands for any reported BFCL score.

## Assumptions

- Day 2 scope is reconstructed from the supplied Day 1 canvas because no separate Day 2 canvas was provided.
- The minimal non-streaming Day 1 integration in M0 is a prerequisite; full Day 1 streaming, grammar, and custom-sampler milestones remain independent and are not silently marked complete.
- The main loop uses native structured tool calls, not textual stop-word ReAct, following Qwen's official guidance.
- Day 2 is synchronous and sequential. Async streaming, concurrent tool execution, human approval, persistent memory, and external web tools are later milestones.
- Tools in the core examples are local and deterministic. Side-effecting tools require explicit caller design and are never automatically retried.
- You will execute GPU notebooks in Colab and share failures; local verification uses real subprocess HTTP tests and scripted model turns but cannot prove Qwen accuracy.
- Existing uncommitted Evening 1 changes will be preserved. Implementation will not automatically commit, push, overwrite notebook edits, or discard user changes.

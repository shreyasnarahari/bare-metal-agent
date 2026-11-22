# Part 1 — Bare loop

**Track:** [Steadyhand, Day 2 of 10 (Nov 8, 2026)](https://agent-track-day-2-of-10-nov-8-2026-mlsg.pages.grok.bot/)
**Milestone:** M0, evening 1
**Package:** `steadyhand`, a sibling of `toolsmith` under `src/`
**Depends on:** the Day 1 Toolsmith endpoint already in this repo
**Produces:** one agent episode that can call mock tools through `ToolsmithService.chat` and append a JSONL journal

This file is part 1 of 7.

| Part | File | Milestone |
|---|---|---|
| 1 | `01-bare-loop.md` | M0 bare loop |
| 2 | `02-registry-and-validation.md` | M1 registry and validation |
| 3 | `03-controller-and-guards.md` | M2 controller and guards |
| 4 | `04-journal-and-replay.md` | M3 replay harness |
| 5 | `05-fault-injector.md` | M4 fault injector |
| 6 | `06-recovery-policies.md` | M5 recovery policies |
| 7 | `07-evaluation-and-writeup.md` | M6 evaluation and M7 write-up |

## Outcome

A small Python package runs one episode: it sends the conversation and tool schemas to the self-hosted model through the existing shim, executes the tool calls that come back, feeds results in, and stops on a final answer or a step limit. Every model turn and tool result is appended to JSONL.

The loop is a control loop around the model. Reliability added in later parts lives in this package, while prompt rendering, native tool-call parsing, grammars, and the OpenAI-compatible HTTP surface stay in `toolsmith`.

## What already exists

Use these as the model boundary. Do not reimplement them.

| Piece | Where | What the loop should call |
|---|---|---|
| Non-streaming chat | `src/toolsmith/api.py` `ToolsmithService.chat` | One model turn. Returns an OpenAI-shaped chat completion dict. |
| Request model | `ChatCompletionRequest` | `messages`, `tools`, `tool_choice="auto"`, `max_tokens`, `temperature`, `seed`, `chat_template_kwargs`. |
| Model profiles | `src/toolsmith/protocols.py` | `QWEN3_PROFILE` (`Qwen/Qwen3-8B`, protocol `qwen3`) and `QWEN35_PROFILE` (`Qwen/Qwen3.5-9B`, protocol `qwen35`). Pass the profile into the service. There is no model-name substring guessing. |
| Tool objects and OpenAI schemas | `src/toolsmith/registry.py` `Tool`, `ToolRegistry` | `Tool.from_function` / `registry.to_openai()`. Handlers are stored on `Tool.handler` and are never executed by the shim. |
| History rendering | `src/toolsmith/prompting.py` `normalize_history`, `ChatPromptRenderer` | The service already renders model-owned templates. Steadyhand only has to append OpenAI-shaped `assistant` and `tool` messages. |
| Parse and schema failures | `APIError` codes from `ToolsmithService.chat` | `tool_call_validation_failed`, `unknown_tool`, `invalid_tool_call_json`, `schema_validation_failed`, `tool_choice_violation`, `empty_model_response`, `completion_truncated`, plus parser failures wrapped as model-output errors. |
| One model-output repair | `src/toolsmith/repair.py` `RepairPolicy` | The service already retries once, non-streaming, for eligible model-output failures. That retry is inside a single `chat()` call. It is not a tool-recovery policy. |
| Redacted JSONL | `src/toolsmith/recording.py` `JsonlRecorder` | Append episode events with the same redaction rules. |
| Canonical JSON hashing | `src/toolsmith/evaluation/models.py` `canonical_hash` | Hash tool arguments the same way later parts hash repeated calls. |

Serving constraints that the loop must respect:

- Fixed tool grammars require `chat_template_kwargs.enable_thinking=false`. The default episode uses grammars and thinking off, so the action is a native tool call.
- Prompt-only thinking is the service flag `use_grammars=False`. Keep that as an explicit episode option for later ablations. It is a different configuration from the grammar path, and the two are not mixed inside one request.
- `response_format=json_schema` cannot be combined with a non-empty `tools` list. Episodes use `tools`. They do not also set `response_format`.
- `json_object` is unsupported. Leave `response_format` unset.
- Streaming never repairs after deltas have been emitted. Part 1 calls `chat()`, not `chat_stream()`.
- The notebook binds services to localhost. Colab secrets hold `HF_TOKEN` and `LOCAL_KEY`. The agent reads the base URL and API key from the environment when it talks to a running server, and it can also call `ToolsmithService.chat` in-process in tests.

## What this part adds

```text
src/steadyhand/__init__.py
src/steadyhand/types.py          # TurnKind, EpisodeStatus, ToolResult, EpisodeResult
src/steadyhand/store.py          # in-memory orders, customers, payments, tickets
src/steadyhand/tools.py          # the ten mock tools, registered on ToolRegistry
src/steadyhand/client.py         # ModelTurn adapter over ToolsmithService.chat
src/steadyhand/executor.py       # sequential handler execution with a timeout
src/steadyhand/loop.py           # the episode loop
src/steadyhand/journal.py        # thin episode log on JsonlRecorder
tests/test_steadyhand_loop.py
tests/test_steadyhand_store.py
```

`pyproject.toml` already discovers packages under `src/`. A `steadyhand` package there is part of the same distribution. Tests stay in `tests/test_*.py`.

Colab work lives in `notebooks/02_steadyhand.ipynb`. It shares llama.cpp and GGUF provisioning, and the Drive caches, with `notebooks/01_toolsmith.ipynb` through `toolsmith.provision`. (The cells were first written into the Toolsmith notebook, then moved to their own notebook after part 7.)

## Design

### Mock store

An in-memory store with copy-on-episode isolation (`store.clone()`), so tests and later fault schedules do not share mutations.

Entities: `Order`, `Customer`, `Payment`, `Ticket`, `Refund`. Seed a deterministic fixture, about a dozen rows, checked into `evaluation/data/steadyhand_store_v1.json`. Load it in `store.load_fixture()`.

Ten tools, matching the curriculum split of five reads, three non-idempotent writes, and two idempotent writes:

| Tool | Effect | Behavior in part 1 |
|---|---|---|
| `get_order` | read | Return the order or a structured not-found result. |
| `get_customer` | read | Return the customer or not-found. |
| `list_payments` | read | Payments for an order. |
| `get_ticket` | read | Ticket or not-found. |
| `search_orders` | read | Filter by `customer_id` and optional `status`. |
| `issue_refund` | non-idempotent write | Append a refund. A second call creates a second refund. |
| `create_ticket` | non-idempotent write | Allocate a new ticket id. |
| `add_order_note` | non-idempotent write | Append a note. Repeating it duplicates the note. |
| `set_ticket_status` | idempotent write | Setting the same status again returns the same ticket and does not append a second transition. |
| `cancel_order` | idempotent write | Cancelling an already-cancelled order returns the cancelled order and does not change anything else. |

Part 1 tools return plain dicts. Policy metadata (timeout, retry, verify probe, alternatives) is part 2. Part 1 passes a single executor timeout, default 5 seconds, around every handler.

Handlers receive the store explicitly (`handler(store, **arguments)`). They do not close over a global. The registry still stores the raw function on `Tool.handler`; the executor binds the store.

### Model turn adapter

`SteadyhandClient.complete(messages, tools, *, tool_choice="auto")` calls `ToolsmithService.chat` and classifies the response:

| `TurnKind` | When |
|---|---|
| `TEXT` | The assistant message has content and no tool calls. This ends the episode as `DONE`. |
| `TOOL_CALLS` | One or more tool calls. Arguments stay JSON strings, as the shim emits them. |
| `PARSE_ERROR` | `chat()` raises `APIError` after the shim's own repair attempt is exhausted, or the backend raises a connection/timeout/HTTP error. |

Carry `prompt_tokens`, `completion_tokens`, `reasoning_content` when the shim returns it, and the `APIError.code` for parse errors. A parse error in part 1 is appended to the journal and ends the episode as `ABORTED`. Part 2 turns the same failure into a typed observation and continues.

Map the shim's message shape without inventing a second tool protocol:

```text
assistant message:
  role, content, tool_calls[{id, type, function:{name, arguments}}], reasoning_content?
tool message:
  role=tool, tool_call_id, content=<JSON string>
```

`tool_call_id` comes from the shim. If a call has no id, synthesize `call_{step}_{index}` and use that same id on the tool message.

### Loop

```text
status = RUNNING
steps = 0
while status == RUNNING:
    if steps >= max_steps:
        status = BUDGET_EXHAUSTED
        break
    turn = client.complete(messages, registry.to_openai())
    journal.append(model_turn=turn)
    if turn.kind == TEXT:
        status = DONE
        break
    if turn.kind == PARSE_ERROR:
        status = ABORTED
        break
    messages.append(turn.assistant_message)
    for call in turn.tool_calls:
        result = executor.run(call, registry, store)
        messages.append(observation_message(call, result))
        journal.append(tool_result=result)
    steps += 1
```

`max_steps` default is 8. Token and wall-clock budgets, the repeated-call guard, and the clean give-up message are part 3. Part 1 only has the step ceiling, so a runaway mock cannot spin forever.

Tool observations in part 1 are a small JSON object, either `{"ok": true, "value": ...}` or `{"ok": false, "error_type": "timeout"|"handler_error"|"not_found", "tool": name, "message": ...}`. Part 2 replaces the failure shape with the typed channel. Keep the success shape stable.

Inside one episode, execute calls in list order. Concurrent reads inside a single assistant message are part 6. Writes in one assistant message run one after another so that episode's store stays single-threaded and replay stays obvious.

Across episodes, the live Colab path is parallel. `LlamaCppServer` already passes `--parallel` through to llama-server (`src/toolsmith/backends/llama_cpp_server.py`). The Day 10 notebook starts that server with 4096 tokens per slot (4 slots on a T4, 8 on an L4). One Steadyhand episode occupies one slot only while `chat()` is in flight. A sweep that runs episodes one at a time leaves the other slots idle. Part 7 sizes the slot count for a T4 or L4 and runs that many episodes at once. Part 1's types have to allow that without a later rewrite:

- Each episode owns its `store.clone()`, its message list, and its journal path. No module-level store and no single global journal file.
- `SteadyhandClient` holds a `ToolsmithService` and may be called from several threads, one episode per call. `LlamaCppClient.complete` only posts on a shared `httpx.Client`, which is safe for concurrent requests. `ToolsmithService._prepare` recompiles a grammar and is not locked today, so each worker gets its own `ToolsmithService` sharing one `LlamaCppClient`. Leave the service recorder unset on those workers. The episode journal is the record.
- `httpx` limits on that client must be at least the server's `parallel` (`max_connections` and `max_keepalive_connections`). The default pool is large enough for the slot counts in part 7. Set the limits explicitly anyway so a later change cannot silently queue inside the client while the GPU waits.
- llama.cpp splits `-c` across slots. Day 10's `-c 16384` with 4 slots on a T4 is about 4096 tokens per episode, not 16384. Part 7 picks `-c` and `--parallel` together so each slot can hold a Steadyhand episode. Part 1 only has to pass `max_tokens` and stop when the shim returns `completion_truncated`, which is `ABORTED`, not a successful answer.

### Journal

`EpisodeJournal` wraps `JsonlRecorder`. One file per episode: `results/steadyhand/<episode_id>.jsonl` locally (`results/` is gitignored). On Colab, append to `/content/steadyhand/<episode_id>.jsonl` (local SSD), then copy the finished file to `MyDrive/toolsmith/records/<gpu>/steadyhand/`. Drive FUSE latency stays off the GPU worker. Part 4 specifies the copy. One JSON object per line:

- `episode_started` with episode id, model profile name, seed, fixture hash, `max_steps`
- `model_turn` with kind, raw assistant message, usage, error code
- `tool_result` with tool name, arguments, ok, value or error
- `episode_finished` with status, steps, final content

Redaction stays inside `JsonlRecorder`. Do not log `LOCAL_KEY`, `HF_TOKEN`, or `Authorization`.

## Tasks

1. Add `src/steadyhand/` and export `run_episode` from the package.
2. Implement the fixture store and the ten handlers. Each handler is a typed function so `Tool.from_function` can build a schema. Use closed objects: the existing schema helper sets Pydantic `extra="forbid"`.
3. Implement `SteadyhandClient` against a `ToolsmithService` instance. Tests pass a fake service whose `chat()` returns canned completion dicts or raises `APIError`.
4. Implement the sequential executor with `concurrent.futures` timeout or a thread, so a handler that sleeps is reported as `timeout` and does not wedge the test run. A timed-out thread must not keep mutating a shared store; handlers used in the timeout test should be a stub, and the real store handlers must stay fast.
5. Implement `run_episode(client, registry, store, messages, *, max_steps, journal)`.
6. Unit-test the three endings: text answer, step budget, parse error. Unit-test a two-step episode where the canned model calls `get_order` and then answers from the tool message.
7. Add one notebook cell that builds the registry and points at the localhost server when `TOOLSMITH_BASE_URL` is set, otherwise skips with a recorded reason. The cell runs two canned-length episodes at once against the already-started server and records whether both completions overlapped in wall clock. It writes journals under `/content/steadyhand/` and copies them to Drive after each episode returns. It does not embed secrets and it does not start a second model.

## Interfaces

```python
class TurnKind(str, Enum):
    TEXT = "text"
    TOOL_CALLS = "tool_calls"
    PARSE_ERROR = "parse_error"

class EpisodeStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ABORTED = "aborted"
    # ESCALATED arrives in part 3

@dataclass(frozen=True)
class ModelTurn:
    kind: TurnKind
    assistant_message: dict[str, Any] | None
    tool_calls: tuple[ToolCall, ...]
    content: str | None
    reasoning_content: str | None
    prompt_tokens: int
    completion_tokens: int
    error_code: str | None

def run_episode(...) -> EpisodeResult: ...
```

`EpisodeResult` holds `status`, `messages`, `steps`, `final_content`, `prompt_tokens`, `completion_tokens`, and `journal_path`.

## Tests

- Fixture load is deterministic. `store.clone()` then `issue_refund` leaves the original store unchanged.
- `cancel_order` twice and `set_ticket_status` to the same value twice leave a single effect. `issue_refund` twice creates two refunds. These lock the effect classes before policy work starts.
- Fake client: text turn finishes `DONE` with zero tool executions.
- Fake client: one tool call then text. The second request's messages contain the tool result JSON.
- Fake client: `APIError(code="schema_validation_failed")` finishes `ABORTED` and the journal contains that code.
- `max_steps=1` with a tool call finishes `BUDGET_EXHAUSTED` after that step.
- A handler that blocks past the timeout returns `ok: false`, `error_type: timeout`.
- Journal lines are strict JSON, secrets replaced, and the file is append-only.

Run `pytest tests/test_steadyhand_loop.py tests/test_steadyhand_store.py` before moving on. The existing Toolsmith suite must stay green. This part does not change `toolsmith` behavior.

## Acceptance

- An episode against canned model output completes without a GPU.
- The live path, when the notebook has a localhost shim, uses `tool_choice="auto"`, native tool calls, and `enable_thinking=false` on the grammar path.
- Handlers run in Steadyhand, after the shim has returned. The FastAPI routes still do not call handlers.
- The journal is enough to reconstruct the message list by hand.

## Pitfalls

- Treating the shim's one-shot `RepairPolicy` as the agent's retry policy. Leave that policy at its current default inside the service. Tool retries start in part 6.
- Putting Thought/Action text in the prompt. The action is the native tool call the parsers in `streaming.py` and `qwen35.py` already understand. Qwen's function-calling guidance warns that reasoning models may emit a stopword inside thinking, which is why this loop does not use a text ReAct template.
- Appending `reasoning_content` back into history by hand. The chat template owns that. Steadyhand appends the assistant message the shim already normalized.
- Executing tools inside `create_app`. The README boundary stays: the shim validates and returns calls; Steadyhand executes them.

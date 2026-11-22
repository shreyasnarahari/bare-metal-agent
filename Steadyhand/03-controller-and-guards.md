# Part 3 — Controller and guards

**Track:** Steadyhand, Day 2 of 10
**Milestone:** M2, evening 4
**Depends on:** `02-registry-and-validation.md`
**Produces:** an explicit episode state machine, step/token/wall-clock budgets, a repeated-call guard, a no-tool path, and a final give-up message

## Outcome

The loop stops for a reason the journal can name. It stops when the model answers in text, when a budget is spent, when the same call repeats too often, or when the controller escalates. A task with a missing parameter can finish by asking the user, with no tool call, instead of inventing an id or an amount.

## What already exists

- `EpisodeStatus` from part 1: `RUNNING`, `DONE`, `BUDGET_EXHAUSTED`, `ABORTED`. This part adds `ESCALATED`.
- `max_steps` in `run_episode`.
- Usage tokens on `ModelTurn`, summed from the shim (`prompt_tokens`, `completion_tokens`).
- `canonical_hash` in `src/toolsmith/evaluation/models.py` for stable hashes of JSON values.
- `ToolCallValidator` already enforces `tool_choice` of `none`, `required`, `auto`, and a named function. The no-tool path is a loop decision about when to send `tool_choice="none"` or to accept a text answer under `auto`. It does not add a new shim feature.
- The shim rejects an empty completion. A whitespace-only constrained response is already a failure in Toolsmith. The controller treats that as `parse_error`, not as `DONE`.

## What this part adds

```text
src/steadyhand/controller.py    # Budgets, LoopGuard, transition table
src/steadyhand/messages.py      # give-up text, ask-user hint
tests/test_steadyhand_controller.py
```

`loop.py` delegates the `while` condition and the per-call repetition check to the controller. The executor and validator stay as they are.

## Design

### State machine

States: `running`, `done`, `budget_exhausted`, `escalated`, `aborted`.

Legal transitions:

| From | Event | To |
|---|---|---|
| `running` | text answer with a visible character | `done` |
| `running` | step, token, or wall-clock budget crossed before the next model call | `budget_exhausted` |
| `running` | repeated identical call reaches the limit | stay `running` for that turn's observation; if the model repeats again after the observation, `escalated` |
| `running` | validation and recovery both say the task cannot proceed (part 6 wires the real signal; the controller exposes `escalate(reason)` now) | `escalated` |
| `running` | backend failure that is not a model-output error (`LlamaCppConnectionError`, HTTP 5xx from the server process, cancellation) | `aborted` |
| `running` | user/task cancellation | `aborted` |

`done` requires the assistant content to contain a visible character. Reuse the same idea as Toolsmith's plain-text rule: a whitespace-only completion is not success.

Terminal states do not call the model again. The result's `final_content` is the model's text for `done`, and the controller's own message for `budget_exhausted` and `escalated`.

### Budgets

```python
@dataclass(frozen=True)
class Budgets:
    max_steps: int = 8
    max_prompt_tokens: int = 24_000
    max_completion_tokens: int = 8_000
    max_wall_clock_s: float | None = None
```

`max_wall_clock_s` defaults to unset. Step and token budgets are the limits that stay fair when several episodes share one GPU. llama-server decodes slots together, so one episode's wall time grows as `parallel` grows even though its token count does not. A fixed 120-second clock would mark healthy batched episodes `budget_exhausted`. Set a wall clock only for a single-slot debug run, and record that it was a single-slot limit.

Check budgets before each model call, using tokens already consumed and `time.monotonic()` from the moment `run_episode` begins on a worker. Queue time spent waiting for a free worker does not count. A call that would start after the budget is exhausted does not start. The in-flight call that crosses the token budget still finishes, then the next iteration stops. Wall clock includes tool time for that episode only. It does not include another episode's generation, and it is not a GPU-wide budget. Token budgets are episode sums, not a single request's `max_tokens`, and they are not shared across the worker pool. The per-request `max_tokens` on `ChatCompletionRequest` stays a generation cap (part 1 default can move to 1024 for episodes). Record both numbers in `episode_started`. The generation cap has to fit in the per-slot context from part 7. A slot of 8192 tokens cannot accept a 24000-token prompt. When the rendered prompt plus `max_tokens` exceeds the slot, abort that episode as `completion_truncated` and record the slot size in the error. Do not raise `max_wall_clock_s` to hide queueing, and do not set one to compensate for slot sharing. Start the clock only when the worker enters `run_episode`, and only when a wall clock is configured.

`budget_exhausted` message names which budget fired and the counts:

```text
Stopped: step budget exhausted (8/8 steps, 12031 prompt tokens, 840 completion tokens, 37.2s).
Incomplete work: <one line from the last user goal>.
```

### Loop guard

Hash `(tool_name, canonical_json(arguments))` with `canonical_hash`. Canonical JSON is `sort_keys=True`, no insignificant whitespace, `allow_nan=False`. Hash the decoded arguments, so `{"b": 1, "a": 2}` and `{"a": 2, "b": 1}` are the same call.

Count per episode. Default limit is 3 identical executions that were actually attempted (validation failures count too, otherwise a model can burn the budget on the same bad payload). On the third sighting, skip the handler and append:

```json
{
  "ok": false,
  "error_type": "repeated_call",
  "tool": "issue_refund",
  "retryable": false,
  "hint": "This exact call already failed. Choose a different tool, change the arguments, or ask the user."
}
```

If the next model turn emits that same hash again, transition to `escalated` with a message that lists the repeated call and the last error type. Near-duplicate detection (tiny numeric or string edits that are still the same intent) is recorded as a follow-on in this part: normalize numbers that are equal as decimals (`120` and `120.0`) and strip surrounding space on strings before hashing. Do not fuzzy-match free text in this part. Fuzzy matching belongs with the pitfall "tiny argument changes" only after exact canonical repeats are tested. Add one explicit normalization test, not a similarity threshold.

### No-tool path

Two situations:

1. **Direct answer.** `tool_choice="auto"` already allows a text turn. The controller accepts it as `done`. Add a system instruction, kept short, that the model should answer directly when the store is not needed and should ask for a missing parameter instead of inventing one. Put that instruction in the episode's initial system message in Steadyhand, not by editing `prompting.py`.
2. **Forced no-tool.** `SteadyhandClient` accepts `tool_choice="none"` for a terminal "ask the user" turn. Use it when the controller escalates or exhausts the budget and a final sentence must be produced without another tool call. If the model server is unavailable in a unit test, the controller's template message is the final content and no extra model call is required. The live notebook may request one `tool_choice="none"` paraphrase; tests assert the template path so they stay GPU-free.

The system instruction for the mock store says which identifiers the user must supply (`order_id`, `customer_id`, amounts) and that the agent must not invent them. Tasks in part 7 include a missing-information category that expects zero writes.

### Give-up message

`escalated` and `budget_exhausted` both end with a message that states what was tried: tool names, error types, and counts from the journal, not a hidden chain of thought. Build that summary from journal events so replay reproduces the same text.

```text
I couldn't complete this.
Tried: issue_refund (invalid_arguments x2), issue_refund (repeated_call x1).
Stopped because the same call was repeated after a correction hint.
```

## Tasks

1. Implement the transition function as a pure function `(state, event) -> state` and test every legal and illegal pair. Illegal pairs raise `ControllerError` rather than falling through.
2. Thread `Budgets` through `run_episode`. Replace the bare `max_steps` check.
3. Implement `LoopGuard` and call it after validation, before `executor.run`. A repeated call does not touch the store.
4. Add the system instruction and the give-up renderer.
5. Extend the journal with `guard_tripped` and `budget_checked` events so part 4 can replay the decision without re-deriving it from scratch. Still store enough inputs (hash, count, budget numbers) that a reader can check the decision.
6. Tests below. No live model.

## Tests

- Text with a visible character ends `done`. Whitespace-only content stays a parse error and consumes a step.
- Step budget 2: two tool turns then `budget_exhausted`, and `client.complete` was called twice.
- Token budget: canned usage that crosses `max_prompt_tokens` prevents the next call.
- Wall clock: a fake clock injected into the controller, not `sleep` in the suite. The default live budget leaves `max_wall_clock_s` unset so a 4-slot decode is not scored as a timeout.
- Three identical `issue_refund` calls: the third does not create a refund; the observation is `repeated_call`; a fourth identical call escalates.
- `{"amount_inr": 120}` and `{"amount_inr": 120.0}` share a hash after normalization.
- Missing-information script: user says "refund my last order" and the canned model asks which order. Status `done`, store unchanged, zero tool calls. A canned model that invents `order_id` is not corrected by the controller in this part; the evaluation in part 7 measures that failure. The controller's job here is to allow the ask-user ending.
- Give-up text for a scripted escalation contains the tool name and the error type, and contains no stack trace.
- Backend `APIError` with `error_type="server_error"` and code `invalid_backend_response` aborts. A `schema_validation_failed` error still follows the part 2 parse/validation observation path.

Run `pytest tests/test_steadyhand_controller.py` and the part 1 and 2 Steadyhand tests.

## Acceptance

- Every episode ends in `done`, `budget_exhausted`, `escalated`, or `aborted`, and the journal's last event matches that status.
- Repeated identical calls stop mutating the store at the configured limit.
- The final message for a non-done ending is reproducible from the journal.
- Asking the user is a successful `done` text turn, not an escalation.

## Pitfalls

- Escalating on the first schema error. Bench2Robust's base-model example hands off after one schema error. Escalation waits until the repeat limit is hit, and, once part 6 exists, until registered retries and alternatives are exhausted. Part 3's `escalate()` is the primitive those policies call.
- Counting only successful executions in the guard. A model can loop on the same invalid arguments and look busy.
- Using wall-clock `datetime.now()` inside semantic checks. The guard and the budgets may read a monotonic clock; the store's business clock stays the fixture clock from part 2.
- Hashing the raw argument string. Key order and `120` versus `120.0` would defeat the guard.

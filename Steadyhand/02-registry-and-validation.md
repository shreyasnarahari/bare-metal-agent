# Part 2 — Registry and validation

**Track:** Steadyhand, Day 2 of 10
**Milestone:** M1, evenings 2 to 3
**Depends on:** `01-bare-loop.md`
**Produces:** policy metadata on every tool, three validation layers, and a typed error observation the model can act on

## Outcome

Before a handler runs, the loop checks that the tool name exists, that the arguments match the JSON Schema, and that semantic rules the schema cannot express also hold. Every failure goes back to the model as a short structured observation: error type, offending fields, whether it is retryable, and a one-line hint. The model can repair the call on the next turn. Parse errors from part 1, which aborted the episode, now continue as the same kind of observation.

## What already exists

| Layer | Where | Keep it |
|---|---|---|
| Name, description, JSON Schema, handler | `src/toolsmith/registry.py` `Tool`, `ToolRegistry.function` | Steadyhand tools are still `Tool` objects. `to_openai()` stays the schema the model sees. |
| Schema generation from annotations | `Tool.from_function` via Pydantic `extra="forbid"` | Use it for the ten mock tools. |
| Name, tool-choice, strict JSON, Draft 2020-12 schema checks | `src/toolsmith/validation.py` `ToolCallValidator` | The shim already runs this before `chat()` returns. A `schema_validation_failed` or `unknown_tool` `APIError` means the shim rejected the model output. |
| Grammar subset | `src/toolsmith/grammar.py` | Closed objects, required-before-optional, scalars, scalar enums, homogeneous arrays, one schema or null. Unsupported keywords fail with a JSON Pointer. Do not put semantic rules into the schema as `pattern`, `minimum`, or `$ref` and expect the grammar to enforce them. |
| Model-output repair prompt | `src/toolsmith/repair.py` `build_repair_messages` | Still the shim's single regeneration of an invalid model message. The observation below is a tool result, a different channel. |

The shim's validator raises. It does not return a tool message. Steadyhand's typed observation is what the model sees on the next turn, after the shim has either returned calls or raised.

## What this part adds

```text
src/steadyhand/policy.py       # Effect, RetrySpec, ToolPolicy
src/steadyhand/registry.py     # SteadyhandRegistry wrapping ToolRegistry
src/steadyhand/validate.py     # semantic checks and the observation builder
src/steadyhand/errors.py       # ErrorObservation
tests/test_steadyhand_validate.py
tests/fixtures/steadyhand_model_turns.json   # recorded invalid and valid calls
```

## Design

### Policy metadata

`Tool` has no field for effect class or retry policy. Add a side table rather than changing the frozen `Tool` dataclass in a way that breaks existing Toolsmith tests.

```python
class Effect(str, Enum):
    READ = "read"
    IDEMPOTENT_WRITE = "idempotent_write"
    NON_IDEMPOTENT_WRITE = "non_idempotent_write"

@dataclass(frozen=True)
class RetrySpec:
    on: tuple[str, ...]          # error types, e.g. timeout, rate_limit, server_error
    max_attempts: int            # extra attempts after the first call
    backoff: str                 # "exp" in part 6; stored now, applied later

@dataclass(frozen=True)
class ToolPolicy:
    effect: Effect
    timeout_s: float
    idempotent: bool
    retry: RetrySpec | None
    verify: str | None           # read-only tool name, used in part 6
    alternatives: tuple[str, ...]
```

`SteadyhandRegistry.register(function, policy, semantic=None)` builds the `Tool` through `ToolRegistry.function` and stores `ToolPolicy` plus an optional semantic callable keyed by tool name. `schemas()` returns `registry.to_openai()`. `policy(name)` and `semantic(name)` are what the executor and validator read.

Declared policies for the ten tools:

| Tool | Effect | timeout_s | retry.on | verify | alternatives |
|---|---|---|---|---|---|
| `get_order`, `get_customer`, `list_payments`, `get_ticket`, `search_orders` | read | 5 | timeout, rate_limit, server_error, max 2 | none | none |
| `set_ticket_status`, `cancel_order` | idempotent write | 5 | timeout, rate_limit, server_error, max 2 | none | none |
| `issue_refund` | non-idempotent write | 5 | none in the automatic sense | `list_payments` | `refund_via_ledger` (registered in part 6; name reserved now) |
| `create_ticket` | non-idempotent write | 5 | none | `get_ticket` once a ticket id is known | none |
| `add_order_note` | non-idempotent write | 5 | none | `get_order` | none |

`idempotent` must match `effect`. A non-idempotent tool with `retry.max_attempts > 0` is a construction error. Part 6 is what consumes `retry`, `verify`, and `alternatives`. Part 2 only stores them and tests that the registry rejects inconsistent policies.

### Three validation layers

Run them in order on each call the client classified as `TOOL_CALLS`. The shim has usually already enforced layers 1 and 2 when `chat()` returns normally. Steadyhand still runs them so replay of a raw model message, and a future client that bypasses the shim's exception path, has one code path.

1. **Name.** `registry.get(name)`. Missing name becomes `unknown_tool`, `retryable: false`, hint `Use one of the registered tools.` Include the registered names only when there are few (the mock store has ten). Cap the hint so it stays one line.
2. **Schema.** `ToolCallValidator` on a one-call `ParsedResponse`, or `Draft202012Validator` on the tool's parameters directly. Map existing codes into the observation: `invalid_tool_call_json`, `schema_validation_failed`. `retryable: true`. Hint names the field and the problem, not the Python traceback.
3. **Semantic.** A callable `(store, arguments) -> list[FieldProblem]`. Schema-valid arguments can still be wrong. Ship these checks with the mock store:
   - `issue_refund.amount_inr` must be `> 0` and must not exceed the order's refundable total (order total minus refunds already recorded).
   - `issue_refund.order_id`, `add_order_note.order_id`, `list_payments.order_id`, `cancel_order.order_id` must exist.
   - `get_customer.customer_id` and `search_orders.customer_id` must exist.
   - `get_ticket.ticket_id` and `set_ticket_status.ticket_id` must exist.
   - `set_ticket_status.status` must be one of `open`, `pending`, `resolved`. Prefer a schema enum for this one so the grammar can enforce it too. Semantic checks are for rules that depend on store state.
   - Dates, where a tool grows one later, must be in the future relative to the episode clock injected into the store, not `datetime.now()` inside the check. The episode clock is fixed in the fixture so replay stays stable.

Semantic failures use `error_type: invalid_arguments`, `retryable: true`, and field details `{field, problem, got}`. A missing order is `not_found` only when the handler would be the thing that discovers it. If the semantic layer already knows the id is absent, fail in the semantic layer and do not call the handler.

Unknown-tool and schema failures are model-side. `not_found` from a read of a well-formed id that the scenario deliberately omits can be a successful tool result (`ok: true` is wrong; return `ok: false`, `error_type: not_found`, `retryable: false`) so the model can ask the user or pick another id. Distinguish "arguments that violate the contract" (`invalid_arguments`, retryable) from "the world does not contain that row" (`not_found`, not a prompt bug).

### Typed observation

This is the agent-computer interface. Keep it short and specific.

```json
{
  "ok": false,
  "error_type": "invalid_arguments",
  "tool": "issue_refund",
  "details": [{"field": "amount_inr", "problem": "must be > 0", "got": -120}],
  "retryable": true,
  "hint": "Fix the listed fields and call issue_refund again."
}
```

Success stays:

```json
{"ok": true, "value": {}}
```

`content` of the `tool` message is `json.dumps(observation, ensure_ascii=False, sort_keys=True)`. Sort keys so journals and replay hashes match.

`ErrorType` values introduced here: `invalid_arguments`, `unknown_tool`, `parse_error`, `not_found`, `timeout`, `handler_error`. Later parts add `repeated_call`, `rate_limit`, `server_error`, `permanent_unavailable`, `outcome_unknown`, `suspicious_result`.

### Parse errors continue

When `SteadyhandClient` returns `PARSE_ERROR`, the loop appends a user-visible tool-free observation as a `user` message, not a `tool` message, because there is no `tool_call_id`:

```json
{
  "ok": false,
  "error_type": "parse_error",
  "tool": null,
  "retryable": true,
  "hint": "Re-emit the call in the native tool format. Do not invent a tool result."
}
```

Bound the quoted bad output the same way `RepairPolicy.max_output_chars` does, and quote it as data. The shim may already have spent its one repair attempt before raising. The loop still gets one chance to show the model the typed hint, then part 3's budgets stop a parse-error loop. Count a parse-error turn against `max_steps`.

### Loop change

```text
for call in turn.tool_calls:
    verdict = validate(call, registry, store)
    if not verdict.ok:
        messages.append(tool_message(call, verdict.observation))
        journal.append(...)
        continue
    result = executor.run(call, registry, store)   # timeout from policy.timeout_s
    messages.append(tool_message(call, result.observation))
```

A validation failure does not call the handler and does not mutate the store.

Semantic checks close over that episode's store clone, not a process-global store. Two workers validating `issue_refund` at the same time must not see each other's refunds. The registry object itself is immutable after startup and may be shared across workers. `schemas()` is the cache key part 7 uses so a fixed tool list is compiled to GBNF once per process instead of once per model turn. Recompiling on every concurrent `chat()` call saturates Colab's few CPUs and delays feeding llama-server slots.

## Tasks

1. Add `ToolPolicy` and `SteadyhandRegistry` without changing `Toolsmith` public constructors. Existing `tests/test_registry.py` stays valid.
2. Attach policies and semantic checks to the ten tools from part 1.
3. Implement `validate_call()` returning either `ok` or an `ErrorObservation`.
4. Change `run_episode` so parse errors and validation failures append observations and continue, until `max_steps`.
5. Record a handful of raw model-shaped calls in `tests/fixtures/steadyhand_model_turns.json`: unknown tool, extra property, negative amount, unknown order id, valid refund. Build them by hand to match `ParsedToolCall` / OpenAI tool-call JSON. They are fixtures, not live model output.
6. Unit-test each error type from those fixtures. Assert the handler was not called when validation fails. Assert the observation JSON matches the shape above, including `retryable` and a non-empty `hint`.
7. Assert a stack trace never appears in `hint` or `details`. Handler exceptions become `handler_error` with a one-line message.

## Tests

- Policy construction rejects a non-idempotent tool with automatic retries, and rejects `verify` pointing at a missing tool or at a write.
- `issue_refund` with `amount_inr: -120` yields the observation in the design section and leaves refunds empty.
- `issue_refund` above the refundable total names `amount_inr` and includes the allowed maximum in `problem`.
- Unknown tool lists no handler invocation.
- Extra JSON property fails schema validation with a JSON Pointer in `details`, consistent with `ToolCallValidator` (`/unexpected`).
- Duplicate semantic problems are capped (`max_errors`, default 5) so a bad call cannot fill the context.
- Episode fixture: canned model emits a bad refund, then a corrected refund, then text. Final store has one refund. Journal contains both observations.
- `Tool.to_openai()` output for these tools still compiles through `GBNFCompiler` for `QWEN3_PROFILE` and `QWEN35_PROFILE`, or the schema is recorded as `SchemaNotSupportedError` with a pointer and the tool is rewritten until it compiles. The mock-tool schemas are in the supported subset on purpose.

Run `pytest tests/test_steadyhand_validate.py tests/test_registry.py tests/test_validation.py`.

## Acceptance

- Every failed call the model could repair produces a field-level observation.
- Semantic checks read the episode store and the fixture clock.
- The OpenAI schemas advertised to the model stay inside the grammar subset already supported by `grammar.py`.
- Inconsistent policy metadata fails at registration time.

## Pitfalls

- Encoding business rules as unsupported JSON Schema keywords. The compiler reports them and refuses the grammar. Post-validation in the shim would also not replace a semantic check that needs the store.
- Returning `{"error": "something went wrong"}` or a Python traceback. Small models need the field, the problem, and whether to retry.
- Treating `not_found` as `invalid_arguments`. One is a bad call. The other is a true fact about the store.
- Letting the semantic layer call writes. Checks are pure reads of the store snapshot.

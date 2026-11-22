# Part 5 — Fault injector

**Track:** Steadyhand, Day 2 of 10
**Milestone:** M4, evenings 6 to 7
**Depends on:** `04-journal-and-replay.md`
**Produces:** a seeded, pinned fault schedule so every recovery policy in part 6 sees the same failures

## Outcome

A wrapper between the executor and the tool handler injects failures from a fixed schedule. The schedule is a pure function of the seed, the tool name, and the call index. Two policy runs with the same seed and the same tasks get the same faults at the same calls. Replay re-executes the injector and gets the same outcomes.

## What already exists

- The executor from part 1, timeouts from `ToolPolicy.timeout_s` in part 2.
- Episode manifest with `fault_schedule_sha256` reserved in part 4.
- `canonical_hash` for the schedule payload.
- Toolsmith's failure taxonomy in the README separates truncation, native parse, schema, semantics, and backend failures. Those are model-output and transport classes. This part adds **tool-side** faults. Keep the two taxonomies in separate fields of the journal (`model_error` versus `injected_fault`). A semantic mistake by the model is not relabeled as an injected fault.

## What this part adds

```text
src/steadyhand/faults.py
tests/test_steadyhand_faults.py
```

The executor gains an optional `FaultInjector`. With no injector, behavior matches parts 1–4 and the golden replays still pass. Those golden manifests keep `fault_schedule_sha256: null`.

## Design

### Two axes, plus lost acknowledgment

Follow the ToolMaze axes the curriculum uses, and add UndoBench's lost-acknowledgment case.

| Mode | What the model sees | What the store does |
|---|---|---|
| `explicit_transient` | `ok: false`, `error_type` one of `timeout`, `rate_limit`, `server_error`, `retryable: true` | Handler does not run. Store unchanged. |
| `explicit_permanent` | `ok: false`, `error_type: permanent_unavailable`, `retryable: false` | Handler does not run. |
| `implicit_transient` | `ok: true` and a well-formed but wrong value | Handler does not run. The injector fabricates the value. |
| `implicit_permanent` | `ok: true` and a wrong value on every call to that tool | Handler does not run. |
| `lost_acknowledgment` | `ok: false`, `error_type: timeout`, `retryable` left to policy (the observation says timeout) | Handler **does** run and commit, then the caller is told it timed out. Journal records `committed: true` on the fault event. The model-facing observation does not say `committed`. |

`outcome_unknown` is the policy-facing name part 6 may use when it refuses to guess. The injector itself still emits `timeout` for the lost-ack case, because that is what a real client would see. The journal's `injected_fault.mode` is `lost_acknowledgment` so evaluation can score it. Tests may read the journal. The model message must not contain the mode name `lost_acknowledgment`, or the model is being handed the answer key.

### Schedule

```python
@dataclass(frozen=True)
class FaultSpec:
    tool: str
    call_index: int          # per-tool, 0-based, counts attempts that reach the injector
    mode: FaultMode
    error_type: str | None   # for explicit modes
    implicit_patch: dict | None

def build_schedule(seed: int, pins: Sequence[FaultSpec]) -> FaultSchedule:
    ...
```

Pins are explicit in the task file so a paper trail says "refund call 0 is lost-ack". The seed only breaks ties or fills an optional random overlay. For the evaluation suite, prefer fully pinned specs. A seeded overlay exists for exploratory runs and must be serialized into the manifest, not re-rolled on replay.

`call_index` counts calls that passed validation and the loop guard, including injected ones, inside that episode only. A validation failure does not consume a fault index. Otherwise a model that fixes its arguments would slide the fault onto a different logical call and ablations would stop matching. The injector keeps the counter on the episode, never on a process-global. Two workers running the same pinned task must each see `call_index` 0 on their first eligible call. The schedule builder is a pure function of `(seed, pins)` and may be called from many threads.

### Implicit patches

Fabricated values must still satisfy the tool's JSON Schema, or the test is about schema violation rather than over-trust. Examples pinned for the mock store:

- `get_order` returns the real order with `total_inr` replaced by `0` or a different status.
- `list_payments` omits the payment that a refund must not double-count, or reports an amount that does not sum to the order total.
- `search_orders` returns an empty list while the customer has orders.

The sanity checker that notices some of these (negative stock, totals that do not add up) is part 6. The injector only produces the lie. Implicit faults are labeled in the journal so metrics can score them. They are not labeled in the tool message.

### Where it sits

```text
validate -> loop guard -> injector -> handler
```

Timeout from `ToolPolicy` still wraps the handler for real hangs. An injected `timeout` returns immediately and does not sleep. Tests stay fast. A real-timeout test from part 1 remains, with the injector absent.

Lost-ack runs the handler inside the same timeout wrapper. If the real handler also times out, the journal says `committed: false` and the mode is recorded as `explicit_transient` with a note `handler_timeout`. Pinned lost-ack tests use handlers that return immediately, so commit succeeds and the observation is still a timeout.

### Journal and manifest

New event `injected_fault`:

```json
{
  "event": "injected_fault",
  "tool": "issue_refund",
  "call_index": 0,
  "mode": "lost_acknowledgment",
  "committed": true,
  "schedule_sha256": "..."
}
```

`fault_schedule_sha256` in the manifest hashes the ordered pins plus the seed. Replay rebuilds the schedule from the task's pins and seed, checks the hash, and re-executes. Recording the fault outcome without re-executing would hide a bug in the injector.

The model-facing observation for an explicit fault uses the part 2 shape. Hints for explicit faults are one line and do not mention that the fault was injected (`"The tool timed out. A retry may succeed."` versus `"This tool is unavailable."`).

### Interaction with the loop guard

An injected failure still counts toward the repeated-call hash, because the model sees the same arguments again if it retries blindly. Part 6's policy retry may bypass the model and retry inside the executor. Those policy retries must be marked `via: "policy"` and must not increment the model's repeated-call counter. They do increment the per-tool `call_index` only if the schedule says policy retries are new calls. **Decision:** policy retries are new injector calls and consume `call_index`, and they are tagged `via: "policy"` so the guard ignores them. Document this in the schedule header. A transient fault pinned at `call_index` 0 then a success on the policy retry means the pin is on index 0 only. If the task wants every attempt to fail, pin indexes 0, 1, and 2.

## Tasks

1. Implement `FaultMode`, `FaultSpec`, `FaultSchedule`, and `FaultInjector.apply(call, handler) -> InjectionResult`.
2. Thread the injector through the executor behind a default of `None`.
3. Add journal events and manifest hashing.
4. Pin fixtures for all five modes on `issue_refund` and one implicit lie on `get_order` and `list_payments`.
5. Extend replay tests: an episode recorded with a pin replays to the same store hash. For lost-ack, the store contains the refund and the model-facing observation is a timeout.
6. Assert the model-facing JSON does not contain `injected`, `lost_acknowledgment`, or `schedule`.

## Tests

- Same seed and pins produce the same `fault_schedule_sha256` on two processes. No `random` calls outside the schedule builder.
- `explicit_*` leaves the store unchanged (compare snapshot hash).
- `lost_acknowledgment` changes the store once. A second naive executor call without an idempotency key changes it twice. That second call is the baseline part 6 will reduce. This part only demonstrates it.
- Implicit `get_order` value validates against the tool schema and differs from the real order in the patched field.
- Validation failure does not advance `call_index`.
- Guard does not count `via: "policy"` attempts. A unit test calls the guard with that tag.
- Golden episode from part 4 still replays with the injector unset.
- Replay with a pin mismatches if the test alters the pin.

Run `pytest tests/test_steadyhand_faults.py tests/test_steadyhand_replay.py`.

## Acceptance

- Faults are pinned, seeded, and hashed in the manifest.
- The model sees either a typed explicit error or a schema-valid lie.
- The journal can tell committed lost-acks from real timeouts.
- CPU replay includes the injector.

## Pitfalls

- Sleeping to simulate timeout. The suite and the Colab bill both suffer, and replay gets slow.
- Telling the model the fault class. Recovery metrics then measure reading the answer key.
- Letting failed validation consume `call_index`. Policies stop being comparable across models that repair arguments at different rates.
- Scoring an injected implicit lie as a handler bug. The results table has a separate column for injected mode.

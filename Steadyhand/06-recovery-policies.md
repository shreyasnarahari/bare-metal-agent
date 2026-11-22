# Part 6 — Recovery policies

**Track:** Steadyhand, Day 2 of 10
**Milestone:** M5, evenings 7 to 9
**Depends on:** `05-fault-injector.md`
**Produces:** six swappable recovery strategies, idempotency for non-idempotent writes, and a small sanity checker for implicit lies

## Outcome

The same episode runner can execute any one of the strategies below. The strategy is a constructor argument, recorded in the manifest, and replayed. Policies differ only in what they do after a tool failure or a suspicious success. They share the controller, the journal, and the fault schedule.

## What already exists

- Typed observations and `ToolPolicy.retry`, `verify`, and `alternatives` from part 2. Those fields were stored and not acted on.
- Loop guard and `escalate(reason)` from part 3. Policy retries are tagged `via: "policy"` so they do not trip the guard (part 5).
- Injector modes from part 5, including lost acknowledgment, where the store commits and the observation is a timeout.
- Shim `RepairPolicy` in `src/toolsmith/repair.py`. That policy regenerates an invalid **model** message once inside `ToolsmithService.chat`. Leave it in place. Do not route tool failures through `build_repair_messages`.
- Effect classes on the ten tools, including which writes are safe to repeat.

## What this part adds

```text
src/steadyhand/recovery.py      # Strategy protocol and the six policies
src/steadyhand/idempotency.py   # key table on the store
src/steadyhand/sanity.py        # invariant checks for a few tools
src/steadyhand/tools_ledger.py  # refund_via_ledger alternative
tests/test_steadyhand_recovery.py
```

`run_episode(..., recovery=Strategy)` selects the policy. The manifest gains `recovery_policy` and `system_prompt_variant` (`plain` or `failure_aware`). The failure-aware prompt is a short system addendum used as an ablation in part 7. ToolMaze reported that such a prompt helped the models they measured. Steadyhand treats that as a factor to measure, not as a guaranteed gain.

## Design

### Strategies

Each strategy implements:

```python
class RecoveryStrategy(Protocol):
    name: str
    def on_observation(
        self, ctx: RecoveryContext, observation: dict
    ) -> RecoveryAction: ...
    def on_success(self, ctx: RecoveryContext, observation: dict) -> RecoveryAction: ...

class RecoveryAction(str, Enum):
    RETURN_TO_MODEL = "return_to_model"
    RETRY = "retry"
    VERIFY = "verify"
    SWITCH = "switch"
    ESCALATE = "escalate"
```

**(a) `raw_error`.** Baseline. Stringify the exception or the injector message into `{"ok": false, "error": "<text>"}` without `error_type`, `details`, or `hint`. Return to the model. No executor retry. This exists so later tables have a weak baseline. Keep it implemented and tested, and keep it out of the default.

**(b) `typed_hint`.** Default. Return the part 2 observation unchanged. No executor-level retry. The model decides whether to call again. This is the first serious policy and the one parts 2 and 3 already approximate.

**(c) `retry_safe`.** For tools whose effect is `read` or `idempotent_write`, if `observation.error_type` is in `policy.retry.on` and attempts remain, retry inside the executor with exponential backoff and jitter. Backoff sleeps with a fake clock in tests (`ctx.sleep`). Live runs cap a single wait at 2 seconds and the total policy wait at 10 seconds so a Colab sweep cannot stall on backoff. Non-idempotent writes are not retried by this strategy. They fall through to `RETURN_TO_MODEL` with the typed observation.

**(d) `idempotent_verify`.** For `non_idempotent_write`:

- Attach an idempotency key derived from `canonical_hash((episode_id, tool, canonical_args))`. The model does not supply the key and cannot forge a second key by rephrasing, because the hash is of the validated arguments.
- The store keeps `idempotency[key] -> receipt`. A repeat with the same key returns the stored receipt and does not apply the effect again.
- On `timeout` or any `retryable` transport error, call the `verify` tool if one is registered (`list_payments` for `issue_refund`, `get_order` for `add_order_note`, `get_ticket` for `create_ticket` when an id exists). If verify shows the effect is present, return that outcome to the model as `ok: true` and record `recovered_by: verify`. If verify shows it is absent, retry once with the **same** idempotency key. If verify itself fails, return `outcome_unknown` with `retryable: false` and let the controller escalate rather than retry blind.
- Lost-ack plus this policy must produce one refund, not two. That is the acceptance test for the whole part.

**(e) `switch`.** If the observation is `permanent_unavailable` and `policy.alternatives` is non-empty, call the next alternative once with mapped arguments. Ship `refund_via_ledger(order_id, amount_inr, reason)` as an idempotent ledger write that records one ledger entry keyed like (d). If the alternative fails, escalate. Do not walk an unbounded chain. One alternative, then stop.

**(f) `escalate`.** Composition of (b), (c), (d), and (e) in that order: typed observation if the model should repair arguments; policy retry for safe effects; verify-and-same-key retry for unsafe writes; one alternative on permanent failure; then `controller.escalate` with the give-up summary from part 3. Argument-validation errors (`invalid_arguments`, `unknown_tool`, `parse_error`) always return to the model. They are not retried at the policy layer and they are not escalated on the first occurrence.

`escalate` is the policy part 7 reports as the full loop. (a) through (e) are ablations.

### Sanity checker

`on_success` runs only for tools that opt in. Start with two:

- `list_payments`: the sum of payment amounts must equal `order.total_inr` within one paisa, unless the order is partially refunded, in which case the checker compares against total minus recorded refunds. A mismatch returns `suspicious_result`, `retryable: false`, hint `Payment list does not match the order total. Call get_order and list_payments again before refunding.` The checker does not invent a corrected total.
- `issue_refund` result: `amount_inr` on the receipt must be `> 0` and the store's refund sum must not exceed the order total. A violation is `suspicious_result`. Because (d) may have just written the row, a suspicious receipt escalates instead of retrying the write.

Implicit faults from part 5 are how these fire in tests. The checker is allowed to miss lies it has no invariant for. Journal `sanity_checked` with `passed: true|false`.

### Failure-aware prompt

A second system addendum, selected by `system_prompt_variant="failure_aware"`, tells the model to read `error_type` and `hint`, to change arguments when `retryable` is true and the error is `invalid_arguments`, and to stop repeating a `repeated_call`. It does not name the injector. Hash the addendum into the manifest. The plain variant is the empty addendum.

### Concurrency inside one episode

Independent **reads** in a single assistant message may run together, up to 4, via a thread pool. That pool is per episode and is capped at 4 so a sweep of many episodes does not spawn unbounded threads. Writes run one at a time, serialized per resource key (`order_id`, `ticket_id`, or `customer_id`). If any call in the message is a write, run the whole message sequentially so observation order matches call order. Parallel reads still append tool messages in the original call order, not completion order. Cancellation of sibling reads is a stretch goal.

The store is not shared across episodes. Inside one episode, take a lock around each handler so the parallel-read path is safe. Mock-store reads are microseconds. They are not what fills an L4. Filling the GPU is part 7: as many episodes in flight as llama-server has slots.

Policy backoff and verify run after a completion has returned, so they do not hold a GPU slot. They do hold a worker. Live waits stay capped (2 seconds per wait, 10 seconds total). Tests use `ctx.sleep` and do not sleep the wall clock. A worker blocked in backoff is a reason part 7 may set the worker count a little above the slot count, only when a smoke test shows slots going idle during verify. It is not a reason to start dozens of episodes.

## Tasks

1. Implement the strategy objects and a `apply_recovery` driver used by the executor.
2. Add the idempotency table to `Store.clone()` so it copies with the store.
3. Implement `refund_via_ledger` and register it as the alternative for `issue_refund`.
4. Implement the two sanity checks.
5. Add the failure-aware prompt variant as data, not as a change to `toolsmith.prompting`.
6. Tests for each strategy against pinned faults. Every test uses `ReplayClient` or a scripted model and the fake sleep clock.
7. Update the manifest and confirm part 4 goldens still replay with `recovery=typed_hint` and no injector. If the default policy name must appear in old manifests, version the journal and keep a reader that treats a missing `recovery_policy` as `typed_hint`.

## Tests

- (a) versus (b) on `invalid_arguments`: (b)'s tool message has `error_type`, `details`, and `hint`; (a)'s does not. Neither calls the handler.
- (c) on a read with `explicit_transient` at call index 0: handler/injector attempt 0 fails, attempt 1 runs the real handler, one model turn sees `ok: true`. Backoff uses the fake clock. A non-idempotent write under (c) is not automatically retried.
- (d) lost-ack on `issue_refund`: one refund row, one idempotency key, model eventually sees success or a single verified receipt. A second episode with the same arguments and a **different** episode id may create a second refund. The same episode id must not.
- (d) when verify cannot tell: observation is `outcome_unknown` and status becomes `escalated`. Zero extra refunds beyond the committed one.
- (e) `permanent_unavailable` on `issue_refund` calls `refund_via_ledger` once and does not call `issue_refund`'s handler.
- (f) first `invalid_arguments` returns to the model; scripted correction then succeeds. A permanent failure with a dead alternative escalates, and the final message names both tools.
- Sanity: implicit `list_payments` whose amounts do not add up yields `suspicious_result` and does not call `issue_refund` on that turn.
- Parallel: three reads in one assistant message return tool messages in call order. Two writes in one message run in order under the lock.
- Manifest `recovery_policy` mismatch fails replay.

Run the full `pytest tests/test_steadyhand_*.py` set.

## Acceptance

- Lost-ack under (d) causes one external effect.
- (a) through (f) are selectable and recorded.
- Non-idempotent writes are not blindly retried by (c).
- Escalation happens after the registered retry and alternative paths, and the final text says what was tried.
- Sanity failures are visible in the journal and are not counted as clean tool successes.

## Pitfalls

- Retrying `issue_refund` because the observation said `timeout`. Under lost-ack that is a double refund. Strategy (c) must refuse. Strategy (d) retries only with the same idempotency key after verify says the effect is absent.
- Letting the model pass its own idempotency key. It will change the key every turn and defeat the table.
- Unlimited alternative chains and unlimited policy retries. Caps are one alternative and `RetrySpec.max_attempts`.
- Treating the shim's model-output repair as strategy (b). Different layer, different budget. Both may happen in one episode: the shim repairs a bad grammar span, then Steadyhand shows a typed tool error. The journal should show both, in order.
- Parallel writes to one order. Serialize on the resource key.

# Part 2 — Capability manifests, deterministic policy, and approvals

**Track:** Bulkhead, Day 3 of 10
**Milestone:** M1, evenings 3–4
**Depends on:** `01-threat-model-and-capability-probe.md`
**Produces:** per-tool sandbox metadata, a pure ALLOW/DENY/ASK engine, monotonic privilege narrowing, and non-bypassable Steadyhand integration

## Outcome

Every schema-valid tool call is authorized by ordinary deterministic code before a handler, fault injector, sandbox process, or network connection runs. A task starts with an explicit maximum privilege set. Automatic transitions may only narrow that set; any requested new tool, domain, writable path, or higher numeric bound produces `ASK` with a stable rule id.

`DENY` gives the model a typed, non-retryable observation and lets it choose a different permitted action. `ASK` stops the episode in Steadyhand’s existing `escalated` state without calling the handler. A human-approved retry is a new linked claimed episode, not a mutation of an append-only finished journal.

## What already exists

- `src/steadyhand/policy.py` stores reliability metadata: `Effect`, timeout, retries, verification, and alternatives. It does not express privileges.
- `src/steadyhand/registry.py` deliberately keeps policy metadata outside the frozen Toolsmith `Tool` object. Extend this side-table pattern instead of changing OpenAI tool schemas.
- `src/steadyhand/loop.py` validates and guard-checks model calls before `_first_attempt`; parallel reads are planned and journaled in call order.
- `src/steadyhand/recovery.py` calls `executor.run` again for retry, verify, and switch. This is a required authorization path, not an exception.
- `src/steadyhand/errors.py` already provides bounded typed observations. Add security error types rather than returning tracebacks.
- `src/steadyhand/controller.py` and `src/steadyhand/messages.py` already provide `escalate(reason)` and deterministic final messages.
- `src/steadyhand/manifest.py` derives claimed episode ids from configuration hashes. Security policy must become part of that identity.
- The current ten store tools have no `exec` effect. The brief’s `effect="exec"` therefore requires an explicit new effect class, not an alias for a read or idempotent write.

## What this part adds

```text
src/bulkhead/manifest.py          # Sandbox, Egress, Limits, SecuritySpec
src/bulkhead/policy.py            # Decision, PolicyRequest, PolicyEngine
src/bulkhead/narrowing.py         # normalized subset/bounds relations
src/bulkhead/approvals.py         # ApprovalRequest, one-shot Grant
src/steadyhand/security.py        # adapter between Bulkhead and Steadyhand
src/steadyhand/tools_exec.py      # schemas/markers only until part 4
tests/test_bulkhead_manifest.py
tests/test_bulkhead_policy.py
tests/test_steadyhand_security.py
tests/fixtures/bulkhead_policies.json
```

Part 3 adds the full security journal schema. Part 2 nevertheless defines the decision record now so integration tests can assert stable fields from the start.

## Manifest design

Use immutable dataclasses with strict validation and canonical JSON serialization. A representative registration is:

```python
SecuritySpec(
    sandbox=Sandbox(
        fs_read=("$WORKSPACE", "/usr", "/lib", "/lib64", "/etc/ssl"),
        fs_write=("$WORKSPACE/out",),
        fs_deny=("$WORKSPACE/.git", "$WORKSPACE/.env"),
        net=Egress(allow=("pypi.internal.corp:443",)),
        limits=Limits(cpu_s=10, mem_mb=512, pids=64, disk_mb=200, out_kb=64),
        secrets=(),
    ),
    output_trust=Trust.UNTRUSTED,
    sinks=(Sink.FILESYSTEM, Sink.NETWORK),
)
```

The exact decorator syntax in the brief is illustrative; this repository’s real extension point is `SteadyhandRegistry.register(...)`. Add an optional `security=` argument and a separate security side table. Register exec tools with `ToolPolicy(effect=Effect.EXEC, idempotent=False, retry=None)` and require their `SecuritySpec` to contain a sandbox; the loop’s existing effect checks then keep exec sequential with writes and recovery cannot retry it automatically. Do not add server-only fields to the model’s JSON Schema.

Rules:

- `fs_write` must be contained in an allowed workspace; `fs_deny` wins over read and write allowances.
- Paths use explicit `$WORKSPACE` expansion, normalized separators, and no ambiguous `..`. Policy normalization is only a decision aid; the OS must enforce the final access.
- Egress entries are normalized `host:port` values, not arbitrary URL prefixes. Empty `allow` means no network, not unrestricted network.
- Limits are positive, finite, and bounded by global maxima. `out_kb` covers combined process output with per-stream subcaps.
- `secrets` must be empty for sandboxed tools. Reject secret names/values at registration instead of filtering them in later.
- `Effect.EXEC` is non-idempotent for recovery purposes: no automatic retry, it runs sequentially with writes, and it requires a sandbox spec. Existing `Effect` idempotence logic and error text must be updated accordingly.
- Existing store tools retain their current behavior when Bulkhead is disabled. In enabled strict mode, a missing security spec denies risky tools; do not silently treat missing metadata as full access.
- `SteadyhandRegistry.security(name)`, `security_spec_sha256()`, and a frozen registry make the metadata available to workers without mutation.

## Policy model

### Inputs and output

```text
PolicyRequest:
  episode_id, call_id, via (model|policy)
  tool, decoded arguments, arguments_sha256
  task policy and registry security spec
  current taint state (part 3)
  exact one-shot grants, if any
  required sandbox tier

Decision:
  verdict: ALLOW | DENY | ASK
  rule_id: stable dotted identifier
  reason_code: machine-stable, short
  normalized_request: canonical, redacted JSON
  requested_grant: present for ASK
```

The engine is pure: no model calls, clock reads, network, filesystem access, randomness, or environment inspection. The caller supplies normalized context. Invalid or contradictory policy configuration fails closed at load/registration time.

### Rule order

1. Reject malformed security metadata or an unavailable required tier.
2. Apply explicit task/tool `DENY`; denied subpaths and destinations always win.
3. Check tool membership, argument-specific rules, path sets, destination sets, and numeric bounds.
4. Apply taint downgrades from part 3: untrusted context changes an otherwise allowed egress or sensitive write to `ASK`; an explicit `DENY` remains `DENY`.
5. Apply task-rule `ASK` and monotonic widening checks.
6. Allow only an exact, unconsumed matching grant or an explicit narrow `ALLOW`.

Use conservative precedence `DENY > ASK > ALLOW`. Never collapse “unknown rule” into `ALLOW`.

### Narrowing relation

Implement and test a total `is_narrower(candidate, current)` relation for the supported rule types:

- Tool and domain sets: candidate is a subset; adding a domain or tool is widening.
- Read/write path sets: every candidate path is beneath an existing allowed path and no denied subpath is removed.
- Numeric argument bounds: lower bounds may rise and upper bounds may fall; crossing in the other direction is widening.
- Resource limits: CPU, memory, PID, disk, output, and time ceilings may only decrease automatically.
- Egress ports and byte limits: allowed authorities and ceilings may only shrink.
- Deny rules and taint requirements cannot be removed automatically.

A narrowing update returns a new immutable task policy and its hash. A widening update does not modify the current policy; it returns `ASK` plus the exact normalized grant needed. No fuzzy path or hostname matching.

### Path policy is not the filesystem boundary

Never authorize a host open with `realpath(...).startswith(workspace)` and then perform a separate open. That is a time-of-check/time-of-use race. In part 4, run file/code operations inside the sandbox. If a future in-process file tool is unavoidable, use a directory file descriptor and `openat2` with `RESOLVE_BENEATH` and appropriate symlink resolution flags, and test hardlink/symlink cases separately.

## Steadyhand integration

### One security path for every call

The order for a model-originated call is:

```text
validate_call -> repeated-call guard -> Bulkhead authorize
  -> DENY/ASK observation (no injector, no handler)
  -> ALLOW: fault injector around the permitted invocation
  -> legacy trusted in-process handler OR selected sandbox backend
  -> output gate (part 6) -> typed observation and audit
```

Recovery retry, verify, and switch must invoke the same adapter before `SequentialExecutor` invokes anything. Add a reusable authorization/execution context to `RecoveryContext`; do not rely on the model path having authorized a different tool or arguments. An alternative tool needs its own decision and audit record.

Security errors are not transient transport failures:

- Add `permission_denied`, `permission_required`, and `sandbox_violation` to `ErrorType`.
- `permission_denied`: `retryable=false`, handler not called, model may choose another action.
- `permission_required`: `retryable=false`, handler not called, episode escalates with the rule id and grant summary.
- `sandbox_violation`: `retryable=false`; recovery must not retry, verify, or switch around it. Return it to the model unless the controller’s security policy requires escalation.
- Add these to recovery’s non-retryable classes. In particular, `escalate` strategy must not turn a denial into an automatic alternative with equivalent privileges.

For parallel reads, make all initial authorization decisions in call order before submitting work. For any call with egress or a sensitive sink, use the sequential path so part 3’s taint update from an earlier result affects the next call. Journal decisions in call order regardless of completion order.

### Approval lifecycle

1. On `ASK`, journal the decision and approval request, append a `permission_required` tool observation so every call has a response, and call `controller.escalate`.
2. The original episode finishes `ESCALATED`; its journal and manifest remain immutable except for the existing atomic completion marker.
3. A decline leaves it escalated and creates no privileged execution.
4. An approval creates a one-shot grant bound to the original episode id, policy hash, tool, canonical arguments hash, normalized grant, expiry, and a unique approval id. It contains no bearer secret and cannot be reused for a different call.
5. The approved retry is a fresh claimed episode with `run_label="approved-<approval-id>"`, a parent episode reference, and the amended security policy hash in its manifest. It replays no privileged prefix from the denied episode and cannot reuse the grant twice.
6. Batch evaluation uses a declared deterministic approval mode (`deny_all` or task-scripted grants), never silent auto-approval.

This design preserves `run_claimed_episode`’s existing crash-resume contract and avoids an interactive human inside replay.

## Journal and manifest fields introduced now

Add a `policy_decision` event with common journal fields plus:

```text
step, tool_call_id, via, tool, arguments_sha256, verdict, rule_id,
reason_code, taint_before, policy_sha256, grant_id, normalized_request_sha256
```

Do not include raw secrets or unbounded arguments; `tool_result.handler_called` remains the execution evidence. The existing `tool_result` still records the model-facing result. `EpisodeManifest` gains optional security fields with defaults for old fixtures: `security_spec_sha256`, `task_policy_sha256`, `bulkhead_version`, and `parent_episode_id`. Behavior-changing security hashes belong to `IDENTITY_FIELDS` when Bulkhead is enabled. In disabled legacy mode, omit these keys from the serialized identity rather than hashing new null fields, so existing episode ids and golden manifests remain stable; old manifests stay readable through optional defaults.

## Tasks

1. Implement strict security manifests and registration-time consistency checks, including `Effect.EXEC` semantics.
2. Implement canonical policy requests, rule evaluation, stable ids, and the narrowing relation with no I/O.
3. Add the Steadyhand adapter and wire every model and recovery call through it. Preserve the fault injector’s existing call-index behavior for allowed calls only.
4. Add security observations and controller handling. Ensure denied/asked calls have `handler_called=false` and leave the store/workspace unchanged.
5. Implement one-shot grant validation and fresh linked approved episodes. No interactive prompt is needed in the unit suite.
6. Extend manifest construction/loading with backward-compatible optional fields and add `policy_decision` journaling.
7. Register `run_python` and `run_shell` schemas as disabled exec tools before part 4. Their handlers must raise if invoked directly. Verify their schemas compile through both existing Qwen grammar profiles.
8. Update affected docstrings in `executor.py`, `loop.py`, `recovery.py`, `registry.py`, and `manifest.py` to describe the security gate.

## Tests

- Empty egress means all network is denied; denied `.git`/`.env` subpaths win over a writable parent.
- Every set, path, numeric, resource, and tool narrowing case has positive and negative tests; all widening requests become `ASK`.
- Rule order tests cover `DENY` versus grant, `DENY` versus `ASK`, unknown rules, and malformed policy.
- A model retry, recovery retry, verify probe, and alternative each receive a fresh policy decision; none can execute after denial.
- Fault injection cannot manufacture an `ALLOW`, execute a denied call, or consume a fault index before authorization.
- `ASK` ends `ESCALATED`, gives a typed observation with a rule id, and leaves store and filesystem untouched.
- Approval is exact and single-use; changed arguments, domain, path, policy hash, or episode id invalidate it.
- Approved retry creates a new linked episode id; the original journal bytes do not change.
- Old golden journals and manifests replay with Bulkhead disabled; enabling or changing security policy changes identity and refuses incompatible resume.
- Parallel allowed reads retain deterministic decision/journal ordering; sensitive sink calls are never batched ahead of taint updates.

Run targeted tests, then `tests/test_steadyhand_loop.py`, `tests/test_steadyhand_recovery.py`, and `tests/test_steadyhand_replay.py`, followed by the full suite.

## Acceptance

- A unit matrix demonstrates that every widening request becomes `ASK` and every narrowing request remains no more privileged.
- No path from model call or recovery policy reaches a handler without a current `ALLOW`.
- Denials and approvals have typed observations, stable rule ids, and auditable manifest identity.
- No model-facing schema, Toolsmith route, or old golden episode changes behavior.
- No operating-system sandbox is claimed until part 4/5 acceptance.

## Pitfalls

- Authorizing only `_first_attempt` and forgetting `RecoveryContext.attempt` or suite code that invokes the executor directly.
- Letting the recovery `switch` strategy choose a different tool to evade a denial.
- Mutating the task policy before an `ASK` is approved, or accepting a broad “approve all future calls” grant.
- Using model-generated text as the approval decision or hiding the rule id in prose only.
- Treating lexical path normalization as protection against symlink/hardlink races.
- Reusing an episode id after changing security policy, or rewriting the denied episode after human approval.

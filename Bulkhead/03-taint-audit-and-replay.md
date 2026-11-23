# Part 3 — Taint state, security audit, and replay

**Track:** Bulkhead, Day 3 of 10
**Milestone:** M2, evening 5
**Depends on:** `02-policy-manifests-and-approvals.md`
**Produces:** coarse episode taint, deterministic taint-aware decisions, versioned security events, and replay that never accidentally repeats external effects

## Outcome

When a tool observation contains outside content, the episode becomes tainted. While tainted, an otherwise permitted egress or sensitive write is downgraded from `ALLOW` to `ASK`; an explicit `DENY` remains a denial. This is deliberately coarse information-flow control, not a claim to implement CaMeL-style per-value capabilities.

Every security decision and execution is reconstructable from the existing Steadyhand JSONL journal. Replay recomputes deterministic policy and taint transitions but, by default, consumes recorded results for process/network tools instead of launching them again. Existing in-memory Steadyhand episodes and golden journals remain valid when Bulkhead is disabled.

## What already exists

- `src/steadyhand/journal.py` has strict `EVENT_FIELDS`, contiguous sequence numbers, exact replay comparison, optional-field defaults, and volatile timing fields.
- `src/steadyhand/manifest.py` stores identity hashes and the full fault schedule needed to rebuild a run.
- `src/steadyhand/replay.py` serves recorded model turns and reexecutes validation, handlers, guards, and recovery. That is safe for the mock store but not for arbitrary shell, filesystem, or network effects.
- `src/steadyhand/loop.py` completes parallel reads in a thread pool but finishes and journals results in call order. Security state must preserve this determinism.
- `src/toolsmith/recording.py` provides redaction; `canonical_hash` provides stable argument and policy hashes.
- Part 2 defines `SecuritySpec.output_trust`, sink classes, `policy_decision`, security manifest hashes, and the shared authorization path.

## What this part adds

```text
src/bulkhead/taint.py             # Trust, Sink, EpisodeTaintState
src/bulkhead/audit.py             # typed security event payloads
src/bulkhead/replay.py            # RecordedExecutionBackend
src/steadyhand/security.py        # stateful per-episode adapter extension
tests/test_bulkhead_taint.py
tests/test_bulkhead_audit.py
tests/test_bulkhead_replay.py
tests/fixtures/bulkhead_episodes/
```

## Taint design

### Labels and state

Start with two labels:

- `TRUSTED`: deterministic internal results explicitly declared trusted by the registry.
- `UNTRUSTED`: web, email, customer files, subprocess stdout/stderr, network responses, and any tool result not explicitly classified otherwise.

`EpisodeTaintState` is per episode, never global or shared between workers. It holds the current label and an ordered set of bounded source descriptors (`tool_call_id`, tool, sequence number), not the outside content itself. State starts `TRUSTED`; a successful or failed observation containing outside content changes it to `UNTRUSTED`. There is no automatic “clean” operation in this track.

Default new external tools to `UNTRUSTED`. Existing internal store reads can be declared trusted only after reviewing their fields. Once outside content has been written into the store, the episode is already tainted; no per-field provenance is implied.

### Where taint applies

`SecuritySpec.sinks` distinguishes at least:

- `NETWORK`: any outbound request, including a nominally read-only fetch with attacker-influenced URL/query/body.
- `FILESYSTEM_SENSITIVE`: writes outside the disposable per-call workspace or to declared sensitive workspace paths.
- `EXTERNAL_EFFECT`: messages, payments, tickets, or other operations declared sensitive by task policy.
- `NONE`: no taint-sensitive sink.

A tainted episode changes `ALLOW` to `ASK` for these sinks only. It does not indiscriminately block every subsequent read or every internal store write. Explicit task rules determine which store effects are sensitive.

Network reads require special care: `GET` is not automatically safe because URL paths, query strings, headers, and DNS names can carry private data. Classify any tool capable of outbound communication as a `NETWORK` sink even if its reliability `Effect` is `READ`.

### Ordering within one assistant message

- Pure parallel reads may be authorized in call order using the same starting taint state only when none has egress or a sensitive sink. Their resulting taint transitions are applied in call order when results finish.
- If any call has egress or a sensitive sink, use sequential planning/execution. Apply each result’s taint transition before authorizing the next call.
- An untrusted read followed by a sensitive write in the same model turn must therefore produce `ASK` for the write; batching must not create a taint bypass.
- A denied or asked call does not create a new content-taint source. Its short deterministic error observation is trusted metadata, though the episode’s previous taint remains.

The policy engine receives `taint_before`; its pure narrowing rules from part 2 decide the outcome. It does not inspect raw observation text or ask a model whether content is malicious.

## Audit schema

Extend `src/steadyhand/journal.py` rather than maintaining a separate unauditable proxy log. Each event below uses the existing common fields (`event`, `episode_id`, `seq`, `recorded_at`) and strict known-field checking.

### `policy_decision`

Defined in part 2. Include step, call id, `via`, tool, arguments hash, verdict, rule id/reason, policy hash, taint before, optional grant id, and normalized request hash. Do not duplicate unbounded arguments or secrets.

### `taint_transition`

```text
step, tool_call_id, source_tool, before, after, source_seq
```

`source_seq` identifies the tool result carrying outside content. A repeated untrusted result need not add a redundant transition, but its source can be retained in a bounded source list outside the model observation.

### `approval_requested`

```text
step, tool_call_id, rule_id, request_sha256, normalized_grant,
parent_episode_id, approval_mode
```

The normalized grant contains privileges, not credentials. Human approval/decline is recorded in the new linked episode’s manifest; do not append a contradictory decision to the original finished journal.

### `sandbox_execution` (consumed from part 4 onward)

```text
step, tool_call_id, backend, tier, config_sha256, request_sha256,
started, exit_code, signal, timed_out, cleanup_status,
stdout_bytes, stderr_bytes, stdout_truncated, stderr_truncated,
violations, resource_usage, elapsed_s
```

`started` is a boolean, not a timestamp. Violations have stable kind/rule ids and bounded sanitized details. Resource usage includes CPU, peak memory, PID high-water mark, and cgroup OOM evidence when available.

### `egress_decision` (consumed from part 6 onward)

```text
step, tool_call_id, request_id, authority, decision, rule_id,
resolved_address_class, bytes_in, bytes_out, truncated, elapsed_s
```

Log normalized host and port, not URL query strings, request bodies, cookies, authorization headers, or credentials. Raw IP destinations are labeled explicitly and denied unless the policy explicitly allows them.

### Manifest and start record

Extend the manifest with canonical `task_policy`, `sandbox_config`, and required-tier information in addition to their hashes. The content is needed to rebuild decisions; the hash detects drift. Include probe semantic hash and helper/backend version where available.

- Behavior-changing policy/spec/config hashes are identity fields.
- Actual selected tier and backend build/probe hashes are checked on resume and reported with results. A sweep may not silently change tier midway.
- A deliberate tier change starts a new sweep/run label rather than resuming an incompatible one.
- `episode_started` receives an optional `bulkhead` summary with a default meaning disabled, preserving old journal readers.

Adding a new event type can retain the current journal version as the existing reader contract permits. Changing required fields on an existing event requires a journal version bump and migration fixtures; do not silently redefine an old event.

## Volatile versus deterministic fields

Policy verdicts, normalized request hashes, rule ids, taint transitions, sandbox configuration, allowlisted authority decisions, and sanitized model observations are deterministic and must compare exactly on replay.

Wall time, elapsed time, CPU usage, memory high-water marks, PID counts, and proxy byte counts may vary in a live reexecution. Mark only the genuinely variable fields volatile. Do not make an entire event volatile just to avoid a replay mismatch.

A replay mismatch must still name the sequence and field. Never “fix” replay by dropping security events or ignoring verdicts.

## Replay modes

### Default: safe deterministic replay

1. Load and match the complete Bulkhead manifest and task policy.
2. Replay model turns, validation, guard decisions, policy decisions, approvals, and taint exactly.
3. Reexecute existing in-memory mock-store handlers as Steadyhand does today when they are explicitly safe for replay.
4. For `Effect.EXEC`, direct filesystem, and network tools, use `RecordedExecutionBackend`. Match the reconstructed request hash and return the recorded typed result/audit outcome without spawning a process or opening a socket.
5. Reconstruct the same tool message and final store/workspace metadata. Verify all recorded events and refuse extra/missing events.

This replay proves that the control plane reproduces its decisions and messages. It does not claim to reprove kernel enforcement. The report and docstrings must say so.

### Explicit live sandbox replay

An opt-in integration mode may reexecute a recorded exec request inside the currently required tier. It must use a fresh fixture workspace, deny network unless the test explicitly enables it, and produce a separate comparison report; it never overwrites the original journal. A mismatch in variable resource usage is not a security failure, but a changed policy verdict or new escape is.

## Redaction and model observations

- Continue using `JsonlRecorder` redaction for known keys and configured secret values. Bulkhead adds no real secrets to sandbox environments, so tests can assert they are absent rather than merely redacted afterwards.
- Error details contain stable codes and bounded field names, not stack traces, host command lines, full paths outside the workspace, or credential material.
- Keep the existing `{"ok": true, "value": ...}` and typed error shapes. Taint metadata and audit events are control-plane records, not a second tool protocol.
- Part 6 defines exact untrusted content delimiters and sanitization. Delimiters help the model distinguish data but are never the enforcement mechanism.

## Tasks

1. Implement `Trust`, sink classes, and per-episode taint state with canonical source descriptors.
2. Extend part 2’s policy input with taint and implement `DENY > ASK > ALLOW` downgrade tests.
3. Change parallel eligibility and sequential processing so sensitive calls cannot outrun a preceding taint transition.
4. Add strict audit payload builders and journal reader support for all new events, including old-journal defaults.
5. Extend manifests with full canonical policy/config content, hashes, backend identity, and parent-approval linkage.
6. Implement `RecordedExecutionBackend` and route exec/filesystem/network replay through it. Keep current mock-store replay behavior intact.
7. Add volatile-field rules at field granularity and replay fixtures for denied, asked/approved, tainted, and clean episodes.
8. Update `replay.py`, `journal.py`, and `manifest.py` docstrings to distinguish control-plane replay from live security verification.

## Tests

- Untrusted web/file/subprocess output followed by sensitive egress or write produces `ASK`; the same call without taint follows its ordinary rule.
- Explicit `DENY` is never weakened to `ASK`; taint does not block an unrelated pure read.
- A same-turn untrusted-read/sensitive-write script cannot bypass through parallel planning.
- Two episodes sharing a registry do not share taint. Parallel episodes and parallel pure reads produce deterministic event order.
- Every new event rejects missing/unknown fields, sequence gaps, unbounded/non-JSON values, and invalid enum labels.
- Old golden journals replay byte-for-byte without new events. New golden fixtures replay without spawning subprocesses or connecting to test servers.
- Editing a verdict, rule id, taint label, request hash, or sanitized observation raises `ReplayMismatch` at the exact field.
- Resource timings may differ only in declared volatile fields; changing a security verdict is never ignored.
- Secrets and authorization values do not appear in new journals, manifests, approval grants, or proxy metadata.
- Approved retries retain the original journal bytes and reconstruct the parent/child linkage.

Run the new tests plus all Steadyhand replay, recovery, fault, and loop tests, then the full suite and `compileall`.

## Acceptance

- Taint is deterministic, episode-local, persistent, and effective within as well as across model turns.
- The journal explains every allow, deny, ask, taint transition, sandbox run, and egress decision without sensitive payloads.
- Replay cannot repeat shell, filesystem, or network side effects by default and clearly states what it does and does not verify.
- Existing golden episodes and Toolsmith behavior remain unchanged.
- No unmeasured security or accuracy claim is added to the README.

## Pitfalls

- Applying taint only after an entire parallel tool turn, allowing a same-turn exfiltration.
- Assuming an HTTP read has no exfiltration sink.
- Logging full URLs, command lines, raw environment, or untrusted unlimited output in the name of auditability.
- Rerunning a recorded shell command or network request during ordinary replay.
- Declaring every field volatile, or bumping/changing journal fields without keeping old fixtures readable.
- Presenting coarse episode taint as complete per-value information-flow control.

# Part 6 — Egress proxy and output gate

**Track:** Bulkhead, Day 3 of 10
**Milestone:** M5, evenings 10–11
**Depends on:** `05-resource-governance-and-tier1.md`
**Produces:** the only outbound network path, exact destination allowlisting and byte caps, sanitized untrusted observations, and network/output attacks

## Outcome

A tier-2 sandbox has no direct external network or DNS. Its only outbound route is a per-call channel to a trusted HTTP CONNECT proxy outside the sandbox. The proxy owns DNS and destination connections, enforces the canonical domain/port allowlist and byte ceilings, and emits correlated audit decisions.

All Bulkhead-managed output passes through a gate that bounds bytes, strips unsafe controls, marks truncation, redacts configured secret values as defense in depth, and labels outside content untrusted before Steadyhand constructs the next model message. Taint enforcement comes from parts 2–3, not from asking the model to respect a wrapper.

## What already exists

- T2 has an isolated network namespace with only loopback and no external route.
- T1 has no direct network by default because Landlock port rules and seccomp cannot provide domain isolation on their own.
- `Egress(allow=...)`, `Limits.out_kb`, `Sink.NETWORK`, taint-aware `ASK`, `egress_decision`, and `sandbox_execution` are defined.
- `ToolResult.observation()` and `content()` provide the single existing tool-message shape. Output handling extends `value`, not the OpenAI tool protocol.
- `JsonlRecorder` redaction and bounded `ErrorObservation` helpers are already used by Steadyhand.

## What this part adds

```text
src/bulkhead/egress.py            # authority parsing, policy binding, decision types
src/bulkhead/proxy.py             # outside CONNECT proxy and byte-capped tunnels
src/bulkhead/bridge.py            # namespace loopback <-> per-call Unix socket
src/bulkhead/output.py            # bounded decoding, sanitization, trusted rendering
src/steadyhand/tools_http.py      # optional trusted proxy-mediated request tool
tests/test_bulkhead_egress.py
tests/test_bulkhead_proxy.py
tests/test_bulkhead_output.py
tests/test_steadyhand_output_gate.py
tests/fixtures/bulkhead_network/
```

No hosted proxy or public endpoint is required. Tests use local HTTP/HTTPS fixtures and an outside local destination listener.

## Egress architecture

### T2 data path

```text
untrusted tool
  -> HTTP_PROXY/HTTPS_PROXY points at fixed loopback bridge
  -> bridge inside network namespace (no external route)
  -> per-call Unix socket, mounted from supervisor
  -> trusted outside CONNECT proxy, bound to this call's canonical policy
  -> DNS resolution and destination connection outside sandbox
```

The loopback bridge exists only so ordinary clients such as `curl` can speak HTTP proxy protocol. It has no host network namespace access and no ability to broaden policy. If the tool kills the bridge, the result is loss of network, not a bypass.

Use a unique Unix socket per execution with mode `0600` and a short abstract/path-safe request identifier. The supervisor binds that socket to the exact call/policy before starting the tool. Do not accept episode ids, policy hashes, or allowlists supplied by the untrusted client as authoritative metadata.

T1 arbitrary exec keeps network disabled. A separately registered `http_request` tool may run as a trusted outside proxy-mediated adapter with its own schema, authorization, taint, and output gate; it must not become a generic host-execution bypass. Do not claim T1 `curl` parity unless a future backend proves it.

### CONNECT policy

- Support only explicit HTTP CONNECT to normalized `host:port`; no arbitrary transparent TCP, UDP, raw sockets, or DNS service inside the sandbox.
- Parse authority strictly: reject userinfo, empty hosts, ambiguous/missing ports, multiple `@`, malformed IPv6, embedded controls, overlong headers, and noncanonical numeric-IP forms.
- Normalize DNS names with lowercasing, IDNA where needed, and trailing-dot handling. Match exact allowed entries by default. Wildcards, if ever added, need separate semantics/tests and are not required here.
- Deny raw IP literals unless that exact IP/port is explicitly allowlisted. Do not treat a hostname allowlist as permission to connect to arbitrary resolved/private addresses.
- Resolve DNS outside the sandbox. Check the resolved address against the destination policy and pin the actual outbound connection to the validated resolution so a later DNS change cannot redirect it.
- Decide whether a task permits loopback, private, link-local, multicast, and metadata address ranges. Default-deny these ranges even for allowed names unless the test/task explicitly needs a local fixture.
- Cap request/header bytes, total bytes out, total bytes in, connection count, idle time, and total tunnel time. Count all tunneled bytes, not just initial headers.
- Do not inject real credentials in the core implementation. Sandbox `secrets` remains empty and tests assert authorization/cookie variables are absent.

For HTTPS, CONNECT carries encrypted bytes; the proxy does not inspect HTTP paths or redirects inside TLS. A redirect to another authority requires a new CONNECT, which is independently denied/allowed. Test this explicitly. A same-authority redirect remains within that authority’s policy and byte ceilings.

DNS tunneling fails because the tool has no external DNS/UDP route and the outside resolver handles only an allowed hostname needed for CONNECT. Arbitrary DNS names and raw-IP connections never reach a resolver/destination.

## Egress audit and observations

The proxy returns bounded structured outcome metadata to the supervisor/adapter, which writes `egress_decision` in episode order. Include allowed/denied authority, stable rule id, request id, address class, byte counts, truncation, and sanitized failure code.

- Correlate proxy and sandbox events with `request_id`; never trust the tool to supply the correlation id.
- Do not log full URLs, query strings, request bodies, headers, cookies, credentials, or response payloads in `egress_decision`.
- Proxy denials become `sandbox_violation` with `retryable=false` and a short allowed-destination hint. The hint must not expose secret policy paths or internal tokens.
- Infrastructure failure of the supervisor/proxy is distinct from a policy denial. It is not counted as a benchmark attack success or as an ordinary model no-call.
- Recovery cannot retry a denial to the same host or switch to an equivalent unapproved network tool. Every new request is authorized independently.

## Output gate

### Capture and decode

- Enforce raw-byte caps while reading, before allocating a giant Python string. Keep per-stream and combined counters and bounded prefixes.
- Decode UTF-8 with a documented replacement policy for invalid/truncated sequences. Never split a valid prefix into an unbounded decoding error.
- Strip NUL, C0/C1 controls, ANSI CSI/OSC escape sequences, terminal title/hyperlink sequences, and bidirectional override controls. Preserve ordinary newline/tab where useful.
- Collapse neither semantic content nor whitespace aggressively; the goal is safe bounded data, not changing program output unnecessarily.
- Record original byte counts, retained byte counts, invalid UTF-8, control stripping, and truncation flags in bounded audit metadata.

### Model-facing result

Retain Steadyhand’s stable outer observation:

```json
{
  "ok": true,
  "value": {
    "exit_code": 0,
    "stdout": {
      "trust": "untrusted",
      "content": "[BEGIN UNTRUSTED OUTPUT <derived-id>]\n...\n[END UNTRUSTED OUTPUT <derived-id>]",
      "truncated": false
    },
    "stderr": {"trust": "untrusted", "content": "", "truncated": false}
  }
}
```

Derive the delimiter id from the request hash so replay is deterministic. Escape/neutralize any line in the data that could forge the exact closing delimiter. Delimiters and `trust` labels are explanatory metadata only; the policy state is already updated by the adapter.

For non-exec structured outside content, bound serialized JSON recursively or return a valid bounded truncation object. Never cut arbitrary JSON text into invalid JSON. Existing un-Bulkheaded mock-store observations remain unchanged for compatibility; all Bulkhead-managed external tools have declared output trust and caps.

A sandbox/seccomp/Landlock/proxy violation uses a short typed error and bounded rule metadata. Do not echo secret file content, environment values, full unauthorized payload, or a traceback in the error.

### Taint transition

The adapter applies output trust and updates episode taint when it finishes the result in call order. It then writes the exact sanitized content to `tool_result`. The next sensitive call sees the updated state, including within a sequential multi-call model turn.

Redaction is a final defense, not proof that reading secrets is safe. If a test canary secret appears in raw process output, the filesystem/policy test has already failed even if redaction hides it from the model.

## Network and output attacks

Add local fixtures for:

- Allowed HTTP and HTTPS CONNECT success, with certificate handling explicit in the fixture.
- Disallowed domain, allowed domain on disallowed port, raw IPv4/IPv6, decimal/hex IP variants, and userinfo/authority confusion.
- DNS rebinding/change between resolution attempts and connection to a disallowed resolved address class.
- Direct external socket, UDP/DNS exfiltration, abstract Unix socket, and access to unrelated host Unix sockets.
- HTTP redirect from allowed to disallowed authority and a redirect on the same authority.
- Header/body/tunnel byte overflow, too many connections, idle/slowloris behavior, and abrupt disconnect.
- Attempted proxy authority/policy override by the tool and reuse of another call’s socket.
- Stdout/stderr floods, invalid UTF-8, terminal escapes, forged delimiter lines, NUL/bidi controls, and oversized structured output.
- Canary secret output: assert both the access failure and that model/journal output does not expose the canary.

All destinations are local controlled fixtures. No test targets a public or internal customer service.

## Tasks

1. Implement strict authority normalization and pure egress decisions with table-driven adversarial tests.
2. Implement the outside per-call CONNECT proxy, validated/pinned DNS, address-class policy, connection/time/byte ceilings, and structured metadata.
3. Implement the T2 loopback/Unix bridge and fixed proxy environment without opening any direct external route.
4. Keep T1 exec network disabled; optionally add the separately authorized `http_request` adapter only after its own tests pass.
5. Implement output capture sanitization, deterministic delimiters, structured truncation, and typed violation rendering.
6. Wire proxy and output events into the existing call-order journal/taint path and recorded replay.
7. Add all network/output fixtures, retain failures, and fix the implementation rather than relaxing the expected verdict.
8. Update execution/recovery/report docstrings to distinguish policy denials, tool exits, resource limits, and proxy infrastructure failures.

## Tests

- Unit-test the proxy state machine with fake sockets/timeouts, then run real loopback integration tests under each supported profile.
- Confirm a packet/socket fixture cannot reach an outside listener directly even when the proxy is running.
- Verify redirect-to-disallowed and raw-IP/DNS attempts fail and have correlated `egress_decision` records.
- Byte caps include both directions and stop further transfer without unbounded buffering.
- Output tests assert exact sanitized model content and replay hashes, not merely that a function returns.
- Tainted network output downgrades the next sensitive call to `ASK`; a proxy denial does not clear taint or authorize an alternative.
- Secret canaries, authorization headers, cookies, query strings, and notebook secrets do not appear in model observations or logs.
- Old golden episodes remain unchanged; new recorded network episodes replay with no proxy, socket, DNS, or process activity.

Run the new tests plus all earlier Bulkhead/Steadyhand security tests and the full repository suite. Real T2 network acceptance remains pending where namespace integration is unavailable.

## Acceptance

- An allowed local host works through the proxy; disallowed host, raw IP, DNS tunnel, redirect second hop, and direct socket all fail with audit evidence.
- There is no direct outbound path from the sandbox and no credential enters it.
- Output is bounded and sanitized before reaching the model or journal, with deterministic untrusted labels and truncation evidence.
- Taint-sensitive writes/egress after outside content require approval under the configured policy.
- Replay produces the same decisions/messages without network activity.
- No public network service, hosted sandbox, or new unverified dependency is required.

## Pitfalls

- Giving the sandbox ordinary network access and relying only on proxy environment variables; tools can ignore them.
- Treating domain allowlisting as string suffix matching or failing to validate the resolved address.
- Claiming to inspect encrypted HTTPS redirects without TLS interception.
- Allowing T1 direct networking because only destination ports are restricted.
- Logging sensitive URLs/headers or accepting policy identity from the untrusted client.
- Decoding/stripping only after buffering unlimited output, or returning truncated invalid JSON.
- Trusting output delimiters as the security boundary instead of deterministic taint and policy.

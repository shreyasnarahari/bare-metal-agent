# Part 1 — Threat model and capability probe

**Track:** [Bulkhead, Day 3 of 10 (Nov 9, 2026)](https://agent-track-day-3-of-10-nov-9-2026-kaw2.pages.grok.bot/)
**Milestone:** M0, evenings 1–2
**Package:** `bulkhead`, a sibling of `toolsmith` and `steadyhand` under `src/`
**Depends on:** the implemented Toolsmith shim and Steadyhand loop
**Produces:** a written threat model, an executable Linux capability probe, explicit sandbox tiers, and reproducible environment records

This file is part 1 of 8. Implement the parts in order and stop at each acceptance gate.

| Part | File | Deliverable |
|---|---|---|
| 1 | `01-threat-model-and-capability-probe.md` | Threat model, capability probe, tier selection |
| 2 | `02-policy-manifests-and-approvals.md` | Sandbox manifests, deterministic ALLOW/DENY/ASK, approvals |
| 3 | `03-taint-audit-and-replay.md` | Taint state, security journal events, exact replay rules |
| 4 | `04-tier2-jailer-and-exec-tools.md` | Namespace/seccomp jailer and `run_python`/`run_shell` |
| 5 | `05-resource-governance-and-tier1.md` | cgroups, tree cleanup, rlimits, Landlock fallback |
| 6 | `06-egress-proxy-and-output-gate.md` | CONNECT allowlisting, byte caps, output sanitization |
| 7 | `07-attack-suite-and-overhead.md` | Escape suite, overhead and usability measurements |
| 8 | `08-agentdojo-evaluation-and-writeup.md` | AgentDojo, live sweeps, results and limitations |

## Outcome

Establish what Bulkhead protects and what the current host can actually enforce before writing a jailer. A single command, `python -m bulkhead.probe`, emits human-readable output and strict JSON describing usable kernel features, failed operations, selected tier, and reasons for rejecting stronger tiers.

No later part may describe a run as “sandboxed” merely because a timeout or a read-only Python variable was used. Tier selection is evidence from real operations in temporary directories, not a guess from kernel version strings.

## What already exists

| Existing component | Current responsibility | Bulkhead boundary |
|---|---|---|
| `src/toolsmith/api.py` | Parses and validates model output; never executes a handler | Leave this boundary unchanged. Bulkhead is behind Steadyhand, not in FastAPI. |
| `src/steadyhand/registry.py` | Frozen registry, argument validators, result schemas, policy side table | Reuse registration and hashes; add security metadata without changing model-facing schemas. |
| `src/steadyhand/validate.py` | Name, JSON Schema, and store-aware semantic checks | These remain the first gates. Valid arguments are not necessarily permitted arguments. |
| `src/steadyhand/executor.py` | Runs handlers on daemon threads with a policy timeout | The timeout is not a security boundary and cannot kill a timed-out thread. Risky execution must move to a separate supervised process. |
| `src/steadyhand/store.py` | Per-episode in-memory store clone | This isolates episodes in Python only; it does not isolate filesystem, processes, or network. |
| `src/steadyhand/recovery.py` | Retries, verification probes, and alternative tools | These calls also reach `SequentialExecutor`; every later security gate must cover them. |
| `src/steadyhand/journal.py` | Strict append-only JSONL, redaction, replay comparison | Reuse this for audit evidence rather than creating an unrelated log format. |
| `src/steadyhand/replay.py` | Rebuilds manifests and replays model turns, handlers, guards, and recovery | Security configuration and external-effect replay need explicit semantics in part 3. |
| `src/steadyhand/eval/suite.py` | Resumable, journaled task sweeps | Reuse manifests, claims, atomic records, and infrastructure-failure handling. |
| `notebooks/02_steadyhand.ipynb` | Existing Steadyhand CPU and live-model workflow | Add Bulkhead sections here. Do not create a per-day notebook. |

The project currently has no sandbox dependency or backend. Its declared Python dependencies are in `pyproject.toml`; bubblewrap, libseccomp, Landlock, and cgroup access must be probed as system capabilities rather than assumed. The current development host is WSL2 and `bwrap` is not initially on `PATH`, so an unavailability result is a valid probe outcome, not permission to claim tier 2.

## What this part adds

```text
src/bulkhead/__init__.py
src/bulkhead/types.py             # SandboxTier, Capability, ProbeReport
src/bulkhead/capabilities.py      # individual non-destructive probes
src/bulkhead/probe.py             # python -m bulkhead.probe
src/bulkhead/errors.py            # CapabilityError, UnsupportedTierError
tests/test_bulkhead_probe.py
tests/fixtures/bulkhead_probe_reports.json
```

A later implementation task adds `Bulkhead/THREAT_MODEL.md` as the maintained one-page threat model. The eight files already in this directory remain the implementation plan; `THREAT_MODEL.md` is a product artifact, not a ninth plan part.

## Threat model

Write down the following before choosing mechanisms.

### Assets

- Customer files and the per-episode workspace, including repository history and configuration.
- Credentials in environment variables, files, agent IPC, proxy configuration, and model-serving configuration.
- Host network access, internal services, metadata endpoints, and allowed external destinations.
- Host CPU, memory, process IDs, disk, file descriptors, and availability.
- Integrity of the store, tool outputs, policy decisions, journals, and evaluation results.

### Attacker and trusted computing base

- Treat all model output as attacker-controlled. The attacker can place text in web pages, emails, customer files, shell output, and any other tool observation.
- A prompt-injected model may select a valid registered tool with schema-valid but malicious arguments. Toolsmith validity does not imply authorization.
- A compromised tool process can fork, create threads, mutate files, open sockets, and attempt to trick the supervisor. It must not be in the trusted computing base.
- Trusted components are the deterministic policy engine, supervisor/jailer, proxy, kernel enforcement, minimal audit adapter, and human approval source. Prompt text and output delimiters are not trusted enforcement.
- Kernel zero-days, malicious host root, a compromised hypervisor, physical attacks, and hardware side channels are out of scope. A host administrator can bypass this design.

### Trust boundaries

1. Toolsmith model boundary: untrusted native calls become typed `ToolCall` values.
2. Steadyhand validation boundary: syntactic and business validation runs before authorization.
3. Bulkhead policy boundary: deterministic code returns `ALLOW`, `DENY`, or `ASK`.
4. OS boundary: namespaces/Landlock restrict visibility and access; seccomp restricts syscalls; cgroups/rlimits restrict resources.
5. Egress boundary: the sandbox has no direct external network; only the outside proxy can connect.
6. Observation boundary: output is capped, sanitized, labeled untrusted, and journaled before another model turn.
7. Approval boundary: only an explicit human grant can widen privileges; the model cannot approve itself.

## Probe design

### Report format

Use a versioned, JSON-serializable `ProbeReport` containing:

- `schema_version`, UTC timestamp, OS/kernel/architecture, and whether the environment appears to be WSL or a container.
- bubblewrap path/version, libseccomp availability/version, and helper/interpreter paths; never dump environment variables.
- Individual capability records with `status` (`ok`, `unavailable`, `denied`, `unsupported`, `failed`), an error code, and a short sanitized detail.
- Landlock ABI and the features actually tested, not just the highest advertised ABI.
- cgroup v2 mount, delegation, leaf creation, controller availability, limit writes, and `cgroup.kill` results.
- Selected tier and a list of requirements that failed for every higher tier.
- A stable `probe_sha256` over semantic capabilities, excluding timestamp, host name, temporary paths, and timings.

The CLI supports `--json <path>` using the repository’s atomic-write pattern and `--require-tier T0|T1|T2`. A requirement mismatch exits nonzero. Human output mirrors the brief’s example but always states `T0 ... NOT a sandbox` when that is the result.

### Test operations, not assumptions

Each probe cleans up its own temporary resources and uses canary files, never real credentials or sensitive host files.

- **User/mount/PID/network namespaces:** run a trivial bubblewrap command with fresh namespaces and `/proc`; distinguish missing binary from `EPERM`, AppArmor, and unsupported flags.
- **seccomp:** install a minimal filter in a child and confirm both an allowed syscall and a deliberately blocked harmless syscall. Record architecture.
- **Landlock:** query ABI, create a ruleset, restrict a temporary directory, and verify a canary read/write decision. Test TCP rules only on ABI 4 or later.
- **cgroup v2:** create a leaf only beneath an explicitly configured delegated parent; apply small memory/PID/CPU limits, read usage, and remove the leaf. Never write to arbitrary system cgroup files.
- **Process cleanup:** verify that the selected cleanup mechanism can terminate a child and grandchild fixture without leaving survivors.
- **Filesystem:** test temporary writable/read-only binds and denied subpaths. Check `openat2` availability for any future in-process file tool.
- **Network:** verify that a namespace fixture cannot reach an outside local listener and that the planned Unix-socket path can be mounted without enabling raw external access.

### Tier contract

| Tier | Required evidence | Permitted use | Required wording |
|---|---|---|---|
| `T2` | bubblewrap namespaces and mount isolation, seccomp, `no_new_privs`, fresh PID namespace, writable cgroup v2 with tree cleanup | Full namespace backend after parts 4–6 pass | “Namespace sandbox” with exact capabilities |
| `T1` | enforced Landlock filesystem rules, seccomp, `no_new_privs`, rlimits, and supported network restrictions or network disabled | Fallback backend after part 5 passes | “Landlock sandbox”; list missing namespace/cgroup properties |
| `T0` | rlimits/supervisor only, or incomplete enforcement | Existing trusted in-memory tools and diagnostics only | “Not a sandbox”; deny risky exec/network tools by default |

Selection is not permission to execute. A caller chooses a required tier and the probe confirms it. Automatic fallback from requested `T2` to `T1` or `T0` is forbidden; a new configuration and manifest are required.

## Tasks

1. Add the `bulkhead` package with no imports from `toolsmith` or `steadyhand` in the probe layer. Keep it independently testable on CPU.
2. Write the threat model and resolve every asset/boundary above. Include the lethal-trifecta case: private data plus untrusted content plus an outbound channel.
3. Implement typed capability records and the versioned report. Reject NaN, unknown enum values, and unstable hashes.
4. Implement each probe with bounded subprocess timeouts and cleanup. Do not use `sudo`, alter host security settings, or install packages silently.
5. Implement strict tier selection and `--require-tier`. Record missing prerequisites with actionable, non-secret details.
6. Add a CPU section to `notebooks/02_steadyhand.ipynb` that runs the same probe and stores the JSON under the existing per-GPU records tree. GPU type is irrelevant to probe selection.
7. Add unit tests with fake probe outcomes and opt-in Linux integration tests. Tests skip with an explicit reason when a capability is absent; a skip is not a successful sandbox test.

## Tests

- Report parsing refuses unknown tiers/statuses, unknown fields, non-finite numbers, and inconsistent tier evidence.
- The same semantic capabilities produce the same `probe_sha256` despite different timestamps and temporary paths.
- Simulated `EPERM`, missing `bwrap`, read-only cgroup, Landlock ABI 0, and seccomp failure select the documented lower tier or fail the requested tier.
- `T0` can never be rendered as `T1`/`T2`, and risky execution rejects `T0` by default.
- Probe timeout and child-cleanup fixtures do not leave processes or cgroup leaves behind.
- Real canary tests never open `~/.ssh`, `/etc/shadow`, `.env`, or any actual credential.
- Existing golden Steadyhand journals still replay unchanged; part 1 does not alter execution.

Run:

```bash
.venv/bin/python -m pytest tests/test_bulkhead_probe.py tests/test_steadyhand_replay.py
.venv/bin/python -m bulkhead.probe --json results/bulkhead/probe.local.json
```

Run the full `.venv/bin/python -m pytest` before advancing. The Colab probe remains an explicit acceptance step for the user’s runtime; do not mark it complete without its JSON output.

## Acceptance

- The threat model names assets, attacker capabilities, trusted components, boundaries, and exclusions.
- The probe prints and writes the selected tier on the development Linux host and, when executed, in Colab.
- Every selected tier is backed by an actual operation and every downgrade has a recorded reason.
- No risky tool runs yet, no system configuration is changed, and no secret is read or logged.
- Existing Toolsmith/Steadyhand tests remain green.

## Pitfalls

- Inferring user-namespace or cgroup support from `uname`, distribution name, or the existence of `/sys/fs/cgroup`.
- Treating “root inside a container” as permission to mount or write cgroups.
- Installing bubblewrap with elevated privileges as a hidden side effect of the probe.
- Calling rlimits, a thread timeout, or a store clone a sandbox.
- Printing environment variables, credential paths with sensitive names, full command lines, or unrelated host details in the report.

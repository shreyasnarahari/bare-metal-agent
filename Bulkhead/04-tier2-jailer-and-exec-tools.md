# Part 4 — Tier-2 jailer and executable tools

**Track:** Bulkhead, Day 3 of 10
**Milestone:** M3, evenings 6–8
**Depends on:** `03-taint-audit-and-replay.md`
**Produces:** a namespace/seccomp backend, isolated workspaces, supervised `run_python` and `run_shell`, and the first 20 blocked attack fixtures

## Outcome

Allowed `Effect.EXEC` calls run in a fresh bubblewrap-based mount/PID/IPC/network/user namespace with a minimal filesystem view, seccomp, no privilege gain on exec, a scrubbed environment, and bounded captured output. The supervisor, not a Python daemon thread, owns the subprocess and its timeout.

This part is the namespace/seccomp phase of tier 2. Until part 5 integrates cgroups and resource cleanup, label integration output `bwrap-namespace-phase`, keep risky execution confined to tests, and do not publish it as the brief’s complete T2 profile.

The backend is selected only when the part 1 probe proves the required features. Failure to construct the sandbox is a `sandbox_violation`/infrastructure error before the tool starts; it is never a fallback to host execution.

## What already exists

- Part 1 defines `T2` evidence and fail-closed tier selection.
- Parts 2–3 provide authorized execution plans, `Effect.EXEC`, full policy/configuration manifests, security observations, and safe replay.
- `src/steadyhand/executor.py` currently binds every handler to `Store` and runs it under a thread timeout. That path remains for trusted in-memory tools; exec markers need a separate process backend before `_call_with_timeout`.
- `src/steadyhand/tools_exec.py` registers `run_python(code: str)` and `run_shell(command: str)` schemas/markers, with no direct host handler.
- `src/toolsmith/grammar.py` and both model profiles can validate that the new tool schemas remain in the supported subset.
- `src/toolsmith/artifacts.py` and `src/toolsmith/recording.py` provide atomic/redacted recording patterns.

## What this part adds

```text
src/bulkhead/backends/__init__.py
src/bulkhead/backends/base.py     # ExecutionRequest, ExecutionResult, Backend protocol
src/bulkhead/backends/bwrap.py    # command/namespace construction, no shell interpolation
src/bulkhead/seccomp.py           # libseccomp filter compilation and architecture checks
src/bulkhead/supervisor.py        # spawn, pipes, timeout, bounded capture, exit classification
src/bulkhead/workspace.py         # per-episode/call views and cleanup
src/steadyhand/tools_exec.py      # real dispatch descriptors and ExecResult
tests/test_bulkhead_bwrap.py
tests/test_bulkhead_seccomp.py
tests/test_bulkhead_supervisor.py
tests/test_steadyhand_exec_tools.py
tests/fixtures/bulkhead_attacks/tier2_first20.json
tests/fixtures/bulkhead_attacks/scripts/
```

The initial implementation can be Python plus the system bubblewrap/libseccomp binaries. Do not add an unverified Python package dependency. `ctypes` may load a probed system libseccomp; bubblewrap’s `--seccomp FD` can receive the compiled filter. A static C++ helper remains a part 8 stretch goal.

## Execution contract

```text
ExecutionRequest:
  request_id, episode_id, tool, arguments_sha256
  program kind (python|shell), bounded input bytes
  canonical Sandbox spec and selected T2 probe hash
  per-call workspace, timeout, output limits, audit context

ExecutionResult:
  started, exit_code, signal, timed_out
  bounded raw stdout/stderr
  violations, cleanup status, elapsed/resource fields
  request_sha256 and config_sha256
```

The adapter converts this to the existing `ToolResult`/`ExecResult`; the backend itself does not import Steadyhand or modify the store. Dispatch occurs only after `ALLOW`. Direct invocation of an exec marker outside the backend raises and is covered by a test.

Pass program text over a bounded stdin descriptor (`python -I -` or shell reading from stdin), not through an outer shell and not in logged command-line arguments. Use fixed argv arrays throughout. The model never supplies launcher flags, environment entries, interpreter paths, or file descriptors.

## Namespace and filesystem view

### Minimal view, not host exposure

Build a fresh namespace with bubblewrap options equivalent to:

- `--unshare-all`, `--die-with-parent`, and a new session.
- Fresh `/proc`, minimal `/dev`, fresh PID namespace with an init that reaps children.
- All capabilities dropped and `NoNewPrivs=1` verified in a fixture.
- Read-only interpreter/runtime binds, certificate roots only when network is later allowed, and a dedicated writable workspace.
- Temporary `/tmp` with an enforced size where supported; no host home, SSH agent, Docker socket, container socket, or credential directory.

The brief mentions `--ro-bind / /` as a starting point. Do not use that as the final least-privilege configuration: read-only host root still exposes readable secrets and makes the “read `~/.ssh`” attack succeed. Use it only as a development comparison. Acceptance requires a minimal bind view in which everything not declared readable is invisible, not merely read-only.

`$WORKSPACE` is a fresh directory owned by the caller with mode `0700`, beneath a configured Bulkhead work root. Do not bind the user’s actual repository or home directory by default. If a task supplies existing files, copy/bound them into a controlled input view rather than exposing the whole parent.

### Read/write/deny enforcement

- Materialize only declared read and write paths. Writable binds are descendants of the controlled workspace, never arbitrary host paths.
- Enforce `fs_deny` in the namespace as well as the policy. Do not bind denied content; where a parent bind would expose it, mask the descendant with an empty inaccessible file/directory of the correct type.
- Specifically test `$WORKSPACE/.git`, `.git/hooks`, and `$WORKSPACE/.env` with canaries. A policy denial alone is insufficient if the process can still read the file.
- Refuse ambiguous host symlink components while constructing binds. Previous calls must be fully reaped before the next call reuses a persistent workspace, so no surviving process can race path setup.
- Hardlink and symlink tests must demonstrate that workspace writes cannot modify a host canary outside the view.

seccomp is not used for path authorization: BPF sees syscall arguments and pointer values, not stable pointed-to path strings. Filesystem visibility/access belongs to the mount namespace here and Landlock in part 5.

## seccomp and process setup

Compile a filter for the actual architecture and pass it to bubblewrap over a dedicated file descriptor. Keep the trusted setup single-threaded at the point where filters/no-new-privileges and exec are applied. Do not use Python `preexec_fn` in Steadyhand’s multithreaded workers; namespace/filter setup belongs in bubblewrap or a dedicated immediately executed helper.

Start with a reviewed syscall set needed by the interpreter and explicitly block bypass/primitive classes from the brief:

- `ptrace`, `process_vm_readv`, `process_vm_writev`, and cross-process memory access.
- `io_uring_setup`, `io_uring_enter`, and `io_uring_register`.
- `bpf`, `perf_event_open`, `keyctl`, and kernel module operations.
- `mount`, `umount2`, `pivot_root`, `setns`, nested namespace creation, and `open_by_handle_at` inside the tool.
- `reboot`, kexec, unsafe `userfaultfd`, and privilege-changing operations not required by the tool contract.

Choose and document whether unsupported syscalls return `EPERM` or kill the process. Prefer `EPERM` for expected denied operations and a logged kill for clearly dangerous architecture mismatches. Test filter behavior directly; a seccomp violation is not inferred from a generic nonzero exit alone.

Close every unneeded inherited descriptor. Set a minimal environment (`PATH`, `HOME` inside workspace, locale, and explicit proxy variables only in part 6). Never inherit `HF_TOKEN`, `LOCAL_KEY`, authorization variables, cloud credentials, SSH agents, or the notebook’s environment.

## Supervisor and timeouts

- Spawn the fixed helper/bubblewrap argv with `subprocess.Popen`, not `shell=True` and not a daemon handler thread.
- Drain stdout and stderr concurrently so a full stderr pipe cannot deadlock a process waiting on stdout.
- Enforce input size, per-stream output caps, and a combined cap while reading; keep only bounded prefixes and counts. Part 6 adds full sanitization and model delimiters.
- Use `time.monotonic()` for timeout and enforce the part 2 tool timeout. Until part 5 adds cgroups, describe cleanup precisely as direct-child/PID-namespace cleanup, not guaranteed cgroup tree kill.
- On timeout or setup failure, terminate the namespace init/direct tree as supported, close pipes, collect the exit status, and remove temporary resources. A nonzero tool exit is a valid tool result; a supervisor/sandbox setup failure is distinct.
- Do not hold `Store.lock` around an external process. Exec tools do not mutate the in-memory store directly.

`ExecResult` contains bounded stdout/stderr, exit code/signal, timeout and truncation flags. Output is marked `UNTRUSTED` through the part 3 adapter even before part 6 adds the final rendering.

## Steadyhand wiring

1. `_plan` authorizes exec calls and resolves a canonical immutable `ExecutionRequest` from the registered spec.
2. `SequentialExecutor` dispatches exec markers to the configured backend rather than `_call_with_timeout`; legacy handlers retain current behavior.
3. Fault injection stays around the authorized backend invocation. A fabricated fault may prevent invocation but cannot bypass it or alter its canonical sandbox spec.
4. The loop writes `sandbox_execution` after `policy_decision` and before the final `tool_result`; recovery calls follow the same order.
5. Default replay uses `RecordedExecutionBackend`. Only explicit integration tests execute fixture scripts.
6. Any exec call is sequential with writes and never automatically retried. A timeout has unknown external effects and follows the security/recovery contract rather than a blind retry.

## Initial attack gate

Implement the first 20 fixtures from the full part 7 matrix, focusing on access and syscall boundaries rather than resource exhaustion:

- Read fake home/SSH and shadow-like canaries; confirm they are invisible/denied.
- Write outside the workspace and through `..`, absolute paths, symlinks, and hardlinks.
- Read/write denied `.env`, `.git`, and `.git/hooks` canaries.
- Attempt `ptrace`, `/proc/<pid>/mem`, `process_vm_*`, and helper-process inspection.
- Attempt `io_uring` socket/setup operations, `bpf`, `setns`, nested namespaces, and mounts.
- Attempt abstract Unix sockets and host service sockets.
- Attempt setuid/capability escalation and unexpected privilege changes.
- Assert credentials and unexpected environment entries are absent.

Each fixture declares expected verdict, required tier, canary evidence, and expected audit violation. Tests use only disposable canaries and local listeners. An unsupported tier is skipped and reported as unsupported, never passed.

## Tasks

1. Implement the backend protocol and canonical request/result serialization; bind them to part 3 replay records.
2. Implement controlled workspaces, minimal bind construction, denied-subpath masking, and cleanup.
3. Implement seccomp compilation and direct filter tests on supported Linux hosts.
4. Implement the single-threaded launch path and verify namespace, capability, descriptor, environment, and `NoNewPrivs` properties from inside fixtures.
5. Implement non-deadlocking bounded process capture, monotonic timeout, and precise exit/setup error classification.
6. Wire exec dispatch through the authorized `SequentialExecutor` path without changing legacy mock handlers or recovery semantics.
7. Add `run_python`/`run_shell` end-to-end tests: benign computation, nonzero exit, timeout, output cap, denied filesystem access, and exact journal sequence.
8. Run the first 20 attacks, preserve every failure as a regression fixture, and fix the backend before advancing. Do not weaken expected verdicts to make the run green.

## Tests and commands

Unit-test argv construction without executing it, then run real Linux integration tests where the probe supports T2. Mock tests do not count as sandbox acceptance.

```bash
.venv/bin/python -m pytest tests/test_bulkhead_bwrap.py tests/test_bulkhead_seccomp.py \
  tests/test_bulkhead_supervisor.py tests/test_steadyhand_exec_tools.py
.venv/bin/python -m bulkhead.probe --require-tier T2
```

Also run all part 2–3 security/replay tests and the full repository suite. A host without T2 keeps these integration tests explicitly skipped; acceptance for the T2 deliverable remains pending until run on a suitable Linux host/VM.

## Acceptance

- Benign Python and shell calls execute only inside the namespace/seccomp phase and return bounded typed results; no result is labeled full T2 until part 5 acceptance.
- The first 20 attack fixtures fail to achieve their attacker objective and produce useful audit evidence.
- The process cannot see undeclared host files, inherit credentials, use blocked syscall classes, or retain unexpected descriptors.
- Direct handler calls, fault injection, recovery, and replay cannot bypass the backend.
- A missing/failed T2 capability never falls back to host execution or gets labeled a successful sandbox test.
- Existing Toolsmith and Steadyhand suites remain green.

## Pitfalls

- Using `--ro-bind / /` and assuming read-only means confidential files cannot be read.
- Applying seccomp to a multithreaded supervisor or relying on unsafe `preexec_fn` in worker threads.
- Using seccomp user notification as a live path-approval mechanism; pointer arguments remain subject to time-of-check/time-of-use races. Reserve it, if used at all, for coarse logging/approval.
- Filtering syscall numbers while assuming seccomp can enforce path strings.
- Leaving `io_uring`, `ptrace`, nested namespaces, or abstract sockets available.
- Logging the submitted code/command or inheriting the notebook environment.
- Holding the store lock during subprocess execution or killing only the direct child while claiming tree cleanup.
- Treating mocked argv tests or skipped integration tests as proof of isolation.

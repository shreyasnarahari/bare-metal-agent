# Bare Metal Agent: Toolsmith

Toolsmith is a from-first-principles function-calling and constrained-decoding library. It serves open-weight models through raw llama.cpp completions and implements the layers normally hidden by an agent framework: prompt rendering, streaming tool-call parsers, JSON Schema-to-GBNF compilation, a pushdown automaton and token-mask engine, validation and repair, and a small OpenAI-compatible FastAPI shim.

**Status:** local implementation and CPU/grammar verification are complete. Real Qwen/Colab acceptance and BFCL/JSONSchemaBench model benchmarks have not been run. This README intentionally contains no inferred accuracy or performance scores.

## Run and verify

On Colab, clone or `git pull` this repository into `/content/bare-metal-agent`, then run `notebooks/01_toolsmith.ipynb` (Toolsmith), `notebooks/02_steadyhand.ipynb` (Steadyhand), or `notebooks/03_bulkhead.ipynb` (Bulkhead) from the synced tree.

Local checks:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src tests
git diff --check
```

There are three notebooks: `notebooks/01_toolsmith.ipynb` for the Toolsmith track, `notebooks/02_steadyhand.ipynb` for the Steadyhand agent loop, and `notebooks/03_bulkhead.ipynb` for the Bulkhead sandbox and permission layer. All three provision llama.cpp and the GGUFs through `toolsmith.provision`, which shares one set of Drive caches. Cells that two notebooks share are kept identical by the notebook tests. On a fresh Colab T4 or L4, run a notebook top to bottom. Do not create per-day notebooks. Put `HF_TOKEN` and `LOCAL_KEY` in Colab Secrets or the protected Drive secrets file, never in committed cells or result records.

### Running on a T4 or L4

Each notebook's shared `hardware` setup cell detects the GPU, vCPUs, RAM, and free disk once (`toolsmith.hardware`). Every setting below comes from those numbers, not from the GPU's name. Without a GPU, with less than 14 GiB of VRAM, or with compute capability below 7.0, the cell stops and says how to switch runtimes.

| Setting | Rule | T4 (2 vCPUs, about 13 GB RAM) | L4 (8 vCPUs) |
|---|---|---|---|
| llama.cpp build jobs | one per vCPU, at most one per 2 GiB of RAM | 2 | 8 |
| Day 10 server | 4096 tokens per slot; more VRAM means more slots, not longer ones | 4 slots, `-c 16384` | 8 slots, `-c 32768` |
| Steadyhand server | preset per GPU and shape, confirmed by the probe | per preset | per preset |
| CPU pools (CPU gate, schema coverage) | one per vCPU, at most 8 | 2 | 8 |
| HF Qwen3-0.6B generation | one fp16 model per process | 2 processes | 6 processes |
| MaskBench | the Toolsmith engine in the kernel, plus single-threaded peers | 1 peer at a time | 5 peers at once |

Each Day 10 request pool sends one request per server slot. Any other GPU uses the T4 or L4 rows that match its VRAM.

- **Overlapped setup.** Downloads, builds, Drive restores, and Drive write-backs run as background jobs (`toolsmith.background`). Each job writes its own log under `/content/background-logs/`.
  - In the Toolsmith notebook, both GGUFs download while b5401 builds.
  - The b8233 build for Qwen3.5 starts after b5401 and runs through Days 2–8 at low priority.
  - The MaskBench peer environment installs during the Day 10 generation runs.
  - In the Steadyhand notebook, provisioning runs during the CPU gate.
  - On a cached runtime, these jobs are short copies from Drive.
- **Drive layout.** These folders under `MyDrive/toolsmith/` are shared by every GPU:
  - `builds/`, holding llama.cpp builds. Their folder names already include the CUDA compute capability.
  - `models/`, holding the GGUFs and the HF datasets cache.

  Records go to `records/<gpu>/` (`t4`, `l4`, or the GPU's name), so a T4 run and an L4 run never resume, overwrite, or skip each other's files.
- **Stamps and manifests.**
  - Every Toolsmith record carries a `runtime` stamp: the GPU, vCPUs, load average, and running background jobs.
  - Day 10 checkpoint manifests also include the compute capability, the llama.cpp tag, and the server shape. If those change, the run stops and lists every field that differs.
- **Timing under concurrency.** Timing cells are not serialized. The MaskBench peers run side by side, and the Days 6–8 benchmarks may overlap a background build. Each timing record states the concurrency it ran under, in the stamp or in `run.json`. Compare timings only between runs with the same concurrency.
- **Steadyhand serving.**
  - The server cell tries KV cache and flash-attention settings in this order: q8_0 with flash attention, f16 with it, then f16 without it.
  - It keeps the first setting that loads, leaves 1 GB of VRAM free, and passes a decode-speed self-test.
  - Running out of VRAM drops one slot.
  - Sweeps mirror their scores to Drive and resume from that copy after a disconnect.

### Disconnects, hangs, and Colab Pro

A Colab session can disconnect, hang, or lose its Drive mount. Recovery is always the same: reconnect, then run all cells from the top.

- **Durable copies.** Checkpoints, scores, and coverage are written on local disk and copied to Drive as whole-line atomic snapshots (`MirroredFile`): every minute for Day 10 checkpoints, every 30 s for Steadyhand scores, and every 2 minutes for coverage. A fresh runtime restores from those copies. A copy cut off mid-line is repaired before anything is appended to it.
- **Flushing Drive.** Colab uploads Drive writes in the background, so each long cell ends with `sync_drive` (`flush_and_unmount`, then mount again), and cleanup flushes and unmounts. If remounting asks you to authorize again, set `TOOLSMITH_DRIVE_SYNC=cleanup` to flush only in cleanup.
- **Serving failures are never results.** A connection error, a timeout, or a 502–504 from the shim is not recorded as a case result. The run stops new work, restarts llama-server once, and resumes. A case that fails this way three times is recorded as `infrastructure_retries_exhausted`. Steadyhand reruns an episode that aborted on infrastructure up to twice, under a new episode ID. It stops the sweep, restarts the server, and resumes when episodes keep aborting.
- **Nothing waits forever.**
  - Clones and builds are killed when their logs stall (10 and 20 minutes).
  - git aborts a transfer that crawls.
  - Hugging Face requests time out.
  - A background job that goes silent is reported.
  - Long cells probe Drive first and remount it if it hung.
  - The Steadyhand journal copier never blocks the sweep workers.

| Section | Resumes at | A disconnect loses |
|---|---|---|
| Setup | Drive caches for the builds and GGUFs | an unfinished llama.cpp build, unless only the kernel restarted |
| Day 10 own set and BFCL | each finished case | cases in flight and the last minute not yet copied |
| JSONSchemaBench generation | each finished split per model and mode | the split in progress |
| HF Qwen3-0.6B generation | each finished item | items in flight |
| Schema coverage | each finished schema | up to 2 minutes of schemas |
| MaskBench | each schema per engine | schemas in progress |
| Steadyhand sweep | each scored episode | episodes in flight and up to 30 s of scores |

Each row can also lose whatever Drive had not yet uploaded since the last `sync_drive`. The report cells read finished runs back from their checkpoints, so they work without rerunning generation and without loading a model.

On Colab Pro (unlike Pro+), a session generally does not keep running in the background once the browser disconnects, so keep the tab open during long cells. The notebooks deliberately do not use keep-alive tricks; making resume cheap is what they rely on.

## Steadyhand: the agent loop

`src/steadyhand/` is the agent loop that runs on top of the Toolsmith shim. The design notes are in `Steadyhand/`. It sends a conversation and tool schemas through `ToolsmithService.chat`, validates each native tool call, and runs it against an in-memory mock store. On a tool failure it applies one of six recovery policies, and it writes one append-only journal per episode. CPU replay of a journal reproduces the episode exactly.

The boundary is unchanged: **handlers run in Steadyhand, after the shim returns. The shim and its FastAPI routes never execute a tool.** Steadyhand runs on Colab from `notebooks/02_steadyhand.ipynb`.

**Status:** the 60-task suite, the fault injector, the policies, replay, and the report generator are implemented. They are checked on CPU with a scripted model and golden journals. **The live sweep has not been run.** No Steadyhand accuracy, duplicate-effect, or throughput number is claimed. Once a sweep runs, `results/steadyhand/RESULTS.md` is generated from its journals by `steadyhand.eval.report`. This section will cite only the measured cells, with their manifests.

## Bulkhead: sandboxed tool execution

`src/bulkhead/` will put Steadyhand's risky tools behind a deterministic policy engine and an OS sandbox. The eight-part plan is in `Bulkhead/`, and the maintained threat model is `Bulkhead/THREAT_MODEL.md`. On Colab, Bulkhead runs from its own notebook, `notebooks/03_bulkhead.ipynb`. It moved out of the Steadyhand notebook after part 8; the part plans still name `02_steadyhand.ipynb`, written before the move.

**Status:** parts 1 to 8 are implemented: the threat model and capability probe, policy and approvals, taint, audit and replay, the tier-2 jailer and exec tools, resource governance with the tier-1 backend, the egress proxy with the output gate, the versioned attack suite with overhead and usability benchmarks, and the AgentDojo adapter with paired runs. **Every backend is verified on the development host:** the bubblewrap (tier-2) backends since bubblewrap 0.9.0 was installed, and the tier-1 backends (with and without a cgroup), together with cgroup governance, the egress proxy, and the output gate. The whole attack corpus runs on `bwrap-t2` inside a delegated cgroup scope with no escape. Tier 2 on Colab is still unverified. **The AgentDojo evaluation's local CPU checks are complete; its live results are pending**, and no AgentDojo score is claimed. See parts 4 to 8 below.

```bash
.venv/bin/python -m bulkhead.probe --json results/bulkhead/probe.local.json
.venv/bin/python -m bulkhead.probe --require-tier T1    # exit 3 unless T1 is proven; never falls back
BULKHEAD_INTEGRATION=1 .venv/bin/python -m pytest tests/test_bulkhead_probe.py
```

The probe proves each capability with a real operation in a temporary directory. It does not infer support from kernel versions or paths, installs nothing, changes no setting, and reads no secret. It then selects a tier:

- **T2, Namespace sandbox:** bubblewrap namespaces, seccomp, `no_new_privs`, rlimits, and a delegated cgroup v2 leaf with `cgroup.kill`.
- **T1, Landlock sandbox:** Landlock, seccomp with inet sockets denied, `no_new_privs`, rlimits, and process-group cleanup.
- **T0, NOT a sandbox:** everything else.

On the WSL2 development host (bubblewrap 0.9.0 installed), the probe selects **T1** on the bare host, because no cgroup parent is configured by default. Inside a delegated user scope the leaf, limit, and `cgroup.kill` probes also pass, and it selects **T2**:

```bash
systemd-run --user --scope -p Delegate=yes bash -c '
  P=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
  mkdir "$P/supervisor" "$P/sandbox"
  echo $$ > "$P/supervisor/cgroup.procs"   # a cgroup that hands controllers to children holds no processes
  echo "+cpu +memory +pids" > "$P/cgroup.subtree_control"
  echo "+cpu +memory +pids" > "$P/sandbox/cgroup.subtree_control"
  .venv/bin/python -m bulkhead.probe --cgroup-parent "$P/sandbox"'
```

The bubblewrap (T2) probes pass against bubblewrap 0.9.0 on the development host. The Colab probe (`bulkhead-probe` in `notebooks/03_bulkhead.ipynb`, written to `records/<gpu>/bulkhead/probe.colab.json`) has not been run.

### Policy gate (part 2)

`run_episode(..., security=SecurityConfig(task_policy=...))` turns on Bulkhead for an episode. Each schema-valid call goes through a pure policy engine (`bulkhead.policy`) before the fault injector or a handler sees it. The order is validate, then the repeated-call guard, then authorize, and only then execute. The engine returns `ALLOW`, `DENY`, or `ASK` with a stable rule id.

- **DENY:** the model gets a non-retryable `permission_denied` observation and may choose another action.
- **ASK:** anything outside the task's privilege set asks: a new tool, path, destination, or a higher bound. The call gets `permission_required` and the episode ends `escalated`.
- **Recovery and faults:** recovery retries, verify probes, and switches to an alternative tool each get their own decision. A denied call never consumes a fault index.

An approval becomes a one-shot grant for that exact call. `run_approved_episode` runs it as a new linked episode (`run_label="approved-<id>"`, with a parent reference). The denied episode's files never change.

Exec tools (`run_python`, `run_shell`, opt-in via `build_registry(exec_tools=True)`) are denied unless the episode has a sandbox backend (part 4). The executor never runs them in-process. With `security=None`, behavior, episode ids, and golden manifests are unchanged.

### Taint, audit, and replay (part 3)

Taint is coarse, episode-level information-flow control. It is not per-value tracking.

- **Labels:** an episode starts trusted. The first tool output from a tool whose results are not declared trusted makes it untrusted, and nothing makes it trusted again. Store reads were reviewed field by field: `search_orders` is trusted; customer, order, ticket, and payment reads are not.
- **Effect:** while the episode is untrusted, a call that reaches a taint-sensitive sink asks instead of being allowed. That covers network, sensitive filesystem writes, and tools the task marks sensitive. An explicit deny stays a deny, and pure reads are unaffected.
- **Ordering:** taint is applied right after each result is journaled, before the next call is decided. A message with a sensitive call runs sequentially, so an untrusted read can't be outrun by a sensitive call in the same message.
- **Journal:** every security event (`policy_decision`, `approval_requested`, `taint_transition`, `sandbox_execution`, `egress_decision`) is strictly checked in the episode's own journal when written and when read.
- **Replay:** default replay re-derives every decision, taint change, and observation. Exec, sandboxed, filesystem, and network calls are served from their recorded `sandbox_execution` runs, so replay never spawns a process or opens a socket. It shows the control plane reproduces; it does not show the kernel enforced anything. `live_replay_report` re-executes under a supplied backend and writes a separate comparison report.
- **Fixtures:** the golden secured episodes in `tests/fixtures/bulkhead_episodes/` cover clean, tainted, same-turn, denied, approved, and sandboxed runs. The sandboxed one uses a scripted test backend, not a sandbox.

### Tier-2 jailer and exec tools (part 4)

`bulkhead.backends.bwrap.BwrapBackend` runs allowed `run_python` and `run_shell` calls in bubblewrap. Its label is `bwrap-namespace-phase`: cgroup limits and cgroup tree cleanup arrive in part 5, so this is not yet the complete T2 profile. The backend refuses to construct unless a probe report proves bubblewrap namespaces, mounts, network isolation, and seccomp on this machine. A sandbox that fails to build for one call is a `sandbox_violation`. Neither case ever falls back to running on the host.

```python
from bulkhead.backends.bwrap import BwrapBackend
from bulkhead.probe import run_probe

backend = BwrapBackend(run_probe(), "work/bulkhead")  # raises BackendUnavailable without proof
run_episode(..., security=SecurityConfig(policy, available_tiers=(SandboxTier.T2,)), backend=backend)
```

- **Namespaces:** fresh user, PID, IPC, UTS, network, and cgroup namespaces, with `--die-with-parent` and `--new-session`. All capabilities are dropped, and bubblewrap's `no_new_privs` is always on.
- **Filesystem view:** only declared paths exist. The read-only paths are the workspace (`/workspace`), `/usr`, and merged-`/usr` symlinks. Only `/workspace/out` is writable, and `.git` and `.env` are covered by empty mode-0000 masks. There is no `--ro-bind / /`. Binding `/`, a home directory, `/etc`, `/proc`, `/sys`, `/dev`, `/run`, temporary directories, or the work root is refused, as are host symlink components. Workspaces are per episode, mode 0700, and seeded by copying files in, never by hardlinking.
- **seccomp** (`bulkhead.seccomp`): the filter is compiled with the system libseccomp through `ctypes` and installed by bubblewrap just before exec. No Python `preexec_fn` runs.
  - A reviewed allowlist. Unlisted syscalls return `ENOSYS`, which makes `clone3` fall back to `clone`, whose flags the filter can inspect.
  - `ptrace`, `process_vm_*`, `pidfd_getfd`, io_uring, `bpf`, `perf_event_open`, keyrings, kernel modules, mounts, `setns`/`unshare`, `clone` with namespace flags, `open_by_handle_at`, reboot/kexec, `userfaultfd`, and privilege, clock, and hostname changes all return `EPERM`.
  - Sockets are allowed for `AF_UNIX` only.
  - A foreign-ABI syscall (x32, i386) kills the process, which is recorded as a seccomp violation.
  - `bulkhead.seccomp.simulate` evaluates the compiled BPF for any syscall without installing it.
- **Supervisor** (`bulkhead.supervisor`): runs a fixed argv with `Popen`, never `shell=True`.
  - Program text goes in over stdin, never on a command line, and the environment is a fixed minimal set (nothing inherited, so no tokens or SSH agent). The program inherits only stdin, stdout, and stderr.
  - Both output streams are drained together with per-stream and combined caps.
  - The deadline is monotonic. At the deadline the process group is killed. After exit, leftover group and session members are killed, and the sandbox's PID namespace takes the rest. A process that escapes all of these is reported as `leaked`, never `clean`.
- **Steadyhand wiring:** after `ALLOW`, `BackendRunner` builds an immutable request from the tool's registered program descriptor, timeout, and sandbox spec. It journals `sandbox_execution` between `policy_decision` and `tool_result`.
  - A nonzero exit is a normal result. A timeout is a `timeout` with unknown effects and is never retried.
  - Exec output is untrusted, so it taints the episode.
  - Faults can prevent a call but never replace it, and no store lock is held while a process runs. Replay serves recorded runs.
- **Exec sandbox spec:** certificate roots were removed from it, since it has no network; they return with egress in part 6.

```bash
.venv/bin/python -m pytest tests/test_bulkhead_bwrap.py tests/test_bulkhead_seccomp.py \
  tests/test_bulkhead_supervisor.py tests/test_steadyhand_exec_tools.py
BULKHEAD_INTEGRATION=1 .venv/bin/python -m pytest tests/test_bulkhead_bwrap.py   # real bubblewrap, where the probe proves it
```

What had been verified on the WSL2 development host when part 4 landed, before bubblewrap was installed (since verified: see *Tier 2 verified with bubblewrap* below):

- **seccomp filter:** on the real kernel, Python threads, subprocesses, and `sh` run normally under it.
- **Blocked primitives:** `ptrace`, `process_vm_readv`, and inet sockets, which this kernel allows unfiltered, are refused with the filter installed.
- **Attack fixtures:** 8 of the first 20 attacks are syscall-only, and all 8 fail under the filter alone. (Part 7 moved every attack into `evaluation/data/bulkhead_attacks_v1.json`; 12 of its attacks now run under the filter alone.)
- **Supervisor:** draining, caps, deadline, cleanup, environment, and descriptor behavior.
- **Steadyhand wiring:** tested end to end with a host-only seccomp backend that is not a sandbox (`tests/exec_support.py`).

Not yet verified anywhere:

- The bubblewrap backend itself.
- The 12 filesystem, process-inspection, and socket attacks that need its namespaces.
- The in-sandbox property checks.

Those 23 integration tests are skipped here as unsupported. Acceptance for this phase is pending a run on a Linux host or VM with bubblewrap.

### Resource governance and tier 1 (part 5)

Every live call now goes through one path, `bulkhead.backends.common.run_governed`:

- **cgroup leaf:** with a delegated parent configured, a fresh leaf with `memory.max`, `memory.swap.max`, `memory.oom.group`, `pids.max`, and `cpu.max` is created before anything is spawned.
- **Helper:** a single-threaded helper (`python -I -m bulkhead.helper`) closes stray descriptors and applies, in order:
  - `no_new_privs`;
  - Landlock (T1);
  - rlimits: `RLIMIT_CPU`, `RLIMIT_FSIZE`, `RLIMIT_NOFILE`, `RLIMIT_CORE=0`, and `RLIMIT_AS` as a backup only;
  - seccomp (T1).

  It then waits at a start barrier until the supervisor has moved it into the leaf and checked the move, and only then execs bubblewrap (T2) or the launcher (T1).
- **While it runs:** the governor polls the workspace's disk total and the leaf's total CPU time.
- **Afterwards:** `cgroup.kill` ends every survivor and the leaf is removed. A leaf that can't be emptied is reported `leaked`, and the next backend start reaps it.

| Profile | Tier | Files | Network | Cleanup | Memory / PIDs |
|---|---|---|---|---|---|
| `bwrap-t2` | T2 | mount namespace: undeclared paths absent | network namespace, no egress | PID namespace and `cgroup.kill` | cgroup |
| `bwrap-namespace-phase` | T2 (partial) | mount namespace | network namespace | PID namespace and process group | none |
| `landlock-t1-cgroup` | T1 | Landlock: access denied, names may be visible | none (`socket()` refused, TCP denied) | process group (`setsid` refused) and `cgroup.kill` | cgroup |
| `landlock-t1` | T1 | Landlock | none | process group only | `RLIMIT_AS` backup only |
| T0 | T0 | not a sandbox: exec and network tools are denied | | | |

`bulkhead.backends.profiles` holds the full description and builds the per-attack expectation matrix from the part 7 dataset (`blocked`, `different_boundary`, `not_applicable`; never `passed`).

- **Construction:** `BwrapBackend` now builds full T2 by default and needs a cgroup parent. The namespace-only profile needs `namespace_phase=True`.
- **`resource_exhausted`:** a memory, PID, disk, or CPU ceiling gives this non-retryable observation, distinct from `timeout` and from a nonzero exit.
- **Disk:**
  - Hard limits: `/tmp` (T2 with `--size`) and any single file (`RLIMIT_FSIZE`).
  - Soft limit: the workspace total is polled about every 50 ms, so a fast writer can overshoot. In the many-file test it stopped at 12 MiB against an 8 MiB limit.

```bash
.venv/bin/python -m pytest tests/test_bulkhead_cgroups.py tests/test_bulkhead_landlock.py tests/test_bulkhead_resource_limits.py
# Destructive resource fixtures run only inside a confirmed delegated parent:
systemd-run --user --scope -p Delegate=yes bash -c '
  P=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
  mkdir "$P/supervisor" "$P/sandbox"
  echo $$ > "$P/supervisor/cgroup.procs"
  echo "+cpu +memory +pids" > "$P/cgroup.subtree_control"
  echo "+cpu +memory +pids" > "$P/sandbox/cgroup.subtree_control"
  BULKHEAD_INTEGRATION=1 BULKHEAD_CGROUP_PARENT="$P/sandbox" .venv/bin/python -m pytest tests/test_bulkhead_resource_limits.py'
```

Verified on the WSL2 development host when part 5 landed (Landlock ABI 7, no bubblewrap yet (since verified: see *Tier 2 verified with bubblewrap* below)):

- **Tier-1 attacks:** the real T1 backend stops all 20 part 4 attacks. Seven fail differently than under T2 (access denied rather than invisible, or no sockets at all rather than a separate network namespace), and the fixture records which and why.
- **Cleanup fixtures:** descriptor exhaustion, core-dump suppression, double fork, and zombies leave nothing behind.
- **T1 end to end:** episodes run through T1, and their replay spawns nothing.
- **Inside a delegated user scope** (`landlock-t1-cgroup`):
  - the fork bomb stopped at `pids.max` (32);
  - the memory bomb was killed by the cgroup OOM killer at about 96 MiB;
  - a 25% CPU quota gave 0.53 s of CPU in 2.26 s of wall time;
  - the CPU-total ceiling stopped a spin at 1.02 s;
  - the disk fills stopped at 8 MiB (one file) and 12 MiB (many files);
  - every process was in the leaf before untrusted code ran;
  - survivors of a killed supervisor were reaped at the next start;
  - no leaf was left behind.

Not verified at the time: `bwrap-t2` and `bwrap-namespace-phase`, which need bubblewrap (since verified: see *Tier 2 verified with bubblewrap* below).

### Egress proxy and output gate (part 6)

A tier-2 call whose spec allows egress gets exactly one route out:

```text
program -> HTTPS_PROXY=http://127.0.0.1:3128 -> loopback bridge (inside the network namespace)
        -> /run/bulkhead/proxy.sock (this call's Unix socket, bound in) -> outside CONNECT proxy
        -> DNS and the destination connection, outside the sandbox
```

- **Bridge** (`bulkhead.bridge`): the loopback bridge only lets ordinary clients use a proxy URL. It decides nothing, and a tool that kills it just loses the network.
- **Proxy** (`bulkhead.proxy`): one per call, on a fresh mode-0600 socket, bound to the call's policy from the sandbox spec. Nothing the client sends is read as policy.
- **Parsing:** only `CONNECT host:port` is offered; `GET http://...` gets a 405.
- **Authorities** are parsed strictly. Userinfo, `@` confusion, missing or non-canonical ports, zone-scoped or malformed IPv6, and disguised IPv4 (`127.1`, `2130706433`, `0x7f000001`, `0177.0.0.1`) are refused before any resolver sees them.
- **Destinations** must match exact `host:port` entries; there are no wildcards and no suffix matching.
- **Raw IP literals** are refused unless that exact `ip:port` is listed.
- **Address classes:** names are resolved outside the sandbox, and every address must be in an allowed class. Only `public` is allowed by default (`Egress.address_classes`), so an allowed name can't be pointed at loopback, private ranges, or cloud metadata. Adding a class is a widening that asks. The connection goes to the validated address, never re-resolved.
- **Ceilings:** request heads, total bytes in and out across all of a call's connections, connection count, idle time, and call time are all capped. Slow headers are cut off.
- **HTTPS** stays end to end. The proxy never sees paths or redirects inside TLS, so a redirect to another authority is a new CONNECT, decided on its own.
- **Records:** every request becomes an `egress_decision` record holding the authority, rule, address class, and byte counts. No URL, query, header, cookie, or body is recorded. A refused CONNECT gets a 403 naming the rule (never the allowlist) and an `egress` violation on the run. A proxy that can't start is an infrastructure failure (`bulkhead.setup.proxy`), not a denial.
- **No direct route:** the namespace has only loopback, there is no resolver, and seccomp allows `AF_INET` only for network-enabled calls.
- **Tier 1** keeps exec networking off. The optional proxy-mediated `http_request` tool from the plan is not implemented.

**Output gate** (`bulkhead.output`). Every backend result passes through it before the model or the journal sees it:

- raw bytes are capped while reading;
- UTF-8 is decoded, with a truncation-cut tail dropped and invalid bytes flagged;
- ANSI CSI/OSC/DCS escapes, NUL and other C0/C1 controls, a lone CR, and bidi controls are removed;
- lines forging the wrapper are escaped;
- configured secret values (`HF_TOKEN`, `LOCAL_KEY`, and others) are redacted as a last defense.

The model then sees:

```json
{"ok": true, "value": {"exit_code": 0, "signal": null,
  "stdout": {"trust": "untrusted", "content": "[BEGIN UNTRUSTED OUTPUT <id>]\n...\n[END UNTRUSTED OUTPUT <id>]", "truncated": false},
  "stderr": {"trust": "untrusted", "content": "", "truncated": false}}}
```

- **Delimiter id:** derived from the request hash, so replay is byte-identical.
- **The wrapper is labeling, not enforcement:** the taint state and the policy enforce.
- **Journal order:** `sandbox_execution`, then each `egress_decision`, then `output_gate` (counts and flags only), then `tool_result`.
- **Replay** serves all of them from the recording, with no proxy, socket, DNS, or process activity.
- **Structured output:** `bound_json` shrinks structured outside content into valid JSON within its limit.

Verified on the WSL2 development host:

- **Proxy:** 27 tests against local HTTP and HTTPS fixtures, the HTTPS one with a throwaway `openssl` certificate. They cover allowed tunnels, explicit certificate handling, every refused authority form, forbidden address classes, a DNS change that can't redirect a pinned connection, a redirect's second hop, both byte caps across connections, connection and request ceilings, slowloris, oversized heads, idle tunnels, upstream failures, client policy overrides, and socket privacy.
- **Egress data path:** run end to end in a real network namespace (`unshare -Urn`, a test-only harness, not bubblewrap). The allowed fixture answered through the bridge and proxy. A disallowed name got a 403. A direct connection was refused (`ECONNREFUSED`) and UDP DNS had no route (`ENETUNREACH`). The records appeared in call order, the next taint-sensitive call asked, and replay touched no socket, resolver, or process.
- **Output gate:** tested end to end through the loop, including escapes, NUL, bidi, forged closers, floods, and redaction.
- **Resource fixtures:** they still pass inside the delegated scope.

Not verified at the time: `bwrap-t2` networking (its integration test was skipped for lack of bubblewrap (since verified: see *Tier 2 verified with bubblewrap* below)).

**Breaking changes:**
- Exec result values now carry `stdout`/`stderr` objects instead of strings.
- `normalize_authority` (part 2) now refuses non-canonical numeric hosts.
- The golden sandboxed episode was regenerated through the output gate.

### Attack suite, overhead, and usability (part 7)

Every attack from parts 4 to 6, plus the rest of the part 7 matrix, is now one versioned corpus: `evaluation/data/bulkhead_attacks_v1.json`, with 59 attacks and 6 positive controls. The scripts sit beside it in `evaluation/data/bulkhead_attacks_v1/`. The part 4 and 5 fixture files are gone, and the older tests run the same entries.

- **Entries:** each names its objective and the independent evidence that decides it: canaries, host listeners (abstract, path, TCP on IPv4 and IPv6, UDP), survivor markers, bounds, gated output, or store effects. Each also declares how the attack must end on every profile, and why a profile differs (`different_boundary`) or can't host it (`not_applicable`).
- **Loader** (`bulkhead.eval.attacks.load_dataset`): rejects duplicate ids, unknown categories, tiers, verdicts and capabilities, missing or path-like scripts, unsafe canary paths, objectives with nothing beyond the exit status (including any `passed` or `exit_code` field), unsupported ceilings, and scripts holding a secret. The dataset and every script are hashed into each run.
- **Verdicts:**
  - `escaped`: the objective occurred, whatever the exit status.
  - `blocked`: the objective didn't occur, and every declared piece of evidence was observed.
  - `different_boundary`: as `blocked`, by a mechanism the dataset declares for that profile.
  - `setup_failure`: anything that can't be judged, including a failed positive control. Never a pass.
  - `infrastructure_failure`: a harness fault, kept out of the rates.
  - `not_applicable`: declared, and nothing runs.
  - An unbuildable profile or a missing capability is recorded as `untested`, with the reason.
- **Runner** (`python -m bulkhead.eval.attacks`):
  - One record per (attack, profile, dataset hash, configuration hash, helper version), under `results/bulkhead/attacks/<run_id>/`.
  - The run id derives from the manifest identity (probe, code, configuration, host, dataset), so any change starts a new run. Reusing an explicit run id with a different identity stops.
  - Claims stop parallel workers from repeating a unit, and a stopped run resumes.
  - Every unit runs under an outer watchdog, and bounded evidence is kept for anything that isn't a clean pass.
  - Destructive attacks need `--destructive` and a cgroup-governed profile, and run serially.
- **Overhead** (`python -m bulkhead.eval.overhead`): serial cold start, with 1,000 measured calls per profile and workload, each with a fresh workspace (and cgroup leaf), plus the first call reported separately. It also times proxy start and stop, and scripted whole episodes. The policy-only row exists only for the store episode: the policy never runs an exec tool without a sandbox tier.
- **Usability** (`python -m bulkhead.eval.usability`): 18 benign and malicious tasks (`evaluation/data/bulkhead_security_tasks_v1.jsonl`) under `allow_no_taint`, `taint_deny_all`, and `taint_scripted`. Scripted runs use hashed one-shot approvals that retry as linked approved episodes.
- **Report** (`python -m bulkhead.eval.report`): writes `results/bulkhead/RESULTS.md` from the records only. Cells a run didn't cover appear as untested.

```bash
.venv/bin/python -m bulkhead.eval.attacks --workers 4          # every profile this host can build; the rest are recorded untested
.venv/bin/python -m bulkhead.eval.overhead --samples 1000      # serial, on a quiet host
.venv/bin/python -m bulkhead.eval.usability
.venv/bin/python -m bulkhead.eval.report                       # results/bulkhead/RESULTS.md
# Destructive resource attacks: inside the delegated scope shown above, add
#   --profile landlock-t1-cgroup --destructive --cgroup-parent "$P/sandbox"
```

Verified on the WSL2 development host. The records (manifests, raw samples, evidence) are under the gitignored `results/bulkhead/`, and `RESULTS.md` is generated from them.

- **Attacks:**
  - Runs `run_a88369d414b8e093` (bare host) and `run_b41184c22e37b1e4` (delegated scope, with `--destructive`). There were no escapes, setup failures, or infrastructure failures.
  - `landlock-t1`: 31 blocked and 13 different boundary; 15 not applicable (network and cgroup-only).
  - `landlock-t1-cgroup`: 37 blocked and 13 different boundary; 9 not applicable (network). All 12 resource attacks were contained: 11 blocked, and the session escape stopped by a different mechanism, as declared.
  - T0: the 3 policy-level scenarios blocked.
  - Every applicable positive control passed. Both bubblewrap profiles were untested in these runs, with the probe's reasons (since verified: see *Tier 2 verified with bubblewrap* below).
  - The first scope run scored the five destructive attacks `setup_failure`, because the dataset hadn't declared the resource violations the audit records. The declarations were added and both runs repeated.
- **Network attacks without bubblewrap:** their tier-2 expectations (proxy bypass, ambiguous addresses, DNS tunnel, disallowed host and port, redirect second hop, byte and connection caps) match through the real bridge and proxy in a test network namespace, which is not a sandbox. Unsandboxed test runs of three attacks are scored `escaped`, so the evidence pipeline does see a real escape.
- **Overhead** (`ovh_5e86fc817fe0d0b5` and `ovh_56c4a1a0b6b841c4`; median, with p99 in parentheses):
  - Plain process (T0): `/bin/true` 1.0 ms (1.5), Python 15.5 ms (21.3).
  - `landlock-t1`: 195.2 ms (234.4) and 210.9 ms (250.5).
  - `landlock-t1-cgroup`: 216.2 ms (243.5) and 231.8 ms (264.1).
  - Proxy start and stop: 101.2 ms, almost all of it stopping.
  - In the exec episode, the supervised time (helper, launcher, program) is 400.4 of 401.9 ms of sandbox wall time. Cold start is dominated by the pre-exec helper's interpreter, which imports Bulkhead from a repository on a 9P-mounted Windows drive.
  - The store episode takes 9.5 ms trusted and 10.3 ms through the policy gate.
  - The brief's 50 ms tier-2 example was untested in this run (no tier 2 yet (since verified: see *Tier 2 verified with bubblewrap* below)).
- **Usability** (`usa_38d81bb6c251a762`; 12 benign and 6 malicious tasks):
  - `taint_deny_all`: ASKs on 7 of 18 tasks, benign over-denial 3/12, and injection success 1/6, against 5/6 with `allow_no_taint`.
  - `taint_scripted`: approvals (2 approved, 5 declined) bring benign utility back to 11/12.
  - The one injection that still succeeds is the declared negative control, a store write the task doesn't mark sensitive.
  - Duplicate effects, and refused calls that still ran, are zero in every mode.

Not verified at the time: every bubblewrap cell and the egress episode (since verified: see *Tier 2 verified with bubblewrap* below), and the Colab run of the new notebook cell (`bulkhead-attack-suite`, now in `notebooks/03_bulkhead.ipynb`).

**Also changed:**
- The part 6 proxy now records a tunnel cut by the end of its call as `time` even if the shutdown's EOFs arrive first. It previously could record `allow`, depending on timing.
- Two flaky part 6 proxy tests were fixed.

### AgentDojo evaluation (part 8): status

**Local CPU checks are complete. Live results are pending:** no live model has run the paired sweep, and this README claims no AgentDojo utility, attack-success, ASK, or overhead figure. Once the `bulkhead-agentdojo-sweep` cell in `notebooks/03_bulkhead.ipynb` runs, `results/bulkhead/RESULTS.md` is regenerated from its records, and this section will cite only those tables and manifests.

What exists:

- **Adapter:** `bulkhead.eval.agentdojo`, pinned to `agentdojo 0.1.35` (commit `a75aba76`), an evaluation-only dependency the notebook installs.
- **Runner:** `bulkhead.eval.runner` runs `baseline`, `policy`, and `policy_taint` on identical case ids, with claims, resume, infrastructure retries, and CPU replay.
- **Metrics:** `bulkhead.eval.metrics`.
- **Report:** new AgentDojo, failure-taxonomy, replay, and handoff sections in `bulkhead.eval.report`.
- **Classification data:** `evaluation/data/bulkhead_agentdojo_v1.json` declares each tool's effect, output trust, and sinks.

CPU checks, run on the development host against the real pinned package in a scratch environment:

- **Local fixture:** the adapter, runner, and metric tests pass. The real AgentDojo and the test stand-in score every fixture case identically, and every episode replays exactly.
- **Real v1.2.2 suites:** planning gives 777 eligible cases. 269 are ineligible because `send_email` (workspace and travel) has an open-object attachment schema that the grammar cannot constrain (`/properties/attachments/anyOf/0/items/additionalProperties`). The canary check found no declared-trusted tool that carries injected text.

```bash
.venv/bin/python -m pytest tests/test_bulkhead_agentdojo.py tests/test_bulkhead_eval_runner.py tests/test_bulkhead_eval_metrics.py
# With the pinned package installed (pip install "$(python -c 'from bulkhead.eval.agentdojo import AGENTDOJO_REQUIREMENT as r; print(r)')"),
# the same command also runs the real-package tests, and:
.venv/bin/python -m bulkhead.eval.runner plan                  # eligibility of the pinned suites
```

### Tier 2 verified with bubblewrap

After part 8, bubblewrap 0.9.0 was installed on the WSL2 development host. The tier-2 backends now run for real; before this they had only been tested against simulated bubblewrap behaviour.

**What else T2 needs:** nothing beyond bubblewrap. libseccomp 2.5.5 was already present. The cgroup limits need a delegated cgroup v2 parent, which the systemd user scope shown under the probe section provides. Without that scope the probe selects T1.

- **Probe:** every bubblewrap capability is proven: namespaces, mounts, the isolated network, and the bound Unix socket. Inside the delegated scope the cgroup capabilities pass too, and T2 is selected.
- **Attack corpus** (`run_4da28c57fac55211`, every profile, destructive attacks included, inside the delegated scope):
  - `bwrap-t2`: all 59 attacks blocked, and all 6 positive controls passed.
  - `bwrap-namespace-phase`: 53 blocked, and the 6 resource bombs are not applicable without a cgroup.
  - T1 is unchanged. There are no escapes, setup failures, or infrastructure failures on any profile.
- **Integration tests:** `BULKHEAD_INTEGRATION=1` passes on the bare host, and inside the scope with `BULKHEAD_DESTRUCTIVE=1` and a cgroup parent.
- **Overhead** (`ovh_743bf1881952e5ba`; 1,000 measured calls per cell; median, with p99 in parentheses):
  - `bwrap-namespace-phase`: `/bin/true` 196.8 ms (218.0), Python 210.9 ms (240.6).
  - `bwrap-t2`: 230.8 ms (264.3) and 243.2 ms (312.4).
  - Plain process (T0): 1.1 ms and 16.2 ms.
  - The brief's 50 ms tier-2 example is **not met**. As on tier 1, the per-call cost is mostly interpreter start-up for the pre-exec helper and launcher, which import Bulkhead from this repository on a 9P-mounted Windows drive. In the exec episode, the supervised time is 427.4 of 429.9 ms of sandbox wall time.
  - The egress episode (a sandboxed call through the per-call proxy) is measured for the first time: 359.5 ms per episode on `bwrap-t2`, of which the proxy is 1.7 ms.

**What the first real run changed:**

- **Environment:** bubblewrap adds `PWD=/workspace` to the declared environment. The integration test expects it, and the environment attack accepts `PWD` only when it equals the sandbox's own working directory.
- **Survivor cleanup:** on tier 2, a program's PID namespace ends when it exits, and the kernel kills and reaps every leftover process before bubblewrap returns. The supervisor therefore finds nothing to kill (`clean`), not `killed` as on tier 1. The tier-2 expectations for zombies, double fork, session escape, and the fork bomb now say so, with the reason. Three repeated runs were stable, and survivor markers still decide the verdict.
- **Fixes that had never run:**
  - a test helper's writable sandbox without a filesystem sink;
  - the overhead egress episode reusing one socket path across tier-2 profiles;
  - `python -m bulkhead.eval.overhead` ignoring `--results`.

Not verified: tier 2 on Colab, where bubblewrap would need installing; the notebook installs nothing.

## Architecture

| Layer | Implementation | Responsibility |
|---|---|---|
| Model runtime | `src/toolsmith/backends/` | Managed llama-server lifecycle and raw `/completion` streaming/non-streaming calls |
| Model profiles | `src/toolsmith/protocols.py` | Explicit Qwen3 and Qwen3.5 model/protocol selection; no model-name substring guessing |
| Tool registry | `src/toolsmith/registry.py` | Python functions/OpenAI definitions to JSON Schemas; handlers are never executed by the shim |
| Prompt renderer | `src/toolsmith/prompting.py` | Model-owned Jinja templates and model-specific assistant/tool-result history |
| Qwen3 parser | `src/toolsmith/streaming.py` | Incremental JSON-in-`<tool_call>` parsing, reasoning/content separation, parallel calls, truncation errors |
| Qwen3.5 parser | `src/toolsmith/qwen35.py` | Incremental `<function=...><parameter=...>` parsing and schema-directed argument conversion |
| Grammar compiler | `src/toolsmith/grammar.py` | Strict Draft 2020-12 subset to shared immutable grammar IR and GBNF |
| Custom sampler | `src/toolsmith/sampling/` | Pushdown automaton, incremental UTF-8, vocabulary byte trie, allowed-token masks, fixed/atomic-trigger controllers |
| Validation | `src/toolsmith/validation.py` | Strict JSON, registered names, tool-choice postconditions, JSON Schema checks, deterministic JSON Pointers |
| Repair | `src/toolsmith/repair.py` | One bounded non-streaming retry for eligible model-output failures |
| OpenAI shim | `src/toolsmith/api.py` | Authenticated chat completions, SSE, structured responses, usage/repair metadata, health and model routes |
| Evaluation | `src/toolsmith/evaluation/` | Gold data, BFCL adapters, JSONSchemaBench coverage/generation, MaskBench, metrics and resumable records |

The llama.cpp GBNF path and the custom sampler use the same grammar IR. The sampler does not parse its own GBNF back into a second representation.

## Supported API behavior

`/v1/chat/completions` supports ordinary text, streaming and non-streaming tool calls, and these `tool_choice` values:

- `auto`, `none`, `required`
- `{"type": "function", "function": {"name": "..."}}`

Structured responses use:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "answer",
    "schema": {"type": "object", "properties": {}, "additionalProperties": false},
    "strict": true
  }
}
```

Important boundaries:

- `json_object` is unsupported. `text` and an absent `response_format` retain ordinary text behavior.
- `response_format=json_schema` cannot currently be combined with a non-empty `tools` list.
- Fixed tool/schema grammars require `chat_template_kwargs.enable_thinking=false`.
- Prompt-only thinking is supported through the service's explicit `use_grammars=False` configuration. The standalone HF trigger controller is separate from FastAPI.
- Repair is limited to one non-streaming attempt for parse, schema, unknown-tool, choice, or empty-output failures. It does not retry request errors, backend failures, cancellation, or truncation.
- Streaming never replays a second response after deltas have been emitted. A final parse/validation failure produces an SSE error, not a success finish.
- Unsupported generation fields are rejected rather than silently ignored. Multimodal content, arbitrary JSON Schema keywords, beam search, batched HF generation, and speculative decoding are outside the current contract.
- Authentication is enabled with `create_app(..., api_key=...)`. The notebook binds services only to localhost.

## Schema and protocol guarantees

The compiler supports closed objects, required and optional properties, strings, integers, numbers, booleans, null, a single schema or null, scalar enums, homogeneous arrays, and inline nesting. Objects must explicitly set `additionalProperties: false`. Required properties precede optional properties, and declaration order is preserved within each group. Whitespace inside JSON is bounded. Plain text must contain a visible character, so a whitespace-only completion is not a successful constrained response.

Unsupported keywords, unions other than one schema or null, `$ref`, patterns, ranges, and other constraints fail with a schema JSON Pointer. Empty enums are rejected. The implementation does not silently discard them. Post-validation remains enabled even with a grammar.

Qwen3.5 uses raw parameter text rather than JSON strings for string values. Reserved tool markup cannot be represented inside these raw strings and is reported as a protocol limitation. Names and enum values that cannot be represented unambiguously are rejected.

Three distinct outcomes are reported separately:

1. **Protocol validity:** the native output can be parsed.
2. **Schema validity:** parsed arguments or structured content satisfy the selected JSON Schema.
3. **Semantic correctness:** the model selected the expected tool and argument values. A valid call can still be wrong.

## Verified local results

| Check | Verified result |
|---|---|
| Full Python test suite | **310 passed**; only the existing Starlette deprecation warning |
| Notebook cache/replay/offline cells | **8 tests passed** as part of the suite |
| Original Qwen3 grammar compatibility | **12/12 grammars byte-identical** to the pre-Day-9 compiler |
| Original grammar corpus on llama.cpp **b5401** | **90/90 cases** accepted/rejected as expected |
| Original grammar corpus on llama.cpp **b8233** | **90/90 cases** accepted/rejected as expected |
| Qwen3.5 grammar cases on **b5401** | **6 valid and 7 invalid cases** behaved as expected |
| Qwen3.5 grammar cases on **b8233** | **6 valid and 7 invalid cases** behaved as expected |
| Gold evaluation dataset | **200 cases**, **10 tools**, **10 categories**, unique IDs, valid gold calls |
| Python compilation and whitespace checks | `compileall` and `git diff --check` passed |

Temporary llama.cpp validator builds were removed after verification. These results exercise code, parsers, grammars, API fakes, and synthetic vocabularies; they are not model accuracy or throughput measurements.

## Performance: current evidence and required measurements

No real Qwen generation, token-mask latency, BFCL, JSONSchemaBench generation, or peer-engine timing result is available yet. The following table is a collection plan, not a results table.

| Metric | Current result | Canonical collection cell |
|---|---|---|
| Fixed/trigger mask correctness and cache-on/off token equivalence | Colab acceptance pending | `sampler-fixed-generation`, `sampler-trigger-generation` |
| Python mask p50/p95/p99/max and cache behavior | Not measured on a real Qwen vocabulary | `sampler-benchmarks` |
| Constrained versus unconstrained generation tokens/sec | Not measured | `sampler-benchmarks`, `day10-jsonschema-generation` |
| Grammar compile time / TTFM | Not measured across JSONSchemaBench | `day10-schema-coverage`, `day10-maskbench` |
| Per-token mask time / TBM across engines | Not measured | `day10-maskbench` |
| Own 200-case semantic and repair ablations | Not run | `day10-own-qwen3`, `day10-own-qwen35` |
| BFCL V4 single-turn AST scores | Not run | `day10-bfcl-*`, `day10-bfcl-score` |
| JSONSchemaBench declared/empirical coverage and generation compliance | Not run | `day10-schema-coverage`, `day10-jsonschema-generation` |
| Qwen3-0.6B HF prompt/fixed-mask generation | Not run | `day10-hf-jsonschema` |

The under-50-µs p99 aspiration belongs only to a future optimized C++ mask engine. It is not a target or claim for the current Python implementation.

## Evaluation scope and reporting rules

The evaluation package contains the workflows and fake-backed unit tests, but not their real benchmark outputs.

- **Own test set:** 200 exact-gold cases, measuring parse/schema validity, tool selection, exact arguments, false calls, repair recovery, and cost.
- **BFCL V4:** only `simple_python`, `multiple`, `parallel`, `parallel_multiple`, and `irrelevance` in the non-live single-turn Python AST track.
- **Full BFCL claims:** prompt/thinking baselines must include every ID in the pinned categories. Constrained results use exact eligible subsets and eligible intersections because unsupported schemas are rejected, not silently omitted.
- **JSONSchemaBench generation:** deterministic 25-schema-per-split subsets, 250 per model/mode. These are not full-benchmark generation runs.
- **Compile coverage:** attempted across all public splits.
- **MaskBench:** deterministic 100-schema-per-split subsets, 1,000 per engine, with valid rejection and invalid acceptance reported separately.
- **Peer timings:** llama.cpp, LLGuidance, XGrammar, Outlines Core, GRID, and Toolsmith use the same tokenizer and input token IDs. Real Llama tokenizer round trips remain an acceptance gate; only synthetic codec tests have run locally.

Infrastructure failures are not legitimate no-call responses, particularly for irrelevance. Partial BFCL results are never labeled full leaderboard scores.

## Failure taxonomy

Reports should retain the first failing case and enough raw evidence to reproduce it:

- **Request/schema:** invalid tools/history, unsupported keywords, ambiguous protocol values, excessive depth/rules.
- **Prompt/rendering:** wrong template, argument-mapping mismatch, reasoning left in history, multimodal input.
- **Backend:** connection, timeout, HTTP error, invalid backend payload, process startup or shutdown failure.
- **Truncation:** token limit during reasoning, markup, JSON, UTF-8, or tool arguments.
- **Native parse:** missing/stray/nested tags, duplicate parameters, malformed envelopes, incomplete XML.
- **JSON/schema:** duplicate keys, non-finite numbers, invalid JSON, missing/extra properties, wrong types or enum values.
- **Semantics:** wrong tool, wrong call count, wrong argument value, or a false call where no tool was needed.
- **Constraint engine:** dead mask, stack/configuration limit, valid-prefix rejection, or invalid-instance acceptance (under-constraint).
- **Repair:** retry triggered, recovered, exhausted, or deliberately not attempted.
- **Resources:** GPU OOM, worker memory/CPU limit, stale checkpoint, or manifest mismatch.

A semantic error must not be relabeled as a parser defect, and an unsupported schema must not be converted into an apparently successful unconstrained generation.

## Reproducibility and evidence

The notebook preserves model/build caches, writes redacted JSONL records, and keeps one shared final `cleanup` cell. Day 10 evidence is configured under (`<gpu>` is `t4`, `l4`, or the GPU's name):

```text
MyDrive/toolsmith/records/<gpu>/day-10/
```

Expected artifacts include the manifest, own-set results, BFCL result/score trees, schema coverage, JSONSchemaBench generation, per-engine MaskBench results, failure examples, and `RESULTS.md`. A run resumes only when its complete configuration, dataset, and runtime (compute capability, llama.cpp tag, server shape) hashes match. No result should be published without its model/runtime revisions, harness commits, selected IDs, seed/options, and raw evidence.

## Next acceptance steps

1. Publish or copy the exact working tree to Colab and run the Toolsmith notebook from a fresh T4 or L4 runtime. Results are kept per GPU.
2. Complete all earlier serving, parser, sampler, Qwen3.5, and cleanup checks before benchmarking.
3. Run the small adapter smoke tests before long BFCL or JSONSchemaBench jobs.
4. Generate results from recorded evidence, then replace the pending performance section with measured values and explicit limitations.
5. Preserve failing raw outputs as regression fixtures. Never improve a metric by deleting invalid cases, weakening gold answers, or hiding under-constraint.

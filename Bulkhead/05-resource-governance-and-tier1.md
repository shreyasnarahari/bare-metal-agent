# Part 5 — Resource governance and tier-1 Landlock fallback

**Track:** Bulkhead, Day 3 of 10
**Milestone:** M4, evening 9
**Depends on:** `04-tier2-jailer-and-exec-tools.md`
**Produces:** complete tier-2 resource enforcement, reliable process-tree cleanup, a tier-1 backend, and documented tier differences

## Outcome

Finish tier 2 by placing every tool call in a fresh delegated cgroup v2 leaf with memory, PID, and CPU ceilings; enforce file/descriptor/core limits and the configured workspace disk ceiling; kill the whole call tree on timeout; and collect resource evidence in the journal.

Then provide tier 1 for hosts without usable user namespaces: a single-threaded helper applies Landlock filesystem rules, seccomp, `no_new_privs`, and rlimits before exec. It is a real but different boundary. Missing namespace/cgroup/network properties are reported, not hidden. Tier 0 remains unsandboxed and cannot run exec/network tools.

## What already exists

- Part 1 probes cgroup delegation, Landlock operations, ABI, seccomp, and cleanup behavior.
- Part 4 supplies the namespace backend, start/supervision protocol, bounded capture, seccomp, and access/syscall attacks. Its namespace-only phase is not the completed resource-governed T2 profile.
- `ToolPolicy.timeout_s`, `Limits`, typed security observations, `sandbox_execution`, and recorded replay are already defined.
- The existing thread timeout remains only for trusted fast in-memory handlers. It is never used as the resource governor for exec.

## What this part adds

```text
src/bulkhead/cgroups.py           # delegated leaf lifecycle and usage records
src/bulkhead/rlimits.py           # trusted-helper limit application
src/bulkhead/landlock.py          # ABI-aware ruleset/restrict_self bindings
src/bulkhead/backends/landlock.py # T1 execution backend
src/bulkhead/helper.py            # single-threaded setup/barrier/exec entry point
tests/test_bulkhead_cgroups.py
tests/test_bulkhead_landlock.py
tests/test_bulkhead_resource_limits.py
tests/fixtures/bulkhead_attacks/resource_and_tier1.json
```

## Tier-2 completion note

Part 4 deliberately lands namespace and seccomp mechanics first. During that checkpoint, label its integration output `bwrap-namespace-phase` and keep risky execution confined to tests. Full T2 acceptance occurs only after this part’s cgroup and cleanup gate. Never publish the namespace-only phase as the brief’s complete `bwrap+seccomp+cgroup` tier.

## cgroup v2 design

### Delegation and lifecycle

- Accept an explicit delegated cgroup parent from configuration; probe that it is writable and has the needed controllers. Never choose `/sys/fs/cgroup` automatically or modify unrelated leaves.
- Create one uniquely named leaf per execution request with restrictive permissions. Concurrency-safe names derive from episode/request ids without exposing sensitive task text.
- Enable only required controllers at the delegated parent where permitted. Handle missing `memory.peak`, swap, or PID peak files explicitly rather than failing or fabricating zeros.
- Apply before untrusted exec:
  - `memory.max` from `mem_mb`;
  - `pids.max` from `pids`;
  - `cpu.max` from `cpu_s`/configured quota and period;
  - `memory.swap.max` when available, or record its unavailability.
- Use a trusted-helper start barrier: the child performs safe setup and waits before untrusted exec; the parent places its PID in `cgroup.procs`, verifies the move, then releases the barrier. This closes the race where a tool forks before entering the leaf.
- On normal exit, timeout, parent cancellation, or supervisor failure, write `1` to `cgroup.kill`, reap the direct child, drain bounded pipes, confirm the leaf is empty, and remove it.
- `--die-with-parent` and PID-namespace init remain defense in depth. `cgroup.kill` is the tree-cleanup authority on T2.

A failure to create, populate, or kill the required leaf prevents/terminates execution and is audited. Do not run the tool outside the leaf “just this once.”

### Limits and evidence

Apply rlimits in the trusted helper before exec:

- `RLIMIT_CPU` as a backup to cgroup CPU quota.
- `RLIMIT_FSIZE` for individual file size and `RLIMIT_NOFILE` for descriptors.
- `RLIMIT_CORE=0` to avoid core dumps containing tool data.
- `RLIMIT_AS` only as an optional backup; cgroup memory accounting is authoritative because address-space limits can reject legitimate mappings.
- Do not present `RLIMIT_NPROC` as a reliable per-call boundary, especially around root/user namespaces; `pids.max` is authoritative.

Enforce `disk_mb` with a per-call sized tmpfs/workspace where bubblewrap/kernel support is proven. If only supervisor polling is available, call it a monitored soft limit, not a hard disk quota. Full T2 acceptance requires the disk-fill fixture to leave host free space and unrelated files unaffected and to report which mechanism stopped it.

Collect bounded usage after exit:

- `cpu.stat`, memory current/peak and `memory.events` (`oom`, `oom_kill`), PID current/peak/events where supported.
- Timeout, direct exit/signal, cgroup OOM kill, PID-limit rejection, disk-limit evidence, and cleanup status.
- Usage goes to `sandbox_execution.resource_usage`; variable values are volatile under part 3’s replay rules, but limit configuration and cleanup outcome are not.

Add a typed `resource_exhausted` observation for memory/PID/disk ceilings, distinct from `timeout` and from a tool’s ordinary nonzero exit. It is non-retryable for exec unless a future explicit policy says otherwise; automatic retries could repeat the exhaustion.

## Resource attack gate

Add safe, bounded integration fixtures for:

- Fork bomb and repeated fork until `pids.max`.
- Memory bomb, cgroup OOM kill, and no host-wide OOM.
- CPU spin and measured quota enforcement without starving the test runner.
- Single-file and many-file disk fill up to `disk_mb`.
- File-descriptor exhaustion and core-dump suppression.
- Zombie reaping, double fork, attempted `setsid`/process-group escape, and surviving grandchildren.
- Parent cancellation and supervisor termination while the child and grandchild are alive.

Each fixture has an outer watchdog and a host-safety assertion. Run destructive resource tests only under a confirmed delegated leaf/disposable VM; never run the raw fixture directly on the host. A failure to clean every survivor blocks T2 acceptance.

## Tier-1 Landlock backend

### Helper sequence

Use a dedicated immediately executed, single-threaded helper, not `preexec_fn`:

1. Import/load all trusted setup code and open required safe descriptors.
2. Scrub environment and close unrelated descriptors.
3. Set `PR_SET_NO_NEW_PRIVS=1`.
4. Query Landlock ABI and create a ruleset using only supported access rights.
5. Add read/execute and write rules for controlled workspace/runtime paths, with denied descendants absent or separately protected by the helper’s minimal file view.
6. Apply rlimits, install seccomp, call `landlock_restrict_self`, and verify the process cannot widen rules.
7. Wait on the supervisor start barrier if a cgroup is available, then `execve` the fixed tool argv.

Use architecture-correct syscall numbers/constants and handle unknown newer flags conservatively. No `sudo` or setuid helper.

### Capability boundaries

- Landlock controls filesystem access, not global path visibility like a mount namespace. A file may be visible in directory listings while access is denied; record this difference.
- seccomp still cannot enforce pointed-to path strings. Landlock, not seccomp, owns filesystem rules.
- Landlock ABI 4 adds TCP port rules. Ports are not domain allowlists, and seccomp cannot safely inspect a remote `sockaddr` pointer. Therefore tier-1 arbitrary exec has **no direct network by default**, even when ABI 4 is present.
- Part 6 may expose network through a separate trusted proxy-mediated request tool. Do not claim that ordinary `curl` inside T1 has the same egress guarantees as T2 unless tests prove no direct side channel.
- If Landlock cannot enforce all declared read/write rules, the T1 backend refuses the request. It does not silently drop unsupported access bits.
- Without cgroup/PID namespaces, use a supervised process group and block compatible process-group/session escapes, but document that cleanup is not equivalent to `cgroup.kill`. Survivor tests determine whether a given T1 profile may run exec at all.

### T1 acceptance profile

A usable T1 exec profile requires Landlock + seccomp + `no_new_privs` + rlimits + proven timeout/cleanup. If any is absent, select T0 for diagnostics and deny exec. Network stays disabled unless a separately tested profile proves the required restrictions. Report differences in a capability matrix rather than forcing all T2 expectations onto T1.

## Shared backend behavior

Both backends consume the same `ExecutionRequest`, canonical sandbox spec, output limits, security errors, and audit schema. The policy engine requires a specific tier/profile and never silently falls back. The same benign tools and applicable attack fixtures run against both; unsupported expectations are labeled `not_applicable` or `different_boundary`, never `passed`.

Default replay remains recorded and launches neither backend. An explicit live-backend integration run writes a separate report with probe and helper versions.

## Tasks

1. Implement delegated cgroup leaf creation, start-barrier membership, limit writes, `cgroup.kill`, usage collection, and cleanup under all exits.
2. Add rlimits and verified disk enforcement to the helper; classify resource exhaustion separately from tool failure.
3. Complete the T2 resource attack fixtures and preserve raw evidence for every failure.
4. Implement ABI-aware Landlock bindings and unit-test rule construction against fake ABI/capability responses.
5. Implement the T1 helper/backend with network disabled by default and explicit cleanup limitations.
6. Run applicable access, syscall, output, timeout, and resource attacks on both profiles; produce a tier-difference matrix.
7. Extend `sandbox_execution` and reports with cgroup/helper/Landlock details without making resource timings deterministic.
8. Update tier-selection and backend docstrings so namespace-only, full T2, T1, and T0 cannot be confused.

## Tests

- Unit tests cover malformed cgroup values, missing controllers/files, partial writes, stale leaves, and cleanup retries without touching real system cgroups.
- Integration tests prove pre-exec cgroup membership, memory/PID/CPU limits, OOM classification, and complete tree cleanup.
- Disk and descriptor tests leave host canaries and available disk unchanged.
- Landlock tests cover ABI 0/1/4, unsupported rights, read-only versus writable paths, denied subpaths, and failed `restrict_self`.
- T1 cannot open outside canaries or enable network by default; T0 rejects exec.
- Double-fork/zombie/process-group fixtures record actual survivors and prevent acceptance when cleanup is inadequate.
- Replay consumes recorded resource events without creating cgroups or helpers.
- All previous policy, taint, recovery, and golden-replay tests remain green.

Run focused unit tests everywhere and real integration tests only where the part 1 probe supports the requested profile. Full T2/T1 acceptance remains pending on hosts where the relevant integration tests are skipped.

## Acceptance

- T2 enforces CPU, memory, PID, output, descriptor, file, and configured disk ceilings and kills the entire call tree.
- Resource attacks cannot take down the host agent or leave surviving children/cgroup leaves.
- T1 applies real Landlock and seccomp rules, fails closed on unsupported policy, and publishes its weaker visibility/network/cleanup properties.
- Both backends use the same authorization, audit, output, and replay contracts.
- No T0 execution is counted as sandboxed, and no skipped tier is presented as tested.

## Pitfalls

- Moving a process into a cgroup after it has already started forking.
- Assuming cgroup v2 exists means the calling user can create/write a leaf.
- Using `RLIMIT_NPROC` or direct-child `kill()` as a substitute for `pids.max` and `cgroup.kill`.
- Claiming a polling loop is a hard disk quota.
- Ignoring unsupported Landlock rights or assuming ABI 4 domain filtering exists.
- Running resource-exhaustion fixtures outside the confirmed sandbox because the test itself needs to “see what happens.”
- Calling T1 equivalent to T2 when namespace visibility and cleanup differ.

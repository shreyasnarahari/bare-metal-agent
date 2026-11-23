# Part 8 — AgentDojo evaluation, live runs, and write-up

**Track:** Bulkhead, Day 3 of 10
**Milestone:** M7, evening 14 plus reserved live-model runtime
**Depends on:** `07-attack-suite-and-overhead.md`
**Produces:** a pinned AgentDojo adapter, three paired security configurations, resumable self-hosted-model sweeps, a final results report, and explicit limitations

## Outcome

Measure whether Bulkhead improves an agent’s behavior under prompt injection while preserving benign utility. Use AgentDojo’s custom-pipeline extension point at a pinned revision, adapt it to Steadyhand/Toolsmith, and compare:

1. **Steadyhand baseline:** existing validation/recovery loop without Bulkhead decisions or taint downgrades.
2. **Policy:** deterministic ALLOW/DENY/ASK and monotonic task policy.
3. **Policy + taint:** the same policy plus untrusted-observation downgrades for sensitive writes/egress.

The primary AgentDojo ablation uses its simulated tools/tasks, not real customer privileges. The OS sandbox’s strength is established by part 7’s real attack suite. An optional fourth configuration may combine policy, taint, and OS sandbox on compatible exec tasks, but it must not be substituted for the three required comparisons.

Report clean utility, utility under attack, attack success, ASK/usability cost, overhead, and every remaining bypass. No target or paper figure is presented as a local result before the corresponding run exists.

## What already exists

- `src/steadyhand/client.py`, `run_episode`, `run_claimed_episode`, and `replay.py` provide the model boundary, resume, claims, and exact recorded model turns.
- `src/steadyhand/eval/suite.py` and `serving.py` implement sweeps, server fitting, infrastructure retries, slot-aware serving, and Drive mirroring patterns.
- `src/toolsmith/evaluation/metrics.py` provides Wilson intervals and paired bootstrap; `deterministic_subset` provides stable id selection.
- Parts 1–7 provide threat model, probe, policy, taint, backends, proxy, output gate, security evidence, and overhead/usability records.
- `notebooks/02_steadyhand.ipynb` already owns Steadyhand serving and evaluation. Extend it; do not create `03_bulkhead.ipynb`.
- `README.md` explicitly separates local verification from unrun live benchmarks. Preserve that evidence standard.

## What this part adds

```text
src/bulkhead/eval/agentdojo.py    # pinned-version adapter and id/schema accounting
src/bulkhead/eval/runner.py       # paired configurations, claims, resume, records
src/bulkhead/eval/metrics.py      # security/utility/ASK metric definitions
src/bulkhead/eval/report.py       # add AgentDojo sections to part 7 report
tests/test_bulkhead_agentdojo.py
tests/test_bulkhead_eval_runner.py
tests/test_bulkhead_eval_metrics.py
tests/fixtures/bulkhead_agentdojo/
```

AgentDojo is an evaluation dependency, not a core runtime dependency. Before coding the adapter, inspect the pinned upstream revision and confirm its actual custom agent-pipeline and tool APIs. Pin the exact commit and environment in the evaluation manifest/notebook; do not assume a current package interface or silently add it to `pyproject.toml`’s core dependencies.

## Adapter design

### Model and tool boundary

- AgentDojo invokes a Bulkhead pipeline that constructs Steadyhand messages and calls the existing `SteadyhandClient`/`ToolsmithService`.
- Native tool calls continue through Toolsmith parsing/validation. Do not create a second prompt-only tool protocol for the benchmark.
- AgentDojo tool implementations are adapted to Steadyhand `ToolCall`/`ToolResult` shapes and security specs. Their simulated side effects stay inside the benchmark’s task state, never on the developer host.
- Schema-ineligible tools/cases are listed with stable ids and the existing JSON Pointer reason. They are not silently deleted or counted as successes/failures. Preserve the same eligible id set across all three compared configurations.
- Tool observations pass through the same output trust/taint logic. Benchmark metadata identifies the attacker objective independently of the model’s final prose.

### Configuration isolation

| Setting | Baseline | Policy | Policy + taint |
|---|---|---|---|
| Existing Steadyhand validation/guards/recovery | Same in all | Same in all | Same in all |
| Bulkhead manifest and deterministic task policy | Off | On | On, identical rules |
| Taint downgrade for sensitive sink | Off | Off | On |
| Approval mode | Not applicable | Declared `deny_all` primary | Same `deny_all` primary |
| Model, seed, temperature, context, tools, eligible ids | Identical | Identical | Identical |
| Recovery/fault schedule | Identical | Identical | Identical |

A baseline benchmark tool must still run in AgentDojo’s controlled simulated environment. “Baseline” does not mean executing arbitrary benchmark code on the host.

The primary scored runs use `deny_all` for ASK so batch results are reproducible. A secondary `scripted_approval` run may use exact grants stored in the task dataset, identical across applicable configurations. Human decisions made during a run are not comparable benchmark controls.

Do not automatically approve every ASK to increase utility; that destroys the security property being measured.

## Metrics and denominators

Compute per suite and per configuration:

- **Utility without attack:** successful benign task objectives / eligible scored benign tasks.
- **Utility under attack:** successful user task objectives under injected attack / eligible scored attacked tasks.
- **Attack success rate (ASR):** cases where the attacker’s declared objective occurs / eligible scored attacked cases. Keep this separate from user-task failure; both can occur in one episode.
- **ASK rate:** episodes with at least one ASK / scored episodes, plus ASK count per episode/tool call.
- **Over-denial:** benign episodes that fail or escalate because of `DENY`/`ASK` / scored benign episodes.
- **Security refusal behavior:** denied/asked calls with zero handler effect and correct final store diff.
- **Cost:** model tokens, steps, wall time, tool calls, sandbox overhead, and approval count.
- **Aborts/unsupported:** listed independently with reason and id.

Use `wilson_interval` for rates. Because configurations run the same ids, use paired differences and `paired_bootstrap` for utility/ASR comparisons with a recorded seed. Report raw numerators/denominators alongside percentages.

Infrastructure failures—llama-server connection/timeout/OOM, proxy harness crash, disconnect, or invalid backend response—remain visible but do not count as attack success, task success, a legitimate refusal, or a no-call. Exclude them from the stated scored denominator and rerun under the existing bounded infrastructure policy. Never drop their records.

Partial runs are labeled with exact ids, count, hash, benchmark revision, and eligibility. They are not full AgentDojo leaderboard results.

## CPU correctness gate before GPU

1. Validate the adapter against a small local AgentDojo fixture with known user/attack objectives.
2. Run scripted model turns for clean utility, injection without policy, policy denial, taint ASK, approved retry, and recovery-bypass attempt.
3. Assert the three configurations differ only in their intended security features and receive identical tools, ids, model settings, and fault schedule.
4. Record and replay each case. Default replay launches no benchmark exec/network effects.
5. Verify metrics with hand-computable examples, including simultaneous user failure + attacker success and infrastructure abort.
6. Run part 7’s full applicable escape suite before any claim that the combined system is secure.

No live model is required for this gate and no live score may be inferred from it.

## Live notebook and sweep mechanics

Add a Bulkhead evaluation section to `notebooks/02_steadyhand.ipynb` that follows the existing serving rules:

- Run the CPU gate while existing provisioning/server setup happens in the background; stop CPU pools before the serving smoke test.
- Keep one Qwen3-8B or Qwen3.5-9B Q4_K_M GGUF resident at a time through the existing llama.cpp/Toolsmith server. Do not load two models or the HF sampler alongside it.
- Use the existing probed `parallel`, total context, and per-slot context; one in-flight episode per server slot, increasing by at most the existing small allowance only when verification/backoff leaves slots idle.
- Grammar-path cells use `enable_thinking=false`; any prompt-only thinking ablation is a separate manifest with `use_grammars=False`.
- Run configuration A, B, and C on the same eligible ids without reloading the model for policy changes. Pair the execution order and record it to reduce time/server drift.
- Store local journals/results under `/content/bulkhead/` and per-GPU durable copies under `MyDrive/toolsmith/records/<gpu>/bulkhead/`. Use atomic snapshots/background copies and drain queues before disconnect.
- Manifest includes benchmark commit, dataset/id hash, code versions, policy/taint mode and hashes, approval mode, sandbox profile/probe/helper hashes, model/quantization, llama.cpp tag, context/slots/KV settings, seed/temperature, and run label.
- Resume only exact matching records. A policy or tier change starts a new run id rather than mixing results.
- Keep agent and tools on the controlled Linux side. Prefer localhost model access. If a tunnel is unavoidable, use the existing random bearer-token secret handling and stop it immediately after the run; never embed the token in cells or records.
- Check the live Colab resource panel. The brief’s roughly 10–15 L4-hour / 15–26-unit estimate is planning context, not a guaranteed cost or official rate.

If the sandbox profile runs on a separate controlled CPU host while the model is in Colab, document and test that authenticated transport boundary explicitly. Do not expose an unauthenticated tool/proxy service.

## Results and final write-up

Generate `results/bulkhead/RESULTS.md` entirely from part 7 and part 8 records. Update the root `README.md` only after real runs, citing the relevant generated tables/manifests. Until then, add only a status stub saying local CPU checks are complete and live results are pending.

The final report contains:

1. **Threat model and architecture:** assets, attacker, boundaries, ALLOW/DENY/ASK flow, taint, jailer, cgroups, proxy, output gate, and audit.
2. **Environment coverage:** local/Colab probe outputs, exact T1/T2/T0 capabilities, helper/build/kernel versions, and untested tiers.
3. **Escape results:** per-tier table and what got through. Include the first failing script, attack id, expected/actual evidence, and remediation/retest.
4. **AgentDojo paired results:** clean utility, utility under attack, ASR, intervals, exact ids/subset, and eligibility.
5. **Usability and overhead:** ASK rate, over-denial, approval mode, cold-start median/p99, whole-episode overhead, and resource/concurrency context.
6. **Failure taxonomy:** distinguish model parse/semantic failures, policy denial, taint ASK, sandbox violation, resource exhaustion, egress denial, output truncation, and infrastructure abort.
7. **Replay evidence:** explain that control-plane replay reproduces decisions/messages and that live attack reruns, not ordinary replay, verify kernel enforcement.
8. **Limitations:** kernel zero-days/host root out of scope; coarse rather than per-value taint; T1 visibility/network/cleanup differences; no absolute proof of security; partial benchmark scope; unrun configurations.
9. **Context references:** Progent, CaMeL, AgentDojo, sandbox-runtime, Codex Linux sandbox, kernel docs, lethal trifecta, GrantBox/MCP Security Bench, and SandboxEscapeBench. Their reported percentages stay in context and name their own harnesses.
10. **Handoff:** identify audit/taint fields that Day 4 context/memory and Day 9 observability/guardrails can consume without trusting model prose.

Include at least one real trajectory for each important residual failure: policy allowed a semantically wrong action, ASK caused benign utility loss, output was truncated, a tier feature was unavailable, or an attack succeeded. Do not print private tool payloads or hidden chain of thought; use bounded journal events.

## Stretch goals, outside core acceptance

- Rewrite the jailer as a static C++ binary using `clone3`, mount setup/pivot, libseccomp, and cgroup lifecycle; benchmark against bubblewrap without weakening tests.
- Implement a C++ `epoll` proxy; `io_uring` is acceptable outside the sandbox only. Add scoped credential injection at the proxy after the no-secret core design is proven.
- Evaluate a warm sandbox pool only with a written state-leak analysis and cross-call isolation tests.
- Add gVisor or Firecracker as tier 3 on a controlled VM and run the same attack/overhead matrix.
- Replace set/bounds narrowing with an SMT-backed relation such as Progent’s, with differential tests against the simple relation.
- Add GrantBox or MCP Security Bench follow-ups after AgentDojo, with their own adapters/manifests and no score conflation.

None of these is required to finish the eight parts and none may delay fixing a core escape.

## Tasks

1. Pin and inspect AgentDojo; implement the smallest adapter against its real pipeline/tool interface and a three-row local fixture.
2. Implement paired configuration construction, eligible-id accounting, approval modes, claims/resume, and infrastructure handling.
3. Implement and hand-test all metrics and intervals, including simultaneous user/attacker outcomes and partial runs.
4. Complete CPU scripted record/replay and the full part 7 security gate.
5. Add the live notebook sweep with one resident model, slot-aware workers, local records, per-GPU Drive copies, and bounded infrastructure retry.
6. Run a small smoke subset through real Toolsmith model calls before the long sweep; inspect raw journals and all three configurations.
7. Run the selected benchmark scope, generate the report from records, and update README with measured cells only.
8. Write the residual-failure/limitations discussion and preserve first failing raw evidence. Leave stretch goals out of the core result table.

## Verification and acceptance

CPU and documentation checks:

```bash
.venv/bin/python -m pytest tests/test_bulkhead_*.py tests/test_steadyhand_*.py
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src tests
git diff --check
```

Live acceptance additionally requires the real notebook smoke/sweep calls and generated records; imports, fake pipelines, or a valid notebook JSON file are not enough.

- The three required configurations run on identical eligible ids and controls with no hidden auto-approval.
- Clean utility, utility under attack, ASR, ASK/over-denial, and overhead have raw numerators/denominators, intervals where applicable, and traceable manifests.
- Infrastructure aborts and ineligible/partial ids are reported without improving denominators.
- Every core escape is fixed/retested or the affected profile is explicitly disabled; no unresolved escape is presented as a passing profile.
- The report states what got through and why, what was not tested, and which claims come from other projects.
- The README contains no invented accuracy, attack-success, security, latency, or Colab-cost figures.

## Pitfalls

- Running the “baseline” with real host privileges or silently changing tools between configurations.
- Auto-approving ASK to improve utility or comparing different task subsets as if they were paired.
- Counting an infrastructure abort or parser crash as a successful defense.
- Publishing a small AgentDojo subset as a full benchmark score or using paper percentages as local measurements.
- Reloading/changing the model, policy, tier, or approval mode mid-run without a new manifest.
- Claiming exact journal replay proves the kernel sandbox, or claiming no measured attack succeeded means the system is unconditionally secure.
- Adding AgentDojo or hosted sandbox services as core runtime dependencies instead of a pinned evaluation adapter.

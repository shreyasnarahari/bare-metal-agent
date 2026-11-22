# Part 4 — Journal and replay

**Track:** Steadyhand, Day 2 of 10
**Milestone:** M3, evening 5
**Depends on:** `03-controller-and-guards.md`
**Produces:** deterministic record/replay of an episode, so later policy work and most regression tests run on CPU

## Outcome

An episode recorded against a live or fake model can be replayed with the same model outputs, the same tool results, and, from part 5 onward, the same injected faults. Replay does not call llama.cpp. A mismatch between the live tool result and the recorded tool result fails the replay loudly.

## What already exists

- `JsonlRecorder` in `src/toolsmith/recording.py`: append-only JSONL, secret redaction, `recorded_at`.
- `EpisodeJournal` from part 1, extended in parts 2 and 3 with validation, guard, and budget events.
- `CheckpointRunner` and `ManifestMismatchError` in `src/toolsmith/evaluation/runner.py`: resumable evaluation refuses to continue when the configuration or dataset hash differs. Replay should use the same idea.
- `canonical_hash` for fixture, tool-schema, and policy hashes.
- `atomic_write_json` in `evaluation/runner.py` for manifests.
- Toolsmith's own records are redacted and are not model-accuracy claims. Steadyhand journals follow that rule.

The shim also records request-level rows when a `JsonlRecorder` is attached to `ToolsmithService`. That record is per completion. The episode journal is the source of truth for replay. Keep both. Do not try to reconstruct an episode only from shim records, because tool execution happens outside the shim.

## What this part adds

```text
src/steadyhand/replay.py
src/steadyhand/manifest.py
tests/test_steadyhand_replay.py
```

Journal files remain one per episode under `results/steadyhand/` locally. On Colab the live file is `/content/steadyhand/<episode_id>.jsonl`, with the manifest beside it. A finished episode is copied to `MyDrive/toolsmith/records/<gpu>/steadyhand/`. Parallel workers never append to the same file. `JsonlRecorder` writes a line by opening the file, and two threads writing one path can tear a line.

Claim a task before running it. Create the manifest with `O_EXCL` (or an equivalent create-new) so two workers cannot start the same `episode_id`. The loser skips. Resume of `completed: false` is also claimed exclusively. Replay never takes a claim and never calls the GPU.

CPU replay of the golden set and, in part 7, of the scripted suite uses a process pool of size `min(os.cpu_count() or 1, 8)`. Each process loads its own fixture. That is the CPU-bound phase and it runs while the Colab GPU is idle or while llama-server is still loading weights. It does not run in parallel with the live sweep on a 2-vCPU T4 runtime, because those cores are better spent preparing prompts for the slots.

## Design

### Episode manifest

Written once at start, rewritten only to set `completed: true` at the end, via `atomic_write_json`.

```json
{
  "episode_id": "ep_...",
  "steadyhand_version": "0.1.0",
  "toolsmith_version": "0.1.0",
  "profile": "qwen35",
  "use_grammars": true,
  "enable_thinking": false,
  "seed": 0,
  "temperature": 0.0,
  "budgets": {},
  "fixture_sha256": "...",
  "schema_sha256": "...",
  "policy_sha256": "...",
  "system_prompt_sha256": "...",
  "fault_schedule_sha256": null,
  "completed": false
}
```

`schema_sha256` hashes `registry.schemas()`. `policy_sha256` hashes the policy table in tool-name order. `fixture_sha256` hashes the store fixture before the episode. Part 5 fills `fault_schedule_sha256`.

Replay loads the manifest and recomputes those hashes from the current code and fixture. On mismatch it raises `ReplayMismatch` with the field name. It does not silently rerun.

### Journal events

Required fields on every line: `event`, `episode_id`, `seq` (monotonic int starting at 0), `recorded_at`.

| Event | Payload used by replay |
|---|---|
| `episode_started` | user and system messages, budgets |
| `model_turn` | full assistant message, kind, usage, error code |
| `observation` | tool call id, observation JSON |
| `guard_tripped` | hash, count |
| `budget_checked` | the numbers that justified continuing or stopping |
| `episode_finished` | status, final content, token sums |

`seq` has no gaps. Replay refuses a gap or a duplicate.

### Recorder client

```python
class ReplayClient:
    def __init__(self, turns: list[ModelTurn]): ...
    def complete(self, messages, tools, *, tool_choice="auto") -> ModelTurn:
        # return the next recorded model_turn
        # if messages hash != recorded request-messages hash: ReplayMismatch
```

Each `model_turn` stores `messages_sha256` computed from the messages Steadyhand sent into `complete()`, after tool observations were appended, before the shim renders the template. Hash the Steadyhand message list, not the rendered prompt string, so a template tweak is visible as a manifest change only when you choose to record the template version. Also store `profile` and `use_grammars` in the manifest so a template change that matters is a deliberate version bump of the manifest fields, not an accidental pass.

On replay:

1. Load fixture store. Do not load a mutated store from the journal.
2. Rebuild the registry from code. Compare schema and policy hashes to the manifest.
3. Run `run_episode` with `ReplayClient`.
4. The real executor and validator run again. Their observations must deep-equal the recorded `observation` payloads.
5. Guard and budget events must deep-equal the recorded events.
6. Final status and final content must match.

A difference means the code no longer implements the recorded episode. Fail the test. Do not update the golden journal inside the test run. Updating a golden file is a separate, reviewed edit when the behavior change is intentional.

### What is hashed versus what is executed

Replay re-executes handlers. It does not stub them, except for the fault injector in part 5, which is itself deterministic and re-executed. If a handler return value differs, the test fails. That is the point: replay is a regression test of the loop plus the tools.

Model tokens are not re-executed. `ReplayClient` never calls `ToolsmithService`. GPU is not required.

### Resume

If a local journal exists with a matching manifest and `completed: false`, the worker that successfully creates the claim may continue from the next missing `seq`, and only when the manifest hashes match. This mirrors `CheckpointRunner`. A second worker that loses the claim leaves the file alone. Replay mode never resumes into a live model. It either matches the whole file or fails.

Write the journal after every event (`JsonlRecorder.append` already flushes by closing the file each call). A disconnect loses the in-flight episode on local disk if its Drive copy has not finished, and loses nothing that has already been copied. Copying is a separate thread fed by a queue of finished episode ids, with depth 2, so a slow Drive mount cannot stall every GPU worker and also cannot fall a whole sweep behind. The sweep does not exit until that queue drains. This part makes the manifest the resume key. The manifest records `server_parallel`, `context_size`, and `per_slot_context` once part 7 has chosen them. A resume refuses a journal whose slot size differs from the server now running.

### Clock and ids

Ticket ids and refund ids come from a counter on the store, seeded from the fixture, not from `uuid4()` or the wall clock. The episode id is `ep_` plus the manifest hash prefix, so the same inputs produce the same id. `recorded_at` is excluded from equality checks.

## Tasks

1. Freeze the event schema in `journal.py` and bump a `JOURNAL_VERSION` constant. Unknown event names fail the reader.
2. Write the manifest beside the journal.
3. Implement `ReplayClient` and `replay_episode(path) -> EpisodeResult`.
4. Record three golden episodes under `tests/fixtures/steadyhand_episodes/` using the fake client from earlier parts: happy path, validation retry then success, repeated-call escalation. Check the JSONL in. It is small and deterministic. Do not record secrets or raw prompts that contain keys.
5. Test that editing one observation byte in a temp copy raises `ReplayMismatch` and names the `seq`.
6. Test that changing a semantic check (temporarily, inside the test, by wrapping the registry) makes replay fail.
7. Test resume: a truncated journal with `completed: false` and matching hashes continues a fake live client from the next seq; a hash mismatch refuses.

## Tests

- Round trip: run with `RecordingClient` (fake model) then `replay_episode`. Status, store snapshot hash, and final content match. Store snapshot hash is `canonical_hash` of the store's public dict.
- Message-list drift: if the test appends an extra system line, `ReplayClient` raises before returning the canned turn.
- `seq` gap fails.
- Redaction: a tool argument that contains the string passed as a secret is stored as `[redacted]`, and replay of that fixture is not required to re-execute the secret. Do not put real secrets in the golden files.
- Version fields: `toolsmith.__version__` and a `steadyhand` version constant are in the manifest.

Run the Steadyhand tests plus `tests/test_recording.py` to be sure redaction still holds.

## Acceptance

- The three golden episodes replay on CPU with no network.
- A behavior change in validation, guards, or handlers breaks replay with a field or `seq` identified.
- Live resume only happens when the manifest matches.
- Golden files contain no API keys and no unredacted authorization headers.

## Pitfalls

- Recording the rendered prompt as the only replay input. Template changes and message-list bugs become impossible to tell apart, and the rendered string can be huge.
- Stubbing tool results during replay. The handlers would stop being tested. Stub the model only.
- Treating a partial BFCL or Toolsmith eval record as an episode journal. Different schemas. Different directories.
- Calling `uuid4()` for ticket ids. Replay would diverge on otherwise identical code.

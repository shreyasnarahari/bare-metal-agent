import dataclasses
import json
import shutil
import threading
from pathlib import Path

import pytest

import steadyhand
import toolsmith
from steadyhand import (
    Budgets,
    EpisodeStatus,
    SteadyhandClient,
    SteadyhandRegistry,
    Store,
    build_registry,
    initial_messages,
    load_fixture,
)
from steadyhand.journal import (
    JOURNAL_VERSION,
    JournalCopier,
    JournalFormatError,
    ReplayMismatch,
    comparable,
    dump_record,
    manifest_path_for,
    read_journal,
)
from steadyhand.manifest import build_manifest, load_manifest
from steadyhand.replay import ReplayClient, replay_episode, replay_many, run_claimed_episode
from steadyhand.tools import TOOL_SPECS
from toolsmith.protocols import QWEN3_PROFILE

GOLDEN = Path(__file__).parent / "fixtures" / "steadyhand_episodes"
GOLDEN_JOURNALS = sorted(GOLDEN.glob("ep_*.jsonl"))
REFUND = {"order_id": "ord_1001", "amount_inr": 120, "reason": "Dented lid"}
BAD_REFUND = {**REFUND, "amount_inr": -120}
GOAL = "Refund 120 INR on order ord_1001 for the dented lid."


class FakeService:
    profile = QWEN3_PROFILE
    use_grammars = True

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def completion(content=None, calls=(), prompt_tokens=50, completion_tokens=8):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            for call_id, name, arguments in calls
        ]
    return {
        "choices": [{"index": 0, "message": message}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


RETRY_TURNS = [
    completion(calls=[("bad", "issue_refund", BAD_REFUND)]),
    completion(calls=[("fixed", "issue_refund", REFUND)]),
    completion("Refunded 120 INR."),
]


def record(directory, turns=RETRY_TURNS, *, goal=GOAL, client=None, **options):
    service = FakeService(*turns)
    result = run_claimed_episode(
        directory,
        client or SteadyhandClient(service),
        build_registry(),
        load_fixture().clone(),
        initial_messages(goal),
        **options,
    )
    return result, service


def golden(label):
    for path in GOLDEN_JOURNALS:
        if load_manifest(manifest_path_for(path)).run_label == label:
            return path
    raise LookupError(label)


def copy_episode(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    journal = Path(shutil.copy(source, directory / source.name))
    shutil.copy(manifest_path_for(source), manifest_path_for(journal))
    return journal


def rewrite(path, records):
    path.write_text("".join(dump_record(record) + "\n" for record in records), encoding="utf-8")


# Golden episodes


def test_there_are_three_golden_episodes():
    labels = sorted(load_manifest(manifest_path_for(path)).run_label for path in GOLDEN_JOURNALS)

    assert labels == ["happy_path", "repeated_call_escalation", "validation_retry"]


@pytest.mark.parametrize(
    ("label", "status"),
    [("happy_path", "done"), ("validation_retry", "done"), ("repeated_call_escalation", "escalated")],
)
def test_golden_episodes_replay_on_cpu(label, status):
    path = golden(label)
    before = path.read_bytes()

    result = replay_episode(path)

    assert result.status.value == status
    assert result.final_content == read_journal(path)[-1]["final_content"]
    assert path.read_bytes() == before  # replay writes nothing


def test_golden_files_hold_no_keys_or_authorization_headers():
    for path in GOLDEN.glob("ep_*"):
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert "bearer " not in lowered
        assert '"authorization"' not in lowered
        assert '"api_key"' not in lowered
        assert "hf_" not in text
        assert "raw_prompt" not in text


def test_golden_episodes_replay_in_a_process_pool():
    outcomes = replay_many(GOLDEN_JOURNALS, workers=2)

    assert [outcome.ok for outcome in outcomes] == [True, True, True]
    assert sorted(outcome.status for outcome in outcomes) == ["done", "done", "escalated"]


# Round trip and manifests


def test_record_then_replay_round_trip(tmp_path):
    result, _ = record(tmp_path)

    replayed = replay_episode(result.journal_path)

    assert replayed.status is result.status is EpisodeStatus.DONE
    assert replayed.final_content == result.final_content
    assert replayed.messages == result.messages
    finished = read_journal(result.journal_path)[-1]
    store = load_fixture().clone()
    store.add_refund("ord_1001", "120", "Dented lid")
    assert finished["store_sha256"] == store.snapshot_sha256()


def test_the_manifest_records_versions_hashes_and_completion(tmp_path):
    result, _ = record(tmp_path)

    manifest = load_manifest(manifest_path_for(result.journal_path))

    assert manifest.completed is True
    assert manifest.steadyhand_version == steadyhand.__version__
    assert manifest.toolsmith_version == toolsmith.__version__
    assert manifest.journal_version == JOURNAL_VERSION
    assert manifest.schema_sha256 == build_registry().schemas_sha256()
    assert manifest.policy_sha256 == build_registry().policies_sha256()
    assert manifest.fixture_sha256 == load_fixture().fixture_sha256
    assert manifest.fault_schedule_sha256 is None
    assert manifest.episode_id == result.episode_id
    assert result.journal_path.name == f"{manifest.episode_id}.jsonl"


def manifest_for(goal=GOAL, settings=None, **options):
    settings = settings or SteadyhandClient(FakeService()).settings()
    return build_manifest(
        settings, build_registry(), load_fixture(), initial_messages(goal),
        budgets=Budgets().as_dict(), guard_limit=3, **options,
    )


def test_episode_ids_are_derived_from_the_inputs():
    same = manifest_for().episode_id

    assert manifest_for().episode_id == same
    assert manifest_for(goal="Something else").episode_id != same
    assert manifest_for(run_label="repeat-2").episode_id != same
    assert manifest_for(server_parallel=4, context_size=16384).episode_id == same
    assert same.startswith("ep_") and len(same) == 19


# Mismatches


def test_editing_one_observation_byte_names_the_seq(tmp_path):
    path = copy_episode(golden("validation_retry"), tmp_path)
    records = read_journal(path)
    target = next(record for record in records if record["event"] == "tool_result")
    target["content"] = target["content"].replace("must be > 0", "must be > 1")
    rewrite(path, records)

    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(path)

    assert caught.value.seq == target["seq"]
    assert caught.value.field == "/content"


def test_a_changed_semantic_check_breaks_replay(tmp_path):
    lenient = SteadyhandRegistry()
    for handler, policy, semantic in TOOL_SPECS:
        lenient.register(handler, policy, (lambda store, arguments: []) if handler.__name__ == "issue_refund" else semantic)
    path = golden("validation_retry")
    first_result = next(record for record in read_journal(path) if record["event"] == "tool_result")

    assert lenient.schemas_sha256() == build_registry().schemas_sha256()
    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(path, registry=lenient.freeze())

    assert caught.value.seq == first_result["seq"]


def test_a_changed_handler_breaks_replay(monkeypatch):
    original = Store.order

    def drifted(self, order_id):
        return dataclasses.replace(original(self, order_id), status="shipped")

    monkeypatch.setattr(Store, "order", drifted)
    path = golden("happy_path")
    result_seq = next(record["seq"] for record in read_journal(path) if record["event"] == "tool_result")

    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(path)

    assert caught.value.seq == result_seq


def test_message_list_drift_fails_before_a_turn_is_returned():
    records = read_journal(golden("happy_path"))
    client = ReplayClient.from_records(records)
    drifted = [*records[0]["messages"], {"role": "system", "content": "Be brief."}]

    with pytest.raises(ReplayMismatch) as caught:
        client.complete(drifted, build_registry().schemas())

    assert caught.value.field == "messages_sha256"
    assert client.remaining == 2
    assert client.complete(records[0]["messages"], build_registry().schemas()).tool_calls[0].name == "get_order"


def test_a_tampered_manifest_hash_is_refused(tmp_path):
    path = copy_episode(golden("happy_path"), tmp_path)
    manifest = json.loads(manifest_path_for(path).read_text())
    manifest["schema_sha256"] = "0" * 64
    manifest_path_for(path).write_text(json.dumps(manifest))

    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(path)

    assert caught.value.field == "schema_sha256"


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda records: records[:2] + records[3:], "seq"),
        (lambda records: records[:3] + records[2:], "seq"),
        (lambda records: [*records[:-1], {**records[-1], "event": "episode_done"}], "event"),
        (lambda records: [*records[:-1], {**records[-1], "surprise": 1}], "fields"),
        (lambda records: [{**records[0], "journal_version": 0}, *records[1:]], "journal_version"),
    ],
)
def test_malformed_journals_are_refused_by_the_reader(tmp_path, mutate, field):
    path = copy_episode(golden("happy_path"), tmp_path)
    rewrite(path, mutate(read_journal(path)))

    with pytest.raises(JournalFormatError) as caught:
        read_journal(path)

    assert caught.value.field == field


def test_a_torn_last_line_is_refused_unless_resuming(tmp_path):
    path = copy_episode(golden("happy_path"), tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "model_')

    with pytest.raises(JournalFormatError):
        read_journal(path)
    assert len(read_journal(path, allow_torn_tail=True)) == 7


def test_an_unfinished_journal_cannot_be_replayed(tmp_path):
    path = copy_episode(golden("happy_path"), tmp_path)
    rewrite(path, read_journal(path)[:4])

    with pytest.raises(JournalFormatError, match="resume"):
        replay_episode(path)


def test_redacted_secrets_are_not_stored_and_such_journals_are_not_replayed(tmp_path):
    secret = "s3cret-token-value"
    turns = [
        completion(calls=[("note", "add_order_note", {"order_id": "ord_1001", "note": f"pin {secret}"})]),
        completion("Noted."),
    ]

    result, _ = record(tmp_path, turns, goal="Add my pin to ord_1001.", secrets=(secret,))

    text = result.journal_path.read_text(encoding="utf-8")
    assert secret not in text
    assert "pin [redacted]" in text
    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(result.journal_path)
    assert caught.value.field == "redacted"


# Controller details survive replay


class FakeClock:
    def __init__(self, step):
        self.now = 0.0
        self.step = step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


def test_wall_clock_decisions_replay_exactly(tmp_path):
    turns = [completion(calls=[(f"c{i}", "get_order", {"order_id": f"ord_100{i + 1}"})]) for i in range(4)]

    result, _ = record(tmp_path, turns, budgets=Budgets(max_wall_clock_s=30.0), clock=FakeClock(12.0))

    assert result.status is EpisodeStatus.BUDGET_EXHAUSTED
    assert replay_episode(result.journal_path).final_content == result.final_content


def test_a_cancelled_episode_replays(tmp_path):
    cancel = threading.Event()
    cancel.set()

    result, _ = record(tmp_path, [], cancel=cancel)

    assert result.error_code == "cancelled"
    assert replay_episode(result.journal_path).status is EpisodeStatus.ABORTED


def test_a_paraphrased_ending_replays_with_its_tool_choice(tmp_path):
    turns = [completion(calls=[(f"c{i}", "issue_refund", BAD_REFUND)]) for i in range(4)]
    turns.append(completion("I could not refund a negative amount."))

    result, _ = record(tmp_path, turns, paraphrase_final=True)

    assert result.final_content == "I could not refund a negative amount."
    assert replay_episode(result.journal_path).final_content == result.final_content


# Claims and resume


def test_a_held_claim_or_a_completed_episode_is_skipped(tmp_path):
    episode_id = manifest_for().episode_id
    claim = tmp_path / f"{episode_id}.claim"
    claim.write_text("other worker\n")

    held, service = record(tmp_path)
    assert held is None and service.requests == []
    assert sorted(path.name for path in tmp_path.iterdir()) == [claim.name]

    claim.unlink()
    first, _ = record(tmp_path)
    again, service = record(tmp_path)
    assert first is not None and again is None and service.requests == []
    assert not claim.exists()


def test_resume_continues_a_truncated_journal_from_the_next_seq(tmp_path):
    full, _ = record(tmp_path / "full")
    original = read_journal(full.journal_path)

    crashed = copy_episode(full.journal_path, tmp_path / "crashed")
    manifest = json.loads(manifest_path_for(crashed).read_text())
    manifest["completed"] = False
    manifest_path_for(crashed).write_text(json.dumps(manifest))
    prefix = original[:4]  # started, budget, first model turn, first tool result
    rewrite(crashed, prefix)
    with crashed.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "budget_che')  # torn by the crash

    resumed, live = record(tmp_path / "crashed", RETRY_TURNS[1:])

    assert resumed.status is EpisodeStatus.DONE
    assert len(live.requests) == 2  # only the turns after the recorded prefix
    after = read_journal(crashed)
    assert [comparable(line) for line in after] == [comparable(line) for line in original]
    assert after[:4] == prefix  # the recorded prefix was kept, not rewritten
    assert load_manifest(manifest_path_for(crashed)).completed is True
    assert not crashed.with_name(f"{resumed.episode_id}.claim").exists()


def test_resume_refuses_a_manifest_that_no_longer_matches(tmp_path):
    full, _ = record(tmp_path)
    manifest_file = manifest_path_for(full.journal_path)
    manifest = json.loads(manifest_file.read_text())
    manifest["completed"] = False
    manifest["policy_sha256"] = "f" * 64
    manifest_file.write_text(json.dumps(manifest))
    before = full.journal_path.read_bytes()

    with pytest.raises(ReplayMismatch) as caught:
        record(tmp_path)

    assert caught.value.field == "policy_sha256"
    assert full.journal_path.read_bytes() == before
    assert not full.journal_path.with_name(f"{full.episode_id}.claim").exists()


def test_resume_refuses_a_different_server_slot(tmp_path):
    full, _ = record(tmp_path)
    manifest_file = manifest_path_for(full.journal_path)
    manifest = json.loads(manifest_file.read_text())
    manifest["completed"] = False
    manifest_file.write_text(json.dumps(manifest))
    client = SteadyhandClient(FakeService(), slot_context=8192)

    with pytest.raises(ReplayMismatch) as caught:
        record(tmp_path, client=client)

    assert caught.value.field == "per_slot_context"


# Copying finished episodes


def test_the_copier_moves_journals_and_manifests_and_drains_on_close(tmp_path):
    first, _ = record(tmp_path / "local")
    second, _ = record(tmp_path / "local", [completion("Hello.")], goal="Hi there.")

    with JournalCopier(tmp_path / "drive", depth=1) as copier:
        copier.submit(first.journal_path)
        copier.submit(second.journal_path)

    names = sorted(path.name for path in (tmp_path / "drive").iterdir())
    assert names == sorted([
        first.journal_path.name, manifest_path_for(first.journal_path).name,
        second.journal_path.name, manifest_path_for(second.journal_path).name,
    ])
    assert copier.errors == []
    assert (tmp_path / "drive" / first.journal_path.name).read_bytes() == first.journal_path.read_bytes()

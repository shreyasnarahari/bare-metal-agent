"""The optional context seam: legacy episodes unchanged, Tidemark episodes hashed, journaled, and replayed exactly."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from steadyhand import Budgets, EpisodeJournal, SteadyhandClient, build_registry, initial_messages, load_fixture, run_episode
from steadyhand.context import RenderedContext, context_config_sha256
from steadyhand.journal import JournalFormatError, ReplayMismatch, comparable, manifest_path_for, read_journal
from steadyhand.manifest import build_manifest, load_manifest, require_match
from steadyhand.replay import replay_episode, run_claimed_episode
from steadyhand.types import EpisodeStatus
from tidemark.baseline import NaiveAppendTruncate
from tidemark.telemetry import backend_timing_fields
from tidemark.tokenizer import RegexTokenizer
from tidemark.types import ContextConfig
from toolsmith.api import APIError
from toolsmith.evaluation.models import canonical_hash

from tidemark_support import FakeService, attempt, completion, qwen_like_render

GOLDEN = Path(__file__).parent / "fixtures" / "steadyhand_episodes"
# The committed golden files; Tidemark must never change a byte of them.
GOLDEN_SHA256 = {
    "ep_07922e66f89c0aa9.jsonl": "6bb37ff557241a1839f73c361ffdf1523380b6ef9a40e1d24784bdcdd23f87f5",
    "ep_07922e66f89c0aa9.manifest.json": "6a051cdbedb0fd1ed9947d3ef2d8928fc107f5dd8e069672989dcbf5dfda8361",
    "ep_8393620aa57df809.jsonl": "2ea6e8e7f297a951e09a5ed6058b2db8b84d02fb0c8998a2f6ef2b0ad26ead2a",
    "ep_8393620aa57df809.manifest.json": "6c44242f1b276097f0bf37139581ed45e7d30af121b485db7634d0fb7affba32",
    "ep_b345550151d9160c.jsonl": "2ca764e29804b506b48e9a92d64f4054333790e9c2e5c7885435b017de35586a",
    "ep_b345550151d9160c.manifest.json": "fb9155852b748bcbc451fa38abcbfc6491f883fe1a1156ce90391ad83fee7298",
}
# The golden manifests' ids predate Steadyhand part 6, which added recovery_policy and system_prompt_variant to
# the identity; unchanged HEAD code already derives these ids for the same scenarios (replay never compares ids).
CURRENT_SCENARIO_IDS = {
    "happy_path": ("ep_268aa8ce0b358631", "ep_8393620aa57df809"),
    "validation_retry": ("ep_b4a01a2f9f338bd8", "ep_b345550151d9160c"),
    "repeated_call_escalation": ("ep_7499cf1f6182975d", "ep_07922e66f89c0aa9"),
}
CONTEXT_EVENTS = {"context_render", "backend_timings"}
CONFIRMED = {"confirmed": True, "new_tokens_field": "prompt_n", "total_tokens_field": None}
GOAL = "What is the current status of order ord_1001?"
STATUS_TURNS = [
    completion(calls=[("call_status", "get_order", {"order_id": "ord_1001"})], attempts=[attempt(prompt_n=40, id_slot=0)]),
    completion("Order ord_1001 was delivered.", attempts=[attempt(prompt_n=12, id_slot=0)]),
]


def config(slot_context=4096, **overrides):
    return ContextConfig(slot_context=slot_context, max_tokens=256, tokenizer=RegexTokenizer().identity(),
                         cache_semantics=CONFIRMED, **overrides)


def naive(slot_context=4096, **overrides):
    return NaiveAppendTruncate(config(slot_context, **overrides), render=qwen_like_render, tokenizer=RegexTokenizer())


def client(*turns, **options):
    return SteadyhandClient(FakeService(*turns), **options)


def events(path):
    return [record["event"] for record in read_journal(path)]


def record_claimed(directory, turns=STATUS_TURNS, *, context=None, goal=GOAL, **options):
    service = FakeService(*turns)
    result = run_claimed_episode(directory, SteadyhandClient(service), build_registry(), load_fixture().clone(),
                                 initial_messages(goal), context=context, **options)
    return result, service


class RecitationSession:
    """Sends the history with a recitation tail written over the newest message (a replaced tail)."""

    strategy = "recitation_test"
    version = "test"
    tail = "[recitation] keep PAY-4471-K verbatim"

    def __init__(self):
        self.renders = []
        self.shaped = []

    def identity(self):
        return {"strategy": self.strategy, "tail": self.tail}

    def render(self, history, *, step, tools, tool_choice, initial_count):
        self.renders.append({"step": step, "tool_choice": tool_choice, "history": len(history), "initial": initial_count})
        messages = [dict(message) for message in history]
        messages[-1] = {**messages[-1], "content": (messages[-1].get("content") or "") + "\n" + self.tail}
        text = qwen_like_render(messages, tools)
        return RenderedContext(
            messages=tuple(messages), tools=tuple(tools), strategy=self.strategy,
            prompt_sha256=hashlib.sha256(text.encode()).hexdigest(), rendered_prompt_tokens=len(RegexTokenizer().encode(text)),
            recitation_sha256=hashlib.sha256(self.tail.encode()).hexdigest(), max_tokens=256, slot_headroom=100,
        )

    def shape_observation(self, message, *, step):
        self.shaped.append(step)
        return {**message, "content": message["content"] + " [shaped]"}

    def backend_timings(self, turn, rendered):
        return backend_timing_fields(turn.metrics, rendered.rendered_prompt_tokens, CONFIRMED)


# Legacy: nothing changes without a session


def test_golden_files_are_byte_identical():
    for name, expected in GOLDEN_SHA256.items():
        assert hashlib.sha256((GOLDEN / name).read_bytes()).hexdigest() == expected, name


def test_re_recording_the_golden_scenarios_reproduces_every_record(tmp_path):
    spec = importlib.util.spec_from_file_location("golden_regenerate", GOLDEN / "regenerate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for label, (goal, completions) in module.SCENARIOS.items():
        result = run_claimed_episode(tmp_path, SteadyhandClient(module.ScriptedService(completions)), build_registry(),
                                     load_fixture().clone(), initial_messages(goal), run_label=label)
        current_id, golden_id = CURRENT_SCENARIO_IDS[label]
        assert result.journal_path.stem == current_id, label
        golden = GOLDEN / f"{golden_id}.jsonl"
        fresh, recorded = read_journal(result.journal_path), read_journal(golden)

        def content(records):
            return [{k: v for k, v in comparable(r).items() if k != "episode_id"} for r in records]

        assert content(fresh) == content(recorded), label
        fresh_manifest = load_manifest(manifest_path_for(result.journal_path)).as_dict()
        golden_manifest = load_manifest(manifest_path_for(golden)).as_dict()
        assert {k: v for k, v in fresh_manifest.items() if k != "episode_id"} == {
            k: v for k, v in golden_manifest.items() if k != "episode_id"
        }
        assert not CONTEXT_EVENTS & {r["event"] for r in fresh} and "context" not in fresh[0]


def test_golden_manifests_and_journals_carry_no_context_fields():
    for path in sorted(GOLDEN.glob("ep_*.jsonl")):
        records = read_journal(path)
        assert not CONTEXT_EVENTS & {r["event"] for r in records} and "context" not in records[0]
        manifest = json.loads(manifest_path_for(path).read_text())
        assert not {"context_config", "context_config_sha256", "tidemark_version", "context_server"} & set(manifest)
        assert "generation_path" not in records[0]["settings"]


def test_a_new_legacy_episode_writes_no_context_events(tmp_path):
    result, service = record_claimed(tmp_path)
    assert result.status is EpisodeStatus.DONE
    assert not CONTEXT_EVENTS & set(events(result.journal_path))
    manifest = load_manifest(manifest_path_for(result.journal_path))
    assert not manifest.contextual and manifest.context_config is None
    # The model saw the logical history itself.
    assert service.requests[1].messages == result.messages[:-1]


# The seam


def test_the_hash_covers_the_exact_messages_sent_including_a_replaced_tail(tmp_path):
    session = RecitationSession()
    journal = EpisodeJournal.create(tmp_path)
    service = FakeService(*STATUS_TURNS)
    result = run_episode(SteadyhandClient(service), build_registry(), load_fixture().clone(), initial_messages(GOAL),
                         journal=journal, context=session)
    records = read_journal(journal.path)
    turns = [r for r in records if r["event"] == "model_turn"]
    for record, request in zip(turns, service.requests):
        assert record["messages_sha256"] == canonical_hash(request.messages)
        assert session.tail in request.messages[-1]["content"]
    # Not the logical history: that never had the tail.
    assert turns[0]["messages_sha256"] != canonical_hash(initial_messages(GOAL))
    assert all(session.tail not in (m.get("content") or "") for m in result.messages)
    sequence = [r["event"] for r in records]
    for index, name in enumerate(sequence):
        if name == "model_turn":
            assert sequence[index - 1] == "context_render" and sequence[index + 1] == "backend_timings"
    render = next(r for r in records if r["event"] == "context_render")
    assert render["recitation_sha256"] == hashlib.sha256(session.tail.encode()).hexdigest()
    assert render["context_config_sha256"] == context_config_sha256(session.identity())
    assert records[0]["context"]["strategy"] == "recitation_test"
    assert session.renders[0] == {"step": 0, "tool_choice": "auto", "history": 2, "initial": 2}


def test_tool_messages_pass_through_the_observation_seam(tmp_path):
    session = RecitationSession()
    journal = EpisodeJournal.create(tmp_path)
    result = run_episode(client(*STATUS_TURNS), build_registry(), load_fixture().clone(), initial_messages(GOAL),
                         journal=journal, context=session)
    tool_messages = [m for m in result.messages if m["role"] == "tool"]
    assert session.shaped == [0] and tool_messages[0]["content"].endswith(" [shaped]")
    tool_record = next(r for r in read_journal(journal.path) if r["event"] == "tool_result")
    assert not tool_record["content"].endswith(" [shaped]")  # the journal keeps the observation as produced


def test_parse_error_follow_ups_and_the_paraphrase_use_the_seam(tmp_path):
    session = RecitationSession()
    journal = EpisodeJournal.create(tmp_path)
    turns = [
        APIError("empty", status_code=500, code="empty_model_response"),
        completion(calls=[("call_status", "get_order", {"order_id": "ord_1001"})], attempts=[attempt(prompt_n=40)]),
        completion("I could not finish.", attempts=[attempt(prompt_n=9)]),
    ]
    result = run_episode(client(*turns), build_registry(), load_fixture().clone(), initial_messages(GOAL),
                         journal=journal, context=session, budgets=Budgets(max_steps=2), paraphrase_final=True)
    assert result.status is EpisodeStatus.BUDGET_EXHAUSTED and result.final_content == "I could not finish."
    assert [r["tool_choice"] for r in session.renders] == ["auto", "auto", "none"]
    assert session.renders[1]["history"] == 3  # the parse-error observation is in the follow-up's history
    sequence = events(journal.path)
    last_observation = len(sequence) - 1 - sequence[::-1].index("observation")
    assert sequence[last_observation:last_observation + 4] == ["observation", "context_render", "model_turn", "backend_timings"]
    assert read_journal(journal.path)[last_observation]["observation"] == {"purpose": "final_paraphrase"}


def test_a_render_that_cannot_fit_aborts_before_any_model_call(tmp_path):
    journal = EpisodeJournal.create(tmp_path)
    service = FakeService(*STATUS_TURNS)
    result = run_episode(SteadyhandClient(service), build_registry(), load_fixture().clone(), initial_messages(GOAL),
                         journal=journal, context=naive(slot_context=400))
    assert result.status is EpisodeStatus.ABORTED and result.error_code == "context_budget_exhausted"
    assert service.requests == []
    records = read_journal(journal.path)
    render = next(r for r in records if r["event"] == "context_render")
    assert render["status"] == "context_budget_exhausted" and render["messages_sha256"] is None
    assert render["slot_headroom"] < 0 and render["max_tokens"] == 256
    assert "model_turn" not in events(journal.path)


def test_a_session_replaces_the_history_only_grows_slot_estimate(tmp_path):
    big_usage = [replace_usage(turn, 300) for turn in STATUS_TURNS]
    legacy = run_episode(client(*big_usage, slot_context=520), build_registry(), load_fixture().clone(),
                         initial_messages(GOAL), journal=EpisodeJournal.create(tmp_path / "legacy"))
    assert legacy.error_code == "completion_truncated"  # 300 + 256 > 520 by the controller's estimate
    tidemark = run_episode(client(*big_usage, slot_context=520), build_registry(), load_fixture().clone(),
                           initial_messages(GOAL), journal=EpisodeJournal.create(tmp_path / "tidemark"),
                           context=naive(slot_context=4096))
    assert tidemark.status is EpisodeStatus.DONE


def replace_usage(response, prompt_tokens):
    return {**response, "usage": {**response["usage"], "prompt_tokens": prompt_tokens}}


# Replay


def test_a_tidemark_episode_replays_with_a_fresh_session(tmp_path):
    result, _ = record_claimed(tmp_path, context=naive())
    replayed = replay_episode(result.journal_path, context=naive())
    assert replayed.status is EpisodeStatus.DONE and replayed.messages == result.messages
    manifest = load_manifest(manifest_path_for(result.journal_path))
    assert manifest.contextual and manifest.context_config == config().as_dict()
    timings = [r for r in read_journal(result.journal_path) if r["event"] == "backend_timings"]
    assert timings[0]["prompt_evaluated_tokens"] == 40 and timings[0]["slot_id"] == 0
    assert timings[0]["reused_tokens"] == timings[0]["rendered_prompt_tokens"] - 40


def test_replay_needs_the_recorded_context_configuration(tmp_path):
    result, _ = record_claimed(tmp_path, context=naive())
    with pytest.raises(ReplayMismatch) as missing:
        replay_episode(result.journal_path)
    assert missing.value.field == "context_config"
    with pytest.raises(ReplayMismatch) as different:
        replay_episode(result.journal_path, context=naive(slot_context=8192))
    assert different.value.field == "context_config"
    golden = sorted(GOLDEN.glob("ep_*.jsonl"))[0]
    with pytest.raises(ReplayMismatch) as legacy:
        replay_episode(golden, context=naive())
    assert legacy.value.field == "context_config"


def tamper(path, seq, change):
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[seq])
    change(record)
    lines[seq] = json.dumps(record, ensure_ascii=False, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def seq_of(path, event, occurrence=0):
    return [r["seq"] for r in read_journal(path) if r["event"] == event][occurrence]


@pytest.mark.parametrize(
    ("event", "field", "change", "expected_field"),
    [
        ("context_render", "prompt_sha256", lambda r: r.update(prompt_sha256="0" * 64), "/prompt_sha256"),
        ("context_render", "rendered_prompt_tokens", lambda r: r.update(rendered_prompt_tokens=r["rendered_prompt_tokens"] + 1),
         "/rendered_prompt_tokens"),
        ("context_render", "dropped_ranges", lambda r: r.update(dropped_ranges=[[1, 2]]), "/dropped_ranges"),
        ("model_turn", "messages_sha256", lambda r: r.update(messages_sha256="0" * 64), "messages_sha256"),
        ("backend_timings", "prompt_evaluated_tokens", lambda r: r.update(prompt_evaluated_tokens=39), "/prompt_evaluated_tokens"),
        ("backend_timings", "rendered_prompt_tokens", lambda r: r.update(rendered_prompt_tokens=r["rendered_prompt_tokens"] + 5),
         "/rendered_prompt_tokens"),
        ("backend_timings", "cache_reuse_fraction", lambda r: r.update(cache_reuse_fraction=0.5), "/cache_reuse_fraction"),
    ],
)
def test_tampering_with_rendered_messages_hashes_or_counts_is_caught(tmp_path, event, field, change, expected_field):
    result, _ = record_claimed(tmp_path, context=naive())
    seq = seq_of(result.journal_path, event)
    tamper(result.journal_path, seq, change)
    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(result.journal_path, context=naive())
    assert caught.value.seq == seq and caught.value.field == expected_field


def test_declared_timings_are_volatile(tmp_path):
    result, _ = record_claimed(tmp_path, context=naive())
    seq = seq_of(result.journal_path, "backend_timings")

    def slower(record):
        record.update(prompt_ms=999.0, predicted_ms=1.0, response_elapsed_ms=5000.0)
        record["attempt_timings"][0]["prompt_ms"] = 999.0

    tamper(result.journal_path, seq, slower)
    assert replay_episode(result.journal_path, context=naive()).status is EpisodeStatus.DONE


def test_tampering_with_the_manifest_context_config_is_caught(tmp_path):
    result, _ = record_claimed(tmp_path, context=naive())
    path = manifest_path_for(result.journal_path)
    manifest = json.loads(path.read_text())
    manifest["context_config"]["safety_margin"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(result.journal_path, context=naive())
    assert caught.value.field == "context_config_sha256"


# Manifests and journal checks


def test_context_configuration_is_part_of_the_episode_id():
    registry, store, messages = build_registry(), load_fixture().clone(), initial_messages(GOAL)
    settings = client().settings()

    def manifest(**context):
        return build_manifest(settings, registry, store, messages, budgets=Budgets().as_dict(), guard_limit=3, **context)

    legacy = manifest()
    first = manifest(context_config=config().as_dict(), context_version="0.1.0")
    other = manifest(context_config=config(generation_path="buffered_stream").as_dict(), context_version="0.1.0")
    newer = manifest(context_config=config().as_dict(), context_version="0.2.0")
    assert len({legacy.episode_id, first.episode_id, other.episode_id, newer.episode_id}) == 4
    assert "context_config_sha256" not in legacy.identity() and first.identity()["tidemark_version"] == "0.1.0"
    serving = manifest(context_config=config().as_dict(), context_version="0.1.0", context_server={"kv_cache": "q8_0"})
    assert serving.episode_id == first.episode_id  # the serving shape is resume-checked, not identity
    with pytest.raises(ReplayMismatch) as caught:
        require_match(serving, replace(serving, context_server={"kv_cache": "f16"}))
    assert caught.value.field == "context_server"
    with pytest.raises(ValueError):
        manifest(context_config=config().as_dict())


def test_resume_refuses_a_changed_context_server(tmp_path):
    first, _ = record_claimed(tmp_path, context=naive(), context_server={"kv_cache": "q8_0"})
    # What a killed process leaves: the journal cut after the first tool result, the manifest not completed.
    cut = seq_of(first.journal_path, "tool_result") + 1
    lines = first.journal_path.read_text(encoding="utf-8").splitlines()[:cut]
    first.journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_file = manifest_path_for(first.journal_path)
    manifest_file.write_text(json.dumps({**json.loads(manifest_file.read_text()), "completed": False}))
    with pytest.raises(ReplayMismatch) as caught:
        record_claimed(tmp_path, context=naive(), context_server={"kv_cache": "f16"})
    assert caught.value.field == "context_server"
    resumed, _ = record_claimed(tmp_path, turns=STATUS_TURNS[1:], context=naive(), context_server={"kv_cache": "q8_0"})
    assert resumed.status is EpisodeStatus.DONE
    assert replay_episode(resumed.journal_path, context=naive()).status is EpisodeStatus.DONE


@pytest.mark.parametrize(
    ("event", "change", "field"),
    [
        ("context_render", lambda r: r.update(rendered_prompt_tokens=-1), "rendered_prompt_tokens"),
        ("context_render", lambda r: r.update(status="truncated"), "status"),
        ("context_render", lambda r: r.update(dropped_ranges=[[3, 5], [4, 6]]), "dropped_ranges"),
        ("context_render", lambda r: r.update(prompt_sha256="abc"), "prompt_sha256"),
        ("backend_timings", lambda r: r.update(ttft_ms=3.0, measurement_path="unavailable"), "ttft_ms"),
        ("backend_timings", lambda r: r.update(cache_reuse_fraction=1.5), "cache_reuse_fraction"),
        ("backend_timings", lambda r: r.update(attempt_timings=[]), "attempt_timings"),
    ],
)
def test_read_journal_rejects_bad_context_values(tmp_path, event, change, field):
    result, _ = record_claimed(tmp_path, context=naive())
    tamper(result.journal_path, seq_of(result.journal_path, event), change)
    with pytest.raises(JournalFormatError) as caught:
        read_journal(result.journal_path)
    assert caught.value.field == field


def test_invalid_context_records_are_never_written(tmp_path):
    journal = EpisodeJournal.create(tmp_path)
    with pytest.raises(ValueError):
        journal.backend_timings(
            step=0, attempt_metrics=[], attempt_timings=[], rendered_prompt_tokens=10, prompt_evaluated_tokens=None,
            reused_tokens=5, cache_reuse_fraction=None, cache_status="available", ttft_ms=None, prompt_ms=None,
            predicted_ms=None, response_elapsed_ms=None, slot_id=None, measurement_path="unavailable",
        )
    assert not journal.path.exists()

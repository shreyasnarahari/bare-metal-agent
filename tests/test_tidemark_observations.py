"""Observation shaping in the loop: raw/shaped separation, order, replay, resume, the corpus, and the pipeline."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from steadyhand import SteadyhandClient, build_registry, initial_messages, load_fixture, run_episode
from steadyhand.context import ObservationSource
from steadyhand.errors import ErrorType
from steadyhand.journal import EpisodeJournal, JournalFormatError, ReplayMismatch, comparable, manifest_path_for, read_journal
from steadyhand.replay import replay_episode, run_claimed_episode
from steadyhand.security import SecurityConfig, store_task_policy
from steadyhand.types import EpisodeStatus, ToolResult, observation_content
from steadyhand.validate import timeout_observation
from tidemark.assembler import PrefixStableSession, SpanRenderer
from tidemark.blobs import BlobMirror, BlobStore, parse_handle
from tidemark.eval.baseline import EvidenceProbe, fact_available
from tidemark.eval.observation_corpus import check as corpus_check
from tidemark.eval.observations import (
    CorpusScriptedModel,
    autotune_shaping,
    build_corpus_registry,
    load_corpus,
    observation_gate,
    replay_benchmark,
    run_observation_benchmark,
    score_entry,
    shaping_candidates,
    validate_corpus,
)
from tidemark.eval.prefix import ChatMLTemplate
from tidemark.observations import (
    DEFAULT_POLICY,
    LOG_RULE,
    ObservationConfig,
    ObservationPolicy,
    ObservationShaper,
    ReducerCache,
    ReducerRule,
    ShapingPool,
    check_projections,
)
from tidemark.recitation import HEADER
from tidemark.reducers import EXEC_STREAMS, JSON_BOUNDED
from tidemark.scheduler import Scheduler, SchedulerPlan
from tidemark.tokenizer import RegexTokenizer
from tidemark.types import ContextConfig
from toolsmith.protocols import QWEN3_PROFILE

from tidemark_support import FakeService, attempt, big_log, completion, shaping_session, tool_registry

CRITICAL = "src/net/socket.c:88:12: error: 'fd' undeclared (first use in this function)"
LOG = big_log()
SECOND_LOG = big_log(2000, 1000, "FATAL [worker] migration 0042 failed: duplicate key")


def read_log(store, log_id: str) -> str:
    """Read one build log by id."""
    return SECOND_LOG if log_id == "second" else LOG


REGISTRY = tool_registry(read_log)
GOAL = "Why did build 7 fail?"


def turns(*steps):
    built = [completion(calls=[(f"c{i}", name, args)], attempts=[attempt(prompt_n=40)]) for i, (name, args) in enumerate(steps)]
    return [*built, completion("The build failed in src/net/socket.c.", attempts=[attempt(prompt_n=12)])]


def episode(tmp_path, steps, *, namespace="ep-ns", security=None, policy=None, session=None, label=None):
    service = FakeService(*turns(*steps))
    session = session or shaping_session(tmp_path / "blobs", namespace, policy=policy)
    result = run_claimed_episode(tmp_path / "journals", SteadyhandClient(service), REGISTRY, load_fixture().clone(),
                                 initial_messages(GOAL), context=session, security=security, run_label=label)
    return result, read_journal(result.journal_path), service


def source(content_value, *, ok=True, tool="read_log", verifying=False, seq=3, error=None):
    observation = {"ok": True, "value": content_value} if ok else error.model_dict()
    return ObservationSource(episode_id="ep_1", step=0, source_seq=seq, tool_call_id="c1", tool=tool, ok=ok,
                             handler_called=True, observation=observation, content=observation_content(observation),
                             verifying=verifying)


def shaper(tmp_path, policy=DEFAULT_POLICY, namespace="ep-ns", **options):
    store = BlobStore(tmp_path / "blobs")
    return ObservationShaper(ObservationConfig.for_store(store, namespace, policy), RegexTokenizer(), store, **options)


# Raw versus shaped, in the loop


def test_the_raw_result_is_journaled_first_then_the_shape_then_taint(tmp_path):
    security = SecurityConfig(task_policy=store_task_policy(available_tools=("read_log", "read_blob")))
    result, records, service = episode(tmp_path, [("read_log", {"log_id": "build-7"})], security=security)
    assert result.status is EpisodeStatus.DONE
    events = [r["event"] for r in records]
    tool = events.index("tool_result")
    assert events[tool + 1] == "observation_shaped" and events[tool + 2] == "taint_transition"
    raw, shaped = records[tool], records[tool + 1]
    assert raw["value"] == LOG and json.loads(raw["content"]) == {"ok": True, "value": LOG}  # the raw result, unchanged
    assert shaped["source_seq"] == raw["seq"] and shaped["reducer"] == "log_head_tail_v1"
    assert shaped["raw_tokens"] > 10 * shaped["shaped_tokens"] and shaped["blob_handle"] == shaped["blob_parts"][0]["handle"]
    assert records[tool + 2]["source_seq"] == raw["seq"]  # taint comes from the raw result's seq
    message = next(m for m in result.messages if m.get("role") == "tool")
    envelope = json.loads(message["content"])
    assert envelope["ok"] is True and envelope["context"]["shape"] == "log_head_tail_v1"
    assert envelope["context"]["blob"] == shaped["blob_handle"] and CRITICAL in envelope["value"]["preview"]
    assert "value" not in shaped and "content" not in shaped  # no payload in the shape event
    sent = service.requests[1].messages
    assert message in sent and all(LOG not in str(m.get("content")) for m in sent)


def test_shaping_happens_once_and_never_rewrites_an_earlier_message(tmp_path):
    result, records, service = episode(tmp_path, [("read_log", {"log_id": "build-7"}), ("read_log", {"log_id": "second"})])
    first = next(m for m in service.requests[1].messages if m.get("role") == "tool")
    later = [m for m in service.requests[2].messages if m.get("role") == "tool"]
    assert later[0] == first and len(later) == 2
    renders = [r for r in records if r["event"] == "context_render"]
    assert [r["lint_status"] for r in renders] == ["cold_start", "expected_append", "expected_append"]
    assert sum(r["event"] == "observation_shaped" for r in records) == 2
    tail = service.requests[2].messages[-1]["content"]
    assert tail.startswith(HEADER) and all(r["blob_handle"] in tail for r in records if r["event"] == "observation_shaped")


def test_typed_errors_and_small_results_reach_the_model_byte_identical(tmp_path):
    steps = [("get_order", {"order_id": 1}), ("get_order", {"order_id": "ord_1001"}), ("read_log", {"log_id": "build-7"})]
    security = SecurityConfig(task_policy=store_task_policy())  # read_log is unlisted: ASK, not shaped
    shaped, shaped_records, _ = episode(tmp_path / "shaped", steps, security=security)
    legacy_journal = EpisodeJournal.create(tmp_path / "legacy")
    legacy = run_episode(SteadyhandClient(FakeService(*turns(*steps))), REGISTRY, load_fixture().clone(),
                         initial_messages(GOAL), journal=legacy_journal, security=security)
    tool_messages = lambda result: [m for m in result.messages if m.get("role") == "tool"]
    assert tool_messages(shaped) == tool_messages(legacy)
    assert [json.loads(m["content"]).get("error_type") for m in tool_messages(shaped)] == ["invalid_arguments", None, "permission_required"]
    assert not [r for r in shaped_records if r["event"] == "observation_shaped"]


@pytest.mark.parametrize("error", [
    timeout_observation("read_log", LOG_RULE and build_registry().policy("get_order")),
])
def test_the_shaper_never_touches_a_typed_error(tmp_path, error):
    out = shaper(tmp_path).shape(source(None, ok=False, error=error))
    assert out.event is None and out.message["content"] == observation_content(error.model_dict())


def test_a_result_under_the_cap_is_left_alone_unless_audit_mode_records_it(tmp_path):
    small = shaper(tmp_path).shape(source("short output"))
    assert small.event is None and small.message["content"] == observation_content({"ok": True, "value": "short output"})
    audit = ObservationPolicy(DEFAULT_POLICY.rules, audit=True)
    result, records, _ = episode(tmp_path / "audit", [("get_order", {"order_id": "ord_1001"})], policy=audit)
    event = next(r for r in records if r["event"] == "observation_shaped")
    assert event["reducer"] == "none" and event["raw_content_sha256"] == event["shaped_content_sha256"]
    assert event["blob_parts"] == [] and event["preview_metadata"] == {"reason": "under_cap"}
    assert replay_episode(result.journal_path, context=shaping_session(tmp_path / "audit" / "blobs", policy=audit, read_only=True),
                          registry=REGISTRY).status is EpisodeStatus.DONE


def test_shaping_never_makes_an_observation_larger(tmp_path):
    policy = DEFAULT_POLICY.with_rules(read_log=ReducerRule("log_head_tail_v1", preview_tokens=20))
    text = "x " * 40  # just over a tiny cap, smaller than any envelope with a handle
    out = shaper(tmp_path, policy).shape(source(text))
    assert out.event is None and out.message["content"] == observation_content({"ok": True, "value": text})
    assert not list((tmp_path / "blobs").glob("ep-ns/*"))


def test_text_that_cannot_be_encoded_is_left_to_the_upstream_gate(tmp_path):
    out = shaper(tmp_path).shape(source("bad \ud800 surrogate " * 2000))
    assert out.event is None


def test_the_same_raw_result_shapes_to_the_same_bytes_a_thousand_times(tmp_path):
    policy = DEFAULT_POLICY.with_rules(read_log=ReducerRule("log_head_tail_v1", preview_tokens=320, head_lines=4, tail_lines=4))
    text = big_log(220, 110, CRITICAL)
    first = shaper(tmp_path, policy).shape(source(text))
    started = time.monotonic()
    for _ in range(1000):
        again = shaper(tmp_path, policy).shape(source(text))
        assert again.message == first.message and again.event == first.event
    assert time.monotonic() - started < 120 and CRITICAL in first.message["content"]


# Replay, resume, and integrity


def files_under(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size) for p in Path(root).rglob("*") if p.is_file())


def test_replay_reproduces_shaped_messages_without_writing_anything(tmp_path):
    result, records, _ = episode(tmp_path, [("read_log", {"log_id": "build-7"}), ("read_blob", {
        "handle": "blob://sha256:" + __import__("hashlib").sha256(LOG.encode()).hexdigest(), "offset": 0, "max_bytes": 300})])
    read = next(r for r in records if r["event"] == "tool_result" and r["tool"] == "read_blob")
    assert read["ok"] and read["value"]["content"] == LOG.encode()[:read["value"]["end_offset"]].decode()
    before = files_under(tmp_path / "blobs")
    replayed = replay_episode(result.journal_path, context=shaping_session(tmp_path / "blobs", read_only=True), registry=REGISTRY)
    assert replayed.status is EpisodeStatus.DONE and replayed.messages == result.messages
    assert files_under(tmp_path / "blobs") == before


def test_replay_refuses_a_deleted_or_edited_blob_before_running(tmp_path):
    result, records, _ = episode(tmp_path, [("read_log", {"log_id": "build-7"})])
    digest = parse_handle(next(r for r in records if r["event"] == "observation_shaped")["blob_handle"])
    path = tmp_path / "blobs" / "ep-ns" / digest[:2] / digest
    original = path.read_bytes()
    path.write_bytes(original.replace(b"step 7:", b"step 8:"))
    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(result.journal_path, context=shaping_session(tmp_path / "blobs", read_only=True), registry=REGISTRY)
    assert caught.value.field == "blob_handle" and "missing or corrupt" in str(caught.value)
    path.unlink()
    with pytest.raises(ReplayMismatch):
        replay_episode(result.journal_path, context=shaping_session(tmp_path / "blobs", read_only=True), registry=REGISTRY)


def test_a_session_without_a_shaper_cannot_replay_shaped_journals(tmp_path):
    result, _, _ = episode(tmp_path, [("read_log", {"log_id": "build-7"})])
    plain = PrefixStableSession(ContextConfig(slot_context=16384, max_tokens=256, strategy="prefix_stable",
                                              tokenizer=RegexTokenizer().identity()),
                                SpanRenderer(ChatMLTemplate(), RegexTokenizer(), protocol=QWEN3_PROFILE.protocol))
    with pytest.raises(ReplayMismatch):
        replay_episode(result.journal_path, context=plain, registry=REGISTRY)


def test_a_changed_reducer_policy_is_refused_rather_than_reshaped(tmp_path):
    result, _, _ = episode(tmp_path, [("read_log", {"log_id": "build-7"})])
    changed = DEFAULT_POLICY.with_rules(read_log=ReducerRule("log_head_tail_v1", preview_tokens=900))
    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(result.journal_path, context=shaping_session(tmp_path / "blobs", read_only=True, policy=changed),
                       registry=REGISTRY)
    assert caught.value.field == "context_config"


@pytest.mark.parametrize("field,value", [("shaped_tokens", 1), ("omitted_ranges", []), ("catalog_revision", 9)])
def test_editing_shape_evidence_fails_at_its_event(tmp_path, field, value):
    result, records, _ = episode(tmp_path, [("read_log", {"log_id": "build-7"})])
    seq = next(r["seq"] for r in records if r["event"] == "observation_shaped")
    lines = result.journal_path.read_text().splitlines()
    item = json.loads(lines[seq])
    item[field] = value
    lines[seq] = json.dumps(item, sort_keys=True)
    result.journal_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(result.journal_path, context=shaping_session(tmp_path / "blobs", read_only=True), registry=REGISTRY)
    assert caught.value.seq in (seq, None)


def test_a_shape_event_must_follow_its_tool_result(tmp_path):
    result, records, _ = episode(tmp_path, [("read_log", {"log_id": "build-7"})])
    seq = next(r["seq"] for r in records if r["event"] == "observation_shaped")
    lines = result.journal_path.read_text().splitlines()
    item = json.loads(lines[seq])
    item["source_seq"] = seq - 2
    lines[seq] = json.dumps(item, sort_keys=True)
    result.journal_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(JournalFormatError) as caught:
        read_journal(result.journal_path)
    assert caught.value.field == "source_seq"


def interrupt_after(result, event):
    """Cut the journal right after the first ``event`` record and mark the manifest incomplete."""
    records = read_journal(result.journal_path)
    seq = next(r["seq"] for r in records if r["event"] == event)
    lines = result.journal_path.read_text().splitlines()
    result.journal_path.write_text("\n".join(lines[:seq + 1]) + "\n")
    manifest = manifest_path_for(result.journal_path)
    data = json.loads(manifest.read_text())
    data["completed"] = False
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    return records


def test_resume_verifies_the_recorded_blobs_and_continues_live(tmp_path):
    steps = [("read_log", {"log_id": "build-7"}), ("read_log", {"log_id": "second"})]
    result, original, _ = episode(tmp_path, steps)
    interrupt_after(result, "observation_shaped")
    live = FakeService(*turns(*steps)[1:])
    resumed = run_claimed_episode(tmp_path / "journals", SteadyhandClient(live), REGISTRY, load_fixture().clone(),
                                  initial_messages(GOAL), context=shaping_session(tmp_path / "blobs"))
    assert resumed.status is EpisodeStatus.DONE and len(live.requests) == 2
    again = read_journal(resumed.journal_path)
    assert [comparable(r) for r in again if r["event"] != "backend_timings"] == [
        comparable(r) for r in original if r["event"] != "backend_timings"]


def test_resume_refuses_a_missing_blob_before_any_model_call(tmp_path):
    steps = [("read_log", {"log_id": "build-7"}), ("read_log", {"log_id": "second"})]
    result, original, _ = episode(tmp_path, steps)
    records = interrupt_after(result, "observation_shaped")
    digest = parse_handle(next(r for r in records if r["event"] == "observation_shaped")["blob_handle"])
    (tmp_path / "blobs" / "ep-ns" / digest[:2] / digest).unlink()
    live = FakeService(*turns(*steps)[1:])
    with pytest.raises(ReplayMismatch):
        run_claimed_episode(tmp_path / "journals", SteadyhandClient(live), REGISTRY, load_fixture().clone(),
                            initial_messages(GOAL), context=shaping_session(tmp_path / "blobs"))
    assert live.requests == []


# Configuration, policy, evidence


def test_the_observation_config_is_part_of_the_identity_only_when_shaping_is_on(tmp_path):
    plain = ContextConfig(slot_context=4096, max_tokens=256, strategy="prefix_stable")
    assert "observations" not in plain.as_dict() and ContextConfig.from_dict(plain.as_dict()) == plain
    session = shaping_session(tmp_path / "blobs")
    observed = session.config
    assert observed.as_dict()["observations"]["namespace"] == "ep-ns"
    assert ContextConfig.from_dict(observed.as_dict()).as_dict() == observed.as_dict()
    assert observed.as_dict()["observations"]["catalog_id"] == BlobStore(tmp_path / "blobs").catalog_id
    with pytest.raises(ValueError):
        PrefixStableSession(plain, session.renderer, observations=session.observations)


def test_projections_were_reviewed_against_the_store_result_schemas_and_unknown_tools_get_bounded_json():
    registry = build_registry()
    assert check_projections(DEFAULT_POLICY, {name: registry.result_schema(name) for name in registry.names()}) == []
    assert DEFAULT_POLICY.rule_for("run_shell").reducer == EXEC_STREAMS == DEFAULT_POLICY.rule_for("run_python").reducer
    assert DEFAULT_POLICY.rule_for("some_new_tool").reducer == JSON_BOUNDED
    broken = DEFAULT_POLICY.with_rules(get_order=ReducerRule("json_projection_v1", pointers=("/no_such_field",)))
    schemas = {name: registry.result_schema(name) for name in registry.names()}
    assert check_projections(broken, schemas) == ["get_order: /no_such_field is not in the result schema"]
    assert "get_ticket: no result schema to review the projection against" in check_projections(DEFAULT_POLICY, {})
    assert ObservationPolicy.from_dict(DEFAULT_POLICY.as_dict()) == DEFAULT_POLICY


def test_evidence_counts_a_blob_only_through_a_handle_the_request_names(tmp_path):
    out = shaper(tmp_path).shape(source(LOG))
    messages = [out.message]
    blobs = BlobStore(tmp_path / "blobs").namespace("ep-ns")
    hidden = "step 77: compiling module_77.c ok"
    assert fact_available(messages, hidden, "blob_handle", blobs=blobs)["via"] == "blob_handle"
    assert fact_available(messages, hidden, "inline", blobs=blobs)["available"] is False
    assert fact_available(messages, CRITICAL, "blob_handle", blobs=blobs)["via"] == "inline"
    assert fact_available(messages, hidden, "blob_handle", blobs=BlobStore(tmp_path / "other").namespace("ep-ns"))["available"] is False


def test_the_evidence_probe_passes_the_part_3_hooks_through(tmp_path):
    session = shaping_session(tmp_path / "blobs")
    from tidemark.eval.baseline import load_long_horizon

    probe = EvidenceProbe(session, load_long_horizon()[0])
    assert probe.blob_reader is not None and probe.shape_result(source(LOG)).event is not None
    probe.check_journal([])


# The pipeline: pools, caches, mirrors, concurrency


def test_the_shaping_pool_admits_in_order_and_bounds_in_flight_bytes():
    pool = ShapingPool(workers=2, max_inflight_bytes=100)
    running, peak, lock = [], [0], threading.Lock()

    def work(weight):
        with lock:
            running.append(weight)
            peak[0] = max(peak[0], sum(running))
        time.sleep(0.02)
        with lock:
            running.remove(weight)
        return weight

    with ThreadPoolExecutor(max_workers=8) as callers:
        results = list(callers.map(lambda w: pool.run(work, w, weight=w), [60, 60, 30, 250, 10, 40]))
    assert results == [60, 60, 30, 250, 10, 40] and pool.stats["oversized"] == 1
    assert peak[0] <= 250 and pool.stats["peak_inflight_bytes"] <= 250
    assert pool.snapshot()["shaping_running"] == 0 and pool.stats["submitted"] == 6
    pool.close()


def test_the_reducer_cache_skips_cpu_work_but_still_catalogs_each_namespace(tmp_path):
    cache = ReducerCache()
    store = BlobStore(tmp_path / "blobs")
    first = ObservationShaper(ObservationConfig.for_store(store, "ns-a"), RegexTokenizer(), store, cache=cache)
    second = ObservationShaper(ObservationConfig.for_store(store, "ns-b"), RegexTokenizer(), store, cache=cache)
    a, b = first.shape(source(LOG)), second.shape(source(LOG))
    assert a.message == b.message and cache.stats()["hits"] == 1 and second.metrics["cache_hits"] == 1
    digest = parse_handle(a.event["blob_handle"])
    assert store.catalog.lookup("ns-a", digest) and store.catalog.lookup("ns-b", digest)
    assert b.event["catalog_revision"] == 2 and a.event["catalog_revision"] == 1


def test_drive_mirroring_never_sits_on_the_shaping_path(tmp_path):
    release = threading.Event()

    def slow(source, destination):
        release.wait(10)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())

    store = BlobStore(tmp_path / "local")
    mirror = BlobMirror(store, tmp_path / "drive", copy=slow)
    store.on_write = mirror.submit
    shaping = ObservationShaper(ObservationConfig.for_store(store, "ep-ns"), RegexTokenizer(), store)
    started = time.monotonic()
    out = shaping.shape(source(LOG))
    assert time.monotonic() - started < 5 and out.event is not None and mirror.copied == 0
    release.set()
    assert mirror.drain()["copied"] == 1
    mirror.close()


class SlowTokenizer(RegexTokenizer):
    def encode(self, text):
        time.sleep(0.001)
        return super().encode(text)


def test_concurrent_episodes_keep_exact_order_bounded_memory_and_replay(tmp_path):
    entries = [entry for entry in load_corpus() if entry.category in ("log", "exec", "json")][:6]
    registry = build_corpus_registry(entries)
    pool = ShapingPool(workers=2, max_inflight_bytes=256 * 1024)
    store = BlobStore(tmp_path / "blobs")

    def session(entry, client=None, *, read_only=False):
        return shaping_session(tmp_path / "blobs", f"bench.{entry.observation_id}", pool=None if read_only else pool,
                               read_only=read_only, tokenizer=SlowTokenizer("regex-fake-v1"))

    plan = SchedulerPlan(agent_slots=2, episode_workers=4)
    with Scheduler(plan, cpu_executor=lambda: ThreadPoolExecutor(max_workers=1)) as scheduler:
        report = run_observation_benchmark(
            entries, run_dir=tmp_path / "run", run_label="bench-test", scheduler=scheduler, registry=registry,
            client_factory=lambda entry, owner: CorpusScriptedModel(entry, delay_s=0.005),
            session_factory=lambda entry, client: session(entry),
        )
    rows = report["rows"]
    assert len(rows) == len(entries) and all(row.status == "done" for row in rows)
    assert all(row.shaped >= 1 for row in rows) and report["shaped_observations"] >= len(entries)
    assert pool.stats["peak_inflight_bytes"] <= 256 * 1024 or pool.stats["oversized"]
    for row in rows:
        records = read_journal(row.journal_path)
        for index, record in enumerate(records):
            if record["event"] == "observation_shaped":
                assert records[index - 1]["event"] == "tool_result" and records[index - 1]["seq"] == record["source_seq"]
    by_id = {entry.observation_id: entry for entry in entries}
    replay = replay_benchmark(rows, registry=registry,
                              session_factory=lambda ident: session(by_id[ident], read_only=True))
    assert replay["passed"], replay["failures"]
    assert store.stats["objects_written"] == 0  # this view wrote nothing; every object was written once by the runs
    objects = [path for path in (tmp_path / "blobs").rglob("*") if path.is_file() and len(path.name) == 64]
    assert len(objects) == len({(path.parent.parent.name, path.name) for path in objects})
    pool.close()


def test_autotune_freezes_the_fastest_correct_profile():
    profiles = shaping_candidates(2, 4096, llama_threads=1)
    assert {profile.workers for profile in profiles} == {1}
    profiles = shaping_candidates(12, 40000, llama_threads=3)
    assert {profile.workers for profile in profiles} == {1, 4, 9}
    measured = {1: 2.0, 4: 5.0, 9: 5.0}
    frozen = autotune_shaping(profiles, lambda p: {"episodes_per_minute": measured[p.workers], "correct": p.workers != 9})
    assert frozen["selected"]["workers"] == 4 and len(frozen["measurements"]) == len(profiles)
    with pytest.raises(RuntimeError):
        autotune_shaping(profiles[:1], lambda p: {"episodes_per_minute": 9.0, "correct": False})


# The corpus


def test_the_committed_corpus_matches_its_builder_and_validates():
    assert corpus_check() == []
    entries = load_corpus()
    summary = validate_corpus(entries)
    assert 30 <= summary["entries"] <= 50 and all(summary["categories"][name] for name in ("pytest", "compiler", "log", "exec", "json", "html"))
    assert summary["labels"] >= 80 and summary["inline_labels"] < summary["labels"]


def test_the_observation_gate_reports_reduction_and_restorability_separately(tmp_path):
    gate = observation_gate(tmp_path / "gate", repeats=1)
    scores = gate["scores"]
    assert gate["passed"] and gate["deterministic"] and scores["survival_or_restorability"] == 1.0
    assert scores["token_reduction"] > 0.9 and scores["must_keep"]["restorable_only"] > 0 and scores["must_keep"]["inline"] > 0
    assert scores["must_keep"]["inline_required"] == scores["must_keep"]["inline_required_met"]
    assert all(trip["equal"] for trip in gate["round_trips"]) and not scores["forbidden_violations"]


def test_a_broken_handle_is_a_failure_not_a_compression_win(tmp_path):
    entry = next(e for e in load_corpus() if e.observation_id == "store-search-650-orders")
    shaping = shaper(tmp_path)
    good = score_entry(entry, shaping)
    assert good["passed"]
    digest = parse_handle(good["blob_handles"][0])
    (tmp_path / "blobs" / "ep-ns" / digest[:2] / digest).write_bytes(b"{}")
    broken = score_entry(entry, shaping)
    assert not broken["passed"] and broken["blob_failures"]
    assert any(not label["passed"] for label in broken["labels"] if label["availability"] == "blob_handle")


def test_reducer_output_is_pinned_so_a_change_needs_a_version_bump(tmp_path):
    """Recorded journals replay only if shaping is byte-stable; regenerate the golden deliberately with a new version."""
    from tidemark.reducers import REDUCER_VERSION

    golden = json.loads((Path(__file__).parent / "fixtures" / "tidemark_observations" / "reducer_goldens.json").read_text())
    assert golden["reducer_version"] == REDUCER_VERSION and golden["policy_sha256"] == DEFAULT_POLICY.sha256()
    entries = {entry.observation_id: entry for entry in load_corpus()}
    shaping = shaper(tmp_path, namespace="golden")
    for observation_id, expected in golden["entries"].items():
        out = shaping.shape(source(entries[observation_id].load_value(), tool=entries[observation_id].tool))
        assert {
            "shaped_content_sha256": __import__("hashlib").sha256(out.message["content"].encode()).hexdigest(),
            "blob_handles": [part["handle"] for part in out.event["blob_parts"]],
            "shaped_tokens": out.event["shaped_tokens"], "omitted_ranges": len(out.event["omitted_ranges"]),
        } == expected, observation_id

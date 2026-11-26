"""PrefixLint: planted cache breakers, legal transitions, floors, attribution, and the local fingerprint store."""

import pytest

from tidemark.assembler import SpanRenderer
from tidemark.eval.prefix import CASES, ChatMLTemplate, planted_prompts, planted_suite
from tidemark.prefix_lint import (
    COLD_START,
    EXPECTED_APPEND,
    EXPECTED_TAIL_ONLY,
    VIOLATION,
    Fingerprint,
    FingerprintStore,
    common_prefix,
    lint,
)
from tidemark.tokenizer import RegexTokenizer
from toolsmith.protocols import QWEN3_PROFILE, QWEN35_PROFILE


def renderer(profile=QWEN3_PROFILE):
    return SpanRenderer(ChatMLTemplate(), RegexTokenizer(), protocol=profile.protocol)


@pytest.mark.parametrize("profile", [QWEN3_PROFILE, QWEN35_PROFILE], ids=lambda p: p.name)
def test_every_planted_breaker_and_legal_change_is_classified(profile):
    suite = planted_suite(renderer(profile), repeats=10)
    assert suite["deterministic"] and suite["span_status"] == "ok"
    failures = [case for case in suite["cases"] if not case["passed"]]
    assert failures == [] and suite["passed"]
    by_case = {case["case"]: case for case in suite["cases"]}
    assert [case for case, _, _ in CASES] == list(by_case)
    for name in ("timestamp_in_frozen", "unsorted_tool_keys", "reordered_tools"):
        assert by_case[name]["first_changed_segment"] == "frozen"
        assert by_case[name]["first_divergent_index"] < suite["base_spans"][0][2]  # inside the frozen span
    assert by_case["pinned_memory_replaced"]["first_changed_segment"] == "pinned"
    assert by_case["recent_append"]["observed_common_prefix_tokens"] >= by_case["recent_append"]["expected_common_prefix_floor"]
    epoch = by_case["epoch_transition"]
    assert epoch["status"] == "expected_epoch" and epoch["expected_common_prefix_floor"] == suite["base_spans"][1][2]


def test_violations_name_the_exact_first_divergent_token():
    prompts = planted_prompts(renderer())
    base = RegexTokenizer().encode(prompts["base"])
    planted = RegexTokenizer().encode(prompts["timestamp_in_frozen"])
    suite = planted_suite(renderer(), repeats=1)
    timestamp = next(case for case in suite["cases"] if case["case"] == "timestamp_in_frozen")
    assert timestamp["first_divergent_index"] == common_prefix(base, planted)
    assert base[: common_prefix(base, planted)] == planted[: common_prefix(base, planted)]


def fingerprint(ids, *, stable, recent=("a",), tail="t1", epoch=0, frozen="f", spans=()):
    return Fingerprint(tuple(ids), f"sha-{len(ids)}-{tail}", stable, 2,
                       {"frozen": frozen, "pinned": "p", "summary": None, "recent": "r", "tail": tail},
                       tuple(recent), epoch, spans)


def test_cold_start_tail_only_and_identical_renders():
    first = fingerprint([1, 2, 3, 9], stable=3)
    assert lint(None, first).status == COLD_START and lint(None, first).first_divergent_index is None
    tail = fingerprint([1, 2, 3, 8], stable=3, tail="t2")
    result = lint(first, tail)
    assert result.status == EXPECTED_TAIL_ONLY and result.first_divergent_index == 3 and result.divergence_cause == "tail_rewrite"
    same = lint(first, fingerprint([1, 2, 3, 9], stable=3))
    assert same.status == EXPECTED_TAIL_ONLY and same.first_divergent_index is None and same.divergence_cause == "unchanged"
    grown = lint(first, fingerprint([1, 2, 3, 4, 5, 9], stable=5, recent=("a", "b")))
    assert grown.status == EXPECTED_APPEND and grown.observed_common_prefix_tokens == 3


def test_a_token_change_before_the_floor_without_a_logical_change_is_a_boundary_merge():
    previous = fingerprint([1, 2, 3, 9], stable=3, spans=(("frozen", 0, 1), ("recent", 1, 3), ("tail", 3, 4)))
    merged = fingerprint([1, 7, 3, 4, 9], stable=4, recent=("a", "b"))
    result = lint(previous, merged)
    assert result.status == VIOLATION and result.divergence_cause == "token_boundary_merge"
    assert result.first_divergent_index == 1 and result.first_changed_segment == "recent"


def test_without_spans_only_logical_changes_decide():
    previous = fingerprint([1, 2, 3], stable=None)
    ok = lint(previous, fingerprint([1, 2, 3, 4], stable=None, recent=("a", "b")))
    assert ok.status == EXPECTED_APPEND and ok.expected_common_prefix_floor is None
    assert ok.detail == {"floor": "span_mapping_unavailable"}
    bad = lint(previous, fingerprint([5, 2, 3], stable=None, frozen="changed"))
    assert bad.status == VIOLATION and bad.divergence_cause == "frozen_changed" and bad.first_divergent_index == 0


def test_fingerprints_round_trip_through_the_compact_local_store(tmp_path):
    store = FingerprintStore(tmp_path / "fingerprints")
    original = fingerprint(range(5000), stable=4990, spans=(("frozen", 0, 10), ("tail", 4990, 5000)))
    path = store.save("episode-1", original)
    assert path.stat().st_size == 5000 * 4  # four bytes per id, not a JSON array
    assert store.load("episode-1") == original and store.load("missing") is None

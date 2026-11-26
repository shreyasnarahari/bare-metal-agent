"""Exact budgets: hard and soft limits, per-segment caps, unavailable spans, and static configuration checks."""

import pytest

from tidemark.budgets import COMPACTION_PENDING, EXHAUSTED, OK, BudgetConfigError, check_static, evaluate
from tidemark.types import ContextConfig


def config(**overrides):
    values = dict(slot_context=4096, max_tokens=256, safety_margin=64, strategy="prefix_stable")
    return ContextConfig(**{**values, **overrides})


SEGMENTS = {"frozen": 900, "pinned": 100, "summary": 0, "recent": 1000, "tail": 120}


def test_the_hard_limit_reserves_generation_and_margin():
    table = evaluate(config(), 2120, SEGMENTS, 12)
    assert table.hard_limit == 4096 - 256 - 64 == 3776 and table.soft_limit == int(0.7 * 3776)
    assert table.decision == OK and table.hard_headroom == 3776 - 2120 and table.spans_available
    assert table.segment("tail").cap == 512 and table.segment("tail").remaining == 392
    assert table.tokens()["prompt"] == 2120 and table.tokens()["generation_prompt"] == 12
    assert table.budgets(config())["completion_reserve"] == 256 and table.budgets(config())["frozen"] is None


def test_soft_limit_and_recent_cap_mark_compaction_pending():
    assert evaluate(config(), 2700, SEGMENTS).decision == COMPACTION_PENDING
    capped = evaluate(config(segment_caps={"recent": 800}), 2120, SEGMENTS)
    assert capped.decision == COMPACTION_PENDING and "recent uses 1000" in capped.reasons[0]
    assert evaluate(config(soft_limit=3000), 2700, SEGMENTS).decision == OK


def test_over_the_hard_limit_or_a_fixed_segment_cap_is_exhausted():
    over = evaluate(config(), 3777, SEGMENTS)
    assert over.decision == EXHAUSTED and "hard limit is 3776" in over.reasons[0]
    assert evaluate(config(), 3776, SEGMENTS).decision == COMPACTION_PENDING  # exactly at the limit still fits
    assert evaluate(config(segment_caps={"pinned": 50}), 2120, SEGMENTS).decision == EXHAUSTED
    assert evaluate(config(segment_caps={"tail": 100}), 2120, SEGMENTS).decision == EXHAUSTED


def test_without_spans_only_the_exact_total_is_judged():
    table = evaluate(config(segment_caps={"recent": 10}), 2120, None)
    assert not table.spans_available and table.decision == OK
    assert all(item.used is None for item in table.segments) and table.tokens()["prompt"] == 2120


def test_a_frozen_segment_too_large_fails_configuration():
    with pytest.raises(BudgetConfigError, match="cap"):
        check_static(config(segment_caps={"frozen": 500}), 900)
    with pytest.raises(BudgetConfigError, match="no history room"):
        check_static(config(), 3300, 100)
    check_static(config(), 900, 100)


def test_configuration_rules():
    with pytest.raises(ValueError):
        config(segment_caps={"middle": 10})
    with pytest.raises(ValueError):
        config(segment_caps={"tail": 0})
    with pytest.raises(ValueError):
        config(overflow="truncate")
    with pytest.raises(ValueError):
        config(allowed_tool_names=())
    restored = ContextConfig.from_dict(config(allowed_tool_names=("get_order",), pinned_memory=("x",)).as_dict())
    assert restored.allowed_tool_names == ("get_order",) and restored.pinned_memory == ("x",)

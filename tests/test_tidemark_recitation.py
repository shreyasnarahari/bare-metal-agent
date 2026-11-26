"""The recitation tail: deterministic, bounded, priority-trimmed, and free of clock or random values."""

import re

from tidemark.recitation import HEADER, PlanItem, RecitationState, bound


def state(**overrides):
    values = dict(completed_step_count=3, prompt_tokens_used=1200, remaining_step_budget=5,
                  current_plan_items=(PlanItem("read ord_1001"), PlanItem("note the code", priority=2)))
    return RecitationState(**{**values, **overrides})


def test_rendering_is_deterministic_and_has_no_clock_or_random_values():
    first, second = state().render(), state().render()
    assert first == second and first.startswith(HEADER) and state().sha256() == state().sha256()
    assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}|[0-9a-f]{16}", first)
    assert "3 steps done; 5 steps left; last prompt 1200 tokens" in first and state().message()["role"] == "user"


def test_unknown_values_render_as_such_not_as_zero():
    text = RecitationState().render()
    assert "unknown steps left" in text and "last prompt n/a tokens" in text and "Plan: none recorded." in text


def test_static_bounds_clip_text_keep_priority_and_recent_items():
    items = tuple(PlanItem(f"item {i} " + "x" * 200, done=(i == 0), priority=(1 if i == 3 else 0)) for i in range(6))
    bounded, evicted = bound(state(current_plan_items=items, open_blob_handles=("b1", "b2", "b3")),
                             max_items=3, max_item_chars=20, max_handles=2)
    kept = [item.text for item in bounded.current_plan_items]
    assert all(len(text) <= 20 for text in kept) and len(kept) == 3
    assert kept[0].startswith("item 3") or any(t.startswith("item 3") for t in kept)  # the priority item survives
    assert not any(t.startswith("item 0") for t in kept)  # a completed item goes first
    assert bounded.open_blob_handles == ("b2", "b3") and "handle:b1" in evicted


def test_eviction_until_it_fits_drops_low_priority_then_handles_and_keeps_required_lines():
    start = state(current_plan_items=(PlanItem("low"), PlanItem("high", priority=5)), open_blob_handles=("h1",))
    bounded, evicted = bound(start, max_items=5, max_item_chars=50, max_handles=4,
                             fits=lambda candidate: not candidate.open_blob_handles)
    assert evicted == ("plan:low", "plan:high", "handle:h1")  # plan items by priority, then handles
    assert "Progress:" in bounded.render() and "Next:" in bounded.render()
    partial, dropped = bound(start, max_items=5, max_item_chars=50, max_handles=4,
                             fits=lambda candidate: len(candidate.current_plan_items) <= 1)
    assert dropped == ("plan:low",) and partial.current_plan_items[0].text == "high"
    never, _ = bound(start, max_items=5, max_item_chars=50, max_handles=4, fits=lambda candidate: False)
    assert never.current_plan_items == () and never.open_blob_handles == ()  # stops with only required lines

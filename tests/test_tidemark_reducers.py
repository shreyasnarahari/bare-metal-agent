"""Typed reducers: logs, text, exec streams, JSON projection and bounding, HTML pages, and the token search."""

import json
from dataclasses import replace

import pytest

from bulkhead.output import render, sanitize
from tidemark.observations import EXEC_RULE, HTML_RULE, LOG_RULE, ReducerRule
from tidemark.reducers import (
    EXEC_STREAMS,
    JSON_BOUNDED,
    JSON_PROJECTION,
    LOG_HEAD_TAIL,
    Candidate,
    ReductionPlan,
    canonical_payload,
    choose,
    line_doc,
    page_text,
    plan_for,
    pointer_tree,
    project,
    render_lines,
    replace_pointer,
    resolve_pointer,
    selections,
    split_lines,
)


def text_of(candidate):
    return json.dumps(candidate.value, ensure_ascii=False)


def preview_of(plan, level):
    return plan.candidates[level]()


def log(lines, **changes):
    return "".join(f"{line}\n" for line in lines)


def test_split_lines_keeps_endings_and_only_splits_on_newline():
    assert split_lines("a\nb\r\nc\rd") == ["a\n", "b\r\n", "c\rd"]
    assert split_lines("x\n") == ["x\n"] and split_lines("") == []


def test_a_critical_line_in_the_middle_is_marked_and_kept_at_every_level():
    lines = [f"step {i} ok" for i in range(4000)]
    lines[2000] = "src/app.py:41: error: name 'fd' is not defined"
    plan = plan_for(LOG_HEAD_TAIL, log(lines), LOG_RULE)
    for level in range(len(plan.candidates) - 1):  # every level but the last (wrapper/forced only)
        candidate = preview_of(plan, level)
        assert "src/app.py:41: error: name 'fd' is not defined" in candidate.value["preview"]
    assert plan.parts[0].payload == log(lines).encode()


def test_separated_identical_failures_both_stay_and_consecutive_duplicates_coalesce():
    lines = ["noise"] * 50 + ["ERROR connection refused"] + ["noise b"] * 50 + ["ERROR connection refused"] + ["tail"] * 50
    doc = line_doc(log(lines), critical=__import__("tidemark.reducers", fromlist=["_compile"])._compile(["ERROR"]))
    assert len(doc.groups) == 5 and doc.groups[0] == (0, 50) and doc.groups[1] == (50, 1)
    preview, ranges, counts = render_lines(doc, frozenset({1, 3}), part="value")
    assert preview.count("ERROR connection refused") == 2
    keep_all, _, _ = render_lines(doc, frozenset(range(len(doc.groups))), part="value")
    assert "previous line repeated 49 more times (2-50)" in keep_all


def test_omitted_ranges_name_the_exact_bytes_left_out():
    lines = [f"line {i} é" for i in range(300)]
    text = log(lines)
    doc = line_doc(text)
    keep = frozenset({0, 1, 299})
    preview, ranges, counts = render_lines(doc, keep, part="value")
    payload = text.encode()
    assert [r["lines"] for r in ranges] == [[2, 299]]
    start, end = ranges[0]["bytes"]
    assert payload[start:end].decode() == "".join(lines[i] + "\n" for i in range(2, 299))
    assert f"read_blob offset {start}" in preview and counts == {"kept_lines": 3, "omitted_lines": 297}


def test_a_long_line_is_clipped_around_its_first_critical_match_with_its_byte_spans():
    body = "x" * 5000 + " FATAL disk full on /var/lib/ledger " + "y" * 5000
    rule = replace(LOG_RULE, max_line_chars=200)
    plan = plan_for(LOG_HEAD_TAIL, log(["start", body, "end"]), rule)
    candidate = preview_of(plan, 0)
    assert "FATAL disk full on /var/lib/ledger" in candidate.value["preview"]
    clipped = [r for r in candidate.omitted_ranges if r["lines"] == [1, 2]]
    assert len(clipped) == 2 and all(r["bytes"][0] < r["bytes"][1] for r in clipped)
    assert "bytes of this line omitted" in candidate.value["preview"]


def test_selections_shrink_head_and_tail_then_drop_tier_two_then_keep_earliest_and_latest_critical():
    lines = ["info"] * 100
    for index in (10, 30, 50, 70, 90):
        lines[index] = f"ERROR number {index}"
    lines[40] = "see src/a.py:12 for details"
    from tidemark.reducers import CRITICAL_PATTERNS, PATH_PATTERNS, _compile

    doc = line_doc(log(lines), critical=_compile(CRITICAL_PATTERNS), secondary=_compile(PATH_PATTERNS))
    picks = selections(doc, 10, 10)
    sizes = [len(pick) for pick in picks]
    assert sizes == sorted(sizes, reverse=True) or sizes[0] >= sizes[-1]
    path_group = next(g for g, (start, _) in enumerate(doc.groups) if start == 40)
    assert path_group in picks[0] and path_group not in picks[-2]
    critical = [g for g, tier in enumerate(doc.tiers) if tier == 1]
    assert {critical[0], critical[-1]} <= picks[-2] and picks[-1] == frozenset()


def test_choose_takes_the_least_aggressive_fitting_candidate_and_reports_over_cap():
    built = []

    def candidate(size):
        def make():
            built.append(size)
            return Candidate(size, {}, (), {"size": size})
        return make

    plan = ReductionPlan("test", (), tuple(candidate(size) for size in (100, 80, 60, 40, 20)))
    choice = choose(plan, lambda c: c.value, 65)
    assert choice.candidate.value == 60 and choice.level == 2 and not choice.over_cap and choice.measured <= 4
    choice = choose(plan, lambda c: c.value, 10)
    assert choice.over_cap and choice.candidate.value == 20


def gated(stdout, stderr="", exit_code=1, signal=None):
    value = {"exit_code": exit_code, "signal": signal}
    for name, text in (("stdout", stdout), ("stderr", stderr)):
        value[name], _ = render(sanitize(text.encode()), request_sha256="0" * 64, stream=name)
    return value


def test_exec_streams_stay_separate_keep_the_gate_wrapper_and_pass_codes_through():
    stdout = log([f"compiling {i}" for i in range(2000)])
    stderr = log(["warning: a"] * 3 + ["src/x.c:9:1: error: bad thing"] + [f"note {i}" for i in range(1500)])
    value = gated(stdout, stderr, exit_code=2, signal=None)
    plan = plan_for(EXEC_STREAMS, value, EXEC_RULE)
    assert plan.reducer == EXEC_STREAMS and [part.part for part in plan.parts] == ["stderr", "stdout"]
    candidate = preview_of(plan, 3)
    shaped = candidate.value
    assert shaped["exit_code"] == 2 and shaped["signal"] is None
    for name in ("stdout", "stderr"):
        content = shaped[name]["content"]
        assert content.startswith("[BEGIN UNTRUSTED OUTPUT ") and content.rstrip("\n").endswith("]")
        assert content.splitlines()[-1].startswith("[END UNTRUSTED OUTPUT ")
        assert shaped[name]["trust"] == "untrusted" and shaped[name]["truncated"] is False
        assert shaped[name]["blob"] == next(p.handle for p in plan.parts if p.part == name)
    assert "src/x.c:9:1: error: bad thing" in shaped["stderr"]["content"]
    assert "src/x.c" not in shaped["stdout"]["content"]  # nothing is flattened across streams
    assert {r["part"] for r in candidate.omitted_ranges} == {"stdout", "stderr"}
    assert plan.parts[0].payload.decode() == value["stderr"]["content"]
    final = preview_of(plan, len(plan.candidates) - 1).value
    assert final["stdout"]["content"].startswith("[BEGIN") and final["stdout"]["content"].rstrip().endswith("]")


def test_an_empty_stream_is_left_alone_and_a_wrong_shape_falls_back_to_bounded_json():
    value = gated(log(["x"] * 10))
    plan = plan_for(EXEC_STREAMS, value, EXEC_RULE)
    assert [part.part for part in plan.parts] == ["stdout"] and preview_of(plan, 0).value["stderr"]["content"] == ""
    odd = {"exit_code": 0, "stdout": "plain"}
    assert plan_for(EXEC_STREAMS, odd, EXEC_RULE).reducer == JSON_BOUNDED


PAYMENTS_RULE = ReducerRule(JSON_PROJECTION, preview_tokens=500, pointers=(
    "/order_id", "/payments/*/payment_id", "/payments/*/amount_inr", "/count",
), required=("/order_id", "/count"))


def payments(count=300):
    return {"order_id": "ord_1", "count": count, "note": "free text " * 50,
            "payments": [{"payment_id": f"pay_{i}", "amount_inr": f"{i}.00", "reason": "r" * 80} for i in range(count)]}


def test_projection_keeps_allowed_pointers_types_and_order_and_counts_omissions():
    value = payments(5)
    projected, omitted = project(value, pointer_tree(PAYMENTS_RULE.pointers))
    assert projected == {"order_id": "ord_1", "count": 5,
                         "payments": [{"payment_id": f"pay_{i}", "amount_inr": f"{i}.00"} for i in range(5)]}
    assert omitted == {"/note": 1, "/payments/*/reason": 5}


def test_projection_levels_cut_arrays_to_a_prefix_but_keep_required_fields():
    plan = plan_for(JSON_PROJECTION, payments(), PAYMENTS_RULE)
    assert plan.reducer == JSON_PROJECTION and json.loads(plan.parts[0].payload) == payments()
    final = preview_of(plan, len(plan.candidates) - 1)
    assert final.value["order_id"] == "ord_1" and final.value["count"] == 300
    middle = preview_of(plan, 4)
    ids = [item["payment_id"] for item in middle.value["payments"]]
    assert ids == [f"pay_{i}" for i in range(len(ids))] and len(ids) < 300
    assert {"pointer": "/payments", "items": [len(ids), 300]} in middle.context["cut"]
    assert middle.context["omitted_fields"] == [{"count": 1, "pointer": "/note"}, {"count": 300, "pointer": "/payments/*/reason"}]
    assert "r" * 80 not in text_of(middle)  # omitted values never appear, only their pointers and counts


def test_bounded_json_takes_scalars_first_and_limits_depth_without_changing_types():
    value = {"z_id": "keep-me", "a_nested": {"b": {"c": {"d": {"e": 1}}}}, "m_list": list(range(100))}
    plan = plan_for(JSON_BOUNDED, value, ReducerRule(JSON_BOUNDED, preview_tokens=100))
    final = preview_of(plan, len(plan.candidates) - 1).value
    assert final["z_id"] == "keep-me" and isinstance(final["a_nested"], dict) and isinstance(final.get("m_list", []), list)
    assert resolve_pointer(final, "/a_nested/b/c") is None


def test_bounded_json_on_a_bare_string_uses_a_typed_text_object():
    plan = plan_for(JSON_BOUNDED, "line\n" * 5000, ReducerRule(JSON_BOUNDED, preview_tokens=100))
    candidate = preview_of(plan, 0)
    assert candidate.value["original_kind"] == "string" and candidate.value["blob"] == plan.parts[0].handle


def test_canonical_payload_is_deterministic_and_refuses_non_finite_numbers():
    assert canonical_payload({"b": 1, "a": [1, 2]}) == canonical_payload({"a": [1, 2], "b": 1})
    with pytest.raises(ValueError):
        canonical_payload({"x": float("nan")})


PAGE = """<!doctype html><html><head><title>Ops &amp; docs</title><style>.a{}</style>
<script>var token = "sk_live_fake";</script></head><body>
<nav><a href="/1">One</a> <a href="/2">Two</a></nav>
<h2>Recover</h2><p>Follow the <a href="https://docs.example.test/run#step-3">runbook</a> &mdash; carefully.</p>
<table><tr><th>Host</th><th>State</th></tr><tr><td>db-1</td><td>down</td></tr></table>
<pre><code>ledgerctl repair --all
  --force</code></pre><ul><li>first</li></ul>
<a href="javascript:alert(1)">click</a><img src="x.png" alt="diagram"></body></html>"""


def test_page_text_keeps_visible_text_links_tables_and_code_and_drops_script_style_and_nav():
    title, lines = page_text(PAGE)
    texts = [line for line, _ in lines]
    assert title == "Ops & docs"
    assert "## Recover" in texts and "Follow the [runbook](https://docs.example.test/run#step-3) — carefully." in texts
    assert "Host | State" in texts and "db-1 | down" in texts
    assert texts[texts.index("```") + 1:texts.index("```") + 3] == ["ledgerctl repair --all", "  --force"]
    assert "- first" in texts and "click[image: diagram]" in texts  # inline elements share their block's line
    joined = "\n".join(texts)
    assert "sk_live_fake" not in joined and ".a{}" not in joined and "[One]" not in joined
    offsets = [offset for _, offset in lines]
    assert offsets == sorted(offsets)
    assert PAGE.encode()[offsets[0]:].decode().startswith("<h2>")


def test_html_plans_keep_the_original_page_as_the_blob_and_replace_only_the_body():
    value = {"url": "https://x.test", "status": 200, "content_type": "text/html", "body": PAGE * 40}
    plan = plan_for("html_text_v1", value, HTML_RULE)
    candidate = preview_of(plan, 0)
    assert plan.parts[0].payload == (PAGE * 40).encode() and plan.parts[0].part == "body"
    assert candidate.value["url"] == "https://x.test" and candidate.value["status"] == 200
    body = candidate.value["body"]
    assert body["original_kind"] == "string" and body["title"] == "Ops & docs" and "sk_live_fake" not in body["preview"]


def test_pointer_helpers():
    value = {"a": [{"b": 1}, {"b": 2}], "c~d": {"e/f": 3}}
    assert resolve_pointer(value, "/a/1/b") == 2 and resolve_pointer(value, "/c~0d/e~1f") == 3
    assert resolve_pointer(value, "/missing") is None and resolve_pointer(value, "") is value
    assert replace_pointer(value, "/a/0/b", 9)["a"][0]["b"] == 9 and value["a"][0]["b"] == 1
    assert pointer_tree(["/a/*/b", "/a/*/c", "/d"]) == {"a": {"*": {"b": True, "c": True}}, "d": True}

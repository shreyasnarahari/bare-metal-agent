"""Prefix-stable assembly: deterministic renders, legal messages, exact spans, the session, masking, and replay."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from steadyhand import Budgets, EpisodeJournal, SteadyhandClient, build_registry, initial_messages, load_fixture, run_episode
from steadyhand.journal import ReplayMismatch, comparable, read_journal
from steadyhand.replay import replay_episode, run_claimed_episode
from steadyhand.types import EpisodeStatus
from tidemark.assembler import ContextAssembler, PrefixStableSession, SpanRenderer, initial_state
from tidemark.budgets import BudgetConfigError
from tidemark.cache import ContentCache
from tidemark.eval.baseline import BaselineSpec, LongHorizonScriptedModel, dataset_sha256, load_long_horizon, run_baseline
from tidemark.eval.prefix import TOOLS, ChatMLTemplate, base_state
from tidemark.recitation import HEADER
from tidemark.scheduler import Scheduler, SchedulerPlan
from tidemark.segments import SegmentError
from tidemark.tokenizer import RegexTokenizer
from tidemark.types import ContextConfig
from toolsmith.api import APIError, ChatCompletionRequest, ToolsmithService
from toolsmith.parsing import Qwen3ToolCallParser
from toolsmith.prompting import SPAN_MAPPING_UNAVAILABLE, render_with_token_spans
from toolsmith.protocols import QWEN3_PROFILE, QWEN35_PROFILE

from tidemark_support import FakeService, attempt, completion

GOAL = "What is the current status of order ord_1001?"
STATUS_TURNS = [
    completion(calls=[("call_status", "get_order", {"order_id": "ord_1001"})], attempts=[attempt(prompt_n=40)]),
    completion("Order ord_1001 was delivered.", attempts=[attempt(prompt_n=12)]),
]


def config(slot_context=16384, **overrides):
    return ContextConfig(slot_context=slot_context, max_tokens=256, strategy="prefix_stable",
                         tokenizer=RegexTokenizer().identity(), **overrides)


def span_renderer(profile=QWEN3_PROFILE, template=None, cache=None):
    return SpanRenderer(template or ChatMLTemplate(), RegexTokenizer(), protocol=profile.protocol, cache=cache)


def session(slot_context=16384, profile=QWEN3_PROFILE, template=None, cache=None, **overrides):
    return PrefixStableSession(config(slot_context, **overrides), span_renderer(profile, template, cache))


# The pure assembler


@pytest.mark.parametrize("profile", [QWEN3_PROFILE, QWEN35_PROFILE], ids=lambda p: p.name)
def test_the_same_state_renders_identically_a_thousand_times(profile):
    assembler = ContextAssembler(span_renderer(profile))
    state = base_state()
    first = assembler.render(state)
    started = time.monotonic()
    for _ in range(1000):
        again = assembler.render(state)
        assert (again.spans.text, again.spans.token_ids, again.spans.spans, again.messages, again.fingerprint) == (
            first.spans.text, first.spans.token_ids, first.spans.spans, first.messages, first.fingerprint)
    assert time.monotonic() - started < 60


def test_the_layout_is_legal_and_ends_with_exactly_one_tail():
    messages, boundaries = base_state().layout()
    ToolsmithService._validate_messages(messages)
    assert [m["role"] for m in messages].count("system") == 1 and messages[0]["role"] == "system"
    assert messages[-1]["content"].startswith(HEADER) and sum(HEADER in (m["content"] or "") for m in messages) == 1
    assert [name for name, _ in boundaries] == ["frozen", "pinned", "summary", "recent", "tail"]


def test_transitions_return_new_states_and_never_rewrite_old_messages():
    state = base_state()
    appended = state.append([{"role": "user", "content": "parse_error"}], [12])
    assert len(state.recent) == 2 and len(appended.recent) == 3 and appended.recent[:2] == state.recent
    next_epoch = appended.advance_epoch({"facts": []}, keep_from=2)
    assert next_epoch.epoch == state.epoch + 1 and next_epoch.recent == appended.recent[2:]
    assert next_epoch.recent_seqs == (12,) and state.sha256() != appended.sha256()


@pytest.mark.parametrize("profile", [QWEN3_PROFILE, QWEN35_PROFILE], ids=lambda p: p.name)
def test_the_production_prompt_is_the_linted_prompt(profile):
    assembled = ContextAssembler(span_renderer(profile)).render(base_state())
    service = ToolsmithService(object(), ChatMLTemplate(), profile=profile)
    request = ChatCompletionRequest(model=profile.model, messages=list(assembled.messages), tools=list(assembled.tools),
                                    chat_template_kwargs={"enable_thinking": False})
    assert service.render_prompt(request) == assembled.spans.text


def test_qwen35_history_is_normalized_exactly_once():
    seen = []

    class Recording(ChatMLTemplate):
        def render(self, messages, tools=(), **options):
            seen.extend(call["function"]["arguments"] for m in messages for call in m.get("tool_calls") or [])
            return super().render(messages, tools, **options)

    ContextAssembler(span_renderer(QWEN35_PROFILE, Recording())).render(base_state())
    assert seen and all(isinstance(arguments, dict) for arguments in seen)
    messages, boundaries = base_state().layout()
    once = [dict(m, tool_calls=[{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
                                for c in m["tool_calls"]]) if m.get("tool_calls") else m for m in messages]
    with pytest.raises(ValueError):  # normalizing twice is refused, not silently accepted
        render_with_token_spans(ChatMLTemplate(), once, TOOLS, boundaries=boundaries, encode=RegexTokenizer().encode,
                                protocol=QWEN35_PROFILE.protocol)


class LastMessageTemplate(ChatMLTemplate):
    """Rewrites the final message's header, as Qwen's template does for a last assistant turn."""

    def render(self, messages, tools=(), **options):
        text = super().render(messages, tools, **options)
        if not options.get("add_generation_prompt", True):
            text = text.replace("<|im_start|>", "<|im_start|>(last)", 1) if messages else text
        return text


def test_ambiguous_spans_are_reported_not_invented():
    assembled = ContextAssembler(span_renderer(template=LastMessageTemplate())).render(base_state())
    assert assembled.spans.status == SPAN_MAPPING_UNAVAILABLE and assembled.spans.spans == ()
    assert assembled.segment_tokens is None and assembled.total == len(assembled.spans.token_ids)
    journal = EpisodeJournal.create("/tmp/tidemark-span-test-" + str(time.monotonic_ns()))
    result = run_episode(SteadyhandClient(FakeService(*STATUS_TURNS)), build_registry(), load_fixture().clone(),
                         initial_messages(GOAL), journal=journal, context=session(template=LastMessageTemplate()))
    render = next(r for r in read_journal(journal.path) if r["event"] == "context_render")
    assert result.status is EpisodeStatus.DONE and render["segment_tokens"]["frozen"] is None
    assert render["lint"] == {"spans": SPAN_MAPPING_UNAVAILABLE} and render["rendered_prompt_tokens"] > 0


# The session inside the loop


def record(tmp_path, turns=STATUS_TURNS, context=None, **options):
    service = FakeService(*turns)
    result = run_claimed_episode(tmp_path, SteadyhandClient(service), build_registry(), load_fixture().clone(),
                                 initial_messages(GOAL), context=context or session(), **options)
    return result, service


def test_every_step_appends_and_rewrites_only_the_tail(tmp_path):
    result, service = record(tmp_path)
    records = read_journal(result.journal_path)
    renders = [r for r in records if r["event"] == "context_render"]
    assert [r["lint_status"] for r in renders] == ["cold_start", "expected_append"]
    assert renders[1]["observed_common_prefix_tokens"] >= renders[1]["expected_common_prefix_floor"]
    assert renders[1]["previous_prompt_sha256"] == renders[0]["prompt_sha256"]
    transitions = [r for r in records if r["event"] == "context_transition"]
    model_seq = next(r["seq"] for r in records if r["event"] == "model_turn")
    tool_seq = next(r["seq"] for r in records if r["event"] == "tool_result")
    assert [t["transition"] for t in transitions] == ["start", "append"]
    assert transitions[1]["appended_seqs"] == [model_seq, tool_seq] and transitions[0]["appended_seqs"] == [0]
    # The request carries one tail, at the end; the old tail is not kept in the history.
    sent = service.requests[1].messages
    assert sent[-1]["content"].startswith(HEADER) and sum(HEADER in (m["content"] or "") for m in sent) == 1
    assert sent[1]["content"].endswith(GOAL) and all(HEADER not in (m.get("content") or "") for m in result.messages)
    table = renders[1]
    assert set(table["segment_tokens"]) == {"frozen", "pinned", "summary", "recent", "tail", "generation_prompt", "prompt"}
    assert table["slot_headroom"] == table["segment_budgets"]["prompt_limit"] - table["rendered_prompt_tokens"]


def test_replay_reproduces_renders_transitions_and_lint(tmp_path):
    result, _ = record(tmp_path)
    assert replay_episode(result.journal_path, context=session()).status is EpisodeStatus.DONE


@pytest.mark.parametrize(
    ("event", "change", "field"),
    [
        ("context_render", lambda r: r.update(lint_status="expected_tail_only"), "/lint_status"),
        ("context_render", lambda r: r.update(observed_common_prefix_tokens=1), "/observed_common_prefix_tokens"),
        ("context_transition", lambda r: r.update(appended_seqs=[1]), "/appended_seqs"),
        ("tool_result", lambda r: r.update(content=r["content"].replace("delivered", "delivered ")), "/content"),
    ],
)
def test_editing_an_old_observation_or_lint_evidence_fails_at_its_event(tmp_path, event, change, field):
    result, _ = record(tmp_path)
    lines = result.journal_path.read_text().splitlines()
    seq = [r["seq"] for r in read_journal(result.journal_path) if r["event"] == event][-1]
    item = json.loads(lines[seq])
    change(item)
    lines[seq] = json.dumps(item, sort_keys=True)
    result.journal_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(result.journal_path, context=session())
    assert caught.value.seq == seq and caught.value.field == field


def test_the_paraphrase_goes_through_the_assembler(tmp_path):
    turns = [STATUS_TURNS[0], completion("I could not finish.", attempts=[attempt(prompt_n=9)])]
    result, service = record(tmp_path, turns, budgets=Budgets(max_steps=1), paraphrase_final=True)
    assert result.status is EpisodeStatus.BUDGET_EXHAUSTED and result.final_content == "I could not finish."
    final = service.requests[-1]
    assert final.tool_choice == "none" and final.messages[-1]["content"].startswith(HEADER)
    assert "Stop working on the task now" in final.messages[-2]["content"]
    renders = [r for r in read_journal(result.journal_path) if r["event"] == "context_render"]
    assert renders[-1]["lint_status"] == "expected_append"
    assert replay_episode(result.journal_path, context=session()).status is EpisodeStatus.BUDGET_EXHAUSTED


def test_a_prompt_over_the_hard_limit_never_reaches_the_model(tmp_path):
    result, service = record(tmp_path, context=session(slot_context=600))
    assert result.status is EpisodeStatus.ABORTED and result.error_code == "context_budget_exhausted"
    assert service.requests == []
    render = next(r for r in read_journal(result.journal_path) if r["event"] == "context_render")
    assert render["status"] == "context_budget_exhausted" and "hard limit" in render["detail"]


def test_passing_the_soft_limit_marks_compaction_pending(tmp_path):
    result, _ = record(tmp_path, context=session(soft_limit=100))
    renders = [r for r in read_journal(result.journal_path) if r["event"] == "context_render"]
    assert all(r["compaction_pending"] for r in renders) and all(r["status"] == "ok" for r in renders)


def test_configuration_errors_fail_explicitly():
    tools = build_registry().schemas()
    with pytest.raises(BudgetConfigError):
        session(segment_caps={"frozen": 10}).validate(initial_messages(GOAL), tools)
    with pytest.raises(SegmentError):
        session(allowed_tool_names=("launch_rocket",)).validate(initial_messages(GOAL), tools)
    with pytest.raises(SegmentError):
        initial_state([{"role": "user", "content": "no system"}], tools, initial_count=1)
    checked = session().validate(initial_messages(GOAL), tools)
    assert checked["frozen_tokens"] > 0 and checked["total"] > checked["frozen_tokens"]


def test_a_changed_tool_list_mid_session_is_refused():
    context = session()
    history = initial_messages(GOAL)
    tools = build_registry().schemas()
    assert context.render(history, step=0, tools=tools, initial_count=2).status == "ok"
    changed = context.render(history, step=1, tools=list(reversed(tools)), initial_count=2)
    assert changed.exhausted and "tool list changed" in changed.detail


# Request-local tool restriction


def test_allowed_tools_reach_the_shim_and_the_journal(tmp_path):
    result, service = record(tmp_path, context=session(allowed_tool_names=("get_order",)))
    renders = [r for r in read_journal(result.journal_path) if r["event"] == "context_render"]
    assert all(r["allowed_tool_names"] == ["get_order"] for r in renders)
    assert service.requests[0].tools == service.requests[1].tools and len(service.requests[0].tools) > 1
    assert service.allowed == [["get_order"], ["get_order"]]  # the grammar subset, not a smaller prompt


class GrammarClient:
    def __init__(self, *contents):
        self.contents = list(contents)
        self.grammars = []

    def complete(self, prompt, **settings):
        self.grammars.append(settings.get("grammar"))
        return {"content": self.contents.pop(0), "timings": {"prompt_n": 5}}


def tool_service(client, cache=None):
    return ToolsmithService(client, ChatMLTemplate(), parser=Qwen3ToolCallParser(lambda: "call_x"), grammar_cache=cache)


def masked_request():
    return ChatCompletionRequest(messages=[{"role": "user", "content": "go"}], tools=TOOLS,
                                 chat_template_kwargs={"enable_thinking": False})


def test_masking_keeps_the_prompt_but_restricts_grammar_and_validation():
    client = GrammarClient('<tool_call>{"name":"get_order","arguments":{"order_id":"ord_1001"}}</tool_call>',
                           '<tool_call>{"name":"add_order_note","arguments":{"order_id":"o","note":"n"}}</tool_call>')
    service = tool_service(client)
    assert service.render_prompt(masked_request()) == service._prepare(masked_request(), allowed_tool_names=["get_order"]).prompt
    response = service.chat(masked_request(), allowed_tool_names=["get_order"])
    assert response["toolsmith"]["allowed_tool_names"] == ["get_order"]
    assert "get_order" in client.grammars[0] and "add_order_note" not in client.grammars[0]
    service.repair_policy = replace(service.repair_policy, max_repairs=0)
    with pytest.raises(APIError) as caught:
        service.chat(masked_request(), allowed_tool_names=["get_order"])
    assert "unregistered tool" in str(caught.value)  # a disallowed call fails validation before any handler
    for bad in ([], ["launch_rocket"], ["get_order", "get_order"]):
        with pytest.raises(APIError) as invalid:
            service.chat(masked_request(), allowed_tool_names=bad)
        assert invalid.value.code == "invalid_allowed_tools"


def test_compiled_grammars_are_cached_by_their_exact_inputs():
    cache = ContentCache("grammars")
    client = GrammarClient(*['<tool_call>{"name":"get_order","arguments":{"order_id":"o"}}</tool_call>'] * 3)
    service = tool_service(client, cache)
    service.chat(masked_request())
    service.chat(masked_request())
    service.chat(masked_request(), allowed_tool_names=["get_order"])
    assert cache.stats()["hits"] == 1 and len(cache) == 2  # the subset is its own key
    assert client.grammars[0] == client.grammars[1] != client.grammars[2]


# Shared caches under concurrency


def test_concurrent_episodes_share_caches_without_changing_any_record(tmp_path):
    tasks = [t for t in load_long_horizon() if t.distance_N <= 10]
    shared = ContentCache("rendered_prefixes", max_entries=64)
    spec = BaselineSpec(run_id="prefix", task_ids=tuple(t.task_id for t in tasks), dataset_sha256=dataset_sha256(tasks),
                        context_config=config().as_dict(), mode="scripted")

    def run(directory, scheduler=None, cache=None):
        return run_baseline(spec, tasks, run_dir=directory, scheduler=scheduler,
                            client_factory=lambda task, owner: LongHorizonScriptedModel(task, max_tokens=256),
                            session_factory=lambda task, client: session(cache=cache))

    sequential = run(tmp_path / "sequential")
    with Scheduler(SchedulerPlan(agent_slots=2, affinity="work_conserving"), cpu_executor=lambda: ThreadPoolExecutor(1)) as scheduler:
        concurrent = run(tmp_path / "concurrent", scheduler, shared)
    assert [s.outcome for s in sequential] == [s.outcome for s in concurrent]
    for left, right in zip(sequential, concurrent):
        def content(path):
            return [{k: v for k, v in comparable(r).items() if k != "episode_id"} for r in read_journal(path)]
        assert content(left.journal_path) == content(right.journal_path)
    assert shared.stats()["hits"] > 0 and len(shared) <= 64


def test_upcoming_episodes_are_prepared_ahead_into_the_shared_cache(tmp_path):
    tasks = [t for t in load_long_horizon() if t.distance_N == 5]
    shared = ContentCache("rendered_prefixes", max_entries=256)
    prepared = []
    spec = BaselineSpec(run_id="prefetch", task_ids=tuple(t.task_id for t in tasks), dataset_sha256=dataset_sha256(tasks),
                        context_config=config().as_dict(), mode="scripted")

    def prepare(task):
        prepared.append(task.task_id)
        session(cache=shared).validate(task.messages(), build_registry().schemas())

    with Scheduler(SchedulerPlan(agent_slots=1, prefetch_depth=2), cpu_executor=lambda: ThreadPoolExecutor(1)) as scheduler:
        scores = run_baseline(spec, tasks, run_dir=tmp_path, scheduler=scheduler, prepare=prepare,
                              client_factory=lambda task, owner: LongHorizonScriptedModel(task, max_tokens=256),
                              session_factory=lambda task, client: session(cache=shared))
    assert prepared == [t.task_id for t in tasks] and all(score.outcome == "success" for score in scores)
    assert shared.stats()["hits"] > 0

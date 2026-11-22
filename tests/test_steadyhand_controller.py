import itertools
import json
import threading
from pathlib import Path

import pytest

from steadyhand import (
    Budgets,
    ControllerError,
    EpisodeController,
    EpisodeJournal,
    EpisodeStatus,
    Event,
    LoopGuard,
    ModelTurn,
    SYSTEM_PROMPT,
    SteadyhandClient,
    TurnKind,
    build_registry,
    initial_messages,
    load_fixture,
    render_final_message,
    run_episode,
    transition,
)
from steadyhand.controller import TRANSITIONS, call_hash, has_visible_character
from toolsmith.api import APIError
from toolsmith.json_utils import strict_json_loads
from toolsmith.protocols import QWEN3_PROFILE
from toolsmith.validation import ToolCallValidationError

REFUND = {"order_id": "ord_1001", "amount_inr": 120, "reason": "Dented lid"}
BAD_REFUND = {"order_id": "ord_1001", "amount_inr": -120, "reason": "Dented lid"}


class FakeService:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.profile = QWEN3_PROFILE
        self.use_grammars = True

    def chat(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def completion(content=None, calls=(), *, prompt_tokens=10, completion_tokens=5):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}
            for call_id, name, arguments in calls
        ]
    return {
        "choices": [{"index": 0, "message": message}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def refund_turn(call_id, arguments=REFUND, **usage):
    return completion(calls=[(call_id, "issue_refund", arguments)], **usage)


def episode(tmp_path, *responses, goal="Refund 120 INR on ord_1001 for the dented lid.", store=None,
            client=None, **options):
    service = FakeService(*responses)
    store = store if store is not None else load_fixture().clone()
    result = run_episode(
        client or SteadyhandClient(service),
        build_registry(),
        store,
        initial_messages(goal),
        journal=EpisodeJournal.create(tmp_path),
        **options,
    )
    return result, service, store


def records(result):
    return [strict_json_loads(line) for line in Path(result.journal_path).read_text(encoding="utf-8").splitlines()]


def events(result, name):
    return [record for record in records(result) if record["event"] == name]


# State machine


EXPECTED = {
    (EpisodeStatus.RUNNING, Event.TEXT_ANSWER): EpisodeStatus.DONE,
    (EpisodeStatus.RUNNING, Event.STEP_COMPLETED): EpisodeStatus.RUNNING,
    (EpisodeStatus.RUNNING, Event.BUDGET_CROSSED): EpisodeStatus.BUDGET_EXHAUSTED,
    (EpisodeStatus.RUNNING, Event.REPEAT_LIMIT): EpisodeStatus.RUNNING,
    (EpisodeStatus.RUNNING, Event.REPEAT_AFTER_HINT): EpisodeStatus.ESCALATED,
    (EpisodeStatus.RUNNING, Event.ESCALATE): EpisodeStatus.ESCALATED,
    (EpisodeStatus.RUNNING, Event.BACKEND_FAILURE): EpisodeStatus.ABORTED,
    (EpisodeStatus.RUNNING, Event.CANCELLED): EpisodeStatus.ABORTED,
}


def test_the_transition_table_is_exactly_the_plan():
    assert TRANSITIONS == EXPECTED


@pytest.mark.parametrize(("state", "event"), list(itertools.product(EpisodeStatus, Event)))
def test_every_state_event_pair_is_legal_or_raises(state, event):
    if (state, event) in EXPECTED:
        assert transition(state, event) is EXPECTED[(state, event)]
    else:
        with pytest.raises(ControllerError):
            transition(state, event)


def test_terminal_states_refuse_further_events():
    controller = EpisodeController()
    controller.escalate("the refund ledger is unavailable")

    assert controller.state is EpisodeStatus.ESCALATED
    assert controller.escalation_reason == "the refund ledger is unavailable"
    with pytest.raises(ControllerError):
        controller.finish_step()


@pytest.mark.parametrize(
    ("text", "visible"),
    [("Done.", True), ("  ok\n", True), ("", False), (" \n\t", False), ("​ ", False), (None, False)],
)
def test_visible_character_rule(text, visible):
    assert has_visible_character(text) is visible


@pytest.mark.parametrize(
    "build",
    [
        lambda: Budgets(max_steps=-1),
        lambda: Budgets(max_prompt_tokens=-5),
        lambda: Budgets(max_wall_clock_s=0),
        lambda: Budgets(max_wall_clock_s=float("nan")),
        lambda: LoopGuard(limit=0),
    ],
)
def test_malformed_budgets_and_guard_limits_are_rejected(build):
    with pytest.raises(ValueError):
        build()


def test_default_budgets_leave_the_wall_clock_unset():
    assert Budgets() == Budgets(8, 24_000, 8_000, None)


# Endings


class ScriptedClient:
    """Returns ModelTurns directly, so the loop's own checks can be tested."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.calls = []

    def settings(self):
        return {"profile": "scripted", "max_tokens": 64}

    def complete(self, messages, tools, *, tool_choice="auto"):
        self.calls.append((list(messages), tool_choice))
        return self.turns.pop(0)


def text_turn(content):
    return ModelTurn(
        kind=TurnKind.TEXT,
        assistant_message={"role": "assistant", "content": content},
        tool_calls=(),
        content=content,
        reasoning_content=None,
        prompt_tokens=5,
        completion_tokens=2,
        error_code=None,
    )


def test_visible_text_ends_done_and_whitespace_is_a_parse_error_that_uses_a_step(tmp_path):
    client = ScriptedClient(text_turn("  \n "), text_turn("Order ord_1001 was delivered."))

    result, _, _ = episode(tmp_path, client=client)

    assert result.status is EpisodeStatus.DONE
    assert result.final_content == "Order ord_1001 was delivered."
    assert result.steps == 1
    feedback = json.loads(result.messages[2]["content"])
    assert feedback["error_type"] == "parse_error"
    assert feedback["previous_output"] == "  \n "
    turns = events(result, "model_turn")
    assert [turn["kind"] for turn in turns] == ["parse_error", "text"]


def test_step_budget_two_stops_after_two_tool_turns(tmp_path):
    result, service, _ = episode(
        tmp_path,
        completion(calls=[("a", "get_order", {"order_id": "ord_1001"})]),
        completion(calls=[("b", "list_payments", {"order_id": "ord_1001"})]),
        completion("never requested"),
        budgets=Budgets(max_steps=2),
    )

    assert result.status is EpisodeStatus.BUDGET_EXHAUSTED
    assert len(service.requests) == 2
    assert result.final_content.splitlines()[0] == (
        "Stopped: step budget exhausted (2/2 steps, 20 prompt tokens, 10 completion tokens)."
    )


@pytest.mark.parametrize(
    ("budgets", "usage", "fired"),
    [
        (Budgets(max_prompt_tokens=100), {"prompt_tokens": 120, "completion_tokens": 5}, "prompt_token"),
        (Budgets(max_completion_tokens=50), {"prompt_tokens": 10, "completion_tokens": 50}, "completion_token"),
    ],
)
def test_a_token_budget_crossed_by_one_call_prevents_the_next(tmp_path, budgets, usage, fired):
    result, service, store = episode(
        tmp_path,
        completion(calls=[("a", "issue_refund", REFUND)], **usage),
        completion("never requested"),
        budgets=budgets,
    )

    assert result.status is EpisodeStatus.BUDGET_EXHAUSTED
    assert len(service.requests) == 1
    assert len(store.refunds_for("ord_1001")) == 1  # the in-flight call still finished
    last_check = events(result, "budget_checked")[-1]
    assert last_check["exhausted"] == fired
    assert last_check["prompt_tokens"] == usage["prompt_tokens"]


class FakeClock:
    def __init__(self, step):
        self.now = 1000.0
        self.step = step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


def test_wall_clock_uses_the_injected_clock(tmp_path):
    turns = [completion(calls=[(f"c{i}", "get_order", {"order_id": f"ord_100{i + 1}"})]) for i in range(4)]

    result, service, _ = episode(
        tmp_path, *turns, budgets=Budgets(max_wall_clock_s=30.0), clock=FakeClock(step=12.0),
    )

    assert result.status is EpisodeStatus.BUDGET_EXHAUSTED
    checks = events(result, "budget_checked")
    assert [check["elapsed_s"] for check in checks] == [12.0, 24.0, 36.0]
    assert checks[-1]["exhausted"] == "wall_clock"
    assert len(service.requests) == 2
    assert "36.0/30.0s" in result.final_content


def test_without_a_wall_clock_the_clock_is_never_read(tmp_path):
    def forbidden_clock():
        raise AssertionError("the clock must not start when no wall clock is configured")

    result, _, _ = episode(tmp_path, completion("Done."), clock=forbidden_clock)

    assert result.status is EpisodeStatus.DONE
    assert events(result, "budget_checked")[0]["elapsed_s"] is None


def test_a_backend_invalid_response_aborts(tmp_path):
    error = APIError(
        "llama.cpp response did not contain string content",
        status_code=502, error_type="server_error", code="invalid_backend_response",
    )
    result, service, _ = episode(tmp_path, error, completion("never requested"))

    assert result.status is EpisodeStatus.ABORTED
    assert result.error_code == "invalid_backend_response"
    assert result.final_content is None
    assert len(service.requests) == 1


def test_a_schema_validation_failure_still_follows_the_parse_path(tmp_path):
    error = APIError("Model output failed validation", status_code=500, error_type="server_error",
                     code="schema_validation_failed")
    error.__cause__ = ToolCallValidationError("bad", code="schema_validation_failed")
    error.raw_output = '<tool_call>{"name": "issue_refund", "arguments": {}}</tool_call>'

    result, service, _ = episode(tmp_path, error, completion("Which amount should I refund?"))

    assert result.status is EpisodeStatus.DONE
    assert json.loads(service.requests[1].messages[-1]["content"])["error_type"] == "parse_error"


def test_cancellation_aborts_before_the_next_model_call(tmp_path):
    cancel = threading.Event()
    cancel.set()

    result, service, _ = episode(tmp_path, completion("never requested"), cancel=cancel)

    assert result.status is EpisodeStatus.ABORTED
    assert result.error_code == "cancelled"
    assert service.requests == []


def test_a_prompt_that_cannot_fit_the_slot_aborts_as_truncated(tmp_path):
    service = FakeService(
        completion(calls=[("a", "get_order", {"order_id": "ord_1001"})], prompt_tokens=3900),
        completion("never requested"),
    )
    client = SteadyhandClient(service, max_tokens=256, slot_context=4096)

    result, _, _ = episode(tmp_path, client=client)

    assert result.status is EpisodeStatus.ABORTED
    assert result.error_code == "completion_truncated"
    assert len(service.requests) == 1
    finished = records(result)[-1]
    assert "4096-token server slot" in finished["error_message"]
    assert events(result, "episode_started")[0]["settings"]["slot_context"] == 4096


def test_a_backend_truncation_names_the_slot_size(tmp_path):
    error = APIError("truncated", status_code=500, error_type="server_error", code="completion_truncated")
    client = SteadyhandClient(FakeService(error), max_tokens=256, slot_context=4096)

    result, _, _ = episode(tmp_path, client=client)

    turn = events(result, "model_turn")[0]
    assert turn["kind"] == "fatal_error"
    assert "server slot holds 4096 tokens" in turn["error_message"]


# Loop guard


def test_identical_calls_hash_alike_after_normalization():
    assert call_hash("issue_refund", '{"amount_inr": 120}') == call_hash("issue_refund", '{"amount_inr": 120.0}')
    assert call_hash("t", '{"b": 1, "a": 2}') == call_hash("t", '{"a":2,"b":1}')
    assert call_hash("t", '{"reason": " Dented lid "}') == call_hash("t", '{"reason": "Dented lid"}')
    assert call_hash("t", '{"amount_inr": 120}') != call_hash("t", '{"amount_inr": 120.5}')
    assert call_hash("t", '{"flag": true}') != call_hash("t", '{"flag": 1}')
    assert call_hash("t", '{"a": 1}') != call_hash("other", '{"a": 1}')
    assert call_hash("t", "{not json") == call_hash("t", " {not json ")


def test_three_identical_refunds_block_the_third_and_escalate_on_the_fourth(tmp_path):
    result, service, store = episode(
        tmp_path,
        refund_turn("r1"),
        refund_turn("r2"),
        refund_turn("r3", {"amount_inr": 120.0, "reason": "Dented lid ", "order_id": "ord_1001"}),
        refund_turn("r4"),
        completion("never requested"),
    )

    assert result.status is EpisodeStatus.ESCALATED
    assert len(service.requests) == 4
    assert [refund.refund_id for refund in store.refunds_for("ord_1001")] == ["ref_7002", "ref_7003"]
    third = json.loads(service.requests[3].messages[-1]["content"])
    assert third == {
        "ok": False,
        "error_type": "repeated_call",
        "tool": "issue_refund",
        "retryable": False,
        "hint": "This exact call already succeeded. Do not repeat it; use the earlier result, "
                "change the arguments, or ask the user.",
        "details": [{"problem": "last outcome: ok"}],
    }
    trips = events(result, "guard_tripped")
    assert [(trip["action"], trip["count"], trip["limit"]) for trip in trips] == [("block", 3, 3), ("escalate", 4, 3)]
    assert trips[0]["call_hash"] == trips[1]["call_hash"]
    blocked = [r for r in events(result, "tool_result") if r["tool_call_id"] == "r3"][0]
    assert blocked["handler_called"] is False


def test_repeated_invalid_calls_count_and_the_give_up_text_names_tool_and_error(tmp_path):
    result, _, store = episode(
        tmp_path,
        refund_turn("r1", BAD_REFUND),
        refund_turn("r2", BAD_REFUND),
        refund_turn("r3", BAD_REFUND),
        refund_turn("r4", BAD_REFUND),
    )

    assert result.status is EpisodeStatus.ESCALATED
    assert store.snapshot_sha256() == store.fixture_sha256
    assert result.final_content == "\n".join([
        "I couldn't complete this.",
        "Tried: issue_refund (invalid_arguments x2), issue_refund (repeated_call x1).",
        "Stopped because the same call was repeated after a correction hint: issue_refund "
        '{"order_id": "ord_1001", "amount_inr": -120, "reason": "Dented lid"} (last error: invalid_arguments).',
    ])
    assert "Traceback" not in result.final_content and 'File "' not in result.final_content
    blocked = json.loads(result.messages[-2]["content"])
    assert blocked["hint"] == (
        "This exact call already failed. Choose a different tool, change the arguments, or ask the user."
    )


def test_changed_arguments_after_a_block_do_not_escalate(tmp_path):
    result, _, store = episode(
        tmp_path,
        refund_turn("r1", BAD_REFUND),
        refund_turn("r2", BAD_REFUND),
        refund_turn("r3", BAD_REFUND),
        refund_turn("r4"),
        completion("Refunded 120 INR."),
    )

    assert result.status is EpisodeStatus.DONE
    assert len(store.refunds_for("ord_1001")) == 1


def test_duplicates_inside_one_turn_are_blocked_not_escalated(tmp_path):
    result, _, store = episode(
        tmp_path,
        completion(calls=[("a", "issue_refund", REFUND), ("b", "issue_refund", REFUND), ("c", "issue_refund", REFUND)]),
        completion("Done."),
    )

    assert result.status is EpisodeStatus.DONE
    assert len(store.refunds_for("ord_1001")) == 2
    assert json.loads(result.messages[-2]["content"])["error_type"] == "repeated_call"


# No-tool path and final messages


def test_asking_for_missing_information_is_a_done_ending(tmp_path):
    result, service, store = episode(
        tmp_path,
        completion("Which order would you like refunded? Please share the order id."),
        goal="Refund my last order.",
    )

    assert result.status is EpisodeStatus.DONE
    assert result.steps == 0
    assert store.snapshot_sha256() == store.fixture_sha256
    assert events(result, "tool_result") == []
    system = service.requests[0].messages[0]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Never invent" in system["content"] and "ask the user" in system["content"]


def test_the_client_passes_a_forced_no_tool_choice_through():
    service = FakeService(completion("I could not finish."))

    SteadyhandClient(service).complete([{"role": "user", "content": "x"}], [], tool_choice="none")

    assert service.requests[0].tool_choice == "none"


@pytest.mark.parametrize("ending", ["budget", "escalated"])
def test_the_final_message_is_reproducible_from_the_journal(tmp_path, ending):
    if ending == "budget":
        result, _, _ = episode(tmp_path, refund_turn("r1"), completion("x"), budgets=Budgets(max_steps=1))
    else:
        result, _, _ = episode(tmp_path, *[refund_turn(f"r{i}", BAD_REFUND) for i in range(4)])

    lines = records(result)
    assert render_final_message(result.status, lines) == result.final_content
    assert lines[-1]["final_content"] == result.final_content


def test_budget_message_matches_the_plan_format():
    lines = [
        {"event": "episode_started", "messages": initial_messages("Refund the dented kettle on ord_1001.")},
        {"event": "tool_result", "tool": "get_order", "ok": True},
        {"event": "budget_checked", "steps": 8, "prompt_tokens": 12031, "completion_tokens": 840,
         "elapsed_s": 37.2, "exhausted": "step",
         "budgets": {"max_steps": 8, "max_prompt_tokens": 24000, "max_completion_tokens": 8000,
                     "max_wall_clock_s": 60.0}},
    ]

    assert render_final_message(EpisodeStatus.BUDGET_EXHAUSTED, lines) == "\n".join([
        "Stopped: step budget exhausted (8/8 steps, 12031 prompt tokens, 840 completion tokens, 37.2s).",
        "Tried: get_order (ok x1).",
        "Incomplete work: Refund the dented kettle on ord_1001.",
    ])


def test_a_generic_escalation_reason_is_rendered():
    lines = [{"event": "escalated", "reason": "the refund ledger is unavailable and no alternative remains"}]

    assert render_final_message(EpisodeStatus.ESCALATED, lines) == "\n".join([
        "I couldn't complete this.",
        "Tried: no tool calls.",
        "Stopped because the refund ledger is unavailable and no alternative remains.",
    ])
    assert render_final_message(EpisodeStatus.DONE, lines) is None


@pytest.mark.parametrize(
    ("responses", "options", "status"),
    [
        ((completion("Done."),), {}, EpisodeStatus.DONE),
        ((refund_turn("r1"), completion("x")), {"budgets": Budgets(max_steps=1)}, EpisodeStatus.BUDGET_EXHAUSTED),
        (tuple(refund_turn(f"r{i}", BAD_REFUND) for i in range(4)), {}, EpisodeStatus.ESCALATED),
        ((APIError("down", status_code=502, code="backend_error"),), {}, EpisodeStatus.ABORTED),
    ],
)
def test_the_last_journal_event_matches_the_status(tmp_path, responses, options, status):
    result, _, _ = episode(tmp_path, *responses, **options)

    last = records(result)[-1]
    assert result.status is status
    assert (last["event"], last["status"]) == ("episode_finished", status.value)


def test_an_optional_paraphrase_uses_tool_choice_none_on_answerable_history(tmp_path):
    responses = [refund_turn(f"r{i}", BAD_REFUND) for i in range(4)]
    responses.append(completion("Sorry, I couldn't issue that refund; the amount must be positive."))

    result, service, _ = episode(tmp_path, *responses, paraphrase_final=True)

    assert result.status is EpisodeStatus.ESCALATED
    assert result.final_content == "Sorry, I couldn't issue that refund; the amount must be positive."
    paraphrase = service.requests[-1]
    assert paraphrase.tool_choice == "none"
    sent = paraphrase.messages
    assert sent[-1]["role"] == "user" and "I couldn't complete this." in sent[-1]["content"]
    assert sent[-2]["role"] == "tool"  # the escalating, unanswered turn is not sent
    assert records(result)[-1]["final_content"] == result.final_content


def test_a_failed_paraphrase_falls_back_to_the_template(tmp_path):
    responses = [refund_turn("r1"), APIError("down", status_code=502, code="backend_error")]

    result, _, _ = episode(tmp_path, *responses, budgets=Budgets(max_steps=1), paraphrase_final=True)

    assert result.final_content.startswith("Stopped: step budget exhausted")

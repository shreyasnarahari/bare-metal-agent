import json
from pathlib import Path

import pytest

from steadyhand import (
    Effect,
    EpisodeJournal,
    EpisodeStatus,
    ErrorType,
    FieldProblem,
    PolicyError,
    RetrySpec,
    SequentialExecutor,
    SteadyhandClient,
    SteadyhandRegistry,
    Store,
    ToolCall,
    ToolPolicy,
    TurnKind,
    build_registry,
    load_fixture,
    run_episode,
    validate_call,
    worker_service,
)
from steadyhand.client import is_model_output_error
from steadyhand.errors import MAX_QUOTED_OUTPUT_CHARS, cap_problems
from steadyhand.tools import TRANSIENT_ERRORS, get_order
from steadyhand.validate import PARSE_ERROR_HINT, must_be_future
from toolsmith.api import APIError
from toolsmith.grammar import GBNFCompiler, resolve_tool_choice
from toolsmith.json_utils import strict_json_loads
from toolsmith.protocols import QWEN3_PROFILE, QWEN35_PROFILE, ToolProtocol
from toolsmith.registry import Tool
from toolsmith.validation import ToolCallValidationError

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "steadyhand_model_turns.json").read_text(encoding="utf-8")
)["turns"]
SYSTEM = {"role": "system", "content": "You are a support agent. Use the tools."}
READ = ToolPolicy(Effect.READ, 5.0, idempotent=True)


def fixture_call(name):
    turn = next(turn for turn in FIXTURES if turn["name"] == name)
    return ToolCall.from_openai(turn["message"]["tool_calls"][0])


def fixture_completion(name):
    turn = next(turn for turn in FIXTURES if turn["name"] == name)
    return completion_from_message(turn["message"])


def completion_from_message(message, *, prompt_tokens=10, completion_tokens=5):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": QWEN3_PROFILE.model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def text_completion(content):
    response = completion_from_message({"role": "assistant", "content": content})
    response["choices"][0]["finish_reason"] = "stop"
    return response


def model_output_error(code, message, raw_output):
    error = APIError(message, status_code=500, error_type="server_error", code=code, repair_attempts=1)
    error.__cause__ = ToolCallValidationError(message, code=code)
    error.raw_output = raw_output
    return error


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


class SpyExecutor(SequentialExecutor):
    def __init__(self):
        self.calls = []

    def run(self, call, registry, store):
        self.calls.append(call.name)
        return super().run(call, registry, store)


def episode(tmp_path, *responses, registry=None, store=None, **options):
    service = FakeService(*responses)
    store = store if store is not None else load_fixture().clone()
    executor = SpyExecutor()
    result = run_episode(
        SteadyhandClient(service),
        registry or build_registry(),
        store,
        [SYSTEM, {"role": "user", "content": "Refund 120 INR on ord_1001 for the dented lid."}],
        journal=EpisodeJournal.create(tmp_path),
        executor=executor,
        **options,
    )
    return result, service, store, executor


def journal_records(path):
    return [strict_json_loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def assert_one_line(observation):
    assert observation["hint"] and "\n" not in observation["hint"]
    for detail in observation.get("details", []):
        assert "\n" not in detail["problem"]
        assert "Traceback" not in detail["problem"] and 'File "' not in detail["problem"]


# Policy metadata


def test_a_non_idempotent_tool_cannot_retry_automatically():
    with pytest.raises(PolicyError, match="non-idempotent"):
        ToolPolicy(
            Effect.NON_IDEMPOTENT_WRITE, 5.0, idempotent=False,
            retry=RetrySpec(on=("timeout",), max_attempts=1),
        )

    allowed = ToolPolicy(
        Effect.NON_IDEMPOTENT_WRITE, 5.0, idempotent=False,
        retry=RetrySpec(on=(), max_attempts=0),
    )
    assert allowed.retry.max_attempts == 0


@pytest.mark.parametrize(
    ("effect", "idempotent"),
    [(Effect.READ, False), (Effect.IDEMPOTENT_WRITE, False), (Effect.NON_IDEMPOTENT_WRITE, True)],
)
def test_idempotent_must_match_the_effect(effect, idempotent):
    with pytest.raises(PolicyError, match="contradicts"):
        ToolPolicy(effect, 5.0, idempotent=idempotent)


@pytest.mark.parametrize(
    "build",
    [
        lambda: ToolPolicy(Effect.READ, 0, idempotent=True),
        lambda: ToolPolicy(Effect.READ, float("inf"), idempotent=True),
        lambda: RetrySpec(on=("timeout",), max_attempts=-1),
        lambda: RetrySpec(on=(), max_attempts=2),
        lambda: RetrySpec(on=("timeout", "timeout"), max_attempts=2),
        lambda: RetrySpec(on=("timeout",), max_attempts=2, backoff="linear"),
        lambda: ToolPolicy(Effect.READ, 5.0, idempotent=True, alternatives=("a", "a")),
    ],
)
def test_malformed_policy_fields_are_rejected(build):
    with pytest.raises(PolicyError):
        build()


def lookup(store: Store, order_id: str) -> dict:
    """Stub read."""
    return {"order_id": order_id}


def write(store: Store, order_id: str) -> dict:
    """Stub non-idempotent write."""
    return {"order_id": order_id}


def other_write(store: Store, order_id: str) -> dict:
    """Second stub write."""
    return {"order_id": order_id}


@pytest.mark.parametrize(
    ("verify", "alternatives", "message"),
    [
        ("missing_read", (), "not registered"),
        ("other_write", (), "must be a read"),
        ("write", (), "itself"),
        (None, ("write",), "its own alternative"),
        (None, ("made_up_tool",), "neither registered nor reserved"),
    ],
)
def test_the_registry_rejects_inconsistent_cross_references(verify, alternatives, message):
    registry = SteadyhandRegistry()
    registry.register(lookup, READ)
    registry.register(other_write, ToolPolicy(Effect.NON_IDEMPOTENT_WRITE, 5.0, idempotent=False))

    with pytest.raises(PolicyError, match=message):
        registry.register(
            write,
            ToolPolicy(Effect.NON_IDEMPOTENT_WRITE, 5.0, idempotent=False,
                       verify=verify, alternatives=alternatives),
        )
    assert "write" not in registry


def test_reserved_alternatives_and_read_verify_tools_register():
    registry = SteadyhandRegistry()
    registry.register(lookup, READ)
    registry.register(
        write,
        ToolPolicy(Effect.NON_IDEMPOTENT_WRITE, 5.0, idempotent=False,
                   verify="lookup", alternatives=("refund_via_ledger",)),
    )

    assert registry.policy("write").verify == "lookup"


def test_a_frozen_registry_refuses_more_tools_and_duplicates_are_rejected():
    registry = SteadyhandRegistry()
    registry.register(lookup, READ)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(lookup, READ)

    registry.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(write, ToolPolicy(Effect.NON_IDEMPOTENT_WRITE, 5.0, idempotent=False))


def test_declared_policies_match_the_plan_table():
    registry = build_registry()
    safe_retry = RetrySpec(on=TRANSIENT_ERRORS, max_attempts=2, backoff="exp")

    assert registry.frozen
    for name in ("get_order", "get_customer", "list_payments", "get_ticket", "search_orders"):
        assert registry.policy(name) == ToolPolicy(Effect.READ, 5.0, idempotent=True, retry=safe_retry)
    for name in ("set_ticket_status", "cancel_order"):
        assert registry.policy(name) == ToolPolicy(
            Effect.IDEMPOTENT_WRITE, 5.0, idempotent=True, retry=safe_retry
        )
    writes = {
        "issue_refund": ("list_payments", ("refund_via_ledger",)),
        "create_ticket": ("get_ticket", ()),
        "add_order_note": ("get_order", ()),
    }
    for name, (verify, alternatives) in writes.items():
        policy = registry.policy(name)
        assert policy.effect is Effect.NON_IDEMPOTENT_WRITE and policy.idempotent is False
        assert policy.retry is None and policy.timeout_s == 5.0
        assert (policy.verify, policy.alternatives) == (verify, alternatives)


def test_schemas_are_fresh_copies_with_stable_hashes():
    registry = build_registry()

    first = registry.schemas()
    first[0]["function"]["name"] = "tampered"

    assert registry.schemas()[0]["function"]["name"] == "get_order"
    assert registry.schemas_sha256() == build_registry().schemas_sha256()
    assert registry.policies_sha256() == build_registry().policies_sha256()


@pytest.mark.parametrize("protocol", list(ToolProtocol))
def test_advertised_schemas_compile_through_the_shim_path(protocol):
    tools = tuple(Tool.from_openai(schema) for schema in build_registry().schemas())

    grammar = GBNFCompiler(protocol=protocol).compile_tools(tools, resolve_tool_choice(tools, "auto"))

    assert len(grammar.tool_names) == 10


# Validation layers, from the recorded fixtures


@pytest.mark.parametrize(
    ("name", "error_type", "retryable", "fields"),
    [
        ("unknown_tool", "unknown_tool", False, ["name"]),
        ("extra_property", "invalid_arguments", True, ["/unexpected"]),
        ("negative_amount", "invalid_arguments", True, ["amount_inr"]),
        ("unknown_order_id", "invalid_arguments", True, ["order_id"]),
        ("invalid_json", "parse_error", True, []),
    ],
)
def test_each_invalid_fixture_yields_its_typed_observation(name, error_type, retryable, fields):
    store = load_fixture().clone()

    verdict = validate_call(fixture_call(name), build_registry(), store)

    observation = verdict.error.as_dict()
    assert verdict.ok is False
    assert observation["ok"] is False
    assert observation["error_type"] == error_type
    assert observation["retryable"] is retryable
    assert [detail.get("field") for detail in observation["details"] if "field" in detail] == fields
    assert_one_line(observation)
    assert store.snapshot_sha256() == store.fixture_sha256


def test_the_valid_fixture_passes_every_layer():
    verdict = validate_call(fixture_call("valid_refund"), build_registry(), load_fixture().clone())

    assert verdict.ok
    assert verdict.arguments == {"order_id": "ord_1001", "amount_inr": 120, "reason": "Dented lid"}
    assert len(verdict.arguments_sha256) == 64


def test_negative_refund_matches_the_design_observation_and_writes_nothing(tmp_path):
    result, service, store, executor = episode(
        tmp_path, fixture_completion("negative_amount"), text_completion("I could not refund that."),
    )

    observation = json.loads(service.requests[1].messages[-1]["content"])
    assert observation == {
        "ok": False,
        "error_type": "invalid_arguments",
        "tool": "issue_refund",
        "details": [{"field": "amount_inr", "problem": "must be > 0", "got": -120}],
        "retryable": True,
        "hint": "Fix the listed fields and call issue_refund again.",
    }
    assert executor.calls == []
    assert store.refunds_for("ord_1001") == []
    assert store.snapshot_sha256() == store.fixture_sha256


def test_a_refund_above_the_refundable_total_names_the_allowed_maximum():
    call = ToolCall("c", "issue_refund", '{"order_id": "ord_1005", "amount_inr": 5499.01, "reason": "x"}')

    observation = validate_call(call, build_registry(), load_fixture().clone()).error.as_dict()

    assert observation["details"] == [
        {
            "field": "amount_inr",
            "problem": "must not exceed the refundable total of 5499.00 INR for ord_1005",
            "got": 5499.01,
        }
    ]
    exact = ToolCall("c", "issue_refund", '{"order_id": "ord_1005", "amount_inr": 5499, "reason": "x"}')
    assert validate_call(exact, build_registry(), load_fixture().clone()).ok


def test_refund_amounts_must_be_whole_paise():
    call = ToolCall("c", "issue_refund", '{"order_id": "ord_1001", "amount_inr": 10.005, "reason": "x"}')

    observation = validate_call(call, build_registry(), load_fixture().clone()).error.as_dict()

    assert observation["details"][0]["problem"] == "must have at most two decimal places"


def test_an_unknown_tool_never_reaches_a_handler(tmp_path):
    result, service, store, executor = episode(
        tmp_path, fixture_completion("unknown_tool"), text_completion("Sorry."),
    )

    observation = json.loads(service.requests[1].messages[-1]["content"])
    assert observation["error_type"] == "unknown_tool"
    assert observation["hint"].startswith("Use one of the registered tools: get_order, get_customer")
    assert executor.calls == []
    tool = next(r for r in journal_records(result.journal_path) if r["event"] == "tool_result")
    assert tool["handler_called"] is False
    assert tool["error"]["error_type"] == "unknown_tool"


def test_schema_details_use_json_pointers_like_the_shim_validator():
    call = ToolCall("c", "issue_refund", '{"order_id": 7, "amount_inr": "a lot"}')

    observation = validate_call(call, build_registry(), load_fixture().clone()).error.as_dict()

    assert observation["details"] == [
        {"field": "/amount_inr", "problem": "must be of type number", "got": "a lot"},
        {"field": "/order_id", "problem": "must be of type string", "got": 7},
        {"field": "/reason", "problem": "is required"},
    ]


def test_an_enum_violation_lists_the_allowed_values():
    call = ToolCall("c", "set_ticket_status", '{"ticket_id": "tkt_9001", "status": "closed"}')

    observation = validate_call(call, build_registry(), load_fixture().clone()).error.as_dict()

    assert observation["details"] == [
        {"field": "/status", "problem": "must be one of ['open', 'pending', 'resolved']", "got": "closed"}
    ]


def flood(store: Store, order_id: str) -> dict:
    """Stub with a noisy semantic check."""
    return {}


def noisy_check(store, arguments):
    return [FieldProblem("order_id", "is wrong", "x")] * 4 + [
        FieldProblem(f"field_{index}", "is wrong", index) for index in range(8)
    ]


def test_semantic_problems_are_deduplicated_and_capped():
    registry = SteadyhandRegistry()
    registry.register(flood, READ, noisy_check)

    default = validate_call(ToolCall("c", "flood", '{"order_id": "x"}'), registry, load_fixture())
    tight = validate_call(ToolCall("c", "flood", '{"order_id": "x"}'), registry, load_fixture(), max_errors=2)

    assert [detail["field"] for detail in default.error.as_dict()["details"]] == [
        "order_id", "field_0", "field_1", "field_2", "field_3",
    ]
    assert len(tight.error.details) == 2
    assert len(cap_problems([FieldProblem("a", "b")] * 10)) == 1


def test_long_values_are_bounded_in_details():
    call = ToolCall("c", "get_customer", json.dumps({"customer_id": "c" * 5000}))

    detail = validate_call(call, build_registry(), load_fixture()).error.as_dict()["details"][0]

    assert len(detail["got"]) <= 200


def test_semantic_checks_read_the_episode_store_only():
    first, second = load_fixture().clone(), load_fixture().clone()
    first.add_refund("ord_1005", "5499.00", "Full refund")
    call = ToolCall("c", "issue_refund", '{"order_id": "ord_1005", "amount_inr": 1, "reason": "x"}')

    assert not validate_call(call, build_registry(), first).ok
    assert validate_call(call, build_registry(), second).ok
    assert second.snapshot_sha256() == second.fixture_sha256


def test_create_ticket_checks_that_the_order_belongs_to_the_customer():
    call = ToolCall("c", "create_ticket", '{"customer_id": "cus_103", "subject": "Lid", "order_id": "ord_1001"}')

    observation = validate_call(call, build_registry(), load_fixture()).error.as_dict()

    assert observation["details"] == [
        {"field": "order_id", "problem": "belongs to cus_101, not cus_103", "got": "ord_1001"}
    ]


@pytest.mark.parametrize(
    ("tool", "arguments", "field"),
    [
        ("get_customer", {"customer_id": "cus_999"}, "customer_id"),
        ("search_orders", {"customer_id": "cus_999"}, "customer_id"),
        ("list_payments", {"order_id": "ord_9999"}, "order_id"),
        ("get_ticket", {"ticket_id": "tkt_1"}, "ticket_id"),
        ("set_ticket_status", {"ticket_id": "tkt_1", "status": "open"}, "ticket_id"),
        ("add_order_note", {"order_id": "ord_9999", "note": "x"}, "order_id"),
        ("cancel_order", {"order_id": "ord_9999"}, "order_id"),
    ],
)
def test_ids_the_semantic_layer_knows_are_absent_fail_before_the_handler(tool, arguments, field):
    verdict = validate_call(ToolCall("c", tool, json.dumps(arguments)), build_registry(), load_fixture())

    assert verdict.error.error_type is ErrorType.INVALID_ARGUMENTS
    assert verdict.error.details[0].field == field


def test_a_missing_order_found_by_the_handler_is_not_found_and_not_retryable():
    registry, store = build_registry(), load_fixture().clone()
    call = ToolCall("c", "get_order", '{"order_id": "ord_9999"}')

    assert validate_call(call, registry, store).ok
    result = SequentialExecutor().run(call, registry, store)

    assert result.handler_called is True
    assert result.observation() == {
        "ok": False,
        "error_type": "not_found",
        "tool": "get_order",
        "retryable": False,
        "hint": "The store has no such row. Check the id with the user or find it another way; "
                "repeating this call will not help.",
        "details": [{"field": "order_id", "problem": "no order with this id exists", "got": "ord_9999"}],
    }


def nested_failure(store: Store, order_id: str) -> dict:
    """Stub whose exception has a real stack under it."""
    def inner():
        raise RuntimeError("ledger\nlocked\n  File \"x.py\", line 1")
    inner()


def test_handler_exceptions_become_one_line_handler_errors():
    registry = SteadyhandRegistry()
    registry.register(nested_failure, READ)

    result = SequentialExecutor().run(ToolCall("c", "nested_failure", '{"order_id": "x"}'), registry, load_fixture())

    observation = result.observation()
    assert observation["error_type"] == "handler_error"
    assert observation["retryable"] is False
    assert observation["details"] == [{"problem": "RuntimeError: ledger locked File \"x.py\", line 1"}]
    assert "\n" not in json.dumps(observation)


def test_a_store_conflict_is_a_handler_error_with_the_store_message():
    registry, store = build_registry(), load_fixture().clone()
    call = ToolCall("c", "cancel_order", '{"order_id": "ord_1001"}')

    result = SequentialExecutor().run(call, registry, store)

    assert result.error.error_type is ErrorType.HANDLER_ERROR
    assert "delivered and can no longer be cancelled" in result.error.details[0].problem
    assert store.snapshot_sha256() == store.fixture_sha256


def test_must_be_future_compares_against_the_fixture_clock():
    store = load_fixture()

    assert must_be_future(store, "due", "2026-11-09T09:00:00+05:30") is None
    assert must_be_future(store, "due", "2026-11-09T09:00:00") is None
    assert must_be_future(store, "due", "2026-11-08T09:59:59+05:30").problem == (
        "must be after the episode clock 2026-11-08T10:00:00+05:30"
    )
    assert must_be_future(store, "due", "tomorrow").problem == "must be an ISO 8601 date or date-time"


# Episodes


def test_a_bad_refund_then_a_corrected_refund_leaves_one_refund(tmp_path):
    corrected = completion_from_message({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_fixed",
            "type": "function",
            "function": {"name": "issue_refund",
                         "arguments": '{"order_id": "ord_1001", "amount_inr": 120, "reason": "Dented lid"}'},
        }],
    })
    result, service, store, executor = episode(
        tmp_path,
        fixture_completion("negative_amount"),
        corrected,
        text_completion("Refunded 120 INR on ord_1001."),
    )

    assert result.status is EpisodeStatus.DONE
    assert result.steps == 2
    assert executor.calls == ["issue_refund"]
    assert [refund.amount_inr for refund in store.refunds_for("ord_1001")] == ["120.00"]
    tools = [r for r in journal_records(result.journal_path) if r["event"] == "tool_result"]
    assert [(r["ok"], r["handler_called"]) for r in tools] == [(False, False), (True, True)]
    assert tools[0]["error"]["details"][0]["field"] == "amount_inr"
    assert tools[1]["value"]["refund_id"] == "ref_7002"


def test_every_invalid_fixture_in_one_message_reaches_no_handler(tmp_path):
    calls = [
        turn["message"]["tool_calls"][0]
        for turn in FIXTURES
        if turn["name"] != "valid_refund"
    ]
    message = {"role": "assistant", "content": None, "tool_calls": calls}
    result, service, store, executor = episode(
        tmp_path, completion_from_message(message), text_completion("None of those worked."),
    )

    observations = [json.loads(m["content"]) for m in service.requests[1].messages[3:]]
    assert [o["error_type"] for o in observations] == [
        "unknown_tool", "invalid_arguments", "invalid_arguments", "invalid_arguments", "parse_error",
    ]
    assert executor.calls == []
    assert store.snapshot_sha256() == store.fixture_sha256


def test_a_parse_error_becomes_a_user_observation_and_the_episode_continues(tmp_path):
    raw = '<tool_call>{"name": "issue_refund", "arguments": {"order_id": "ord_1001"}}</tool_call>'
    error = model_output_error(
        "schema_validation_failed",
        "Model output failed validation: Arguments for 'issue_refund' failed schema validation: "
        "/: 'amount_inr' is a required property",
        raw,
    )
    result, service, store, executor = episode(tmp_path, error, text_completion("Which amount?"))

    assert result.status is EpisodeStatus.DONE
    assert result.steps == 1
    feedback = service.requests[1].messages[-1]
    assert feedback["role"] == "user"
    observation = json.loads(feedback["content"])
    assert observation == {
        "ok": False,
        "error_type": "parse_error",
        "tool": None,
        "retryable": True,
        "hint": PARSE_ERROR_HINT,
        "details": [{"problem": error.args[0]}],
        "previous_output": raw,
    }
    records = journal_records(result.journal_path)
    turn = next(r for r in records if r["event"] == "model_turn")
    assert (turn["kind"], turn["error_code"], turn["raw_output"]) == ("parse_error", "schema_validation_failed", raw)
    feedback_record = next(r for r in records if r["event"] == "observation")
    assert feedback_record["content"] == feedback["content"]


def test_quoted_bad_output_is_bounded(tmp_path):
    error = model_output_error("invalid_tool_call_json", "Could not parse model output", "x" * 5000)

    result, service, _, _ = episode(tmp_path, error, text_completion("Done."))

    quoted = json.loads(service.requests[1].messages[-1]["content"])["previous_output"]
    assert quoted == "x" * MAX_QUOTED_OUTPUT_CHARS + "\n[truncated by Steadyhand]"


def test_repeated_parse_errors_stop_at_the_step_budget(tmp_path):
    errors = [model_output_error("invalid_tool_call", "Could not parse model output", "<tool_call>") for _ in range(5)]

    result, service, _, _ = episode(tmp_path, *errors, max_steps=3)

    assert result.status is EpisodeStatus.BUDGET_EXHAUSTED
    assert result.steps == 3
    assert len(service.requests) == 3


@pytest.mark.parametrize(
    "error",
    [
        APIError("Message 2 arrived before every tool call had a result"),
        APIError("truncated", status_code=500, error_type="server_error", code="completion_truncated"),
    ],
)
def test_request_and_truncation_errors_still_abort(tmp_path, error):
    result, service, _, _ = episode(tmp_path, error, text_completion("never requested"))

    assert result.status is EpisodeStatus.ABORTED
    assert result.error_code == error.code
    assert len(service.requests) == 1


def test_model_output_errors_are_recognised_by_cause_or_empty_response():
    caused = model_output_error("unknown_tool", "bad", "raw")
    empty = APIError("empty", status_code=500, code="empty_model_response")
    plain = APIError("bad request")

    assert is_model_output_error(caused) and is_model_output_error(empty)
    assert not is_model_output_error(plain)


class PromptEchoRenderer:
    def render(self, messages, tools=(), *, enable_thinking=None):
        return json.dumps({"messages": messages})


class ScriptedLlama:
    def __init__(self, *contents):
        self.contents = list(contents)
        self.prompts = []

    def complete(self, prompt, **settings):
        self.prompts.append(json.loads(prompt))
        return {"content": self.contents.pop(0), "tokens_evaluated": 20, "tokens_predicted": 5}


@pytest.mark.parametrize("profile", [QWEN3_PROFILE, QWEN35_PROFILE])
def test_a_real_shim_rejection_flows_back_to_the_model(tmp_path, profile):
    bad = '<tool_call>{"name": "refund_order", "arguments": {}}</tool_call>'
    if profile is QWEN35_PROFILE:
        bad = "<tool_call>\n<function=refund_order>\n</function>\n</tool_call>"
    llama = ScriptedLlama(bad, bad, "I can only use the registered tools.")
    service = worker_service(llama, PromptEchoRenderer(), profile)

    result = run_episode(
        SteadyhandClient(service),
        build_registry(),
        load_fixture().clone(),
        [SYSTEM, {"role": "user", "content": "Refund ord_1001."}],
        journal=EpisodeJournal.create(tmp_path),
    )

    assert result.status is EpisodeStatus.DONE
    assert len(llama.prompts) == 3  # one shim repair inside chat(), then the loop's observation
    observation = json.loads(result.messages[2]["content"])
    assert observation["error_type"] == "parse_error"
    assert observation["previous_output"] == bad
    turn = next(r for r in journal_records(result.journal_path) if r["event"] == "model_turn")
    assert turn["kind"] == TurnKind.PARSE_ERROR.value
    assert turn["repair_attempts"] == 1


def test_the_journal_reconstructs_messages_with_validation_and_parse_feedback(tmp_path):
    error = model_output_error("invalid_tool_call", "Could not parse model output", "<tool_call>{")
    result, _, _, _ = episode(
        tmp_path,
        error,
        fixture_completion("negative_amount"),
        fixture_completion("valid_refund"),
        text_completion("Refunded."),
    )

    messages = []
    for record in journal_records(result.journal_path):
        if record["event"] == "episode_started":
            messages.extend(record["messages"])
        elif record["event"] == "model_turn" and record["assistant_message"] is not None:
            messages.append(record["assistant_message"])
        elif record["event"] == "tool_result":
            messages.append({"role": "tool", "tool_call_id": record["tool_call_id"], "content": record["content"]})
        elif record["event"] == "observation":
            messages.append({"role": record["role"], "content": record["content"]})

    assert messages == result.messages
    started = journal_records(result.journal_path)[0]
    assert started["schema_sha256"] == build_registry().schemas_sha256()
    assert started["policy_sha256"] == build_registry().policies_sha256()


def test_get_order_still_answers_from_the_store():
    store = load_fixture()
    result = SequentialExecutor().run(
        ToolCall("c", "get_order", '{"order_id": "ord_1001"}'), build_registry(), store,
    )

    assert result.observation() == {"ok": True, "value": get_order(store, "ord_1001")}

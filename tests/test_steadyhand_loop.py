import json
import threading
import time
from pathlib import Path

import pytest

import steadyhand
from steadyhand import (
    Budgets,
    Effect,
    EpisodeJournal,
    EpisodeStatus,
    LiveConfigError,
    LiveSettings,
    SequentialExecutor,
    SteadyhandClient,
    SteadyhandRegistry,
    Store,
    ToolCall,
    ToolPolicy,
    TurnKind,
    build_llama_client,
    build_registry,
    load_fixture,
    run_episode,
    worker_service,
)
from steadyhand.journal import manifest_path_for
from steadyhand.manifest import load_manifest
from steadyhand.tools import get_order
from toolsmith.api import APIError, ToolsmithService
from toolsmith.backends import LlamaCppConnectionError, LlamaCppTimeoutError
from toolsmith.json_utils import strict_json_loads
from toolsmith.prompting import ChatPromptRenderer
from toolsmith.protocols import QWEN3_PROFILE, QWEN35_PROFILE

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "01_toolsmith.ipynb"
SYSTEM = {"role": "system", "content": "You are a support agent. Use the tools."}


class FakeService:
    """Stands in for ToolsmithService: canned completion dicts or raised errors."""

    def __init__(self, *responses, profile=QWEN3_PROFILE, use_grammars=True):
        self.responses = list(responses)
        self.requests = []
        self.profile = profile
        self.use_grammars = use_grammars

    def chat(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def completion(content=None, calls=(), *, prompt_tokens=10, completion_tokens=5, repair_attempts=0):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for call_id, name, arguments in calls
        ]
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": QWEN3_PROFILE.model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "toolsmith": {"repair_attempts": repair_attempts, "grammar_sha256": None, "protocol": "qwen3"},
    }


def episode(tmp_path, *responses, question="What is the status of ord_1001?", registry=None,
            store=None, secrets=(), **options):
    service = FakeService(*responses)
    store = store if store is not None else load_fixture().clone()
    journal = EpisodeJournal.create(tmp_path, secrets=secrets)
    result = run_episode(
        SteadyhandClient(service),
        registry or build_registry(),
        store,
        [SYSTEM, {"role": "user", "content": question}],
        journal=journal,
        **options,
    )
    return result, service, store


def journal_records(path):
    return [strict_json_loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def events(path):
    return [record["event"] for record in journal_records(path)]


# Endings


def test_text_turn_finishes_done_with_zero_tool_executions(tmp_path):
    result, service, store = episode(tmp_path, completion("Order ord_1001 was delivered."))

    assert result.status is EpisodeStatus.DONE
    assert result.final_content == "Order ord_1001 was delivered."
    assert result.steps == 0
    assert (result.prompt_tokens, result.completion_tokens) == (10, 5)
    assert result.messages[-1] == {"role": "assistant", "content": "Order ord_1001 was delivered."}
    assert events(result.journal_path) == [
        "episode_started", "budget_checked", "model_turn", "episode_finished",
    ]
    assert store.snapshot_sha256() == store.fixture_sha256

    request = service.requests[0]
    assert request.model == QWEN3_PROFILE.model
    assert request.tool_choice == "auto"
    assert request.chat_template_kwargs == {"enable_thinking": False}
    assert request.response_format is None
    assert request.stream is False
    assert request.tools == build_registry().schemas()
    assert (request.temperature, request.seed) == (0.0, 0)


def test_tool_call_then_text_feeds_the_tool_result_into_the_second_request(tmp_path):
    result, service, store = episode(
        tmp_path,
        completion(calls=[("call_a", "get_order", {"order_id": "ord_1001"})],
                   prompt_tokens=30, completion_tokens=12),
        completion("Order ord_1001 was delivered.", prompt_tokens=80, completion_tokens=9),
    )

    assert result.status is EpisodeStatus.DONE
    assert result.steps == 1
    assert (result.prompt_tokens, result.completion_tokens) == (110, 21)

    second = service.requests[1].messages
    assert [message["role"] for message in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0]["function"] == {
        "name": "get_order",
        "arguments": '{"order_id": "ord_1001"}',
    }
    assert second[3]["tool_call_id"] == "call_a"
    assert json.loads(second[3]["content"]) == {
        "ok": True,
        "value": get_order(load_fixture(), "ord_1001"),
    }
    assert result.messages[:4] == second
    assert result.messages[4]["content"] == "Order ord_1001 was delivered."
    assert events(result.journal_path) == [
        "episode_started", "budget_checked", "model_turn", "tool_result",
        "budget_checked", "model_turn", "episode_finished",
    ]


def test_a_fatal_error_aborts_and_the_journal_keeps_the_code(tmp_path):
    error = APIError("Assistant call 0 needs a unique non-empty id", code="invalid_message_history")
    result, service, store = episode(tmp_path, error)

    assert result.status is EpisodeStatus.ABORTED
    assert result.error_code == "invalid_message_history"
    assert result.final_content is None
    assert result.messages == [SYSTEM, {"role": "user", "content": "What is the status of ord_1001?"}]

    records = journal_records(result.journal_path)
    turn = next(record for record in records if record["event"] == "model_turn")
    assert turn["kind"] == "fatal_error"
    assert turn["error_code"] == "invalid_message_history"
    assert turn["assistant_message"] is None
    assert records[-1]["status"] == "aborted"
    assert records[-1]["error_code"] == "invalid_message_history"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (LlamaCppConnectionError("connection refused"), "backend_connection_error"),
        (LlamaCppTimeoutError("timed out"), "backend_timeout"),
        (APIError("truncated", status_code=500, code="completion_truncated"), "completion_truncated"),
        (APIError("unsupported", code="unsupported_schema"), "unsupported_schema"),
    ],
)
def test_backend_and_truncation_failures_abort(tmp_path, error, code):
    result, _, _ = episode(tmp_path, error)

    assert result.status is EpisodeStatus.ABORTED
    assert result.error_code == code


def test_step_budget_stops_after_the_last_allowed_step(tmp_path):
    result, service, store = episode(
        tmp_path,
        completion(calls=[("call_a", "get_order", {"order_id": "ord_1001"})]),
        completion("never requested"),
        budgets=Budgets(max_steps=1),
    )

    assert result.status is EpisodeStatus.BUDGET_EXHAUSTED
    assert result.steps == 1
    assert len(service.requests) == 1
    assert result.final_content.startswith("Stopped: step budget exhausted (1/1 steps, ")
    assert events(result.journal_path) == [
        "episode_started", "budget_checked", "model_turn", "tool_result",
        "budget_checked", "episode_finished",
    ]


def test_a_runaway_model_stops_at_the_default_step_ceiling(tmp_path):
    # Distinct arguments, so the repeated-call guard never trips before the step ceiling.
    orders = ["ord_1001", "ord_1002", "ord_1003", "ord_1004", "ord_1005"]
    responses = [
        completion(calls=[(f"call_{index}", "get_order", {"order_id": orders[index % 5]})])
        for index in range(20)
    ]
    result, service, _ = episode(tmp_path, *responses)

    assert result.status is EpisodeStatus.BUDGET_EXHAUSTED
    assert result.steps == Budgets().max_steps == 8
    assert len(service.requests) == 8


def test_zero_steps_never_calls_the_model(tmp_path):
    result, service, _ = episode(tmp_path, budgets=Budgets(max_steps=0))

    assert result.status is EpisodeStatus.BUDGET_EXHAUSTED
    assert service.requests == []


# Calls and observations


def test_missing_call_ids_are_synthesized_and_reused_on_the_tool_message(tmp_path):
    result, service, _ = episode(
        tmp_path,
        completion(calls=[("", "get_order", {"order_id": "ord_1001"}),
                          ("", "get_customer", {"customer_id": "cus_101"})]),
        completion("Done."),
    )

    second = service.requests[1].messages
    assert [call["id"] for call in second[2]["tool_calls"]] == ["call_0_0", "call_0_1"]
    assert [message["tool_call_id"] for message in second[3:]] == ["call_0_0", "call_0_1"]
    turn = next(r for r in journal_records(result.journal_path) if r["event"] == "model_turn")
    assert turn["assistant_message"]["tool_calls"][0]["id"] == "call_0_0"


def test_calls_in_one_message_run_in_list_order(tmp_path):
    result, service, store = episode(
        tmp_path,
        completion(calls=[
            ("call_a", "issue_refund", {"order_id": "ord_1001", "amount_inr": 120, "reason": "Dent"}),
            ("call_b", "list_payments", {"order_id": "ord_1001"}),
        ]),
        completion("Refunded."),
    )

    refund, payments = (json.loads(m["content"]) for m in service.requests[1].messages[3:5])
    assert refund["value"]["refund_id"] == "ref_7002"
    assert [r["refund_id"] for r in payments["value"]["refunds"]] == ["ref_7002"]
    assert len(store.refunds_for("ord_1001")) == 1


def test_unknown_order_returns_a_not_found_observation(tmp_path):
    result, service, _ = episode(
        tmp_path,
        completion(calls=[("call_a", "get_order", {"order_id": "ord_9999"})]),
        completion("I could not find that order."),
    )

    observation = json.loads(service.requests[1].messages[3]["content"])
    assert observation["error_type"] == "not_found"
    assert observation["retryable"] is False
    assert observation["details"] == [
        {"field": "order_id", "problem": "no order with this id exists", "got": "ord_9999"}
    ]


def slow_lookup(store: Store, order_id: str) -> dict:
    """Stub that blocks until the test releases it. It never touches the store."""
    RELEASE.wait(5)
    return {"order_id": order_id}


RELEASE = threading.Event()


def stub_registry(handler, timeout_s=5.0):
    registry = SteadyhandRegistry()
    registry.register(handler, ToolPolicy(Effect.READ, timeout_s, idempotent=True))
    return registry.freeze()


def test_a_blocking_handler_is_reported_as_a_timeout_from_its_policy():
    RELEASE.clear()
    store = load_fixture().clone()
    started = time.monotonic()
    try:
        result = SequentialExecutor().run(
            ToolCall("call_a", "slow_lookup", '{"order_id": "ord_1001"}'),
            stub_registry(slow_lookup, timeout_s=0.05),
            store,
        )
    finally:
        RELEASE.set()

    assert time.monotonic() - started < 2
    assert result.observation() == {
        "ok": False,
        "error_type": "timeout",
        "tool": "slow_lookup",
        "retryable": True,
        "hint": "slow_lookup did not finish in time. Calling it again is safe.",
        "details": [{"problem": "no result within 0.05 seconds"}],
    }
    assert result.handler_called is True
    assert result.arguments_sha256 is not None
    assert store.snapshot_sha256() == store.fixture_sha256


def slow_write(store: Store, order_id: str) -> dict:
    """Non-idempotent stub that blocks until released. It never touches the store."""
    RELEASE.wait(5)
    return {"order_id": order_id}


def test_a_timed_out_non_idempotent_write_is_not_retryable_and_points_at_verify():
    registry = SteadyhandRegistry()
    registry.register(slow_lookup, ToolPolicy(Effect.READ, 5.0, idempotent=True))
    registry.register(
        slow_write,
        ToolPolicy(Effect.NON_IDEMPOTENT_WRITE, 0.05, idempotent=False, verify="slow_lookup"),
    )
    RELEASE.clear()
    try:
        result = SequentialExecutor().run(ToolCall("c", "slow_write", '{"order_id": "x"}'), registry, load_fixture())
    finally:
        RELEASE.set()

    observation = result.observation()
    assert observation["retryable"] is False
    assert observation["hint"] == (
        "slow_write did not finish in time and may have taken effect. "
        "Check with slow_lookup before calling it again."
    )


def test_a_timeout_inside_an_episode_reaches_the_model_and_the_episode_continues(tmp_path):
    RELEASE.clear()
    try:
        result, service, _ = episode(
            tmp_path,
            completion(calls=[("call_a", "slow_lookup", {"order_id": "ord_1001"})]),
            completion("The lookup timed out."),
            registry=stub_registry(slow_lookup, timeout_s=0.05),
        )
    finally:
        RELEASE.set()

    assert result.status is EpisodeStatus.DONE
    assert json.loads(service.requests[1].messages[3]["content"])["error_type"] == "timeout"


def exploding(store: Store, order_id: str) -> dict:
    """Stub that fails with a multi-line message."""
    raise RuntimeError("boom\n  Traceback (most recent call last):\n    detail")


@pytest.mark.parametrize(
    ("call", "error_type", "problem"),
    [
        (ToolCall("c", "exploding", '{"order_id": "x"}'), "handler_error",
         "RuntimeError: boom Traceback (most recent call last): detail"),
        (ToolCall("c", "exploding", '{"order_id": '), "parse_error", None),
        (ToolCall("c", "exploding", "[1]"), "parse_error", "arguments must be a JSON object"),
        (ToolCall("c", "missing_tool", "{}"), "unknown_tool", "no tool with this name is registered"),
        (ToolCall("c", "exploding", '{"wrong": 1}'), "handler_error", None),
    ],
)
def test_the_executor_turns_every_failure_into_a_one_line_typed_error(call, error_type, problem):
    result = SequentialExecutor().run(call, stub_registry(exploding), load_fixture().clone())

    observation = result.observation()
    assert observation["ok"] is False
    assert observation["error_type"] == error_type
    assert "\n" not in json.dumps(observation)
    if problem is not None:
        assert observation["details"][0]["problem"] == problem


# Journal


def test_journal_lines_are_strict_json_with_sequence_numbers(tmp_path):
    result, _, _ = episode(
        tmp_path,
        completion(calls=[("call_a", "get_order", {"order_id": "ord_1001"})]),
        completion("Delivered."),
    )

    records = journal_records(result.journal_path)
    assert [record["seq"] for record in records] == list(range(len(records)))
    assert {record["episode_id"] for record in records} == {result.episode_id}
    assert all("recorded_at" in record for record in records)
    assert result.journal_path == tmp_path / f"{result.episode_id}.jsonl"

    started = records[0]
    assert started["settings"]["profile"] == "qwen3"
    assert started["settings"]["seed"] == 0
    assert started["budgets"]["max_steps"] == 8
    assert started["guard_limit"] == 3
    assert started["fixture_sha256"] == load_fixture().fixture_sha256
    assert started["settings"]["use_grammars"] is True and started["settings"]["enable_thinking"] is False
    assert started["journal_version"] == 1
    tool = next(record for record in records if record["event"] == "tool_result")
    assert tool["tool"] == "get_order"
    assert tool["arguments"] == '{"order_id": "ord_1001"}'
    assert tool["ok"] is True and tool["value"]["order_id"] == "ord_1001"
    assert tool["handler_called"] is True
    assert len(tool["arguments_sha256"]) == 64
    finished = records[-1]
    assert finished["status"] == "done"
    assert finished["final_content"] == "Delivered."
    assert finished["steps"] == 1


def test_journal_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_KEY", "local-env-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-env-secret")
    result, _, _ = episode(
        tmp_path,
        completion("Done."),
        question="Keys: passed-secret, local-env-secret, hf-env-secret",
        secrets=("passed-secret",),
    )
    journal = EpisodeJournal(result.journal_path, episode_id=result.episode_id)
    journal.append("headers", headers={"Authorization": "Bearer passed-secret"})

    raw = result.journal_path.read_text(encoding="utf-8")
    for secret in ("passed-secret", "local-env-secret", "hf-env-secret"):
        assert secret not in raw
    records = journal_records(result.journal_path)
    assert records[0]["messages"][1]["content"] == "Keys: [redacted], [redacted], [redacted]"
    assert records[-1]["headers"]["Authorization"] == "[redacted]"


def test_journal_is_append_only_and_never_reuses_a_file(tmp_path):
    result, _, _ = episode(tmp_path, completion("Done."))
    before = result.journal_path.read_text(encoding="utf-8").splitlines()

    EpisodeJournal(result.journal_path, episode_id=result.episode_id).append("extra")
    after = result.journal_path.read_text(encoding="utf-8").splitlines()

    assert after[: len(before)] == before
    assert len(after) == len(before) + 1
    with pytest.raises(FileExistsError):
        EpisodeJournal.create(tmp_path, result.episode_id)
    with pytest.raises(ValueError):
        EpisodeJournal(tmp_path / "x.jsonl", episode_id="../escape")


def test_default_journal_directory_is_the_gitignored_results_tree():
    journal = EpisodeJournal.create(episode_id="ep_default_dir_check")

    expected = Path(__file__).parent.parent / "results" / "steadyhand" / "ep_default_dir_check.jsonl"
    assert journal.path == expected.resolve()
    assert not journal.path.exists()


def test_the_journal_reconstructs_the_message_list(tmp_path):
    result, _, _ = episode(
        tmp_path,
        completion(calls=[("call_a", "get_order", {"order_id": "ord_1001"}),
                          ("call_b", "get_ticket", {"ticket_id": "tkt_404"})]),
        completion(calls=[("call_c", "add_order_note", {"order_id": "ord_1001", "note": "Asked about lid"})]),
        completion("Noted."),
    )

    messages = []
    for record in journal_records(result.journal_path):
        if record["event"] == "episode_started":
            messages.extend(record["messages"])
        elif record["event"] == "model_turn" and record["assistant_message"] is not None:
            messages.append(record["assistant_message"])
        elif record["event"] == "tool_result":
            messages.append({"role": "tool", "tool_call_id": record["tool_call_id"], "content": record["content"]})

    assert messages == result.messages


def test_an_unexpected_exception_still_finishes_the_journal(tmp_path):
    class BrokenClient:
        def settings(self):
            return {"profile": "fake"}

        def complete(self, messages, tools, *, tool_choice="auto"):
            raise RuntimeError("bug")

    journal = EpisodeJournal.create(tmp_path)
    with pytest.raises(RuntimeError):
        run_episode(BrokenClient(), build_registry(), load_fixture().clone(),
                    [{"role": "user", "content": "hi"}], journal=journal)

    finished = journal_records(journal.path)[-1]
    assert finished["event"] == "episode_finished"
    assert (finished["status"], finished["error_code"]) == ("aborted", "internal_error")


# Client


def test_client_refuses_thinking_on_the_grammar_path():
    with pytest.raises(ValueError, match="enable_thinking"):
        SteadyhandClient(FakeService(), enable_thinking=True)


def test_prompt_only_thinking_is_an_explicit_service_option():
    service = FakeService(completion("Hi."), use_grammars=False)
    client = SteadyhandClient(service)

    turn = client.complete([{"role": "user", "content": "hi"}], [])

    assert turn.kind is TurnKind.TEXT
    assert service.requests[0].chat_template_kwargs == {"enable_thinking": True}
    assert client.settings()["use_grammars"] is False


def test_client_carries_reasoning_and_repair_metadata():
    response = completion("Hi.", repair_attempts=1)
    response["choices"][0]["message"]["reasoning_content"] = "thinking"

    turn = SteadyhandClient(FakeService(response, use_grammars=False)).complete([{"role": "user", "content": "hi"}], [])

    assert turn.reasoning_content == "thinking"
    assert turn.repair_attempts == 1
    assert turn.assistant_message["reasoning_content"] == "thinking"


@pytest.mark.parametrize(
    ("response", "kind", "code"),
    [
        (completion("   \n"), TurnKind.PARSE_ERROR, "empty_model_response"),
        ({"choices": []}, TurnKind.FATAL_ERROR, "invalid_backend_response"),
    ],
)
def test_unusable_completions_are_classified(response, kind, code):
    turn = SteadyhandClient(FakeService(response)).complete([{"role": "user", "content": "hi"}], [])

    assert turn.kind is kind
    assert turn.error_code == code


# Live configuration


def test_live_settings_read_a_localhost_url_and_key_from_the_environment():
    settings = LiveSettings.from_env({"TOOLSMITH_BASE_URL": "http://127.0.0.1:8080/", "LOCAL_KEY": "k"})

    assert settings.base_url == "http://127.0.0.1:8080"
    assert settings.api_key == "k"
    assert "k'" not in repr(settings)
    assert LiveSettings.from_env({"TOOLSMITH_BASE_URL": "http://localhost:8080"}, api_key="x").api_key == "x"


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({}, "TOOLSMITH_BASE_URL is not set"),
        ({"TOOLSMITH_BASE_URL": "http://example.com:8080", "LOCAL_KEY": "k"}, "localhost"),
        ({"TOOLSMITH_BASE_URL": "ftp://127.0.0.1", "LOCAL_KEY": "k"}, "localhost"),
        ({"TOOLSMITH_BASE_URL": "http://127.0.0.1:8080"}, "LOCAL_KEY"),
    ],
)
def test_live_settings_reject_missing_or_remote_configuration(environ, message):
    with pytest.raises(LiveConfigError, match=message):
        LiveSettings.from_env(environ)


def test_shared_llama_client_pool_admits_the_server_parallel_count():
    client = build_llama_client(LiveSettings("http://127.0.0.1:8080", "k"), connections=4)
    try:
        pool = client._client._transport._pool
        assert pool._max_connections == 4
        assert pool._max_keepalive_connections == 4
    finally:
        client.close()


# Real ToolsmithService, fake llama-server


class PromptEchoRenderer:
    """Renders the message list as JSON so the fake backend can read the conversation."""

    def render(self, messages, tools=(), *, enable_thinking=None):
        return json.dumps({"messages": messages, "enable_thinking": enable_thinking})


QWEN35_GET_ORDER = (
    "<tool_call>\n<function=get_order>\n<parameter=order_id>\nord_1001\n</parameter>\n"
    "</function>\n</tool_call>"
)


class ScriptedLlama:
    """Fake LlamaCppClient: answers with a native tool call, then text once a tool result exists."""

    def __init__(self, tool_call_text=None, *, barrier=None):
        self.tool_call_text = tool_call_text
        self.barrier = barrier
        self.lock = threading.Lock()
        self.calls = []
        self.closed = False

    def complete(self, prompt, **settings):
        payload = json.loads(prompt)
        with self.lock:
            self.calls.append((payload, settings))
        messages = payload["messages"]
        if messages[-1]["role"] == "tool":
            observation = json.loads(messages[-1]["content"])
            return {"content": f"Answer: {json.dumps(observation['value'])[:40]}",
                    "tokens_evaluated": 60, "tokens_predicted": 9}
        if self.barrier is not None:
            self.barrier.wait()
        text = self.tool_call_text
        if text is None:
            call = (
                {"name": "list_payments", "arguments": {"order_id": "ord_1003"}}
                if "ord_1003" in messages[-1]["content"]
                else {"name": "get_order", "arguments": {"order_id": "ord_1001"}}
            )
            text = f"<tool_call>{json.dumps(call)}</tool_call>"
        return {"content": text, "tokens_evaluated": 40, "tokens_predicted": 12}

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("profile", "tool_call_text"),
    [(QWEN3_PROFILE, None), (QWEN35_PROFILE, QWEN35_GET_ORDER)],
)
def test_real_service_round_trip_uses_native_calls_grammars_and_no_thinking(tmp_path, profile, tool_call_text):
    llama = ScriptedLlama(tool_call_text)
    service = worker_service(llama, PromptEchoRenderer(), profile)
    assert isinstance(service, ToolsmithService) and service.recorder is None

    result = run_episode(
        SteadyhandClient(service),
        build_registry(),
        load_fixture().clone(),
        [SYSTEM, {"role": "user", "content": "Status of ord_1001?"}],
        journal=EpisodeJournal.create(tmp_path),
    )

    assert result.status is EpisodeStatus.DONE
    assert result.steps == 1
    assert (result.prompt_tokens, result.completion_tokens) == (100, 21)
    assert result.messages[2]["tool_calls"][0]["function"]["name"] == "get_order"
    assert json.loads(result.messages[3]["content"])["value"]["status"] == "delivered"
    for payload, settings in llama.calls:
        assert payload["enable_thinking"] is False
        assert settings["grammar"]
        assert settings["temperature"] == 0.0


# Notebook cell


def notebook_source(cell_id):
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
    return "".join(cell["source"])


def run_notebook_cell(tmp_path):
    namespace = {
        "DRIVE_RECORDS": tmp_path / "drive",
        "HF_TOKEN": "hf-test-secret",
        "LOCAL_KEY": "local-test-secret",
    }
    source = notebook_source("steadyhand-bare-loop")
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("base_url", "reason"),
    [(None, "TOOLSMITH_BASE_URL is not set"), ("http://10.0.0.5:8080", "localhost")],
)
def test_notebook_cell_skips_with_a_recorded_reason(tmp_path, monkeypatch, base_url, reason):
    monkeypatch.delenv("TOOLSMITH_BASE_URL", raising=False)
    if base_url is not None:
        monkeypatch.setenv("TOOLSMITH_BASE_URL", base_url)
    monkeypatch.setenv("STEADYHAND_LOCAL_DIR", str(tmp_path / "local"))

    namespace = run_notebook_cell(tmp_path)

    record = namespace["steadyhand_record"]
    assert record["status"] == "skipped"
    assert reason in record["reason"]
    saved = journal_records(tmp_path / "drive" / "steadyhand" / "part1-smoke.jsonl")
    assert saved[-1]["status"] == "skipped"
    assert not (tmp_path / "local").exists()


def test_notebook_cell_runs_two_overlapping_episodes_and_copies_journals(tmp_path, monkeypatch):
    llama = ScriptedLlama(barrier=threading.Barrier(2, timeout=10))
    built = {}

    def fake_build_llama_client(settings, *, connections, timeout):
        built.update(base_url=settings.base_url, api_key=settings.api_key, connections=connections)
        return llama

    monkeypatch.setenv("TOOLSMITH_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("STEADYHAND_LOCAL_DIR", str(tmp_path / "local"))
    monkeypatch.delenv("STEADYHAND_PROFILE", raising=False)
    monkeypatch.delenv("TOOLSMITH_PARALLEL", raising=False)
    monkeypatch.setattr(steadyhand, "build_llama_client", fake_build_llama_client)
    monkeypatch.setattr(ChatPromptRenderer, "from_pretrained", classmethod(lambda cls, *a, **k: PromptEchoRenderer()))

    namespace = run_notebook_cell(tmp_path)

    record = namespace["steadyhand_record"]
    assert record["status"] == "ran"
    assert record["completions_overlapped"] is True
    assert record["copy_errors"] == []
    assert record["overlap_seconds"] > 0
    assert built == {"base_url": "http://127.0.0.1:8080", "api_key": "local-test-secret", "connections": 4}
    assert llama.closed
    assert [run["status"] for run in record["episodes"]] == ["done", "done"]
    assert len({run["episode_id"] for run in record["episodes"]}) == 2
    for run in record["episodes"]:
        local, copy = Path(run["journal"]), Path(run["drive_copy"])
        assert local.parent == tmp_path / "local"
        assert copy.parent == tmp_path / "drive" / "steadyhand"
        assert local.read_bytes() == copy.read_bytes()
        assert load_manifest(manifest_path_for(copy)).completed is True
        assert {r["episode_id"] for r in journal_records(local)} == {run["episode_id"]}
        assert "local-test-secret" not in local.read_text(encoding="utf-8")
    assert all(settings["grammar"] for _, settings in llama.calls)

import json

import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion

from toolsmith.api import APIError, ChatCompletionRequest, ToolsmithService, create_app
from toolsmith.backends import CompletionTruncatedError
from toolsmith.parsing import Qwen3ToolCallParser
from toolsmith.recording import JsonlRecorder
from toolsmith.repair import RepairPolicy


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def render(self, messages, tools=(), *, enable_thinking=None):
        self.calls.append((messages, tools, enable_thinking))
        return "rendered prompt"


class FakeClient:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, prompt, **settings):
        self.calls.append((prompt, settings))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def make_service(*responses, recorder=None):
    client = FakeClient(*responses)
    renderer = FakeRenderer()
    parser = Qwen3ToolCallParser(lambda: "call_test")
    service = ToolsmithService(
        client,
        renderer,
        parser=parser,
        recorder=recorder,
        repair_policy=RepairPolicy(max_repairs=0),
    )
    return service, client, renderer


def request(**overrides):
    values = {
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [WEATHER_TOOL],
        "max_tokens": 128,
        "temperature": 0.0,
        "seed": 7,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    values.update(overrides)
    return ChatCompletionRequest(**values)


def test_text_response_matches_openai_sdk_model():
    service, client, renderer = make_service(
        {"content": "It is sunny.", "timings": {"prompt_n": 20, "predicted_n": 4}}
    )

    response = service.chat(request(tools=[]))
    parsed = ChatCompletion.model_validate(response)

    assert parsed.choices[0].message.content == "It is sunny."
    assert parsed.choices[0].finish_reason == "stop"
    assert parsed.usage.total_tokens == 24
    assert client.calls[0] == (
        "rendered prompt",
        {"max_tokens": 128, "temperature": 0.0, "seed": 7},
    )
    assert renderer.calls[0][1] == []


def test_tool_call_and_tool_result_round_trip():
    service, client, renderer = make_service(
        {
            "content": '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
            "tokens_evaluated": 30,
            "tokens_predicted": 12,
        },
        {"content": "Paris is 22 degrees and sunny."},
    )

    first = service.chat(
        request(chat_template_kwargs={"enable_thinking": False})
    )
    assistant = first["choices"][0]["message"]
    assert first["choices"][0]["finish_reason"] == "tool_calls"
    assert assistant["tool_calls"] == [
        {
            "id": "call_test",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }
    ]
    assert renderer.calls[0][1] == [WEATHER_TOOL]
    assert renderer.calls[0][2] is False

    messages = [
        {"role": "user", "content": "Weather in Paris?"},
        assistant,
        {
            "role": "tool",
            "tool_call_id": "call_test",
            "content": json.dumps({"temperature": 22, "condition": "sunny"}),
        },
    ]
    second = service.chat(request(messages=messages))

    assert second["choices"][0]["message"]["content"] == "Paris is 22 degrees and sunny."
    assert renderer.calls[1][0] == messages
    assert len(client.calls) == 2


def test_recorder_saves_raw_and_openai_responses(tmp_path):
    path = tmp_path / "records" / "gate.jsonl"
    recorder = JsonlRecorder(path, secrets=["secret-key"])
    service, _, _ = make_service({"content": "Done"}, recorder=recorder)

    service.chat(request(tools=[]))

    record = json.loads(path.read_text())
    assert record["event"] == "chat_completion"
    assert record["raw_prompt"] == "rendered prompt"
    assert record["raw_response"] == {"content": "Done"}
    assert record["response"]["choices"][0]["message"]["content"] == "Done"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stream_options": {"include_usage": True}}, "stream_options"),
        ({"tool_choice": "sometimes"}, "tool_choice"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"chat_template_kwargs": {"unknown": True}}, "Unsupported"),
    ],
)
def test_unsupported_options_fail_loudly(overrides, message):
    service, _, _ = make_service({"content": "unused"})

    with pytest.raises(APIError, match=message):
        service.chat(request(**overrides))


def test_truncation_is_an_explicit_api_error():
    service, _, _ = make_service(CompletionTruncatedError({"content": "partial"}))

    with pytest.raises(APIError) as caught:
        service.chat(request())

    assert caught.value.code == "completion_truncated"


def test_fastapi_route_and_openai_error_shape():
    service, _, _ = make_service({"content": "Done"})
    client = TestClient(create_app(service))

    valid = client.post(
        "/v1/chat/completions",
        json={"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hi"}]},
    )
    invalid = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "tool_choice": "required"},
    )
    missing_messages = client.post("/v1/chat/completions", json={})

    assert valid.status_code == 200
    assert valid.json()["object"] == "chat.completion"
    assert invalid.status_code == 400
    assert invalid.json()["error"]["type"] == "invalid_request_error"
    assert missing_messages.status_code == 400
    assert missing_messages.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    ("choice", "content", "root", "finish_reason"),
    [
        ("auto", "No tool needed", "root ::= plain-text | tool-calls", "stop"),
        ("none", "No tool needed", "root ::= plain-text", "stop"),
        (
            "required",
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
            "root ::= tool-calls",
            "tool_calls",
        ),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
            "root ::= tool-",
            "tool_calls",
        ),
    ],
)
def test_tool_choice_modes_send_expected_grammar(choice, content, root, finish_reason):
    service, client, _ = make_service({"content": content})

    result = service.chat(request(tool_choice=choice))

    settings = client.calls[0][1]
    assert settings["grammar"].startswith(root)
    assert settings["max_tokens"] == 128
    assert result["choices"][0]["finish_reason"] == finish_reason


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (
            '<tool_call>{"name":"get_weather","arguments":{"city":3}}</tool_call>',
            "schema_validation_failed",
        ),
        (
            '<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
            "unknown_tool",
        ),
        ("No call", "tool_choice_violation"),
    ],
)
def test_required_output_validation_failures_are_server_errors(content, code):
    service, _, _ = make_service({"content": content})

    with pytest.raises(APIError) as caught:
        service.chat(request(tool_choice="required"))

    assert caught.value.code == code
    assert caught.value.status_code == 500


def test_unknown_tool_is_rejected_in_auto_mode():
    service, _, _ = make_service(
        {"content": '<tool_call>{"name":"unknown","arguments":{}}</tool_call>'}
    )

    with pytest.raises(APIError) as caught:
        service.chat(request())

    assert caught.value.code == "unknown_tool"


@pytest.mark.parametrize(
    "overrides",
    [
        {"chat_template_kwargs": {}},
        {"chat_template_kwargs": {"enable_thinking": True}},
    ],
)
def test_tool_grammars_require_thinking_to_be_disabled(overrides):
    service, client, renderer = make_service({"content": "unused"})

    with pytest.raises(APIError) as caught:
        service.chat(request(**overrides))

    assert caught.value.code == "thinking_with_grammar_unsupported"
    assert client.calls == []
    assert renderer.calls == []


def test_invalid_named_choice_and_unsupported_schema_fail_before_generation():
    service, client, _ = make_service({"content": "unused"})

    with pytest.raises(APIError) as caught:
        service.chat(request(tool_choice={"type": "function", "function": {"name": "missing"}}))
    assert caught.value.code == "invalid_tool_choice"

    unsupported = json.loads(json.dumps(WEATHER_TOOL))
    unsupported["function"]["parameters"]["properties"]["city"]["pattern"] = "^Paris$"
    with pytest.raises(APIError) as caught:
        service.chat(request(tools=[unsupported]))
    assert caught.value.code == "unsupported_schema"
    assert "/properties/city/pattern" in str(caught.value)
    assert client.calls == []


def test_required_without_tools_fails_before_generation():
    service, client, _ = make_service({"content": "unused"})

    with pytest.raises(APIError) as caught:
        service.chat(request(tools=[], tool_choice="required"))

    assert caught.value.code == "invalid_tool_choice"
    assert client.calls == []


def test_recorder_saves_grammar_and_validation_metadata(tmp_path):
    recorder = JsonlRecorder(tmp_path / "grammar.jsonl")
    service, client, _ = make_service(
        {
            "content": '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
        },
        recorder=recorder,
    )

    service.chat(request(tool_choice="required"))

    record = json.loads((tmp_path / "grammar.jsonl").read_text())
    assert record["grammar_sha256"]
    assert record["grammar"] == client.calls[0][1]["grammar"]
    assert record["grammar_mode"] == "required"
    assert record["tool_choice"] == {"mode": "required"}
    assert record["validation"] == "passed"
    assert record["outcome"] == "success"

import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from toolsmith.api import APIError, ChatCompletionRequest, ToolsmithService, create_app
from toolsmith.backends import CompletionTruncatedError
from toolsmith.deltas import aggregate_chunks
from toolsmith.parsing import Qwen3ToolCallParser
from toolsmith.protocols import QWEN35_PROFILE
from toolsmith.recording import JsonlRecorder


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["answer", "count"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "answer", "schema": SCHEMA, "strict": True},
}


class Renderer:
    def __init__(self):
        self.calls = []

    def render(self, messages, tools=(), *, enable_thinking=None):
        self.calls.append((deepcopy(messages), deepcopy(tools), enable_thinking))
        return f"prompt-{len(self.calls)}"


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, prompt, **settings):
        self.calls.append((prompt, settings))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def health(self):
        return {"status": "ok"}


def request(**overrides):
    values = {
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [WEATHER_TOOL],
        "tool_choice": "required",
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
    }
    values.update(overrides)
    return ChatCompletionRequest(**values)


def qwen_call(city="Paris"):
    arguments = json.dumps({"city": city}, separators=(",", ":"))
    return (
        '<tool_call>{"name":"get_weather","arguments":'
        + arguments
        + "}</tool_call>"
    )


def qwen35_call(city="Paris"):
    return (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\n"
        f"{city}\n</parameter>\n</function>\n</tool_call>"
    )


def test_non_streaming_repair_renders_error_reuses_grammar_and_sums_usage(tmp_path):
    client = Client(
        {"content": qwen_call(3), "tokens_evaluated": 10, "tokens_predicted": 4},
        {"content": qwen_call(), "tokens_evaluated": 20, "tokens_predicted": 5},
    )
    renderer = Renderer()
    recorder = JsonlRecorder(tmp_path / "repair.jsonl")
    service = ToolsmithService(
        client,
        renderer,
        parser=Qwen3ToolCallParser(lambda: "call_test"),
        recorder=recorder,
    )
    original = request()

    response = service.chat(original)

    assert len(client.calls) == 2
    assert client.calls[0][1]["grammar"] == client.calls[1][1]["grammar"]
    assert response["usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 9,
        "total_tokens": 39,
    }
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"city":"Paris"}'
    assert original.messages == [{"role": "user", "content": "Weather?"}]
    assert "untrusted JSON string" in renderer.calls[1][0][-1]["content"]
    records = [json.loads(line) for line in (tmp_path / "repair.jsonl").read_text().splitlines()]
    assert [record["outcome"] for record in records] == ["retry", "repaired"]
    assert records[-1]["repair_attempts"] == 1


def test_repair_exhaustion_preserves_final_code_and_attempt_count():
    service = ToolsmithService(
        Client({"content": qwen_call(3)}, {"content": qwen_call(4)}),
        Renderer(),
    )

    with pytest.raises(APIError) as caught:
        service.chat(request())

    assert caught.value.code == "schema_validation_failed"
    assert caught.value.repair_attempts == 1
    assert caught.value.body()["error"]["repair_attempts"] == 1


def test_truncation_is_not_repaired():
    client = Client(CompletionTruncatedError({"content": "partial"}))
    service = ToolsmithService(client, Renderer())

    with pytest.raises(APIError) as caught:
        service.chat(request())

    assert caught.value.code == "completion_truncated"
    assert len(client.calls) == 1


def test_structured_response_uses_schema_prompt_grammar_and_openai_shape():
    client = Client({"content": '{"answer":"sunny","count":2}'})
    renderer = Renderer()
    service = ToolsmithService(client, renderer)
    original_messages = [{"role": "user", "content": "Return the answer"}]

    response = service.chat(
        request(
            messages=original_messages,
            tools=[],
            tool_choice="auto",
            response_format=RESPONSE_FORMAT,
        )
    )

    parsed = ChatCompletion.model_validate(response)
    assert json.loads(parsed.choices[0].message.content) == {"answer": "sunny", "count": 2}
    assert parsed.choices[0].finish_reason == "stop"
    assert client.calls[0][1]["grammar"].startswith("root ::= ")
    assert "Schema name: answer" in renderer.calls[0][0][0]["content"]
    assert original_messages == [{"role": "user", "content": "Return the answer"}]


def test_official_openai_sdk_parse_consumes_structured_response():
    class Answer(BaseModel):
        answer: str
        count: int

    app = create_app(
        ToolsmithService(Client({"content": '{"answer":"sunny","count":2}'}), Renderer())
    )
    test_client = TestClient(app)

    def transport(request):
        response = test_client.post(
            str(request.url),
            content=request.content,
            headers=dict(request.headers),
        )
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(transport))
    sdk = OpenAI(base_url="http://testserver/v1", api_key="test", http_client=http_client)

    result = sdk.chat.completions.parse(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": "Return JSON"}],
        response_format=Answer,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    assert result.choices[0].message.parsed == Answer(answer="sunny", count=2)


def test_structured_stream_validates_before_success_finish():
    class StreamingClient:
        def stream_complete(self, prompt, **settings):
            yield {"content": '{"answer":"sun', "stop": False}
            yield {"content": 'ny","count":2}', "stop": True, "truncated": False}

    service = ToolsmithService(StreamingClient(), Renderer())
    parts = list(
        service.chat_stream(
            request(
                tools=[],
                tool_choice="auto",
                stream=True,
                response_format=RESPONSE_FORMAT,
                stream_options={"include_usage": True},
            )
        )
    )
    chunks = [
        json.loads(part[6:])
        for part in parts
        if part.startswith("data: ") and part != "data: [DONE]\n\n"
    ]
    result = aggregate_chunks(chunks)

    assert json.loads(result["message"]["content"]) == {"answer": "sunny", "count": 2}
    assert result["finish_reason"] == "stop"


def test_invalid_structured_output_is_repaired_once():
    client = Client(
        {"content": '{"answer":"sunny","count":"two"}'},
        {"content": '{"answer":"sunny","count":2}'},
    )
    service = ToolsmithService(client, Renderer())

    response = service.chat(
        request(tools=[], tool_choice="auto", response_format=RESPONSE_FORMAT)
    )

    assert json.loads(response["choices"][0]["message"]["content"])["count"] == 2
    assert len(client.calls) == 2


def test_qwen35_service_parses_native_call_and_compiles_xml_grammar():
    client = Client({"content": qwen35_call()})
    service = ToolsmithService(client, Renderer(), profile=QWEN35_PROFILE)

    response = service.chat(request(model=QWEN35_PROFILE.model))

    call = response["choices"][0]["message"]["tool_calls"][0]
    assert call["function"] == {"name": "get_weather", "arguments": '{"city":"Paris"}'}
    assert "<function=get_weather>" in client.calls[0][1]["grammar"]
    assert '"name"' not in client.calls[0][1]["grammar"]


def test_qwen35_stream_aggregates_native_arguments():
    class StreamingClient:
        def stream_complete(self, prompt, **settings):
            for character in qwen35_call("Zürich"):
                yield {"content": character, "stop": False}
            yield {"content": "", "stop": True, "truncated": False}

    service = ToolsmithService(StreamingClient(), Renderer(), profile=QWEN35_PROFILE)
    parts = list(service.chat_stream(request(model=QWEN35_PROFILE.model, stream=True)))
    chunks = [
        json.loads(part[6:])
        for part in parts
        if part.startswith("data: ") and part != "data: [DONE]\n\n"
    ]
    result = aggregate_chunks(chunks)

    assert result["message"]["tool_calls"][0]["function"]["arguments"] == '{"city":"Zürich"}'
    assert result["finish_reason"] == "tool_calls"


def test_qwen35_history_decodes_arguments_only_for_its_template():
    renderer = Renderer()
    service = ToolsmithService(Client({"content": "Done"}), renderer, profile=QWEN35_PROFILE)
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]

    service.chat(
        request(model=QWEN35_PROFILE.model, messages=messages, tool_choice="auto")
    )

    history_arguments = renderer.calls[0][0][1]["tool_calls"][0]["function"]["arguments"]
    assert history_arguments == {"city": "Paris"}
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"city":"Paris"}'


def test_authentication_health_models_and_invalid_history():
    service = ToolsmithService(Client({"content": "Done"}), Renderer())
    client = TestClient(create_app(service, api_key="secret"))

    unauthorized = client.get("/v1/models")
    models = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    health = client.get("/health", headers={"Authorization": "Bearer secret"})

    assert unauthorized.status_code == 401
    assert models.json()["data"][0]["id"] == "Qwen/Qwen3-8B"
    assert health.json() == {"status": "ok"}

    with pytest.raises(APIError):
        service.chat(request(messages=[{"role": "tool", "tool_call_id": "missing", "content": "x"}]))


def test_unsupported_response_formats_and_tools_conflict_make_no_backend_call():
    client = Client({"content": "unused"})
    service = ToolsmithService(client, Renderer())

    with pytest.raises(APIError) as caught:
        service.chat(request(tools=[], tool_choice="auto", response_format={"type": "json_object"}))
    assert caught.value.code == "unsupported_response_format"
    with pytest.raises(APIError) as caught:
        service.chat(request(response_format=RESPONSE_FORMAT))
    assert caught.value.code == "incompatible_response_format"
    assert client.calls == []

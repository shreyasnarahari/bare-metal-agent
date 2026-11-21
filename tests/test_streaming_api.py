import json

import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionChunk

from toolsmith.api import APIError, ChatCompletionRequest, ToolsmithService, create_app
from toolsmith.backends import CompletionTruncatedError
from toolsmith.deltas import DONE, aggregate_chunks
from toolsmith.parsing import Qwen3IncrementalParser
from toolsmith.recording import JsonlRecorder


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
    def render(self, messages, tools=(), *, enable_thinking=None):
        return "rendered prompt"


class TrackingUpstream:
    def __init__(self, items, owner):
        self.items = iter(items)
        self.owner = owner

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.items)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.owner.closed = True


class FakeStreamingClient:
    def __init__(self, items):
        self.items = items
        self.calls = []
        self.closed = False

    def stream_complete(self, prompt, **settings):
        self.calls.append((prompt, settings))
        return TrackingUpstream(self.items, self)


def make_service(items, *, recorder=None):
    client = FakeStreamingClient(items)
    service = ToolsmithService(
        client,
        FakeRenderer(),
        parser_factory=lambda thinking: Qwen3IncrementalParser(
            lambda: "call_test", expect_thinking=thinking
        ),
        recorder=recorder,
    )
    return service, client


def request(**overrides):
    values = {
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    values.update(overrides)
    return ChatCompletionRequest(**values)


def decode_stream(stream):
    chunks = []
    done_count = 0
    for part in stream:
        assert part.startswith("data: ")
        payload = part[len("data: "):]
        if payload == "[DONE]\n\n":
            done_count += 1
        else:
            chunks.append(json.loads(payload))
    return chunks, done_count


def test_text_stream_has_stable_chunks_finish_usage_and_done():
    events = [
        {"content": "Hello ", "stop": False},
        {"content": "world", "stop": False},
        {
            "content": "",
            "stop": True,
            "truncated": False,
            "timings": {"prompt_n": 2, "predicted_n": 3},
        },
    ]
    service, client = make_service(events)

    chunks, done_count = decode_stream(
        service.chat_stream(request(stream_options={"include_usage": True}))
    )

    for chunk in chunks:
        ChatCompletionChunk.model_validate(chunk)
    aggregated = aggregate_chunks(chunks)
    assert aggregated["message"]["content"] == "Hello world"
    assert aggregated["finish_reason"] == "stop"
    assert aggregated["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert done_count == 1
    assert client.calls[0][1] == {"max_tokens": 256, "temperature": 0.6, "seed": None}
    assert client.closed


def test_tool_stream_reconstructs_arguments():
    text = '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
    events = [{"content": character, "stop": False} for character in text]
    events.append({"content": "", "stop": True, "truncated": False})
    service, client = make_service(events)

    chunks, done_count = decode_stream(
        service.chat_stream(request(tools=[WEATHER_TOOL], tool_choice="required"))
    )
    aggregated = aggregate_chunks(chunks)

    assert aggregated["message"]["tool_calls"] == [
        {
            "id": "call_test",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }
    ]
    assert aggregated["finish_reason"] == "tool_calls"
    assert client.calls[0][1]["grammar"].startswith("root ::= tool-calls")
    assert done_count == 1


def test_midstream_truncation_emits_error_not_success_finish():
    service, _ = make_service(
        [
            {"content": "partial", "stop": False},
            CompletionTruncatedError({"content": "partial", "truncated": True}),
        ]
    )

    chunks, done_count = decode_stream(service.chat_stream(request()))

    assert chunks[-1]["error"]["code"] == "completion_truncated"
    assert all(choice.get("finish_reason") is None for chunk in chunks for choice in chunk.get("choices", []))
    assert done_count == 1


def test_early_close_closes_upstream_and_records_cancellation(tmp_path):
    events = [
        {"content": "first", "stop": False},
        {"content": "second", "stop": False},
    ]
    recorder = JsonlRecorder(tmp_path / "stream.jsonl")
    service, client = make_service(events, recorder=recorder)
    stream = service.chat_stream(request())
    next(stream)
    next(stream)

    stream.close()

    assert client.closed
    record = json.loads((tmp_path / "stream.jsonl").read_text())
    assert record["outcome"] == "cancelled"


def test_stream_request_validation_happens_before_iteration():
    service, _ = make_service([])

    with pytest.raises(APIError, match="tool_choice"):
        service.chat_stream(request(tool_choice="required"))


def test_fastapi_route_returns_event_stream():
    service, _ = make_service(
        [{"content": "done", "stop": True, "truncated": False}]
    )
    client = TestClient(create_app(service))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "chat.completion.chunk" in body
    assert DONE in body


def test_stream_schema_validation_failure_has_no_success_finish(tmp_path):
    text = '<tool_call>{"name":"get_weather","arguments":{"city":3}}</tool_call>'
    events = [{"content": text, "stop": False}, {"content": "", "stop": True}]
    recorder = JsonlRecorder(tmp_path / "invalid-stream.jsonl")
    service, client = make_service(events, recorder=recorder)

    chunks, done_count = decode_stream(
        service.chat_stream(request(tools=[WEATHER_TOOL], tool_choice="required"))
    )

    assert chunks[-1]["error"]["code"] == "schema_validation_failed"
    assert all(
        choice.get("finish_reason") is None
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert done_count == 1
    assert client.closed
    record = json.loads((tmp_path / "invalid-stream.jsonl").read_text())
    assert record["validation"] == "failed"
    assert record["outcome"] == "error"
    assert record["grammar_sha256"]


def test_stream_named_choice_uses_exact_named_tool_grammar():
    events = [
        {
            "content": '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
            "stop": True,
        }
    ]
    service, client = make_service(events)
    named = {"type": "function", "function": {"name": "get_weather"}}

    chunks, _ = decode_stream(
        service.chat_stream(request(tools=[WEATHER_TOOL], tool_choice=named))
    )

    grammar = client.calls[0][1]["grammar"]
    assert grammar.startswith("root ::= tool-")
    assert '\\"get_weather\\"' in grammar
    assert "plain-text" not in grammar
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

import json

from openai.types.chat import ChatCompletionChunk

from toolsmith.deltas import DONE, OpenAIStreamAdapter, aggregate_chunks, sse_json
from toolsmith.parsing import Qwen3IncrementalParser


def test_delta_adapter_reconstructs_tool_calls_and_usage():
    adapter = OpenAIStreamAdapter(
        "test-model",
        completion_id="chatcmpl-test",
        created=123,
        include_usage=True,
    )
    parser = Qwen3IncrementalParser(lambda: "call_1", expect_thinking=True)
    text = (
        "<think>Need weather</think>Answer"
        '<tool_call>{"name":"weather","arguments":{"city":"Paris","note":"café"}}</tool_call>'
    )
    chunks = [adapter.role_chunk()]
    for character in text:
        for event in parser.feed(character):
            chunk = adapter.convert(event)
            if chunk is not None:
                chunks.append(chunk)
    for event in parser.finish():
        chunk = adapter.convert(event)
        if chunk is not None:
            chunks.append(chunk)
    chunks.append(adapter.finish_chunk(parser.response))
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    chunks.append(adapter.usage_chunk(usage))

    for chunk in chunks:
        ChatCompletionChunk.model_validate(chunk)
    aggregated = aggregate_chunks(chunks)

    assert aggregated["message"]["content"] == "Answer"
    assert aggregated["message"]["reasoning_content"] == "Need weather"
    assert aggregated["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "weather",
                "arguments": '{"city":"Paris","note":"café"}',
            },
        }
    ]
    assert aggregated["finish_reason"] == "tool_calls"
    assert aggregated["usage"] == usage
    assert {chunk["id"] for chunk in chunks} == {"chatcmpl-test"}
    assert {chunk["created"] for chunk in chunks} == {123}
    assert chunks[-1]["choices"] == []


def test_aggregate_chunks_accepts_null_tool_calls_from_sdk_dump():
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "", "tool_calls": None},
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": None,
                        "function": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": None,
                                "function": {"name": None, "arguments": '{"city":"Paris"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    ]

    aggregated = aggregate_chunks(chunks)

    assert aggregated["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }
    ]
    assert aggregated["finish_reason"] == "tool_calls"
    assert aggregated["usage"]["total_tokens"] == 3


def test_usage_chunk_is_optional_and_sse_preserves_unicode():
    adapter = OpenAIStreamAdapter("test-model")

    assert adapter.usage_chunk({}) is None
    assert sse_json({"content": "café"}) == 'data: {"content":"café"}\n\n'
    assert DONE == "data: [DONE]\n\n"

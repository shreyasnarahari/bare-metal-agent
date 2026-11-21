import json
import random
from pathlib import Path

import pytest

from toolsmith.qwen35 import Qwen35IncrementalParser, Qwen35ToolCallParser
from toolsmith.registry import Tool
from toolsmith.streaming import (
    ContentDelta,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallParseError,
    ToolCallStart,
)

CASES = json.loads((Path(__file__).parent / "fixtures" / "qwen35_cases.json").read_text())


def tools():
    return [
        Tool(
            "create_profile",
            "Create a profile",
            {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"enum": ["c", "f"]},
                    "count": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "active": {"type": "boolean"},
                    "nothing": {"type": "null"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "location": {
                        "type": "object",
                        "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}},
                        "required": ["lat", "lon"],
                        "additionalProperties": False,
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        ),
        Tool("ping", "Ping", {"type": "object", "properties": {}, "additionalProperties": False}),
    ]


def ids():
    values = (f"call_{index}" for index in range(20))
    return lambda: next(values)


def chunkings(text):
    yield [text]
    yield list(text)
    for size in (2, 3, 7, 13):
        yield [text[index:index + size] for index in range(0, len(text), size)]
    randomizer = random.Random(19)
    chunks = []
    while text:
        size = randomizer.randint(1, 11)
        chunks.append(text[:size])
        text = text[size:]
    yield chunks


def parse_chunks(chunks, expect_thinking):
    parser = Qwen35IncrementalParser(tools(), ids(), expect_thinking=expect_thinking)
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    return parser.response, events


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_non_streaming_and_arbitrary_chunks_match(case):
    expected = Qwen35ToolCallParser(
        tools(), ids(), expect_thinking=case["expect_thinking"]
    ).parse(case["text"])
    call = expected.tool_calls[0]

    assert call.name == case["expected"]["name"]
    assert json.loads(call.arguments) == case["expected"]["arguments"]
    assert expected.reasoning_content == case["expected"].get("reasoning")
    assert expected.content == case["expected"].get("content")

    for chunks in chunkings(case["text"]):
        response, events = parse_chunks(chunks, case["expect_thinking"])
        assert response == expected
        starts = [event for event in events if isinstance(event, ToolCallStart)]
        ends = [event for event in events if isinstance(event, ToolCallEnd)]
        assert len(starts) == len(ends) == 1
        arguments = "".join(
            event.text for event in events if isinstance(event, ToolCallArgumentsDelta)
        )
        assert json.loads(arguments) == case["expected"]["arguments"]
        assert "".join(event.text for event in events if isinstance(event, ContentDelta)).strip() == (case["expected"].get("content") or "")
        assert "".join(event.text for event in events if isinstance(event, ReasoningDelta)).strip() == (case["expected"].get("reasoning") or "")


def test_every_split_point_and_parallel_calls():
    text = (
        "<tool_call>\n<function=ping>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=create_profile>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    )
    expected = Qwen35ToolCallParser(tools(), ids(), expect_thinking=False).parse(text)

    for position in range(len(text) + 1):
        response, events = parse_chunks(
            (text[:position], text[position:]), expect_thinking=False
        )
        assert response == expected
        assert [event.index for event in events if isinstance(event, ToolCallStart)] == [0, 1]
        assert [event.index for event in events if isinstance(event, ToolCallEnd)] == [0, 1]


def test_plain_text_and_omitted_thinking():
    plain = Qwen35ToolCallParser(tools(), expect_thinking=False).parse("No tool needed")
    thinking = Qwen35ToolCallParser(tools(), expect_thinking=True).parse(
        "Reason first.</think>Answer"
    )

    assert plain.content == "No tool needed"
    assert thinking.reasoning_content == "Reason first."
    assert thinking.content == "Answer"


def test_string_null_and_invalid_typed_values_are_not_coerced():
    text = (
        "<tool_call><function=create_profile>"
        "<parameter=city>null</parameter>"
        "<parameter=count>not-an-int</parameter>"
        "<parameter=active>yes</parameter>"
        "</function></tool_call>"
    )

    response = Qwen35ToolCallParser(tools(), expect_thinking=False).parse(text)

    assert json.loads(response.tool_calls[0].arguments) == {
        "city": "null",
        "count": "not-an-int",
        "active": "yes",
    }


def test_null_branch_with_extra_keywords_is_not_nullable():
    tool = Tool(
        "get_weather",
        "Weather",
        {
            "type": "object",
            "properties": {
                "unit": {"anyOf": [{"type": "string"}, {"type": "null", "const": None}]},
            },
            "additionalProperties": False,
        },
    )
    text = (
        "<tool_call><function=get_weather>"
        "<parameter=unit>null</parameter>"
        "</function></tool_call>"
    )

    parsed = Qwen35ToolCallParser([tool], expect_thinking=False).parse(text)

    assert json.loads(parsed.tool_calls[0].arguments) == {"unit": "null"}


def test_nullable_parameter_decodes_json_null_and_string():
    tool = Tool(
        "get_weather",
        "Weather",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    )
    text = (
        "<tool_call><function=get_weather>"
        "<parameter=city>Paris</parameter>"
        "<parameter=unit>null</parameter>"
        "</function></tool_call>"
    )

    parsed = Qwen35ToolCallParser([tool], expect_thinking=False).parse(text)

    assert json.loads(parsed.tool_calls[0].arguments) == {"city": "Paris", "unit": None}


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("<tool_call><function=ping></tool_call>", "invalid_tool_call_xml"),
        ("<tool_call><function=create_profile><parameter=city>A<parameter=city>B</parameter></function></tool_call>", "invalid_tool_call_xml"),
        ("<tool_call><function=create_profile><parameter=city>A</parameter><parameter=city>B</parameter></function></tool_call>", "duplicate_tool_call_parameter"),
        ("<tool_call><function=ping></function><tool_call></function></tool_call>", "nested_tool_call"),
        ("</tool_call>", "unexpected_tool_call_close"),
        ("<tool_call><function=ping></function>", "unclosed_tool_call"),
        ("<think>unfinished", "unclosed_thinking_block"),
    ],
)
def test_malformed_xml_and_truncation_fail(text, code):
    with pytest.raises(ToolCallParseError) as caught:
        Qwen35ToolCallParser(tools()).parse(text)

    assert caught.value.code == code

import json
import random
from pathlib import Path

import pytest

from toolsmith.parsing import Qwen3ToolCallParser
from toolsmith.streaming import (
    ContentDelta,
    Qwen3IncrementalParser,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallParseError,
    ToolCallStart,
)

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "qwen3_responses.json").read_text()
)


def id_source():
    return (f"call_{index}" for index in range(20))


def ids():
    values = id_source()
    return lambda: next(values)


def chunkings(text):
    yield [text]
    yield list(text)
    for size in (2, 3, 5, 11):
        yield [text[index:index + size] for index in range(0, len(text), size)]
    for seed in (0, 17):
        randomizer = random.Random(seed)
        chunks = []
        remaining = text
        while remaining:
            size = randomizer.randint(1, 9)
            chunks.append(remaining[:size])
            remaining = remaining[size:]
        yield chunks


def parse_chunks(text, chunks, expect_thinking):
    parser = Qwen3IncrementalParser(ids(), expect_thinking=expect_thinking)
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    return parser.response, events


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture["name"])
def test_streaming_matches_non_streaming_for_all_chunkings(fixture):
    expected = Qwen3ToolCallParser(ids()).parse(fixture["text"])

    for chunks in chunkings(fixture["text"]):
        response, events = parse_chunks(
            fixture["text"], chunks, fixture["expect_thinking"]
        )
        assert response == expected
        content = "".join(event.text for event in events if isinstance(event, ContentDelta))
        reasoning = "".join(
            event.text for event in events if isinstance(event, ReasoningDelta)
        )
        assert (content.strip() or None) == expected.content
        assert (reasoning.strip() or None) == expected.reasoning_content

        starts = [event for event in events if isinstance(event, ToolCallStart)]
        ends = [event for event in events if isinstance(event, ToolCallEnd)]
        assert [event.index for event in starts] == list(range(len(expected.tool_calls)))
        assert [event.index for event in ends] == list(range(len(expected.tool_calls)))
        for call_index, call in enumerate(expected.tool_calls):
            raw_arguments = "".join(
                event.text
                for event in events
                if isinstance(event, ToolCallArgumentsDelta) and event.index == call_index
            )
            assert json.loads(raw_arguments) == json.loads(call.arguments)


def test_content_deltas_match_the_stripped_message():
    text = "  A final answer.  "
    response, events = parse_chunks(text, ["  A fi", "nal answer.  "], False)
    content = "".join(event.text for event in events if isinstance(event, ContentDelta))

    assert content == response.content == "A final answer."
    assert Qwen3ToolCallParser(ids()).parse(" \n\t").content is None


def test_every_single_split_preserves_a_tool_call():
    text = '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>'
    expected = Qwen3ToolCallParser(ids()).parse(text)

    for position in range(len(text) + 1):
        response, _ = parse_chunks(text, [text[:position], text[position:]], False)
        assert response == expected


def test_arguments_before_name_flush_without_losing_whitespace():
    text = '<tool_call>{"arguments":{"x":"tag <tool_call> text"},"name":"later"}</tool_call>'
    _, events = parse_chunks(text, list(text), False)
    start = next(index for index, event in enumerate(events) if isinstance(event, ToolCallStart))
    argument_text = "".join(
        event.text for event in events if isinstance(event, ToolCallArgumentsDelta)
    )

    assert argument_text == '{"x":"tag <tool_call> text"}'
    assert all(not isinstance(event, ToolCallEnd) for event in events[:start])


def test_duplicate_generated_ids_are_rejected():
    text = (
        '<tool_call>{"name":"a","arguments":{}}</tool_call>'
        '<tool_call>{"name":"b","arguments":{}}</tool_call>'
    )
    parser = Qwen3IncrementalParser(lambda: "duplicate", expect_thinking=False)

    with pytest.raises(ToolCallParseError) as caught:
        parser.feed(text)

    assert caught.value.code == "duplicate_tool_call_id"


def test_feed_after_finish_and_after_error_is_rejected():
    parser = Qwen3IncrementalParser(expect_thinking=False)
    parser.feed("done")
    parser.finish()
    with pytest.raises(RuntimeError, match="finished"):
        parser.feed("more")

    failed = Qwen3IncrementalParser(expect_thinking=False)
    with pytest.raises(ToolCallParseError):
        failed.feed("<tool_call>{bad}</tool_call>")
    with pytest.raises(RuntimeError, match="failed"):
        failed.feed("more")


def test_truncated_tool_call_prefixes_never_succeed():
    text = '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>'

    for length in range(len("<tool_call>"), len(text)):
        parser = Qwen3IncrementalParser(ids(), expect_thinking=False)
        with pytest.raises(ToolCallParseError):
            try:
                parser.feed(text[:length])
                parser.finish()
            except ToolCallParseError:
                raise


def test_known_content_is_emitted_before_finish():
    parser = Qwen3IncrementalParser(expect_thinking=False)

    events = parser.feed("plain text")

    assert [event.text for event in events if isinstance(event, ContentDelta)] == ["plain text"]

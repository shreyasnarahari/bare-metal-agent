import pytest

from toolsmith.parsing import Qwen3ToolCallParser, ToolCallParseError


def parser_with_ids(*ids):
    values = iter(ids)
    return Qwen3ToolCallParser(lambda: next(values))


def test_plain_text_response():
    parsed = Qwen3ToolCallParser().parse("  A final answer.  ")

    assert parsed.content == "A final answer."
    assert parsed.reasoning_content is None
    assert parsed.tool_calls == []


def test_thinking_and_multiple_tool_calls():
    parser = parser_with_ids("call_1", "call_2")
    parsed = parser.parse(
        "<think>Need weather and a search.</think>\n"
        "Before\n"
        '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>\n'
        '<tool_call>{"name":"search","arguments":{"query":"café"}}</tool_call>\n'
        "After"
    )

    assert parsed.reasoning_content == "Need weather and a search."
    assert parsed.content == "Before\n\n\nAfter"
    assert [call.to_openai() for call in parsed.tool_calls] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query":"café"}'},
        },
    ]


def test_qwen_thinking_response_without_opening_tag():
    parsed = parser_with_ids("call_1").parse(
        'Reasoning only</think><tool_call>{"name":"ping","arguments":{}}</tool_call>'
    )

    assert parsed.reasoning_content == "Reasoning only"
    assert parsed.content is None
    assert parsed.tool_calls[0].name == "ping"


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("<tool_call>{bad json}</tool_call>", "invalid_tool_call_json"),
        ('<tool_call>{"name":"x","arguments":{}}', "unclosed_tool_call"),
        ("</tool_call>", "unexpected_tool_call_close"),
        ("<think>unfinished", "unclosed_thinking_block"),
        ('<tool_call>["not", "object"]</tool_call>', "invalid_tool_call"),
        ('<tool_call>{"name":"","arguments":{}}</tool_call>', "invalid_tool_call"),
        ('<tool_call>{"name":"x","arguments":[]}</tool_call>', "invalid_tool_call"),
    ],
)
def test_malformed_outputs_have_specific_errors(text, code):
    with pytest.raises(ToolCallParseError) as caught:
        Qwen3ToolCallParser().parse(text)

    assert caught.value.code == code


@pytest.mark.parametrize(
    "arguments",
    [
        '{"city":"Paris","city":"London"}',
        '{"city":NaN}',
        '{"city":1e999}',
    ],
)
def test_qwen3_arguments_cannot_hide_non_strict_json_before_validation(arguments):
    text = (
        '<tool_call>{"name":"get_weather","arguments":'
        + arguments
        + "}</tool_call>"
    )

    with pytest.raises(ToolCallParseError) as caught:
        Qwen3ToolCallParser().parse(text)

    assert caught.value.code == "invalid_tool_call_json"

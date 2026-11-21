import pytest

from toolsmith.grammar import ToolChoice
from toolsmith.parsing import ParsedResponse, ParsedToolCall
from toolsmith.registry import Tool
from toolsmith.validation import ToolCallValidationError, ToolCallValidator


def tool():
    return Tool(
        "get_weather",
        "Weather",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"enum": ["c", "f"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    )


def response(*calls):
    return ParsedResponse(tool_calls=[ParsedToolCall(f"call_{index}", name, arguments) for index, (name, arguments) in enumerate(calls)])


def test_valid_arguments_and_auto_choice_pass():
    validator = ToolCallValidator([tool()], ToolChoice("auto"))

    validator.validate(response(("get_weather", '{"city":"Paris","unit":"c"}')))
    validator.validate(ParsedResponse(content="No tool needed"))


def test_nested_validation_error_has_json_pointer():
    validator = ToolCallValidator([tool()], ToolChoice("auto"))

    with pytest.raises(ToolCallValidationError) as caught:
        validator.validate(response(("get_weather", '{"city":3}')))

    assert caught.value.code == "schema_validation_failed"
    assert caught.value.details[0].startswith("/city:")


def test_unknown_tool_and_invalid_json_are_rejected():
    validator = ToolCallValidator([tool()], ToolChoice("auto"))

    with pytest.raises(ToolCallValidationError) as caught:
        validator.validate(response(("unknown", "{}")))
    assert caught.value.code == "unknown_tool"

    with pytest.raises(ToolCallValidationError) as caught:
        validator.validate(response(("get_weather", "{")))
    assert caught.value.code == "invalid_tool_call_json"


@pytest.mark.parametrize(
    ("choice", "parsed", "message"),
    [
        (ToolChoice("none"), response(("get_weather", '{"city":"Paris"}')), "none"),
        (ToolChoice("required"), ParsedResponse(content="text"), "no tool call"),
        (ToolChoice("named", "get_weather"), ParsedResponse(content="text"), "exactly one"),
        (
            ToolChoice("named", "get_weather"),
            response(
                ("get_weather", '{"city":"Paris"}'),
                ("get_weather", '{"city":"London"}'),
            ),
            "exactly one",
        ),
    ],
)
def test_tool_choice_postconditions_are_enforced(choice, parsed, message):
    with pytest.raises(ToolCallValidationError, match=message) as caught:
        ToolCallValidator([tool()], choice).validate(parsed)

    assert caught.value.code == "tool_choice_violation"

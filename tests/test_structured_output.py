import pytest

from toolsmith.repair import RepairPolicy, build_repair_messages
from toolsmith.validation import (
    StructuredOutputValidationError,
    StructuredOutputValidator,
    ToolCallValidationError,
    ToolCallValidator,
    strict_json_loads,
)
from toolsmith.grammar import ToolChoice
from toolsmith.parsing import ParsedResponse, ParsedToolCall
from toolsmith.registry import Tool


SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "integer"}},
        "nested": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    "required": ["items", "nested"],
    "additionalProperties": False,
}


def test_structured_output_returns_decoded_value():
    value = StructuredOutputValidator(SCHEMA).validate(
        '{"items":[1,-2],"nested":{"name":"Ada"}}'
    )
    assert value == {"items": [1, -2], "nested": {"name": "Ada"}}


@pytest.mark.parametrize("text", ['{"items":[1,],"nested":{"name":"Ada"}}', "NaN", "1e999", '{"a":1,"a":2}', "{} trailing"])
def test_strict_json_rejects_malformed_nonfinite_and_duplicate_values(text):
    with pytest.raises(StructuredOutputValidationError) as caught:
        StructuredOutputValidator({"type": "object"}).validate(text)
    assert caught.value.code == "invalid_structured_output_json"


def test_structured_schema_errors_have_stable_nested_json_pointers():
    with pytest.raises(StructuredOutputValidationError) as caught:
        StructuredOutputValidator(SCHEMA).validate(
            '{"items":["bad",2],"nested":{}}'
        )
    assert caught.value.code == "schema_validation_failed"
    assert caught.value.details == (
        "/items/0: 'bad' is not of type 'integer'",
        "/nested: 'name' is a required property",
    )


def test_tool_validation_mixed_object_and_array_paths_do_not_break_sorting():
    tool = Tool("create", "Create", SCHEMA)
    response = ParsedResponse(
        tool_calls=[
            ParsedToolCall(
                "call_1",
                "create",
                '{"items":["bad"],"nested":{}}',
            )
        ]
    )
    with pytest.raises(ToolCallValidationError) as caught:
        ToolCallValidator([tool], ToolChoice("required")).validate(response)
    assert caught.value.details[0].startswith("/items/0:")
    assert caught.value.details[1].startswith("/nested:")


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', "Infinity", "[1,]"])
def test_shared_strict_json_loader(text):
    with pytest.raises(ValueError):
        strict_json_loads(text)


def test_repair_messages_are_bounded_and_do_not_mutate_original():
    original = [{"role": "user", "content": "Call the tool"}]
    policy = RepairPolicy(max_output_chars=20, max_error_chars=30)

    messages = build_repair_messages(original, "x" * 100, "bad" * 100, policy)

    assert original == [{"role": "user", "content": "Call the tool"}]
    assert len(messages) == 2
    assert messages[-1]["role"] == "user"
    assert "[truncated by Toolsmith]" in messages[-1]["content"]
    assert "untrusted JSON string" in messages[-1]["content"]


def test_repair_policy_rejects_invalid_limits():
    with pytest.raises(ValueError):
        RepairPolicy(max_repairs=-1)
    with pytest.raises(ValueError):
        RepairPolicy(max_output_chars=0)

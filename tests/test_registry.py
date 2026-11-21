from typing import Literal

import pytest

from toolsmith.registry import Tool, ToolRegistry


def get_weather(city: str, unit: Literal["c", "f"] = "c") -> dict:
    """Get the current weather."""

    return {"city": city, "unit": unit}


def test_function_registration_generates_openai_schema():
    registry = ToolRegistry()
    returned = registry.function(get_weather)

    assert returned is get_weather
    assert len(registry) == 1
    definition = registry.to_openai()[0]
    function = definition["function"]
    assert definition["type"] == "function"
    assert function["name"] == "get_weather"
    assert function["description"] == "Get the current weather."
    assert function["parameters"]["required"] == ["city"]
    assert function["parameters"]["additionalProperties"] is False
    assert function["parameters"]["properties"]["city"]["type"] == "string"
    assert function["parameters"]["properties"]["unit"]["enum"] == ["c", "f"]
    assert function["parameters"]["properties"]["unit"]["default"] == "c"
    assert registry.get("get_weather").handler is get_weather


def test_explicit_schema_and_openai_round_trip():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def echo(value: int) -> int:
        return value

    tool = Tool.from_function(echo, name="echo_value", parameters=schema)
    restored = Tool.from_openai(tool.to_openai())

    assert restored.name == "echo_value"
    assert restored.parameters == schema
    assert restored.handler is None


def test_duplicate_tool_names_are_rejected():
    registry = ToolRegistry([Tool.from_function(get_weather)])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(Tool.from_function(get_weather))


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("not a name", {"type": "object"}),
        ("valid_name", {"type": "string"}),
        ("valid_name", {"type": "object", "properties": {"x": {"type": "unknown"}}}),
    ],
)
def test_invalid_tool_definitions_are_rejected(name, parameters):
    with pytest.raises((TypeError, ValueError)):
        Tool(name, "description", parameters)


def test_function_parameters_need_type_annotations():
    def untyped(value):
        return value

    with pytest.raises(TypeError, match="type annotation"):
        Tool.from_function(untyped)

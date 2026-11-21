import json
from pathlib import Path
from typing import Literal

import pytest
from jsonschema import Draft202012Validator

from toolsmith.grammar import (
    GBNFCompiler,
    GrammarConfigurationError,
    SchemaNotSupportedError,
    resolve_tool_choice,
)
from toolsmith.parsing import Qwen3ToolCallParser
from toolsmith.registry import Tool, ToolRegistry
from toolsmith.validation import ToolCallValidator

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "grammar_cases.json").read_text()
)


def weather_tool():
    return Tool.from_openai(
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": FIXTURES["supported"][0]["schema"],
            },
        }
    )


def search_tool():
    return Tool(
        "search_docs",
        "Search documents",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )


@pytest.mark.parametrize("case", FIXTURES["supported"], ids=lambda case: case["name"])
def test_supported_schemas_compile_deterministically(case):
    compiler = GBNFCompiler()

    first = compiler.compile_schema(case["schema"])
    second = compiler.compile_schema(case["schema"])

    assert first.text == second.text
    assert first.sha256 == second.sha256
    assert first.text.startswith("root ::= ")
    assert first.text.endswith("\n")
    for sample in case["valid"]:
        value = json.loads(sample["text"])
        Draft202012Validator(case["schema"]).validate(value)


@pytest.mark.parametrize("case", FIXTURES["unsupported"], ids=lambda case: case["name"])
def test_unsupported_schemas_report_keyword_and_pointer(case):
    with pytest.raises(SchemaNotSupportedError) as caught:
        GBNFCompiler().compile_schema(case["schema"])

    assert caught.value.keyword == case["keyword"]
    assert caught.value.pointer == case["pointer"]
    assert case["pointer"] in str(caught.value)


def test_registry_generated_optional_enum_schema_compiles():
    def get_weather(city: str, unit: Literal["c", "f"] = "c") -> dict:
        return {"city": city, "unit": unit}

    registry = ToolRegistry()
    registry.function(get_weather)

    grammar = GBNFCompiler().compile_tools(list(registry), "required")

    assert '\\"get_weather\\"' in grammar.text
    assert '\\"city\\"' in grammar.text
    assert '"\\"c\\"" | "\\"f\\""' in grammar.text
    assert "plain-text" not in grammar.text


def test_tool_choice_modes_generate_distinct_envelopes():
    tools = [weather_tool(), search_tool()]
    compiler = GBNFCompiler()

    auto = compiler.compile_tools(tools, "auto")
    none = compiler.compile_tools(tools, "none")
    required = compiler.compile_tools(tools, "required")
    named = compiler.compile_tools(
        tools,
        {"type": "function", "function": {"name": "get_weather"}},
    )

    assert auto.text.startswith("root ::= plain-text | tool-calls\n")
    assert none.text.startswith("root ::= plain-text\n")
    assert required.text.startswith("root ::= tool-calls\n")
    assert named.mode == "named"
    assert named.tool_names == ("get_weather",)
    assert '\\"get_weather\\"' in named.text
    assert '\\"search_docs\\"' not in named.text
    assert "plain-text" not in named.text
    assert len({auto.sha256, none.sha256, required.sha256, named.sha256}) == 4


def test_required_tool_grammar_output_parses_and_validates():
    tools = [weather_tool()]
    compiler = GBNFCompiler()
    grammar = compiler.compile_tools(tools, "required")
    text = (
        '<tool_call>{"name":"get_weather","arguments":{"city":"Paris","unit":"c"}}'
        "</tool_call>"
    )
    parsed = Qwen3ToolCallParser(lambda: "call_test").parse(text)
    choice = resolve_tool_choice(tools, "required")

    ToolCallValidator(tools, choice).validate(parsed)

    assert parsed.tool_calls[0].name == "get_weather"
    assert grammar.mode == "required"


def test_escaped_property_names_and_enum_strings_are_gbnf_literals():
    schema = {
        "type": "object",
        "properties": {
            'quote"slash\\line\n': {
                "enum": ['quote"', "slash\\", "line\n", "雪"]
            }
        },
        "required": ['quote"slash\\line\n'],
        "additionalProperties": False,
    }

    grammar = GBNFCompiler().compile_schema(schema)

    assert r'"\"quote\\\"slash\\\\line\\n\""' in grammar.text
    assert r'"\"雪\""' in grammar.text


def test_numeric_rules_use_json_syntax():
    compiler = GBNFCompiler()

    integer = compiler.compile_schema({"type": "integer"})
    number = compiler.compile_schema({"type": "number"})

    assert '"-"? ([0] | [1-9] [0-9]*)' in integer.text
    assert '("." [0-9]+)?' in number.text
    assert '([eE] [+-]? [0-9]+)?' in number.text


def test_no_tool_text_modes_do_not_need_a_grammar():
    compiler = GBNFCompiler()

    assert compiler.compile_tools([], None) is None
    assert compiler.compile_tools([], "auto") is None
    assert compiler.compile_tools([], "none") is None


@pytest.mark.parametrize(
    ("tools", "choice", "message"),
    [
        ([], "required", "at least one"),
        ([], {"type": "function", "function": {"name": "missing"}}, "at least one"),
        ([weather_tool()], "sometimes", "Unsupported tool_choice"),
        ([weather_tool()], {"type": "function", "function": {"name": "missing"}}, "Unknown tool"),
        ([weather_tool()], {"type": "function", "function": {}}, "only a name"),
        ([weather_tool()], {"type": "other", "function": {"name": "get_weather"}}, "named tool_choice"),
    ],
)
def test_invalid_tool_choices_fail_before_compilation(tools, choice, message):
    with pytest.raises(GrammarConfigurationError, match=message):
        GBNFCompiler().compile_tools(tools, choice)


def test_boolean_nonfinite_and_untyped_schemas_fail_loudly():
    compiler = GBNFCompiler()

    with pytest.raises(SchemaNotSupportedError, match="JSON object"):
        compiler.compile_schema(True)
    with pytest.raises(SchemaNotSupportedError, match="finite"):
        compiler.compile_schema({"enum": [float("nan")]})
    with pytest.raises(SchemaNotSupportedError, match="type or enum"):
        compiler.compile_schema({"description": "anything"})


def test_depth_rule_and_tool_limits_are_enforced():
    nested = current = {"type": "string"}
    for _ in range(3):
        current = {"type": "array", "items": current}
    nested = current

    with pytest.raises(SchemaNotSupportedError, match="depth"):
        GBNFCompiler(max_depth=2).compile_schema(nested)
    with pytest.raises(GrammarConfigurationError, match="rules"):
        GBNFCompiler(max_rules=2).compile_schema({"type": "string"})
    with pytest.raises(GrammarConfigurationError, match="tools"):
        GBNFCompiler(max_tools=1).compile_tools([weather_tool(), search_tool()])


def test_only_draft_2020_12_is_accepted():
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "string",
        "title": "Example",
        "description": "Annotations do not constrain values",
        "default": "ok",
        "examples": ["ok"],
    }

    assert GBNFCompiler().compile_schema(schema).mode == "schema"

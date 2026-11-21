import json

import pytest

from toolsmith.grammar import GBNFCompiler, SchemaNotSupportedError, ToolChoice
from toolsmith.qwen35 import Qwen35ToolCallParser
from toolsmith.registry import Tool
from toolsmith.sampling.automaton import PushdownAutomaton
from toolsmith.sampling.mask import TokenMaskEngine
from toolsmith.sampling.vocabulary import Vocabulary, VocabularyEntry
from toolsmith.validation import ToolCallValidator


def tools():
    return [
        Tool(
            "get_weather",
            "Weather",
            {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"enum": ["c", "f"]},
                    "details": {
                        "type": "object",
                        "properties": {
                            "days": {"type": "array", "items": {"type": "integer"}},
                            "active": {"type": "boolean"},
                        },
                        "required": ["days", "active"],
                        "additionalProperties": False,
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        ),
        Tool(
            "ping",
            "Ping",
            {"type": "object", "properties": {}, "additionalProperties": False},
        ),
    ]


def call(city="Paris", suffix=""):
    if suffix:
        suffix = "\n" + suffix
    return (
        "<tool_call>\n<function=get_weather>\n"
        f"<parameter=city>\n{city}\n</parameter>{suffix}\n"
        "</function>\n</tool_call>"
    )


def accepted(grammar, text):
    automaton = PushdownAutomaton.from_compiled(grammar)
    state = automaton.advance_bytes(automaton.initial_state(), text.encode("utf-8"))
    return state is not None and automaton.can_finish(state)


def test_required_xml_grammar_parses_validates_and_accepts_every_prefix():
    grammar = GBNFCompiler(protocol="qwen35").compile_tools(tools(), "required")
    text = call(
        "Zürich",
        "<parameter=unit>\nc\n</parameter>\n"
        "<parameter=details>\n{\"days\":[1,-2],\"active\":true}\n</parameter>",
    )
    parsed = Qwen35ToolCallParser(tools(), expect_thinking=False).parse(text)
    ToolCallValidator(tools(), ToolChoice("required")).validate(parsed)
    automaton = PushdownAutomaton.from_compiled(grammar)
    state = automaton.initial_state()

    for byte in text.encode("utf-8"):
        state = automaton.advance_bytes(state, bytes((byte,)))
        assert state is not None
    assert automaton.can_finish(state)
    assert json.loads(parsed.tool_calls[0].arguments)["details"]["days"] == [1, -2]
    assert grammar.protocol == "qwen35"
    assert grammar.limitations


@pytest.mark.parametrize(
    "text",
    [
        call(),
        call("", "<parameter=unit>\nf\n</parameter>"),
        call("A <tool_cat> B"),
        "<tool_call>\n<function=ping>\n</function>\n</tool_call>",
        call() + "\n<tool_call>\n<function=ping>\n</function>\n</tool_call>",
    ],
)
def test_valid_xml_outputs_accept(text):
    grammar = GBNFCompiler(protocol="qwen35").compile_tools(tools(), "required")
    assert accepted(grammar, text)


@pytest.mark.parametrize(
    "text",
    [
        call("Paris</parameter><parameter=city>London"),
        call("Paris</function>"),
        call("Paris<tool_call>"),
        call("Paris", "<parameter=unknown>\nx\n</parameter>"),
        call("Paris", "<parameter=unit>\nk\n</parameter>"),
        "<tool_call>\n<function=ping>\n<parameter=extra>\nx\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
    ],
)
def test_invalid_xml_outputs_reject(text):
    grammar = GBNFCompiler(protocol="qwen35").compile_tools(tools(), "required")
    assert not accepted(grammar, text)


def test_choice_modes_and_default_qwen3_protocol():
    qwen35 = GBNFCompiler(protocol="qwen35")
    qwen3 = GBNFCompiler()

    auto = qwen35.compile_tools(tools(), "auto")
    none = qwen35.compile_tools(tools(), "none")
    named = qwen35.compile_tools(
        tools(), {"type": "function", "function": {"name": "ping"}}
    )

    assert auto.text.startswith("root ::= plain-text | tool-calls")
    assert none.text.startswith("root ::= plain-text")
    assert named.text.startswith("root ::= tool-")
    assert "get_weather" not in named.text
    assert qwen3.compile_tools(tools(), "required").text.startswith("root ::= tool-calls")
    assert '<tool_call>' in qwen3.compile_tools(tools(), "required").text


@pytest.mark.parametrize(
    ("schema", "pointer"),
    [
        (
            {
                "type": "object",
                "properties": {"bad>name": {"type": "string"}},
                "additionalProperties": False,
            },
            "/properties/bad>name",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"enum": ["1", 1]}},
                "additionalProperties": False,
            },
            "/properties/value/enum/1",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"enum": ["</parameter>"]}},
                "additionalProperties": False,
            },
            "/properties/value/enum/0",
        ),
        (
            {"type": "object", "properties": {"city": {"type": "string"}}},
            "/additionalProperties",
        ),
    ],
)
def test_unrepresentable_xml_schemas_fail_with_pointer(schema, pointer):
    tool = Tool("bad", "Bad", schema)

    with pytest.raises(SchemaNotSupportedError) as caught:
        GBNFCompiler(protocol="qwen35").compile_tools([tool], "required")

    assert caught.value.pointer == pointer


def test_trie_mask_matches_brute_force_for_xml_prefixes():
    grammar = GBNFCompiler(protocol="qwen35").compile_tools(tools()[:1], "required")
    automaton = PushdownAutomaton.from_compiled(grammar)
    text = call("Par", "<parameter=unit>\nc\n</parameter>").encode()
    tokens = sorted(set(bytes((byte,)) for byte in text)) + [
        b"<tool_call>",
        b"</tool_call>",
        b"invalid",
        b"ar>",
    ]
    vocabulary = Vocabulary(
        [VocabularyEntry(index, raw) for index, raw in enumerate(tokens)],
        model_vocabulary_size=len(tokens) + 1,
        tokenizer_vocabulary_size=len(tokens),
        eos_token_ids=frozenset({len(tokens)}),
        special_token_ids=frozenset({len(tokens)}),
        excluded_empty_tokens=0,
        build_seconds=0,
    )
    engine = TokenMaskEngine(vocabulary, automaton)

    for position in range(0, len(text) + 1, 7):
        state = automaton.advance_bytes(automaton.initial_state(), text[:position])
        assert engine.mask(state).allowed == engine.brute_force_mask(state)

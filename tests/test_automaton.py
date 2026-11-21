import itertools
import json
import math
import random
from pathlib import Path

import pytest

from toolsmith.grammar import GBNFCompiler
from toolsmith.registry import Tool
from toolsmith.sampling.automaton import (
    AutomatonLimitError,
    PushdownAutomaton,
    UTF8State,
)

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "grammar_cases.json").read_text()
)
AUTOMATON_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "automaton_cases.json").read_text()
)


def advance_one_byte_at_a_time(automaton, text):
    state = automaton.initial_state()
    for byte in text.encode("utf-8"):
        state = automaton.advance_bytes(state, bytes((byte,)))
        assert state is not None
    return state


@pytest.mark.parametrize("case", FIXTURES["supported"], ids=lambda case: case["name"])
def test_valid_corpus_accepts_every_byte_prefix_and_invalid_corpus_rejects(case):
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema(case["schema"])
    )

    for sample in case["valid"]:
        state = advance_one_byte_at_a_time(automaton, sample["text"])
        assert automaton.can_finish(state), sample["name"]
        chunked = automaton.initial_state()
        data = sample["text"].encode("utf-8")
        for offset in range(0, len(data), 3):
            chunked = automaton.advance_bytes(chunked, data[offset : offset + 3])
            assert chunked is not None
        assert automaton.can_finish(chunked)
    for sample in case["invalid"]:
        state = automaton.advance_bytes(
            automaton.initial_state(), sample["text"].encode("utf-8")
        )
        assert state is None or not automaton.can_finish(state), sample["name"]


def test_random_byte_chunks_have_identical_acceptance():
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema(FIXTURES["supported"][1]["schema"])
    )
    text = FIXTURES["supported"][1]["valid"][1]["text"]
    data = text.encode("utf-8")
    randomizer = random.Random(19)

    for _ in range(30):
        state = automaton.initial_state()
        offset = 0
        while offset < len(data):
            size = randomizer.randint(1, 8)
            state = automaton.advance_bytes(state, data[offset : offset + size])
            assert state is not None
            offset += size
        assert automaton.can_finish(state)


def tools():
    return [
        Tool("get_weather", "Weather", FIXTURES["supported"][0]["schema"]),
        Tool(
            "search_docs",
            "Search",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
    ]


def qwen_call(name, arguments):
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
        + "\n</tool_call>"
    )


@pytest.mark.parametrize(
    ("mode", "text", "expected"),
    [
        ("auto", "plain café text", True),
        ("auto", "<tool_call>broken", False),
        ("none", "2 < 3 and 4 > 1", True),
        ("required", "plain text", False),
    ],
)
def test_tool_text_and_choice_roots(mode, text, expected):
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_tools(tools(), mode)
    )
    state = automaton.advance_bytes(automaton.initial_state(), text.encode())

    assert (state is not None and automaton.can_finish(state)) is expected


def test_tool_calls_parallel_named_and_protocol_text_inside_strings():
    compiler = GBNFCompiler()
    weather = qwen_call("get_weather", {"city": "</tool_call>"})
    search = qwen_call("search_docs", {"query": "grammars"})

    required = PushdownAutomaton.from_compiled(compiler.compile_tools(tools(), "required"))
    named = PushdownAutomaton.from_compiled(
        compiler.compile_tools(
            tools(), {"type": "function", "function": {"name": "get_weather"}}
        )
    )

    state = advance_one_byte_at_a_time(required, weather + "\n" + search)
    assert required.can_finish(state)
    state = advance_one_byte_at_a_time(named, weather)
    assert named.can_finish(state)
    state = named.advance_bytes(named.initial_state(), search.encode())
    assert state is None or not named.can_finish(state)
    state = named.advance_bytes(named.initial_state(), (weather + weather).encode())
    assert state is None or not named.can_finish(state)


@pytest.mark.parametrize("schema_type", ["integer", "number"])
def test_short_numeric_strings_match_independent_json_parser(schema_type):
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema({"type": schema_type})
    )
    alphabet = "01-+eE."

    for length in range(5):
        for characters in itertools.product(alphabet, repeat=length):
            text = "".join(characters)
            try:
                value = json.loads(text)
                expected = isinstance(value, int if schema_type == "integer" else (int, float))
                expected = expected and not isinstance(value, bool)
                if isinstance(value, float):
                    expected = expected and math.isfinite(value)
            except (json.JSONDecodeError, ValueError):
                expected = False
            state = automaton.advance_bytes(automaton.initial_state(), text.encode())
            actual = state is not None and automaton.can_finish(state)
            assert actual == expected, (schema_type, text)


def feed_utf8(data):
    state = UTF8State()
    codepoints = []
    for byte in data:
        decoded = state.feed(byte)
        if decoded is None:
            return None
        state, codepoint = decoded
        if codepoint is not None:
            codepoints.append(codepoint)
    return state, codepoints


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(bytes.fromhex(case["hex"]), id=case["name"])
        for case in AUTOMATON_FIXTURES["invalid_utf8"]
    ],
)
def test_invalid_utf8_is_rejected(data):
    assert feed_utf8(data) is None


@pytest.mark.parametrize(
    "case",
    AUTOMATON_FIXTURES["valid_utf8"],
    ids=lambda case: case["name"],
)
def test_valid_utf8_fixture_decodes_exact_codepoints(case):
    state, codepoints = feed_utf8(case["text"].encode("utf-8"))

    assert not state.pending
    assert "".join(chr(codepoint) for codepoint in codepoints) == case["text"]


def test_partial_utf8_can_cross_token_boundary_but_not_eos():
    state, codepoints = feed_utf8("A😀é".encode())
    assert not state.pending
    assert codepoints == [ord("A"), ord("😀"), ord("é")]

    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema({"type": "string"})
    )
    state = automaton.advance_bytes(automaton.initial_state(), b'"\xc3')
    assert state is not None and state.utf8.pending
    assert not automaton.can_finish(state)
    state = automaton.advance_bytes(state, b'\xa9"')
    assert state is not None and automaton.can_finish(state)


def test_plain_text_uses_tail_calls_and_long_strings_have_bounded_stacks():
    plain = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_tools(tools(), "none")
    )
    state = plain.advance_bytes(plain.initial_state(), b"x" * 10_000)
    assert state is not None and plain.can_finish(state)
    assert state.max_stack_depth <= 1

    string = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema({"type": "string"})
    )
    state = string.advance_bytes(string.initial_state(), b'"' + b"x" * 10_000)
    assert state is not None
    assert state.max_stack_depth <= 3


def test_nesting_stack_limit_is_explicit():
    schema = {"type": "string"}
    for _ in range(6):
        schema = {"type": "array", "items": schema}
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema(schema),
        max_stack_depth=3,
    )

    with pytest.raises(AutomatonLimitError, match="stack"):
        state = automaton.initial_state()
        for _ in range(6):
            state = automaton.advance_bytes(state, b"[")
            assert state is not None


def test_state_description_is_bounded_and_does_not_include_prompt():
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema({"type": "integer"})
    )
    state = automaton.advance_bytes(automaton.initial_state(), b"-")

    description = automaton.describe(state)

    assert "byte 1" in description
    assert "expected" in description
    assert len(description) < 500

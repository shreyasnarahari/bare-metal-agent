import json
from pathlib import Path

import pytest

from toolsmith.grammar import GBNFCompiler
from toolsmith.registry import Tool
from toolsmith.sampling.automaton import PushdownAutomaton
from toolsmith.sampling.mask import DeadEndError, MaskLimitError, TokenMaskEngine
from toolsmith.sampling.vocabulary import Vocabulary, VocabularyEntry

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "grammar_cases.json").read_text()
)


def vocabulary(raw_tokens, *, eos=900, size=1000):
    return Vocabulary(
        [VocabularyEntry(index, raw) for index, raw in enumerate(raw_tokens)],
        model_vocabulary_size=size,
        tokenizer_vocabulary_size=len(raw_tokens) + 1,
        eos_token_ids=frozenset({eos}),
        special_token_ids=frozenset({eos}),
        excluded_empty_tokens=0,
        build_seconds=0,
    )


def weather_automaton():
    tool = Tool("get_weather", "Weather", FIXTURES["supported"][0]["schema"])
    return PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_tools([tool], "required")
    )


def test_trie_mask_matches_brute_force_at_every_byte_prefix():
    call = (
        '<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris","unit":"c"}}'
        "\n</tool_call>"
    )
    tokens = [
        b"<",
        b"<tool_call>",
        b"<tool_call>\n",
        b"<tool_call>bad",
        b'{"name":',
        b'"get_weather"',
        b',"arguments":',
        b'{"city":"Paris","unit":"c"}',
        b"\n</tool_call>",
        b"bad",
        b" ",
        b"\n",
        b"{",
        b"}",
        b'"',
        b":",
        b",",
        b"city",
        b"Paris",
        b"unit",
        b"c",
        *(bytes((byte,)) for byte in sorted(set(call.encode()))),
    ]
    vocab = vocabulary(tokens)
    automaton = weather_automaton()
    engine = TokenMaskEngine(vocab, automaton)

    assert 3 not in engine.mask(automaton.initial_state()).allowed
    for offset in range(len(call.encode()) + 1):
        prefix = call.encode()[:offset]
        state = automaton.advance_bytes(automaton.initial_state(), prefix)
        assert state is not None
        result = engine.mask(state)
        assert result.allowed == engine.brute_force_mask(state), offset


def test_whole_token_can_cross_many_json_grammar_boundaries():
    complete = b'{"city":"Paris","unit":"c"}'
    vocab = vocabulary(
        [complete, b'{"city":3}', b'{"city":"Paris",', b'{"city":"Paris"}', b" "]
    )
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema(FIXTURES["supported"][0]["schema"])
    )
    engine = TokenMaskEngine(vocab, automaton)

    allowed = engine.mask(automaton.initial_state()).allowed

    assert 0 in allowed
    assert 1 not in allowed
    assert 2 in allowed
    assert 3 in allowed
    assert allowed == engine.brute_force_mask(automaton.initial_state())


def test_partial_utf8_token_and_eos_boundaries():
    vocab = vocabulary([b'"', b"\xc3", b"\xa9", b"\xc3\xa9", b"\xa9\"", b'x"'])
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema({"type": "string"})
    )
    engine = TokenMaskEngine(vocab, automaton)

    state = engine.advance_token(automaton.initial_state(), 0)
    allowed = engine.mask(state).allowed
    assert {1, 3, 5} <= allowed
    assert 900 not in allowed
    state = engine.advance_token(state, 1)
    assert state.utf8.pending
    allowed = engine.mask(state).allowed
    assert 2 in allowed
    assert 4 in allowed
    assert 900 not in allowed
    state = engine.advance_token(state, 2)
    state = engine.advance_token(state, 0)
    assert automaton.can_finish(state)
    assert 900 in engine.mask(state).allowed


def test_cache_reuses_language_state_not_byte_offset_and_evicts():
    vocab = vocabulary([b"0", b"1", b"-", b"01"])
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema({"type": "integer"})
    )
    engine = TokenMaskEngine(vocab, automaton, cache_size=1)

    first = automaton.initial_state()
    second = engine.advance_token(first, 1)
    third = engine.advance_token(second, 1)
    initial_result = engine.mask(first)
    second_result = engine.mask(second)
    third_result = engine.mask(third)

    assert third_result.cache_hit
    assert third_result.allowed == second_result.allowed
    assert engine.cache_info()["evictions"] == 1
    assert initial_result.allowed != second_result.allowed
    assert engine.mask(automaton.initial_state()).cache_hit is False


def test_cache_off_and_on_return_identical_results():
    vocab = vocabulary([b"0", b"1", b"-", b"01"])
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema({"type": "integer"})
    )
    state = automaton.advance_bytes(automaton.initial_state(), b"12")
    cached = TokenMaskEngine(vocab, automaton, cache_size=10)
    uncached = TokenMaskEngine(vocab, automaton, cache_size=0)

    assert cached.mask(state).allowed == uncached.mask(state).allowed


def test_special_padded_and_empty_tokens_are_never_allowed():
    vocab = vocabulary([b"0", b"1"])
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema({"type": "integer"})
    )
    engine = TokenMaskEngine(vocab, automaton)

    allowed = engine.mask(automaton.initial_state()).allowed

    assert 900 not in allowed
    assert all(token_id in {0, 1} for token_id in allowed)
    with pytest.raises(DeadEndError, match="not an ordinary"):
        engine.advance_token(automaton.initial_state(), 900)


def test_empty_mask_and_traversal_limit_are_explicit_errors():
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema({"type": "integer"})
    )
    empty_engine = TokenMaskEngine(vocabulary([b"x"]), automaton)
    with pytest.raises(DeadEndError, match="No token"):
        empty_engine.mask(automaton.initial_state())

    limited = TokenMaskEngine(
        vocabulary([b"0", b"1"]), automaton, max_visited_states=1
    )
    with pytest.raises(MaskLimitError, match="exceeded"):
        limited.mask(automaton.initial_state())

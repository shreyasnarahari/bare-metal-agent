from dataclasses import FrozenInstanceError

import pytest

from toolsmith.grammar import CompiledGrammar, GBNFCompiler
from toolsmith.sampling.automaton import InvalidGrammarError, PushdownAutomaton
from toolsmith.sampling.ir import (
    CharRange,
    CharSet,
    GrammarProgram,
    Literal,
    Repeat,
    RuleRef,
    sequence,
)


def test_compiled_grammar_contains_deterministic_immutable_ir():
    compiler = GBNFCompiler()

    first = compiler.compile_schema({"type": "integer"})
    second = compiler.compile_schema({"type": "integer"})

    assert isinstance(first.program, GrammarProgram)
    assert first.program == second.program
    assert first.program.render_gbnf() == first.text
    assert first.text == (
        "root ::= json-ws json-integer json-ws\n"
        "json-ws ::= [ \\t\\r\\n]{0,4}\n"
        'json-integer ::= "-"? ([0] | [1-9] [0-9]*)\n'
    )
    with pytest.raises(FrozenInstanceError):
        first.program.root = "other"


def test_ir_charset_matches_codepoints_and_negation():
    characters = CharSet((CharRange(ord("a"), ord("z")),), negated=True)

    assert characters.matches(ord("A"))
    assert not characters.matches(ord("q"))


def test_program_requires_unique_rules_and_root():
    with pytest.raises(ValueError, match="unique"):
        GrammarProgram((("root", Literal("a")), ("root", Literal("b"))))
    with pytest.raises(ValueError, match="Missing root"):
        GrammarProgram((("value", Literal("a")),))


def test_manual_gbnf_text_without_ir_is_not_silently_reparsed():
    grammar = CompiledGrammar('root ::= "a"\n', "schema")

    with pytest.raises(InvalidGrammarError, match="no IR"):
        PushdownAutomaton.from_compiled(grammar)


def test_nullable_repeat_validation_is_shared_with_automaton_compiler():
    program = GrammarProgram(
        (
            ("root", Repeat(RuleRef("maybe"))),
            ("maybe", sequence()),
        )
    )

    with pytest.raises(InvalidGrammarError, match="nullable"):
        PushdownAutomaton(program)

"""Toolsmith's token-mask engine behind MaskBench's engine interface.

MaskBench times each engine on the same schemas and token sequences: compile the
grammar (TTFM, time to first mask), then compute the mask and commit one token
per step (TBM, time between masks). This adapter wires in Toolsmith's grammar
compiler, automaton, and trie-walking mask engine, using the slow Llama
tokenizer MaskBench uses for every engine.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from ..grammar import GBNFCompiler
from ..sampling.automaton import PushdownAutomaton
from ..sampling.mask import TokenMaskEngine
from ..sampling.vocabulary import Vocabulary


class ToolsmithMaskEngine:
    """The methods MaskBench calls: init, compile_grammar, reset, compute_mask, commit_token."""

    def __init__(self, *, cache_size: int = 4096) -> None:
        self.multi = False
        self.debug = False
        self.cache_size = cache_size
        self.tokenizer = None
        self.tokenizer_model_id = ""

    # MaskBench assigns ``tokenizer`` first, then calls init().
    def init(self) -> None:
        if self.tokenizer is None:
            raise RuntimeError("Assign a slow Llama tokenizer before init()")
        self.vocabulary = Vocabulary.from_llama_tokenizer(self.tokenizer)

    def get_id(self) -> str:
        return "toolsmith"

    def get_name(self) -> str:
        return "Toolsmith PDA/trie"

    def get_module(self) -> str:
        return "toolsmith"

    def get_version(self) -> str:
        try:
            return importlib.metadata.version(self.get_module())
        except importlib.metadata.PackageNotFoundError:
            return "0.1.0"

    # Timed by MaskBench as TTFM: schema to grammar to automaton, plus a fresh mask engine.
    def compile_grammar(self, schema: dict[str, Any]) -> None:
        self.compiled = GBNFCompiler().compile_schema(schema)
        self.automaton = PushdownAutomaton.from_compiled(self.compiled)
        self.engine = TokenMaskEngine(
            self.vocabulary,
            self.automaton,
            cache_size=self.cache_size,
        )
        self.reset()

    def reset(self) -> None:
        self.state = self.automaton.initial_state()
        self.allowed = frozenset()
        self.eos_reached = False

    def compute_mask(self) -> frozenset[int]:
        self.allowed = self.engine.mask(self.state).allowed
        return self.allowed

    # False means the token was not allowed (MaskBench counts it against the engine).
    def commit_token(self, token: int) -> bool:
        if self.eos_reached or token not in self.allowed:
            return False
        if token in self.vocabulary.eos_token_ids:
            if not self.automaton.can_finish(self.state):
                return False
            self.eos_reached = True
            return True
        self.state = self.engine.advance_token(self.state, token)
        return True

    # Extra fields MaskBench stores with each result.
    def get_status_extras(self) -> dict[str, Any]:
        return {
            "grammar_sha256": self.compiled.sha256,
            "grammar_rules": self.compiled.rule_count,
            "vocabulary_fingerprint": self.vocabulary.fingerprint,
            "cache": self.engine.cache_info(),
        }

    def log_single(self, text: str) -> None:
        if not self.multi:
            print(text)


__all__ = ["ToolsmithMaskEngine"]

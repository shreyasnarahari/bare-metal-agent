from __future__ import annotations

import importlib.metadata
from typing import Any

from ..grammar import GBNFCompiler
from ..sampling.automaton import PushdownAutomaton
from ..sampling.mask import TokenMaskEngine
from ..sampling.vocabulary import Vocabulary


class ToolsmithMaskEngine:
    def __init__(self, *, cache_size: int = 4096) -> None:
        self.multi = False
        self.debug = False
        self.cache_size = cache_size
        self.tokenizer = None
        self.tokenizer_model_id = ""

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

"""Compute which tokens a grammar allows next (the "token mask").

At every decoding step the sampler needs the set of token ids whose bytes keep
the grammar alive. Checking all ~150k tokens one by one is slow, so
``TokenMaskEngine.mask`` walks the vocabulary trie and the automaton together:
each trie edge advances the automaton by one byte, and a branch is abandoned the
moment the automaton rejects it, which skips every token under that prefix.

Masks depend only on the automaton state, so they are cached (LRU). A state that
allows no token at all is a dead end and raises, because generation could only
continue by breaking the grammar. ``brute_force_mask`` is the slow, obviously
correct version the tests compare against.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, replace

from .automaton import AutomatonState, PushdownAutomaton
from .vocabulary import Vocabulary


class DeadEndError(RuntimeError):
    """No token can continue the grammar from this state (or the chosen token is not allowed)."""


class MaskLimitError(RuntimeError):
    """The trie walk visited more states than allowed (a guard against pathological grammars)."""


@dataclass(frozen=True)
class MaskResult:
    """The allowed token ids for one state, with traversal counts and timing for benchmarks."""

    allowed_token_ids: tuple[int, ...]
    visited_trie_nodes: int
    automaton_transitions: int
    elapsed_seconds: float
    cache_hit: bool = False

    @property
    def allowed(self) -> frozenset[int]:
        return frozenset(self.allowed_token_ids)


class TokenMaskEngine:
    """Masks for one vocabulary and one grammar automaton, with an LRU cache keyed by state."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        automaton: PushdownAutomaton,
        *,
        cache_size: int = 4096,
        max_visited_states: int = 1_000_000,
    ) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")
        if max_visited_states < 1:
            raise ValueError("max_visited_states must be positive")
        self.vocabulary = vocabulary
        self.automaton = automaton
        self.cache_size = cache_size
        self.max_visited_states = max_visited_states
        self._cache: OrderedDict[tuple[object, object], MaskResult] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

    @staticmethod
    def _key(state: AutomatonState) -> tuple[object, object]:
        # The byte offset is left out: two states with the same configurations allow the same tokens.
        return state.configurations, state.utf8

    def mask(self, state: AutomatonState) -> MaskResult:
        """The tokens allowed after ``state``. EOS is allowed only if the grammar can finish here."""
        started = time.perf_counter()
        key = self._key(state)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return replace(
                cached,
                cache_hit=True,
                elapsed_seconds=time.perf_counter() - started,
            )
        self.cache_misses += 1
        # Depth-first walk of the trie: (trie node, automaton state after that node's prefix).
        allowed: set[int] = set()
        pending = [(0, state)]
        visited = 0
        transitions = 0
        while pending:
            visited += 1
            if visited > self.max_visited_states:
                raise MaskLimitError(
                    f"Mask traversal exceeded {self.max_visited_states} states"
                )
            trie_node, current = pending.pop()
            node = self.vocabulary.trie.nodes[trie_node]
            allowed.update(node.token_ids)  # tokens ending here keep the grammar viable
            for byte, child in reversed(tuple(node.children.items())):
                transitions += 1
                advanced = self.automaton.advance_bytes(current, bytes((byte,)))
                if advanced is not None:  # otherwise the whole subtree is pruned
                    pending.append((child, advanced))
        if self.automaton.can_finish(state):
            allowed.update(self.vocabulary.eos_token_ids)
        result = MaskResult(
            tuple(sorted(allowed)),
            visited,
            transitions,
            time.perf_counter() - started,
        )
        if not result.allowed_token_ids:
            raise DeadEndError(
                "No token can extend the current grammar state: "
                + self.automaton.describe(state)
            )
        # Least recently used entries are evicted once the cache is full.
        if self.cache_size:
            self._cache[key] = result
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
                self.cache_evictions += 1
        return result

    def brute_force_mask(self, state: AutomatonState) -> frozenset[int]:
        """The same answer by trying every token separately: slow, used as the test oracle."""
        allowed = set()
        for entry in self.vocabulary.entries:
            if self.automaton.advance_bytes(state, entry.raw_bytes) is not None:
                allowed.add(entry.token_id)
        if self.automaton.can_finish(state):
            allowed.update(self.vocabulary.eos_token_ids)
        return frozenset(allowed)

    # The state after a sampled token; a token outside the mask is a dead end.
    def advance_token(
        self,
        state: AutomatonState,
        token_id: int,
    ) -> AutomatonState:
        try:
            raw = self.vocabulary.raw_bytes(token_id)
        except KeyError as exc:
            raise DeadEndError(f"Token {token_id} is not an ordinary vocabulary token") from exc
        advanced = self.automaton.advance_bytes(state, raw)
        if advanced is None:
            raise DeadEndError(
                f"Token {token_id} is not valid after {self.automaton.describe(state)}"
            )
        return advanced

    def cache_info(self) -> dict[str, int]:
        """Cache size and hit, miss, and eviction counts, for benchmark records."""
        return {
            "size": len(self._cache),
            "capacity": self.cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "evictions": self.cache_evictions,
        }

    def clear_cache(self) -> None:
        self._cache.clear()


__all__ = ["DeadEndError", "MaskLimitError", "MaskResult", "TokenMaskEngine"]

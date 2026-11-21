from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, replace

from .automaton import AutomatonState, PushdownAutomaton
from .vocabulary import Vocabulary


class DeadEndError(RuntimeError):
    pass


class MaskLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaskResult:
    allowed_token_ids: tuple[int, ...]
    visited_trie_nodes: int
    automaton_transitions: int
    elapsed_seconds: float
    cache_hit: bool = False

    @property
    def allowed(self) -> frozenset[int]:
        return frozenset(self.allowed_token_ids)


class TokenMaskEngine:
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
        return state.configurations, state.utf8

    def mask(self, state: AutomatonState) -> MaskResult:
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
            allowed.update(node.token_ids)
            for byte, child in reversed(tuple(node.children.items())):
                transitions += 1
                advanced = self.automaton.advance_bytes(current, bytes((byte,)))
                if advanced is not None:
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
        if self.cache_size:
            self._cache[key] = result
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
                self.cache_evictions += 1
        return result

    def brute_force_mask(self, state: AutomatonState) -> frozenset[int]:
        allowed = set()
        for entry in self.vocabulary.entries:
            if self.automaton.advance_bytes(state, entry.raw_bytes) is not None:
                allowed.add(entry.token_id)
        if self.automaton.can_finish(state):
            allowed.update(self.vocabulary.eos_token_ids)
        return frozenset(allowed)

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

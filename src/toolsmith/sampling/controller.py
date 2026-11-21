from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from .automaton import AutomatonState, PushdownAutomaton
from .mask import DeadEndError, TokenMaskEngine
from .vocabulary import Vocabulary


class UnsupportedGenerationError(ValueError):
    pass


class IncompleteConstrainedGeneration(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        token_ids: tuple[int, ...],
        raw_bytes: bytes,
        state: str,
    ) -> None:
        super().__init__(message)
        self.token_ids = token_ids
        self.raw_bytes = raw_bytes
        self.state = state


@dataclass(frozen=True)
class MaskDecision:
    allowed_token_ids: tuple[int, ...] | None
    blocked_token_ids: tuple[int, ...]
    elapsed_seconds: float
    active: bool
    cache_hit: bool = False
    visited_trie_nodes: int = 0
    candidate_tokens: int = 0


class BatchOneSynchronizer:
    def __init__(self) -> None:
        self.previous_length: int | None = None
        self.prompt_length: int | None = None

    def synchronize(self, input_ids: object) -> int | None:
        shape = getattr(input_ids, "shape", None)
        if shape is None or len(shape) != 2 or int(shape[0]) != 1:
            raise UnsupportedGenerationError(
                "Constrained generation requires a single batch row and no beam expansion"
            )
        row = input_ids[0]
        length = int(shape[1])
        if self.previous_length is None:
            self.previous_length = length
            self.prompt_length = length
            return None
        if length != self.previous_length + 1:
            raise UnsupportedGenerationError(
                "Expected exactly one appended token; beam search, rewind, and speculative decoding are unsupported"
            )
        self.previous_length = length
        return int(row[-1])


class _BaseController:
    def __init__(self, vocabulary: Vocabulary) -> None:
        self.vocabulary = vocabulary
        self.generated_token_ids: list[int] = []
        self.eos_reached = False
        self.mask_seconds: list[float] = []
        self.mask_calls = 0

    @property
    def generated_ids(self) -> tuple[int, ...]:
        return tuple(self.generated_token_ids)

    def generated_bytes(self) -> bytes:
        return self.vocabulary.reconstruct_bytes(self.generated_token_ids)

    def can_finish(self) -> bool:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def finalize(self, stop_reason: str) -> None:
        if not self.can_finish():
            raise IncompleteConstrainedGeneration(
                f"Generation ended with an incomplete constrained response ({stop_reason})",
                token_ids=self.generated_ids,
                raw_bytes=self.generated_bytes(),
                state=self.describe(),
            )


class FixedMaskController(_BaseController):
    def __init__(
        self,
        vocabulary: Vocabulary,
        automaton: PushdownAutomaton,
        *,
        cache_size: int = 4096,
    ) -> None:
        super().__init__(vocabulary)
        self.automaton = automaton
        self.engine = TokenMaskEngine(
            vocabulary,
            automaton,
            cache_size=cache_size,
        )
        self.state = automaton.initial_state()

    @property
    def active(self) -> bool:
        return True

    def mask(self) -> MaskDecision:
        result = self.engine.mask(self.state)
        self.mask_calls += 1
        self.mask_seconds.append(result.elapsed_seconds)
        return MaskDecision(
            result.allowed_token_ids,
            (),
            result.elapsed_seconds,
            True,
            result.cache_hit,
            result.visited_trie_nodes,
        )

    def commit(self, token_id: int) -> None:
        if token_id in self.vocabulary.eos_token_ids:
            if not self.automaton.can_finish(self.state):
                raise DeadEndError("EOS is not valid before the grammar can finish")
            self.eos_reached = True
        else:
            self.state = self.engine.advance_token(self.state, token_id)
        self.generated_token_ids.append(token_id)

    def can_finish(self) -> bool:
        return self.automaton.can_finish(self.state)

    def describe(self) -> str:
        return self.automaton.describe(self.state)


@dataclass(frozen=True)
class _FreeState:
    suffix: bytes = b""


@dataclass(frozen=True)
class _TriggeredState:
    free: _FreeState | None
    active: AutomatonState | None


class TriggeredMaskController(_BaseController):
    def __init__(
        self,
        vocabulary: Vocabulary,
        automaton: PushdownAutomaton,
        *,
        trigger: bytes = b"<tool_call>",
        cache_size: int = 4096,
    ) -> None:
        super().__init__(vocabulary)
        if not trigger or len(trigger) > 256:
            raise ValueError("trigger must contain between 1 and 256 bytes")
        self.automaton = automaton
        self.trigger = trigger
        self.engine = TokenMaskEngine(
            vocabulary,
            automaton,
            cache_size=cache_size,
        )
        self.initial_state = automaton.initial_state()
        self.state = self.initial_state
        self.free_state = _FreeState()
        self.activation_step: int | None = None
        self.cache_size = cache_size
        self._free_cache: OrderedDict[bytes, tuple[int, ...]] = OrderedDict()
        self.free_cache_hits = 0
        self.free_cache_misses = 0
        self._unusable = tuple(
            token_id
            for token_id in range(vocabulary.model_vocabulary_size)
            if vocabulary.raw_tokens[token_id] is None
            and token_id not in vocabulary.eos_token_ids
        )
        self._contains_trigger_start = frozenset(
            entry.token_id
            for entry in vocabulary.entries
            if trigger[:1] in entry.raw_bytes
        )

    @property
    def active(self) -> bool:
        return self.free_state is None

    def _longest_suffix(self, data: bytes) -> bytes:
        maximum = min(len(self.trigger) - 1, len(data))
        for size in range(maximum, 0, -1):
            candidate = self.trigger[:size]
            if data.endswith(candidate):
                return candidate
        return b""

    def _advance_free(
        self,
        free: _FreeState,
        raw: bytes,
    ) -> _TriggeredState | None:
        suffix = free.suffix
        for index, byte in enumerate(raw):
            combined = suffix + bytes((byte,))
            if combined == self.trigger:
                active = self.automaton.advance_bytes(self.initial_state, self.trigger)
                if active is None:
                    return None
                if index + 1 < len(raw):
                    active = self.automaton.advance_bytes(active, raw[index + 1 :])
                    if active is None:
                        return None
                return _TriggeredState(None, active)
            suffix = self._longest_suffix(combined)
        return _TriggeredState(_FreeState(suffix), None)

    def _advance(self, raw: bytes) -> _TriggeredState | None:
        if self.free_state is not None:
            return self._advance_free(self.free_state, raw)
        active = self.automaton.advance_bytes(self.state, raw)
        if active is None:
            return None
        return _TriggeredState(None, active)

    def _relevant(self, token_id: int, raw: bytes) -> bool:
        if token_id in self._contains_trigger_start:
            return True
        suffix = self.free_state.suffix if self.free_state is not None else b""
        return bool(suffix) and len(suffix) < len(self.trigger) and raw.startswith(
            self.trigger[len(suffix) : len(suffix) + 1]
        )

    def _free_blocked(self) -> tuple[tuple[int, ...], int, bool]:
        suffix = self.free_state.suffix if self.free_state is not None else b""
        cached = self._free_cache.get(suffix)
        if cached is not None:
            self.free_cache_hits += 1
            self._free_cache.move_to_end(suffix)
            return cached, 0, True
        self.free_cache_misses += 1
        blocked = set(self._unusable)
        candidates = 0
        for entry in self.vocabulary.entries:
            if not self._relevant(entry.token_id, entry.raw_bytes):
                continue
            candidates += 1
            if self._advance_free(_FreeState(suffix), entry.raw_bytes) is None:
                blocked.add(entry.token_id)
        result = tuple(sorted(blocked))
        if self.cache_size:
            self._free_cache[suffix] = result
            if len(self._free_cache) > self.cache_size:
                self._free_cache.popitem(last=False)
        return result, candidates, False

    def mask(self) -> MaskDecision:
        started = time.perf_counter()
        self.mask_calls += 1
        if self.free_state is None:
            result = self.engine.mask(self.state)
            decision = MaskDecision(
                result.allowed_token_ids,
                (),
                result.elapsed_seconds,
                True,
                result.cache_hit,
                result.visited_trie_nodes,
            )
        else:
            blocked, candidates, cache_hit = self._free_blocked()
            decision = MaskDecision(
                None,
                blocked,
                time.perf_counter() - started,
                False,
                cache_hit,
                0,
                candidates,
            )
        self.mask_seconds.append(decision.elapsed_seconds)
        return decision

    def commit(self, token_id: int) -> None:
        if token_id in self.vocabulary.eos_token_ids:
            if not self.can_finish():
                raise DeadEndError("EOS is not valid before the triggered call is complete")
            self.eos_reached = True
        else:
            was_free = self.free_state is not None
            advanced = self._advance(self.vocabulary.raw_bytes(token_id))
            if advanced is None:
                raise DeadEndError(
                    f"Token {token_id} is invalid at trigger/grammar position: {self.describe()}"
                )
            if was_free and advanced.free is None:
                self.activation_step = len(self.generated_token_ids)
            self.free_state = advanced.free
            if advanced.active is not None:
                self.state = advanced.active
        self.generated_token_ids.append(token_id)

    def can_finish(self) -> bool:
        return self.free_state is not None or self.automaton.can_finish(self.state)

    def describe(self) -> str:
        if self.free_state is not None:
            return f"free text, trigger suffix {self.free_state.suffix!r}"
        return "trigger active; " + self.automaton.describe(self.state)

    def cache_info(self) -> dict[str, int]:
        return {
            **self.engine.cache_info(),
            "free_size": len(self._free_cache),
            "free_hits": self.free_cache_hits,
            "free_misses": self.free_cache_misses,
        }


__all__ = [
    "BatchOneSynchronizer",
    "FixedMaskController",
    "IncompleteConstrainedGeneration",
    "MaskDecision",
    "TriggeredMaskController",
    "UnsupportedGenerationError",
]

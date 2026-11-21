from __future__ import annotations

import hashlib
import re
import sys
import time
from dataclasses import dataclass
from typing import Any


class UnsupportedTokenizerError(ValueError):
    pass


def qwen_byte_decoder() -> dict[str, int]:
    """Inverse of the Qwen and GPT-2 byte-to-unicode map."""
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = values[:]
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            characters.append(256 + extra)
            extra += 1
    return {chr(character): byte for byte, character in zip(values, characters)}


@dataclass(frozen=True)
class VocabularyEntry:
    token_id: int
    raw_bytes: bytes


@dataclass(slots=True)
class TrieNode:
    children: dict[int, int]
    token_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class _TrieBuilderNode:
    children: dict[int, int]
    token_ids: list[int]


class ByteTrie:
    def __init__(self, entries: list[VocabularyEntry]) -> None:
        builders = [_TrieBuilderNode({}, [])]
        for entry in sorted(entries, key=lambda item: (item.raw_bytes, item.token_id)):
            if not entry.raw_bytes:
                continue
            node_index = 0
            for byte in entry.raw_bytes:
                node = builders[node_index]
                child = node.children.get(byte)
                if child is None:
                    child = len(builders)
                    builders.append(_TrieBuilderNode({}, []))
                    node.children[byte] = child
                node_index = child
            builders[node_index].token_ids.append(entry.token_id)
        self.nodes = tuple(
            TrieNode(dict(sorted(node.children.items())), tuple(sorted(node.token_ids)))
            for node in builders
        )

    def child(self, node: int, byte: int) -> int | None:
        return self.nodes[node].children.get(byte)


@dataclass(frozen=True)
class VocabularyStats:
    model_vocabulary_size: int
    tokenizer_vocabulary_size: int
    usable_tokens: int
    trie_nodes: int
    duplicate_byte_tokens: int
    excluded_special_tokens: int
    excluded_empty_tokens: int
    padded_token_ids: int
    approximate_bytes: int
    build_seconds: float

    def as_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


class Vocabulary:
    def __init__(
        self,
        entries: list[VocabularyEntry],
        *,
        model_vocabulary_size: int,
        tokenizer_vocabulary_size: int,
        eos_token_ids: frozenset[int],
        special_token_ids: frozenset[int],
        excluded_empty_tokens: int,
        build_seconds: float,
    ) -> None:
        self.model_vocabulary_size = model_vocabulary_size
        self.eos_token_ids = eos_token_ids
        self.special_token_ids = special_token_ids
        self.entries = tuple(sorted(entries, key=lambda entry: entry.token_id))
        self.raw_tokens: tuple[bytes | None, ...] = tuple(
            _raw_tokens(entries, model_vocabulary_size)
        )
        self.trie = ByteTrie(list(self.entries))
        fingerprint = hashlib.sha256()
        for token_id, raw in enumerate(self.raw_tokens):
            fingerprint.update(token_id.to_bytes(4, "big", signed=False))
            if raw is None:
                fingerprint.update(b"\xff")
            else:
                fingerprint.update(len(raw).to_bytes(4, "big"))
                fingerprint.update(raw)
        self.fingerprint = fingerprint.hexdigest()
        duplicate_tokens = sum(
            max(0, len(node.token_ids) - 1) for node in self.trie.nodes
        )
        self.stats = VocabularyStats(
            model_vocabulary_size,
            tokenizer_vocabulary_size,
            len(self.entries),
            len(self.trie.nodes),
            duplicate_tokens,
            len(special_token_ids),
            excluded_empty_tokens,
            model_vocabulary_size - tokenizer_vocabulary_size,
            self._approximate_bytes(),
            build_seconds,
        )

    def _approximate_bytes(self) -> int:
        total = sys.getsizeof(self.raw_tokens) + sys.getsizeof(self.entries)
        total += sum(sys.getsizeof(raw) for raw in self.raw_tokens if raw is not None)
        for node in self.trie.nodes:
            total += sys.getsizeof(node) + sys.getsizeof(node.children)
            total += sys.getsizeof(node.token_ids)
            total += sum(sys.getsizeof(byte) + sys.getsizeof(child) for byte, child in node.children.items())
        return total

    def raw_bytes(self, token_id: int) -> bytes:
        if not 0 <= token_id < len(self.raw_tokens):
            raise KeyError(f"Token ID outside model vocabulary: {token_id}")
        raw = self.raw_tokens[token_id]
        if raw is None:
            raise KeyError(f"Token ID has no ordinary byte representation: {token_id}")
        return raw

    def reconstruct_bytes(self, token_ids: list[int] | tuple[int, ...]) -> bytes:
        parts = []
        for token_id in token_ids:
            if token_id in self.special_token_ids or token_id in self.eos_token_ids:
                continue
            parts.append(self.raw_bytes(token_id))
        return b"".join(parts)

    @classmethod
    def from_qwen_tokenizer(
        cls,
        tokenizer: Any,
        *,
        model_vocabulary_size: int | None = None,
        eos_token_ids: frozenset[int] | set[int] | None = None,
    ) -> Vocabulary:
        started = time.perf_counter()
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if get_vocab is None:
            raise UnsupportedTokenizerError(
                "The tokenizer must expose a Qwen byte-level vocabulary"
            )
        vocabulary = get_vocab()
        if not isinstance(vocabulary, dict):
            raise UnsupportedTokenizerError("Tokenizer get_vocab() did not return a mapping")
        byte_decoder = getattr(tokenizer, "byte_decoder", None)
        if not isinstance(byte_decoder, dict):
            byte_decoder = qwen_byte_decoder()
        added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
        if eos_token_ids is None:
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is None:
                eos_ids = set()
            elif isinstance(eos, (list, tuple, set, frozenset)):
                eos_ids = set(eos)
            else:
                eos_ids = {eos}
        else:
            eos_ids = set(eos_token_ids)
        special_ids = set(eos_ids)
        for token_id, token in added.items():
            if bool(getattr(token, "special", False)):
                special_ids.add(int(token_id))

        maximum_id = max(vocabulary.values(), default=-1)
        size = max(maximum_id + 1, model_vocabulary_size or 0)
        if size < maximum_id + 1:
            raise ValueError("model_vocabulary_size is smaller than the tokenizer vocabulary")
        entries = []
        excluded_empty = 0
        for token, token_id in sorted(vocabulary.items(), key=lambda item: item[1]):
            token_id = int(token_id)
            if token_id in special_ids:
                continue
            added_token = added.get(token_id)
            if added_token is not None:
                content = getattr(added_token, "content", token)
                raw = str(content).encode("utf-8")
            else:
                try:
                    raw = bytes(byte_decoder[character] for character in token)
                except KeyError as exc:
                    raise UnsupportedTokenizerError(
                        f"Could not recover raw bytes for token {token_id}: {token!r}"
                    ) from exc
            if not raw:
                excluded_empty += 1
                continue
            entries.append(VocabularyEntry(token_id, raw))
        return cls(
            entries,
            model_vocabulary_size=size,
            tokenizer_vocabulary_size=len(vocabulary),
            eos_token_ids=frozenset(eos_ids),
            special_token_ids=frozenset(special_ids),
            excluded_empty_tokens=excluded_empty,
            build_seconds=time.perf_counter() - started,
        )

    @classmethod
    def from_llama_tokenizer(
        cls,
        tokenizer: Any,
        *,
        model_vocabulary_size: int | None = None,
        eos_token_ids: frozenset[int] | set[int] | None = None,
    ) -> Vocabulary:
        started = time.perf_counter()
        if getattr(tokenizer, "is_fast", False):
            raise UnsupportedTokenizerError(
                "Use the slow Llama tokenizer for raw-byte extraction"
            )
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if get_vocab is None:
            raise UnsupportedTokenizerError("Llama tokenizer get_vocab() is unavailable")
        vocabulary = get_vocab()
        if not isinstance(vocabulary, dict):
            raise UnsupportedTokenizerError("Llama tokenizer vocabulary is not a mapping")
        added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
        if eos_token_ids is None:
            eos = getattr(tokenizer, "eos_token_id", None)
            eos_ids = set(eos) if isinstance(eos, (list, tuple, set, frozenset)) else {eos}
        else:
            eos_ids = set(eos_token_ids)
        eos_ids.discard(None)
        special_ids = set(eos_ids)
        special_ids.update(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []))
        for token_id, token in added.items():
            if bool(getattr(token, "special", False)):
                special_ids.add(int(token_id))

        maximum_id = max(vocabulary.values(), default=-1)
        size = max(maximum_id + 1, model_vocabulary_size or 0)
        if size < maximum_id + 1:
            raise ValueError("model_vocabulary_size is smaller than the tokenizer vocabulary")
        entries = []
        excluded_empty = 0
        for token, token_id in sorted(vocabulary.items(), key=lambda item: item[1]):
            token_id = int(token_id)
            if token_id in special_ids:
                continue
            added_token = added.get(token_id)
            if added_token is not None:
                content = getattr(added_token, "content", token)
                raw = str(content).encode("utf-8")
            else:
                fallback = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", token)
                if fallback:
                    raw = bytes((int(fallback.group(1), 16),))
                else:
                    raw = token.replace("▁", " ").encode("utf-8")
            if not raw:
                excluded_empty += 1
                continue
            entries.append(VocabularyEntry(token_id, raw))
        return cls(
            entries,
            model_vocabulary_size=size,
            tokenizer_vocabulary_size=len(vocabulary),
            eos_token_ids=frozenset(eos_ids),
            special_token_ids=frozenset(special_ids),
            excluded_empty_tokens=excluded_empty,
            build_seconds=time.perf_counter() - started,
        )


def _raw_tokens(
    entries: list[VocabularyEntry],
    model_vocabulary_size: int,
) -> list[bytes | None]:
    raw_tokens: list[bytes | None] = [None] * model_vocabulary_size
    for entry in entries:
        if not 0 <= entry.token_id < model_vocabulary_size:
            raise ValueError(f"Token ID outside model vocabulary: {entry.token_id}")
        if raw_tokens[entry.token_id] is not None:
            raise ValueError(f"Duplicate token ID: {entry.token_id}")
        raw_tokens[entry.token_id] = entry.raw_bytes
    return raw_tokens


__all__ = [
    "ByteTrie",
    "UnsupportedTokenizerError",
    "Vocabulary",
    "VocabularyEntry",
    "VocabularyStats",
]

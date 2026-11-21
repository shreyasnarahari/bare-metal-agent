from dataclasses import dataclass

import pytest

from toolsmith.sampling.vocabulary import (
    ByteTrie,
    UnsupportedTokenizerError,
    Vocabulary,
    VocabularyEntry,
)


def bytes_to_unicode():
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
    return dict(zip(values, (chr(value) for value in characters)))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {character: byte for byte, character in BYTE_ENCODER.items()}


def encoded(raw):
    return "".join(BYTE_ENCODER[byte] for byte in raw)


@dataclass
class AddedToken:
    content: str
    special: bool = False


class FakeQwenTokenizer:
    is_fast = False
    eos_token_id = 90

    def __init__(self):
        self.encoder = {
            encoded(b"a"): 0,
            encoded(b" "): 1,
            encoded(bytes([0xC3])): 2,
            encoded(bytes([0xA9])): 3,
            encoded("é".encode()): 4,
            encoded(b""): 5,
        }
        self.byte_decoder = BYTE_DECODER
        self.added_tokens_decoder = {
            80: AddedToken("<tool_call>"),
            81: AddedToken("雪"),
            90: AddedToken("<|im_end|>", True),
        }

    def get_vocab(self):
        return {
            **self.encoder,
            **{token.content: token_id for token_id, token in self.added_tokens_decoder.items()},
        }


def test_qwen_vocab_recovers_exact_raw_bytes_and_added_tokens():
    vocabulary = Vocabulary.from_qwen_tokenizer(
        FakeQwenTokenizer(), model_vocabulary_size=100
    )

    assert vocabulary.raw_bytes(0) == b"a"
    assert vocabulary.raw_bytes(1) == b" "
    assert vocabulary.raw_bytes(2) == bytes([0xC3])
    assert vocabulary.raw_bytes(3) == bytes([0xA9])
    assert vocabulary.raw_bytes(4) == "é".encode()
    assert vocabulary.raw_bytes(80) == b"<tool_call>"
    assert vocabulary.raw_bytes(81) == "雪".encode()
    assert vocabulary.raw_tokens[90] is None
    assert vocabulary.raw_tokens[99] is None
    assert vocabulary.stats.excluded_empty_tokens == 1
    assert vocabulary.stats.excluded_special_tokens == 1
    assert vocabulary.stats.usable_tokens == 7
    assert vocabulary.stats.trie_nodes > 1
    assert vocabulary.reconstruct_bytes([0, 90, 4]) == b"a" + "é".encode()


def test_trie_preserves_duplicate_byte_sequences():
    trie = ByteTrie(
        [
            VocabularyEntry(3, b"same"),
            VocabularyEntry(1, b"same"),
            VocabularyEntry(2, b"sand"),
        ]
    )

    node = 0
    for byte in b"same":
        node = trie.child(node, byte)
    assert trie.nodes[node].token_ids == (1, 3)


def test_unknown_special_empty_and_padded_ids_have_no_raw_bytes():
    vocabulary = Vocabulary.from_qwen_tokenizer(FakeQwenTokenizer())

    for token_id in (5, 90, 99):
        with pytest.raises(KeyError):
            vocabulary.raw_bytes(token_id)


def test_fast_and_non_qwen_tokenizers_fail_loudly():
    tokenizer = FakeQwenTokenizer()
    tokenizer.is_fast = True
    with pytest.raises(UnsupportedTokenizerError, match="slow"):
        Vocabulary.from_qwen_tokenizer(tokenizer)

    class MissingCodec:
        is_fast = False

        def get_vocab(self):
            return {"a": 0}

    with pytest.raises(UnsupportedTokenizerError, match="byte-level"):
        Vocabulary.from_qwen_tokenizer(MissingCodec())


def test_unencodable_normal_token_is_not_silently_dropped():
    tokenizer = FakeQwenTokenizer()
    tokenizer.encoder["😀"] = 6

    with pytest.raises(UnsupportedTokenizerError, match="raw bytes"):
        Vocabulary.from_qwen_tokenizer(tokenizer)


def test_duplicate_token_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate token ID"):
        Vocabulary(
            [VocabularyEntry(1, b"a"), VocabularyEntry(1, b"b")],
            model_vocabulary_size=2,
            tokenizer_vocabulary_size=2,
            eos_token_ids=frozenset(),
            special_token_ids=frozenset(),
            excluded_empty_tokens=0,
            build_seconds=0,
        )

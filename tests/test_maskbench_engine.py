from types import SimpleNamespace

from toolsmith.evaluation.maskbench_engine import ToolsmithMaskEngine
from toolsmith.evaluation.maskbench_runner import timing_summary
from toolsmith.sampling.vocabulary import UnsupportedTokenizerError, Vocabulary

import pytest


class Added:
    def __init__(self, content, special):
        self.content = content
        self.special = special


class FakeLlamaTokenizer:
    is_fast = False
    eos_token_id = 6
    all_special_ids = [6]
    added_tokens_decoder = {
        6: Added("<|endoftext|>", True),
        7: Added("<tool_call>", False),
    }

    def get_vocab(self):
        return {
            "1": 0,
            "2": 1,
            "x": 2,
            "▁": 3,
            "<0xC3>": 4,
            "<0xA9>": 5,
            "<|endoftext|>": 6,
            "<tool_call>": 7,
        }


def test_llama_codec_recovers_bytes_specials_and_padding():
    vocabulary = Vocabulary.from_llama_tokenizer(
        FakeLlamaTokenizer(), model_vocabulary_size=9
    )

    assert vocabulary.raw_bytes(3) == b" "
    assert vocabulary.reconstruct_bytes([4, 5]) == "é".encode()
    assert vocabulary.raw_bytes(7) == b"<tool_call>"
    assert vocabulary.raw_tokens[6] is None
    assert vocabulary.raw_tokens[8] is None
    assert vocabulary.stats.padded_token_ids == 1
    with pytest.raises(UnsupportedTokenizerError):
        Vocabulary.from_llama_tokenizer(SimpleNamespace(is_fast=True))


def test_mask_commit_reset_and_disallowed_token_does_not_advance():
    engine = ToolsmithMaskEngine(cache_size=0)
    engine.tokenizer = FakeLlamaTokenizer()
    engine.init()
    engine.compile_grammar({"type": "integer"})

    allowed = engine.compute_mask()
    assert 0 in allowed and 1 in allowed
    assert 2 not in allowed and 6 not in allowed
    assert not engine.commit_token(2)
    assert engine.commit_token(1)
    assert 6 in engine.compute_mask()
    assert engine.commit_token(6)
    assert not engine.commit_token(0)

    engine.reset()
    engine.compute_mask()
    assert engine.commit_token(0)
    assert engine.eos_reached is False


def test_timing_summary_keeps_validation_and_invalidation_separate():
    summary = timing_summary([
        {
            "ttfm_us": 10,
            "all_mask_us": [2, 4],
            "num_tokens": 2,
        },
        {
            "ttfm_us": 20,
            "all_mask_us": [8],
            "num_tokens": 1,
            "validation_error": "test #0: should reject but didn't",
        },
    ])
    assert summary["schemas"] == 2
    assert summary["invalidation_errors"] == 1
    assert summary["validation_errors"] == 0
    assert summary["tokens"] == 3
    assert summary["tbm_us"]["p50"] == 4

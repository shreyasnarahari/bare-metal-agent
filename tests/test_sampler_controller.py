import pytest

from toolsmith.grammar import GBNFCompiler
from toolsmith.sampling.automaton import PushdownAutomaton
from toolsmith.sampling.controller import (
    BatchOneSynchronizer,
    FixedMaskController,
    IncompleteConstrainedGeneration,
    UnsupportedGenerationError,
)
from toolsmith.sampling.mask import DeadEndError
from toolsmith.sampling.vocabulary import Vocabulary, VocabularyEntry


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index):
        return self.rows[index]


def integer_controller():
    vocabulary = Vocabulary(
        [VocabularyEntry(0, b"0"), VocabularyEntry(1, b"1"), VocabularyEntry(2, b"-")],
        model_vocabulary_size=4,
        tokenizer_vocabulary_size=4,
        eos_token_ids=frozenset({3}),
        special_token_ids=frozenset({3}),
        excluded_empty_tokens=0,
        build_seconds=0,
    )
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_schema({"type": "integer"})
    )
    return FixedMaskController(vocabulary, automaton)


def test_batch_one_synchronizer_commits_exactly_one_new_token():
    synchronizer = BatchOneSynchronizer()

    assert synchronizer.synchronize(FakeTensor([[10, 11]])) is None
    assert synchronizer.synchronize(FakeTensor([[10, 11, 4]])) == 4
    assert synchronizer.synchronize(FakeTensor([[10, 11, 4, 5]])) == 5
    assert synchronizer.prompt_length == 2


def test_batch_beam_rewind_and_speculative_lengths_are_rejected():
    synchronizer = BatchOneSynchronizer()
    synchronizer.synchronize(FakeTensor([[10, 11]]))

    with pytest.raises(UnsupportedGenerationError, match="single batch"):
        synchronizer.synchronize(FakeTensor([[10, 11, 1], [10, 11, 2]]))
    with pytest.raises(UnsupportedGenerationError, match="one appended"):
        synchronizer.synchronize(FakeTensor([[10, 11]]))
    synchronizer = BatchOneSynchronizer()
    synchronizer.synchronize(FakeTensor([[10]]))
    with pytest.raises(UnsupportedGenerationError, match="one appended"):
        synchronizer.synchronize(FakeTensor([[10, 1, 2]]))


def test_fixed_controller_commits_tokens_and_finishes_with_eos():
    controller = integer_controller()

    decision = controller.mask()
    assert set(decision.allowed_token_ids) == {0, 1, 2}
    controller.commit(1)
    assert controller.can_finish()
    assert 3 in controller.mask().allowed_token_ids
    controller.commit(3)
    controller.finalize("eos")
    assert controller.generated_ids == (1, 3)
    assert controller.generated_bytes() == b"1"


def test_incomplete_fixed_controller_preserves_partial_output():
    controller = integer_controller()
    controller.commit(2)

    with pytest.raises(IncompleteConstrainedGeneration) as caught:
        controller.finalize("length")

    assert caught.value.token_ids == (2,)
    assert caught.value.raw_bytes == b"-"
    assert "byte 1" in caught.value.state


def test_eos_before_acceptance_is_rejected():
    controller = integer_controller()
    controller.commit(2)

    with pytest.raises(DeadEndError, match="EOS"):
        controller.commit(3)

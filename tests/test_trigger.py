import json

import pytest

from toolsmith.grammar import GBNFCompiler
from toolsmith.registry import Tool
from toolsmith.sampling.automaton import PushdownAutomaton
from toolsmith.sampling.controller import (
    IncompleteConstrainedGeneration,
    TriggeredMaskController,
)
from toolsmith.sampling.vocabulary import Vocabulary, VocabularyEntry

MARKER = b"<tool_call>"
REST = b'\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>'


def build_controller(extra_tokens=(), *, cache_size=32):
    raw_tokens = [
        b"reasoning",
        MARKER,
        MARKER + b"{",
        MARKER + b"bad",
        b"prefix" + MARKER + b"bad",
        REST,
        b"<tool_cat>",
        b"plain",
    ]
    for size in range(1, len(MARKER)):
        raw_tokens.extend([MARKER[:size], MARKER[size:]])
    raw_tokens.extend(extra_tokens)
    unique = list(dict.fromkeys(raw_tokens))
    eos = len(unique) + 2
    vocabulary = Vocabulary(
        [VocabularyEntry(index, raw) for index, raw in enumerate(unique)],
        model_vocabulary_size=eos + 3,
        tokenizer_vocabulary_size=len(unique) + 1,
        eos_token_ids=frozenset({eos}),
        special_token_ids=frozenset({eos}),
        excluded_empty_tokens=0,
        build_seconds=0,
    )
    tool = Tool(
        "get_weather",
        "Weather",
        {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )
    automaton = PushdownAutomaton.from_compiled(
        GBNFCompiler().compile_tools([tool], "required")
    )
    controller = TriggeredMaskController(
        vocabulary, automaton, cache_size=cache_size
    )
    ids = {raw: index for index, raw in enumerate(unique)}
    return controller, ids, eos


def test_atomic_marker_plus_invalid_suffix_is_blocked_before_emission():
    controller, ids, eos = build_controller()

    decision = controller.mask()

    assert ids[MARKER + b"bad"] in decision.blocked_token_ids
    assert ids[b"prefix" + MARKER + b"bad"] in decision.blocked_token_ids
    assert ids[MARKER] not in decision.blocked_token_ids
    assert ids[MARKER + b"{"] not in decision.blocked_token_ids
    assert ids[b"<tool_cat>"] not in decision.blocked_token_ids
    assert eos not in decision.blocked_token_ids


def test_atomic_marker_activates_and_complete_call_allows_eos():
    controller, ids, eos = build_controller()

    controller.commit(ids[MARKER])
    assert controller.active
    assert controller.activation_step == 0
    assert ids[REST] in controller.mask().allowed_token_ids
    controller.commit(ids[REST])
    assert controller.can_finish()
    assert eos in controller.mask().allowed_token_ids
    controller.commit(eos)
    controller.finalize("eos")
    assert controller.generated_bytes() == MARKER + REST


@pytest.mark.parametrize("split", range(1, len(MARKER)))
def test_every_marker_split_is_handled(split):
    controller, ids, _ = build_controller()

    controller.commit(ids[MARKER[:split]])
    assert not controller.active
    assert ids[MARKER[split:]] not in controller.mask().blocked_token_ids
    controller.commit(ids[MARKER[split:]])
    assert controller.active
    controller.commit(ids[REST])
    assert controller.can_finish()


def test_token_containing_prefix_marker_and_valid_arguments_is_atomic():
    combined = b"think" + MARKER + REST
    controller, ids, _ = build_controller([combined])

    decision = controller.mask()
    assert ids[combined] not in decision.blocked_token_ids
    controller.commit(ids[combined])
    assert controller.active
    assert controller.can_finish()
    assert controller.generated_bytes() == combined


def test_false_prefix_returns_to_free_mode_and_partial_marker_can_end_as_text():
    controller, ids, eos = build_controller()

    controller.commit(ids[b"<tool_cat>"])
    assert not controller.active
    assert controller.can_finish()
    controller.commit(eos)
    controller.finalize("eos")

    partial, ids, _ = build_controller()
    partial.commit(ids[b"<tool_"])
    assert not partial.active
    assert partial.can_finish()
    partial.finalize("length")


def test_active_incomplete_call_has_explicit_truncation():
    controller, ids, _ = build_controller()
    controller.commit(ids[MARKER])

    with pytest.raises(IncompleteConstrainedGeneration) as caught:
        controller.finalize("length")

    assert caught.value.raw_bytes == MARKER
    assert "trigger active" in caught.value.state


def test_parallel_calls_and_marker_text_inside_arguments_are_supported():
    call = MARKER + REST
    second = b'\n<tool_call>\n{"name":"get_weather","arguments":{"city":"</tool_call>"}}\n</tool_call>'
    controller, ids, _ = build_controller([call + second])

    controller.commit(ids[call + second])

    assert controller.active
    assert controller.can_finish()
    generated = controller.generated_bytes()
    decoded = json.loads(
        b"{" + generated.split(b"{", 1)[1].split(b"\n</tool_call>", 1)[0]
    )
    assert decoded["name"] == "get_weather"
    assert generated.count(b"\n</tool_call>") == 2


def test_free_cache_and_uncached_trigger_have_same_blocked_set():
    cached, ids, _ = build_controller(cache_size=4)
    uncached, _, _ = build_controller(cache_size=0)

    first = cached.mask()
    second = cached.mask()

    assert second.cache_hit
    assert first.blocked_token_ids == uncached.mask().blocked_token_ids
    assert cached.cache_info()["free_hits"] == 1

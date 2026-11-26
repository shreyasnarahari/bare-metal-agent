"""Planted cache breakers and the determinism check behind the PrefixLint gate and the live prefix smoke.

``planted_suite`` renders a base state, then one variant per case, and lints
each variant against the base. Breakers:

- a timestamp inserted into the frozen system prompt;
- unsorted tool-schema keys, and reordered tool definitions;
- whitespace changed in an old tool observation;
- an old assistant call's arguments re-serialized differently;
- pinned memory replaced mid-epoch;
- an old summary edited without advancing the epoch.

Legal changes: a recent append, a tail-only rewrite, and an epoch transition.
Each case names its expected status and cause. The suite passes only if every
case matches, and only if the base state renders identically on every one of
``repeats`` renders.

``ChatMLTemplate`` is a deterministic stand-in for CPU checks. Like the real
Qwen template, it serializes tool schemas in insertion order, so unsorted
keys are a real breaker. The live smoke runs the same suite with the model's
template and the parity-checked tokenizer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..assembler import ContextAssembler, ContextState, SpanRenderer
from ..prefix_lint import lint
from ..recitation import RecitationState
from ..segments import freeze_tools

TOOLS = [
    {"type": "function", "function": {"name": "get_order", "description": "Look up one order by id.",
                                      "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                                                     "required": ["order_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "add_order_note", "description": "Attach a note to an order.",
                                      "parameters": {"type": "object",
                                                     "properties": {"order_id": {"type": "string"}, "note": {"type": "string"}},
                                                     "required": ["order_id", "note"], "additionalProperties": False}}},
]
CASES = (
    # (case, expected status, expected cause)
    ("timestamp_in_frozen", "violation", "frozen_changed"),
    ("unsorted_tool_keys", "violation", "frozen_changed"),
    ("reordered_tools", "violation", "frozen_changed"),
    ("observation_whitespace", "violation", "recent_rewritten"),
    ("arguments_reserialized", "violation", "recent_rewritten"),
    ("pinned_memory_replaced", "violation", "pinned_changed_mid_epoch"),
    ("summary_edited_same_epoch", "violation", "summary_changed_without_epoch"),
    ("recent_append", "expected_append", "recent_append"),
    ("tail_rewrite", "expected_tail_only", "tail_rewrite"),
    ("epoch_transition", "expected_epoch", "epoch_transition"),
)


class ChatMLTemplate:
    """A deterministic ChatML-style template (the ``PromptTemplate`` interface) for CPU checks."""

    def render(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] = (), *,
               enable_thinking: bool | None = None, add_generation_prompt: bool = True) -> str:
        parts = []
        for index, message in enumerate(messages):
            content = message.get("content") or ""
            if index == 0 and message.get("role") == "system" and tools:
                content += "\n\n# Tools\n" + "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
            for call in message.get("tool_calls") or []:
                function = call["function"]
                arguments = function["arguments"]
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                content += f"\n<tool_call>\n{{\"name\": \"{function['name']}\", \"arguments\": {arguments}}}\n</tool_call>"
            role = message["role"]
            if role == "tool":
                role, content = "user", f"<tool_response>\n{content}\n</tool_response>"
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False else ""))
        return "".join(parts)


def base_state() -> ContextState:
    """Epoch 1 with a summary, pinned memory, and one tool round trip in the recent segment."""
    call = {"content": None, "role": "assistant", "tool_calls": [
        {"function": {"arguments": '{"order_id": "ord_1001"}', "name": "get_order"}, "id": "call_0_0", "type": "function"}]}
    result = {"content": '{"ok": true, "value": {"status": "delivered"}}', "role": "tool", "tool_call_id": "call_0_0"}
    return ContextState(
        frozen_system="You are a careful support agent.",
        frozen_tools=freeze_tools(TOOLS),
        task_charter="Check order ord_1001, then note PAY-4471-K on it.",
        pinned_memory=("The customer prefers email.",),
        epoch=1,
        summary={"facts": [{"text": "ord_1001 is a kettle", "cites": [3]}], "compacted_through_seq": 9},
        recitation=RecitationState(completed_step_count=1, remaining_step_budget=7),
    ).append([call, result], [10, 11])


def _variant(state: ContextState, case: str) -> ContextState:
    if case == "timestamp_in_frozen":
        return replace(state, frozen_system=state.frozen_system + " Current time: 2026-11-26T10:00:00Z.")
    if case == "unsorted_tool_keys":
        # Toolsmith rebuilds a tool's outer keys in a fixed order, but passes its parameter schema through,
        # and templates serialize it in insertion order: unsorted schema keys reach the prompt.
        first = dict(state.frozen_tools[0])
        parameters = first["function"]["parameters"]
        unsorted = {key: parameters[key] for key in ("type", "properties", "required", "additionalProperties")}
        changed = {**first, "function": {**first["function"], "parameters": unsorted}}
        return replace(state, frozen_tools=(changed, *state.frozen_tools[1:]))
    if case == "reordered_tools":
        return replace(state, frozen_tools=tuple(reversed(state.frozen_tools)))
    if case == "observation_whitespace":
        recent = list(state.recent)
        recent[1] = {**recent[1], "content": recent[1]["content"].replace(": ", ":  ", 1)}
        return replace(state, recent=tuple(recent))
    if case == "arguments_reserialized":
        recent = list(state.recent)
        call = dict(recent[0]["tool_calls"][0])
        call["function"] = {**call["function"], "arguments": '{"order_id":"ord_1001"}'}
        recent[0] = {**recent[0], "tool_calls": [call]}
        return replace(state, recent=tuple(recent))
    if case == "pinned_memory_replaced":
        return replace(state, pinned_memory=("The customer prefers phone calls.",))
    if case == "summary_edited_same_epoch":
        return replace(state, summary={**dict(state.summary or {}), "compacted_through_seq": 8})
    if case == "recent_append":
        call = {"content": None, "role": "assistant", "tool_calls": [
            {"function": {"arguments": '{"note": "PAY-4471-K", "order_id": "ord_1001"}', "name": "add_order_note"},
             "id": "call_1_0", "type": "function"}]}
        result = {"content": '{"ok": true}', "role": "tool", "tool_call_id": "call_1_0"}
        return state.append([call, result], [13, 14]).with_recitation(replace(state.recitation, completed_step_count=2))
    if case == "tail_rewrite":
        return state.with_recitation(replace(state.recitation, next_action="answer the user"))
    if case == "epoch_transition":
        return state.advance_epoch({"facts": [{"text": "ord_1001 checked", "cites": [10, 11]}],
                                    "compacted_through_seq": 11}, keep_from=2)
    raise ValueError(f"Unknown planted case {case!r}")


def planted_suite(renderer: SpanRenderer, *, repeats: int = 25) -> dict[str, Any]:
    """Determinism over ``repeats`` renders, then every planted case linted against the base render."""
    assembler = ContextAssembler(renderer)
    state = base_state()
    base = assembler.render(state)
    deterministic = all(
        (again := assembler.render(state)).spans.text == base.spans.text
        and again.spans.token_ids == base.spans.token_ids and again.spans.spans == base.spans.spans
        and again.messages == base.messages
        for _ in range(repeats)
    )
    previous = base.fingerprint
    results = []
    for case, expected_status, expected_cause in CASES:
        variant = _variant(state, case).with_previous(previous)
        rendered = assembler.render(variant)
        outcome = lint(previous, rendered.fingerprint)
        results.append({
            "case": case, "expected_status": expected_status, "status": outcome.status,
            "expected_cause": expected_cause, "cause": outcome.divergence_cause,
            "first_divergent_index": outcome.first_divergent_index,
            "first_changed_segment": outcome.first_changed_segment,
            "observed_common_prefix_tokens": outcome.observed_common_prefix_tokens,
            "expected_common_prefix_floor": outcome.expected_common_prefix_floor,
            "prompt_tokens": rendered.total, "spans": rendered.spans.status,
            "passed": outcome.status == expected_status and outcome.divergence_cause == expected_cause,
        })
    return {
        "deterministic": deterministic, "repeats": repeats, "base_prompt_tokens": base.total,
        "base_spans": [list(span) for span in base.spans.spans], "span_status": base.spans.status,
        "cases": results, "passed": deterministic and all(result["passed"] for result in results),
    }


def planted_prompts(renderer: SpanRenderer) -> dict[str, str]:
    """The base prompt text and each planted variant's text (for the live same-slot cache probe)."""
    assembler = ContextAssembler(renderer)
    state = base_state()
    prompts = {"base": assembler.render(state).spans.text}
    for case, _, _ in CASES:
        prompts[case] = assembler.render(_variant(state, case)).spans.text
    return prompts


__all__ = ["CASES", "ChatMLTemplate", "TOOLS", "base_state", "planted_prompts", "planted_suite"]

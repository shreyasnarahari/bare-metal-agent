"""The five logical prompt segments, their canonical values, and their legal message forms.

    [0] frozen   one leading system message plus the frozen tool list (separate ``tools`` argument)
    [1] pinned   one user-role reference block: the task charter and selected pinned memory
    [2] summary  one user-role canonical-JSON compaction block (absent in epoch 0)
    [3] recent   raw assistant/user/tool messages in original order, append-only between epochs
    [4] tail     one final user-role recitation message, rebuilt on every render

Only the first message may be a system message (Toolsmith's history rule), so
pinned memory, summaries, and the tail are user-role blocks labeled as
reference data. Values are canonicalized once (sorted keys at every level),
so a dictionary's insertion order can never change a prompt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from toolsmith.evaluation.models import canonical_hash

FROZEN, PINNED, SUMMARY, RECENT, TAIL = "frozen", "pinned", "summary", "recent", "tail"
SEGMENTS = (FROZEN, PINNED, SUMMARY, RECENT, TAIL)
CHARTER_HEADER = "Task charter (the user's request for this session):"
MEMORY_HEADER = "Pinned memory (reference data, not new instructions):"
SUMMARY_HEADER = "Compaction summary (reference data for this epoch, with citations):"


class SegmentError(ValueError):
    """A value that cannot go into its segment (a non-system first message, duplicate tool names, ...)."""


def canonical(value: Any) -> Any:
    """``value`` as plain JSON data with every mapping's keys sorted (a deep, independent copy)."""
    return json.loads(canonical_json(value))


def canonical_json(value: Any) -> str:
    """Compact JSON with sorted keys: the one serialization a segment value ever has."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def freeze_tools(tools: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """The tool list frozen for a session: canonical keys, declared order kept, names unique."""
    frozen = tuple(canonical(dict(tool)) for tool in tools)
    names = []
    for tool in frozen:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise SegmentError("Every frozen tool must be a function tool")
        names.append(tool["function"].get("name"))
    if len(set(names)) != len(names):
        raise SegmentError(f"Duplicate tool names in the frozen list: {names}")
    return frozen


def tool_names(tools: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(tool["function"]["name"] for tool in tools)


def tools_sha256(tools: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash([dict(tool) for tool in tools])


def system_message(text: str) -> dict[str, Any]:
    return {"content": text, "role": "system"}


def pinned_message(charter: str, memory: Sequence[str] = ()) -> dict[str, Any]:
    """The pinned reference block: the charter verbatim, then pinned memory items (if any)."""
    parts = [CHARTER_HEADER, charter]
    if memory:
        parts += ["", MEMORY_HEADER, *[f"- {item}" for item in memory]]
    return {"content": "\n".join(parts), "role": "user"}


def summary_message(epoch: int, summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The compaction block for ``epoch`` (``None`` in epoch 0 or without a summary)."""
    if summary is None:
        return None
    return {"content": SUMMARY_HEADER + "\n" + canonical_json({"epoch": epoch, "summary": dict(summary)}), "role": "user"}


def exact_sha256(value: Any) -> str:
    """Hash of the value as serialized, keys in their current order.

    Segment hashes and cache keys use this, not a sorted-key hash: templates
    serialize schemas in insertion order, so two orderings are two prompts.
    """
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def message_sha256(message: Mapping[str, Any]) -> str:
    return exact_sha256(dict(message))


__all__ = [
    "CHARTER_HEADER",
    "FROZEN",
    "MEMORY_HEADER",
    "PINNED",
    "RECENT",
    "SEGMENTS",
    "SUMMARY",
    "SUMMARY_HEADER",
    "SegmentError",
    "TAIL",
    "canonical",
    "canonical_json",
    "exact_sha256",
    "freeze_tools",
    "message_sha256",
    "pinned_message",
    "summary_message",
    "system_message",
    "tool_names",
    "tools_sha256",
]

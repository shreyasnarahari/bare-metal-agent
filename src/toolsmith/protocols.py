"""Model profiles: which model, tokenizer, and native tool-call protocol a run uses.

Each supported model family writes tool calls differently. Qwen3 emits a JSON
object inside ``<tool_call>`` tags; Qwen3.5 emits ``<function=...>`` blocks with
one ``<parameter=...>`` per argument. Callers pick a profile explicitly; nothing
guesses the protocol from a model name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ToolProtocol(str, Enum):
    """The tool-call markup a model family emits (and the parser that reads it)."""

    QWEN3 = "qwen3"
    QWEN35 = "qwen35"


@dataclass(frozen=True)
class ModelProfile:
    """Everything needed to talk to one model: names, tokenizer, protocol, thinking default."""

    name: str  # short id used in file names and records ("qwen3")
    model: str  # Hugging Face model id, also the model name the shim reports
    tokenizer: str
    protocol: ToolProtocol
    thinking_by_default: bool


QWEN3_PROFILE = ModelProfile(
    name="qwen3",
    model="Qwen/Qwen3-8B",
    tokenizer="Qwen/Qwen3-8B",
    protocol=ToolProtocol.QWEN3,
    thinking_by_default=True,
)

QWEN35_PROFILE = ModelProfile(
    name="qwen35",
    model="Qwen/Qwen3.5-9B",
    tokenizer="Qwen/Qwen3.5-9B",
    protocol=ToolProtocol.QWEN35,
    thinking_by_default=True,
)


__all__ = [
    "ModelProfile",
    "QWEN3_PROFILE",
    "QWEN35_PROFILE",
    "ToolProtocol",
]

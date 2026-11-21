from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ToolProtocol(str, Enum):
    QWEN3 = "qwen3"
    QWEN35 = "qwen35"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
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

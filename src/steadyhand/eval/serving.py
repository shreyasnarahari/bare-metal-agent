"""Server shapes and startup-probe helpers for the live sweep.

The presets are capacity guesses to confirm with the probe, not measured
throughputs. llama.cpp splits ``-c`` across ``--parallel`` slots, so the two
are always set as a pair and ``per_slot_context = context_size // parallel``.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ServerPreset:
    gpu: str
    profile: str
    shape: str  # "wide" (many short episodes) or "tall" (fewer, longer)
    context_size: int
    parallel: int
    tasks: str

    @property
    def per_slot_context(self) -> int:
        return self.context_size // self.parallel

    @property
    def name(self) -> str:
        return f"{self.gpu}/{self.profile}/{self.shape}"

    def shrink(self) -> ServerPreset:
        """One slot fewer, same per-slot context: the response to low VRAM or a load-time OOM."""
        if self.parallel <= 1:
            raise ValueError("Already at one slot")
        parallel = self.parallel - 1
        return replace(self, parallel=parallel, context_size=self.per_slot_context * parallel)

    def as_dict(self) -> dict[str, Any]:
        return {
            "preset": self.name,
            "gpu": self.gpu,
            "profile": self.profile,
            "shape": self.shape,
            "context_size": self.context_size,
            "parallel": self.parallel,
            "per_slot_context": self.per_slot_context,
        }


PRESETS = (
    ServerPreset("T4", "qwen3", "wide", 32768, 4, "own suite, explicit and lost-ack faults"),
    ServerPreset("T4", "qwen3", "tall", 32768, 2, "tasks whose prompt exceeds 8192 tokens"),
    ServerPreset("T4", "qwen35", "wide", 24576, 3, "same wide tasks, only if the probe stays under the ceiling"),
    ServerPreset("L4", "qwen35", "wide", 65536, 4, "own suite and fault grid"),
    ServerPreset("L4", "qwen35", "tall", 65536, 2, "long-context multi-turn"),
    ServerPreset("L4", "qwen3", "wide", 65536, 8, "comparison model, short tasks"),
)


def gpu_family(name: str) -> str | None:
    for family in ("L4", "T4"):
        if re.search(rf"\b{family}\b", name):
            return family
    return None


def preset_for(gpu_name: str, profile: str, shape: str) -> ServerPreset:
    family = gpu_family(gpu_name)
    for preset in PRESETS:
        if (preset.gpu, preset.profile, preset.shape) == (family, profile, shape):
            return preset
    raise LookupError(f"No preset for GPU {gpu_name!r}, profile {profile!r}, shape {shape!r}")


def flash_attention_args(help_text: str) -> list[str]:
    """Whichever flash-attention flag this binary's ``--help`` documents, or nothing."""
    if re.search(r"--flash-attn\s+(\[|\{|<)?\s*(on|auto)", help_text) or "--flash-attn FA" in help_text:
        return ["--flash-attn", "on"]
    if re.search(r"(^|\s)-fa(,|\s)", help_text) or "--flash-attn" in help_text:
        return ["-fa"]
    return []


def server_extra_args(
    help_text: str,
    *,
    cpu_count: int,
    kv_cache: str = "q8_0",
    batch_size: int = 2048,
    ubatch_size: int = 512,
) -> list[str]:
    """``extra_args`` for ``LlamaCppServer``: q8 KV cache, flash attention if supported, batches, threads."""
    threads = str(max(1, cpu_count))
    return [
        "--cache-type-k", kv_cache,
        "--cache-type-v", kv_cache,
        *flash_attention_args(help_text),
        "--batch-size", str(batch_size),
        "--ubatch-size", str(ubatch_size),
        "-t", threads,
        "--threads-batch", threads,
    ]


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Rows of ``--query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits``."""
    rows = []
    for fields in csv.reader(io.StringIO(text.strip())):
        if not fields:
            continue
        name, total, used, utilization = (field.strip() for field in fields[:4])
        rows.append({
            "name": name,
            "memory_total_mib": int(float(total)),
            "memory_used_mib": int(float(used)),
            "utilization_gpu": int(float(utilization)),
        })
    return rows


def needs_smaller_shape(row: Mapping[str, Any], min_free_mib: int = 1024) -> bool:
    return row["memory_total_mib"] - row["memory_used_mib"] < min_free_mib


def slots_busy(slots: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Busy and total slots from llama-server ``GET /slots`` (``is_processing`` or a non-zero ``state``)."""
    busy = sum(1 for slot in slots if slot.get("is_processing") or slot.get("state", 0) not in (0, "idle"))
    return busy, len(slots)


def tokens_per_second(completion_tokens: int, seconds: float) -> float | None:
    return completion_tokens / seconds if seconds > 0 else None


__all__ = [
    "PRESETS",
    "ServerPreset",
    "flash_attention_args",
    "gpu_family",
    "needs_smaller_shape",
    "parse_nvidia_smi",
    "preset_for",
    "server_extra_args",
    "slots_busy",
    "tokens_per_second",
]

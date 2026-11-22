"""Server shapes and startup-probe helpers for the live sweep.

The presets are capacity guesses to confirm with the probe, not measured
throughputs. llama.cpp splits ``-c`` across ``--parallel`` slots, so the two
are always set as a pair and ``per_slot_context = context_size // parallel``.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
    ServerPreset("T4", "qwen35", "tall", 32768, 2, "long-context multi-turn, only if the probe stays under the ceiling"),
    ServerPreset("L4", "qwen35", "wide", 65536, 4, "own suite and fault grid"),
    ServerPreset("L4", "qwen35", "tall", 65536, 2, "long-context multi-turn"),
    ServerPreset("L4", "qwen3", "wide", 65536, 8, "comparison model, short tasks"),
)


L4_VRAM_MIB = 22 * 1024  # an L4 reports about 23000 MiB
T4_VRAM_MIB = 14 * 1024  # a T4 reports 15360 MiB


def gpu_family(name: str) -> str | None:
    for family in ("L4", "T4"):
        if re.search(rf"\b{family}\b", name):
            return family
    return None


def preset_family(gpu_name: str, vram_mib: int | None = None) -> str | None:
    """The preset rows to use: the named family, else the rows for the VRAM this card has."""
    family = gpu_family(gpu_name)
    if family or vram_mib is None:
        return family
    if vram_mib >= L4_VRAM_MIB:
        return "L4"
    if vram_mib >= T4_VRAM_MIB:
        return "T4"
    return None


def preset_for(gpu_name: str, profile: str, shape: str, vram_mib: int | None = None) -> ServerPreset:
    """The preset for this GPU; another card (A100, V100) borrows the T4 or L4 rows by VRAM.

    A borrowed preset is a starting point: the probe-and-shrink loop corrects it.
    """
    family = preset_family(gpu_name, vram_mib)
    for preset in PRESETS:
        if (preset.gpu, preset.profile, preset.shape) == (family, profile, shape):
            return preset
    raise LookupError(f"No preset for GPU {gpu_name!r} ({vram_mib} MiB), profile {profile!r}, shape {shape!r}")


def flash_attention_style(help_text: str) -> str | None:
    """``"value"`` for ``--flash-attn on|off|auto``, ``"switch"`` for a bare ``-fa``, else ``None``."""
    if re.search(r"--flash-attn\s+(\[|\{|<)?\s*(on|off|auto)", help_text) or "--flash-attn FA" in help_text:
        return "value"
    if re.search(r"(^|\s)-fa(,|\s)", help_text) or "--flash-attn" in help_text:
        return "switch"
    return None


def flash_attention_args(help_text: str, enabled: bool = True) -> list[str]:
    """The flag this binary's ``--help`` documents. Off is explicit where the default may be ``auto``."""
    style = flash_attention_style(help_text)
    if style == "value":
        return ["--flash-attn", "on" if enabled else "off"]
    if style == "switch" and enabled:
        return ["-fa"]
    return []


def kv_ladder(help_text: str) -> list[dict[str, Any]]:
    """KV cache and flash-attention settings to try, cheapest memory first.

    q8_0 with flash attention halves the KV cache, but whether a GPU (for
    example a T4, sm_75) runs it at full speed depends on the build. If the
    self-test fails, the next rung is tried. A quantized V cache needs flash
    attention, so the last rung is f16 without it.
    """
    rungs = []
    if flash_attention_style(help_text):
        rungs += [{"kv_cache": "q8_0", "flash_attn": True}, {"kv_cache": "f16", "flash_attn": True}]
    rungs.append({"kv_cache": "f16", "flash_attn": False})
    return rungs


def server_extra_args(
    help_text: str,
    *,
    cpu_count: int,
    kv_cache: str = "q8_0",
    flash_attn: bool = True,
    batch_size: int = 2048,
    ubatch_size: int = 512,
) -> list[str]:
    """``extra_args`` for ``LlamaCppServer``: KV cache type, flash attention if supported, batches, threads."""
    threads = str(max(1, cpu_count))
    return [
        "--cache-type-k", kv_cache,
        "--cache-type-v", kv_cache,
        *flash_attention_args(help_text, flash_attn),
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


MIN_DECODE_TPS = 5.0  # a GPU-resident 8B or 9B Q4 decodes far faster; below this an op fell back to CPU
SELF_TEST_PROMPT = "Count upward from 1, separated by spaces: 1 2 3 4"
_OUT_OF_MEMORY = re.compile(r"out of memory|failed to allocate|cudaMalloc|unable to allocate|\bOOM\b", re.IGNORECASE)


def decode_speed(response: Mapping[str, Any], seconds: float) -> float | None:
    """Decode tokens per second from llama-server ``timings``, else tokens over wall time."""
    timings = response.get("timings") if isinstance(response.get("timings"), Mapping) else {}
    speed = timings.get("predicted_per_second")
    if speed:
        return float(speed)
    tokens = int(timings.get("predicted_n") or response.get("tokens_predicted") or 0)
    return tokens / seconds if tokens and seconds > 0 else None


def is_out_of_memory(text: str) -> bool:
    return bool(_OUT_OF_MEMORY.search(text))


@dataclass
class FittedServer:
    server: Any
    preset: ServerPreset
    rung: dict[str, Any]
    extra_args: list[str]
    gpu: dict[str, Any]
    decode_tps: float | None
    attempts: list[dict[str, Any]] = field(default_factory=list)


def fit_server(
    preset: ServerPreset,
    rungs: Sequence[Mapping[str, Any]],
    *,
    start: Callable[[ServerPreset, list[str]], Any],
    extra_args_for: Callable[[Mapping[str, Any]], list[str]],
    query_gpu: Callable[[], Mapping[str, Any]],
    probe: Callable[[Any], float | None],
    min_decode_tps: float = MIN_DECODE_TPS,
    min_free_mib: int = 1024,
    log: Callable[[str], None] = print,
) -> FittedServer:
    """Start the first (KV rung, slot count) that loads, leaves VRAM free, and decodes at GPU speed.

    ``start`` launches a server and waits until it is healthy (stopping it and
    raising on failure). Running out of memory, at load or after it, costs one
    slot; any other failure, or a self-test below ``min_decode_tps``, moves to
    the next rung, because fewer slots cannot fix an unsupported kernel.
    """
    attempts: list[dict[str, Any]] = []
    for rung in rungs:
        current = preset
        extra_args = extra_args_for(rung)
        while True:
            server, gpu, speed, action, reason = None, {}, None, None, None
            try:
                server = start(current, extra_args)
                gpu = dict(query_gpu())
                if needs_smaller_shape(gpu, min_free_mib):
                    action, reason = "shrink", f"less than {min_free_mib} MiB of VRAM free after load"
                else:
                    speed = probe(server)
                    if speed is not None and speed < min_decode_tps:
                        action, reason = "next_rung", f"decode {speed:.1f} tok/s is below {min_decode_tps:g}"
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                action = "shrink" if is_out_of_memory(message) else "next_rung"
                reason = message[-600:]
            attempts.append({**rung, "parallel": current.parallel, "context_size": current.context_size,
                             "decode_tps": speed, "result": action or "ok", "reason": reason})
            if action is None:
                return FittedServer(server, current, dict(rung), extra_args, gpu, speed, attempts)
            if server is not None:
                server.stop(timeout=30)
            log(f"{current.name} kv={rung['kv_cache']} flash_attn={rung['flash_attn']} parallel={current.parallel}: {reason}")
            if action == "next_rung" or current.parallel <= 1:
                break
            current = current.shrink()
    raise RuntimeError(f"No server configuration fits and decodes at GPU speed: {attempts}")


__all__ = [
    "MIN_DECODE_TPS",
    "PRESETS",
    "SELF_TEST_PROMPT",
    "FittedServer",
    "ServerPreset",
    "decode_speed",
    "fit_server",
    "flash_attention_args",
    "flash_attention_style",
    "gpu_family",
    "is_out_of_memory",
    "kv_ladder",
    "needs_smaller_shape",
    "parse_nvidia_smi",
    "preset_family",
    "preset_for",
    "server_extra_args",
    "slots_busy",
    "tokens_per_second",
]

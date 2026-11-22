"""Detect the runtime once and derive every concurrency setting from what is there.

Both notebooks call :func:`detect_hardware` in their shared ``hardware`` setup
cell. Build jobs, server slots, CPU pools, and process counts all come from the
detected vCPUs, RAM, and VRAM, never from the GPU's name alone, so a T4, an L4,
or any other card gets settings it can hold.

Standard library only: the setup cells import this before ``pip install``.
"""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

MIB_PER_GIB = 1024
MIN_COMPUTE_CAP = 70  # Volta; Colab offers T4 (75), L4 (89), and A100 (80)
MIN_VRAM_MIB = 14 * MIB_PER_GIB  # a T4 reports 15360 MiB
LARGE_VRAM_MIB = 22 * MIB_PER_GIB  # an L4 reports about 23000 MiB
SWITCH_RUNTIME = "In Colab choose Runtime > Change runtime type, pick a T4 or L4 GPU, then run from the top."


class UnsupportedRuntime(RuntimeError):
    """The runtime cannot run the notebooks; the message says how to switch."""


def gpu_family(name: str) -> str | None:
    """``"T4"`` or ``"L4"`` when the name says so, else ``None``."""
    for family in ("L4", "T4"):
        if re.search(rf"\b{family}\b", name or ""):
            return family
    return None


def visible_cpus() -> int:
    """vCPUs this process may run on (cgroup and affinity aware)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def total_ram_mib(meminfo: str | Path = "/proc/meminfo") -> int:
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    except (AttributeError, ValueError, OSError):
        return 0


def free_disk_mib(path: str | Path = "/content") -> int:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free // (1024 * 1024)


def build_jobs_for(cpu_count: int, ram_mib: int) -> int:
    """Parallel compile jobs: one per vCPU, but no more than one per 2 GiB (nvcc is memory hungry)."""
    return max(1, min(cpu_count, ram_mib // (2 * MIB_PER_GIB)))


def parse_gpu_rows(text: str) -> list[dict[str, Any]]:
    """Rows of ``--query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits``."""
    rows = []
    for fields in csv.reader(io.StringIO(text.strip())):
        if len(fields) < 3:
            continue
        name, cap, memory = (field.strip() for field in fields[:3])
        rows.append({
            "name": name,
            "compute_cap": cap.replace(".", ""),
            "vram_mib": int(float(memory)),
        })
    return rows


def query_gpus() -> str:
    """Raw ``nvidia-smi`` output, or ``""`` when there is no usable NVIDIA GPU."""
    if shutil.which("nvidia-smi") is None:
        return ""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60,
    )
    return result.stdout if result.returncode == 0 else ""


def vram_free_mib(output: str | None = None) -> int | None:
    """Free VRAM on the first GPU right now, or ``None`` when ``nvidia-smi`` cannot say."""
    if output is None:
        if shutil.which("nvidia-smi") is None:
            return None
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return None
        output = result.stdout
    try:
        total, used = (int(float(value)) for value in output.strip().splitlines()[0].split(",")[:2])
    except (IndexError, ValueError):
        return None
    return total - used


@dataclass(frozen=True)
class Hardware:
    gpu_name: str  # "" when no GPU is visible
    compute_cap: str  # "75", "89"; "" without a GPU
    vram_mib: int
    cpu_count: int
    ram_mib: int
    disk_free_mib: int

    @property
    def family(self) -> str | None:
        return gpu_family(self.gpu_name)

    @property
    def records_slug(self) -> str:
        """Per-GPU records folder name: ``t4``, ``l4``, or the GPU name made path-safe."""
        if self.family:
            return self.family.lower()
        name = re.sub(r"\b(nvidia|tesla)\b", "", self.gpu_name.lower())
        slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
        return slug or "unknown-gpu"

    def build_jobs(self) -> int:
        return build_jobs_for(self.cpu_count, self.ram_mib)

    def cpu_workers(self, cap: int = 8) -> int:
        """Processes for CPU-bound pools; each spawned worker is budgeted 1 GiB, leaving 2 GiB for the kernel."""
        ram_limited = (self.ram_mib - 2 * MIB_PER_GIB) // MIB_PER_GIB
        return max(1, min(self.cpu_count, cap, ram_limited))

    def day10_slots(self) -> int:
        """llama-server slots for Day 10, at a fixed 4096-token context per slot."""
        if self.family == "T4":
            return 4
        if self.family == "L4":
            return 8
        return 8 if self.vram_mib >= LARGE_VRAM_MIB else 4

    def hf_processes(self, cap: int = 6) -> int:
        """Processes that each hold an fp16 Qwen3-0.6B (about 2 GiB with activations)."""
        vram_limited = (self.vram_mib - 2 * MIB_PER_GIB) // (2 * MIB_PER_GIB)
        return max(1, min(self.cpu_count, vram_limited, cap))

    def peer_workers(self) -> int:
        """MaskBench peer processes alongside the in-kernel Toolsmith engine."""
        return max(1, self.cpu_count - 1)

    def identity(self) -> dict[str, Any]:
        """The fields that change results; stamped on records and manifests."""
        return {
            "gpu": self.gpu_name,
            "gpu_family": self.family,
            "compute_cap": self.compute_cap,
            "vram_mib": self.vram_mib,
            "cpu_count": self.cpu_count,
            "ram_mib": self.ram_mib,
        }

    def summary(self) -> str:
        gpu = f"{self.gpu_name} (sm_{self.compute_cap}, {self.vram_mib / MIB_PER_GIB:.1f} GiB)" if self.gpu_name else "no GPU"
        return (
            f"{gpu}; {self.cpu_count} vCPUs; {self.ram_mib / MIB_PER_GIB:.1f} GiB RAM; "
            f"{self.disk_free_mib / MIB_PER_GIB:.0f} GiB free disk. Build jobs {self.build_jobs()}, "
            f"CPU workers {self.cpu_workers()}, Day 10 slots {self.day10_slots()}, "
            f"HF processes {self.hf_processes()}, MaskBench peers {self.peer_workers()}."
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "family": self.family, "records_slug": self.records_slug}


def detect_hardware(
    *,
    gpu_output: str | None = None,
    cpu_count: int | None = None,
    ram_mib: int | None = None,
    disk_path: str | Path = "/content",
) -> Hardware:
    """The first GPU plus host resources. Arguments override detection (tests use them)."""
    rows = parse_gpu_rows(query_gpus() if gpu_output is None else gpu_output)
    gpu = rows[0] if rows else {"name": "", "compute_cap": "", "vram_mib": 0}
    return Hardware(
        gpu_name=gpu["name"],
        compute_cap=gpu["compute_cap"],
        vram_mib=gpu["vram_mib"],
        cpu_count=visible_cpus() if cpu_count is None else cpu_count,
        ram_mib=total_ram_mib() if ram_mib is None else ram_mib,
        disk_free_mib=free_disk_mib(disk_path),
    )


def require_supported(hardware: Hardware) -> None:
    if not hardware.gpu_name:
        raise UnsupportedRuntime(f"No NVIDIA GPU is visible. {SWITCH_RUNTIME}")
    if not hardware.compute_cap.isdigit() or int(hardware.compute_cap) < MIN_COMPUTE_CAP:
        raise UnsupportedRuntime(
            f"{hardware.gpu_name} has compute capability {hardware.compute_cap or 'unknown'}; "
            f"the llama.cpp builds need {MIN_COMPUTE_CAP} or newer. {SWITCH_RUNTIME}"
        )
    if hardware.vram_mib < MIN_VRAM_MIB:
        raise UnsupportedRuntime(
            f"{hardware.gpu_name} has {hardware.vram_mib} MiB of VRAM; an 8B or 9B Q4 model with a usable "
            f"KV cache needs at least {MIN_VRAM_MIB} MiB. {SWITCH_RUNTIME}"
        )


def runtime_stamp(hardware: Hardware, jobs: Any = None, session_id: str | None = None) -> dict[str, Any]:
    """Hardware identity plus how busy the machine is right now, for timing records.

    ``session_id`` (one per kernel) tells records written by a rerun after a
    disconnect apart from the first run's.
    """
    try:
        load = [round(value, 2) for value in os.getloadavg()]
    except OSError:
        load = None
    active: Callable[[], list[str]] | None = getattr(jobs, "active", None)
    return {
        "runtime": {
            **hardware.identity(),
            "loadavg": load,
            "background_jobs": active() if active else [],
            "session_id": session_id,
        },
    }


__all__ = [
    "Hardware",
    "UnsupportedRuntime",
    "build_jobs_for",
    "detect_hardware",
    "free_disk_mib",
    "gpu_family",
    "parse_gpu_rows",
    "require_supported",
    "runtime_stamp",
    "total_ram_mib",
    "visible_cpus",
    "vram_free_mib",
]

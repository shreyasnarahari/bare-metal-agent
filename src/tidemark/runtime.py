"""The hardware envelope, the resource budget derived from it, and the runtime resource sampler.

``HardwareEnvelope`` extends ``toolsmith.hardware.Hardware`` with what capacity
planning needs right now: free VRAM and RAM, local free disk, and what the
pinned llama.cpp build can do. ``ResourceBudget`` turns that into margins and
CPU/IO partitions. GPU names never decide anything here; measured values do.

``ResourceSampler`` writes ``resource_samples.jsonl`` beside a run: a volatile
file, separate from the deterministic journals. Each sample holds system and
per-process-group CPU (from ``/proc``), load average, RAM, local disk, the
latest row of one long-lived ``nvidia-smi`` sampler, and the scheduler's slot
and queue counts. It never reads another process's environment or command line.

``summarize_utilization`` reports utilization only where the relevant work was
ready (see ``classify_sample``)::

    gpu_slot_occupancy    = busy_slots / usable_slots            over model-ready samples
    cpu_ready_utilization = allocated CPU cores in use / capacity over CPU-ready samples
    ready_but_idle        = a free slot with a waiting model call, or idle CPU capacity with queued CPU work

Idle time with no ready work (sequential dependencies, setup, drain) is
labeled, not counted as a scheduler failure. The targets (95% occupancy, 85%
CPU) are tuning targets: a miss is reported with the bottleneck the samples
show, never filled with artificial work.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from toolsmith.evaluation.models import canonical_hash
from toolsmith.hardware import Hardware, detect_hardware, free_disk_mib, gpu_family, vram_free_mib

MIB = 1024 * 1024
DEFAULT_VRAM_MARGIN_MIB = 1024  # the existing fit_server free-VRAM floor
DEFAULT_RAM_MARGIN_MIB = 2048
WORKER_RAM_MIB = 1024  # each spawned CPU worker's budget (matches Hardware.cpu_workers)
GPU_OCCUPANCY_TARGET = 0.95
CPU_UTILIZATION_TARGET = 0.85
KV_TYPES = ("f16", "q8_0", "q4_0")


@dataclass(frozen=True)
class LlamaCapabilities:
    """What the pinned llama.cpp build documents (from ``--help``) and what a live probe measured."""

    build: str | None = None
    flash_attn: str | None = None  # "value" (--flash-attn on|off|auto), "switch" (-fa), or None
    kv_cache_types: tuple[str, ...] = ()
    slot_save: bool = False
    slots_endpoint: bool | None = None  # measured: GET /slots answered
    id_slot: bool | None = None  # measured: a request pinned to a slot reported that slot

    @classmethod
    def from_help(cls, help_text: str, *, build: str | None = None) -> LlamaCapabilities:
        from steadyhand.eval.serving import flash_attention_style

        documented = tuple(kind for kind in KV_TYPES if kind in help_text) if "--cache-type-k" in help_text else ()
        return cls(
            build=build,
            flash_attn=flash_attention_style(help_text),
            kv_cache_types=documented or (("f16",) if help_text else ()),
            slot_save="--slot-save-path" in help_text,
        )

    def with_probe(self, *, slots_endpoint: bool | None, id_slot: bool | None) -> LlamaCapabilities:
        return LlamaCapabilities(self.build, self.flash_attn, self.kv_cache_types, self.slot_save, slots_endpoint, id_slot)


def read_meminfo(path: str | Path = "/proc/meminfo") -> dict[str, int]:
    """``MemTotal`` and ``MemAvailable`` in MiB (0 when unreadable)."""
    values = {"total_mib": 0, "available_mib": 0}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                values["total_mib" if key == "MemTotal" else "available_mib"] = int(rest.split()[0]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return values


@dataclass(frozen=True)
class HardwareEnvelope:
    """The machine as capacity planning sees it, at one moment."""

    gpu_name: str
    compute_cap: str
    vram_total_mib: int
    vram_free_mib: int
    cpu_count: int
    ram_total_mib: int
    ram_free_mib: int
    disk_free_mib: int
    llama: LlamaCapabilities = field(default_factory=LlamaCapabilities)

    @property
    def family(self) -> str | None:
        """The GPU family name, used only to pick starting candidates."""
        return gpu_family(self.gpu_name)

    @classmethod
    def from_hardware(
        cls,
        hardware: Hardware,
        *,
        vram_free: int | None = None,
        ram_free: int | None = None,
        llama: LlamaCapabilities | None = None,
    ) -> HardwareEnvelope:
        return cls(
            gpu_name=hardware.gpu_name,
            compute_cap=hardware.compute_cap,
            vram_total_mib=hardware.vram_mib,
            vram_free_mib=hardware.vram_mib if vram_free is None else vram_free,
            cpu_count=hardware.cpu_count,
            ram_total_mib=hardware.ram_mib,
            ram_free_mib=hardware.ram_mib if ram_free is None else ram_free,
            disk_free_mib=hardware.disk_free_mib,
            llama=llama or LlamaCapabilities(),
        )

    @classmethod
    def detect(
        cls,
        *,
        hardware: Hardware | None = None,
        disk_path: str | Path = "/content",
        help_text: str = "",
        build: str | None = None,
    ) -> HardwareEnvelope:
        """Measure now: ``detect_hardware`` plus free VRAM, available RAM, and free disk."""
        hardware = hardware or detect_hardware(disk_path=disk_path)
        meminfo = read_meminfo()
        envelope = cls.from_hardware(
            hardware,
            vram_free=vram_free_mib(),
            ram_free=meminfo["available_mib"] or None,
            llama=LlamaCapabilities.from_help(help_text, build=build),
        )
        return envelope

    def hardware_key(self) -> dict[str, Any]:
        """The stable part (no free-memory readings), for caching capacity plans."""
        return {
            "gpu": self.gpu_name, "family": self.family, "compute_cap": self.compute_cap,
            "vram_total_mib": self.vram_total_mib, "cpu_count": self.cpu_count, "ram_total_mib": self.ram_total_mib,
        }

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["family"] = self.family
        value["llama"]["kv_cache_types"] = list(self.llama.kv_cache_types)
        return value


@dataclass(frozen=True)
class ResourceBudget:
    """Margins and CPU/IO partitions for one envelope. Every number is derived, none hardcoded per GPU."""

    usable_vram_mib: int
    vram_margin_mib: int
    ram_budget_mib: int
    ram_margin_mib: int
    disk_budget_mib: int
    setup_cpu_workers: int
    llama_threads: int
    orchestration_cpus: int
    live_cpu_workers: int
    io_threads: int
    embedding_workers: int
    embedding_threads: int
    worker_ram_mib: int = WORKER_RAM_MIB

    @classmethod
    def from_envelope(
        cls,
        envelope: HardwareEnvelope,
        *,
        vram_margin_mib: int = DEFAULT_VRAM_MARGIN_MIB,
        ram_margin_mib: int = DEFAULT_RAM_MARGIN_MIB,
        worker_ram_mib: int = WORKER_RAM_MIB,
        disk_reserve_mib: int = 2048,
    ) -> ResourceBudget:
        """Setup pools may use every RAM-safe vCPU before the server starts; live pools keep room for
        llama-server and the orchestrating process."""
        cpus = max(1, envelope.cpu_count)
        ram_budget = max(0, envelope.ram_free_mib - ram_margin_mib)
        ram_workers = max(1, ram_budget // max(1, worker_ram_mib))
        llama_threads = max(1, cpus // 4)
        orchestration = 1
        return cls(
            usable_vram_mib=max(0, envelope.vram_free_mib - vram_margin_mib),
            vram_margin_mib=vram_margin_mib,
            ram_budget_mib=ram_budget,
            ram_margin_mib=ram_margin_mib,
            disk_budget_mib=max(0, envelope.disk_free_mib - disk_reserve_mib),
            setup_cpu_workers=max(1, min(cpus, ram_workers)),
            llama_threads=llama_threads,
            orchestration_cpus=orchestration,
            live_cpu_workers=max(1, min(cpus - llama_threads - orchestration, ram_workers)),
            io_threads=min(4, max(2, cpus // 2)),
            embedding_workers=1,
            embedding_threads=max(1, min(2, cpus - llama_threads - orchestration)),
            worker_ram_mib=worker_ram_mib,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Sampling


def read_cpu_times(path: str | Path = "/proc/stat") -> tuple[int, int] | None:
    """System-wide ``(busy, total)`` jiffies from the first ``cpu`` line, or None."""
    try:
        fields = Path(path).read_text().splitlines()[0].split()
    except (OSError, IndexError):
        return None
    if not fields or fields[0] != "cpu":
        return None
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
    return sum(values) - idle, sum(values)


def read_process_cpu_seconds(pid: int, proc: str | Path = "/proc") -> float | None:
    """utime + stime of one process (from ``/proc/<pid>/stat``, never its environment or command line)."""
    try:
        text = (Path(proc) / str(pid) / "stat").read_text()
    except OSError:
        return None
    try:
        after_name = text[text.rindex(")") + 2:].split()
        utime, stime = int(after_name[11]), int(after_name[12])
    except (ValueError, IndexError):
        return None
    return (utime + stime) / os.sysconf("SC_CLK_TCK")


class GpuSampler:
    """One managed ``nvidia-smi -lms`` process; a reader thread keeps only the newest row.

    ``lines`` replaces the subprocess in tests (any iterable of CSV rows).
    """

    QUERY = "utilization.gpu,memory.used,memory.total"

    def __init__(self, *, interval_ms: int = 500, lines: Any = None) -> None:
        self.latest: dict[str, int] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._lines = lines
        self._interval_ms = interval_ms
        self._thread: threading.Thread | None = None

    def start(self) -> GpuSampler:
        source = self._lines
        if source is None:
            if shutil.which("nvidia-smi") is None:
                return self
            self._process = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits", f"-lms={self._interval_ms}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            source = self._process.stdout
        self._thread = threading.Thread(target=self._read, args=(source,), name="tidemark-gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def _read(self, source: Any) -> None:
        for line in source:
            parts = [part.strip() for part in line.split(",")]
            try:
                utilization, used, total = (int(float(part)) for part in parts[:3])
            except ValueError:
                continue
            self.latest = {"gpu_util": utilization, "gpu_mem_used_mib": used, "gpu_mem_total_mib": total}

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)


class ResourceSampler:
    """Append one resource sample every ``interval`` seconds to ``path`` (JSONL) on a daemon thread.

    ``snapshot`` returns the scheduler's counts (``Scheduler.snapshot``);
    ``pid_groups`` returns named process groups to account CPU for (for example
    ``{"llama_server": [pid], "cpu_pool": [...]}``); ``drive_lag`` returns the
    seconds the oldest pending Drive copy has waited.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        interval: float = 0.5,
        snapshot: Callable[[], Mapping[str, Any]] | None = None,
        gpu: GpuSampler | None = None,
        pid_groups: Callable[[], Mapping[str, Sequence[int]]] | None = None,
        drive_lag: Callable[[], float | None] | None = None,
        disk_path: str | Path = "/content",
        clock: Callable[[], float] = time.monotonic,
        proc: str | Path = "/proc",
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.path = Path(path)
        self.interval = interval
        self._snapshot = snapshot
        self._gpu = gpu
        self._pid_groups = pid_groups
        self._drive_lag = drive_lag
        self._disk_path = disk_path
        self._clock = clock
        self._proc = Path(proc)
        self._phase = "setup"
        self._started = clock()
        self._last_time = self._started
        self._last_cpu = read_cpu_times(self._proc / "stat")
        self._last_pid_cpu: dict[int, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.samples_written = 0

    def set_phase(self, phase: str) -> None:
        """Label later samples: ``setup``, ``autotune``, ``scored``, ``drain``, or a custom phase."""
        self._phase = phase

    def sample(self) -> dict[str, Any]:
        """Take and append one sample now."""
        with self._lock:
            now = self._clock()
            elapsed = max(1e-9, now - self._last_time)
            cpu = read_cpu_times(self._proc / "stat")
            cpu_util = None
            if cpu is not None and self._last_cpu is not None and cpu[1] > self._last_cpu[1]:
                cpu_util = (cpu[0] - self._last_cpu[0]) / (cpu[1] - self._last_cpu[1])
            self._last_cpu = cpu
            groups: dict[str, float | None] = {}
            for name, pids in (self._pid_groups() if self._pid_groups else {}).items():
                used = 0.0
                known = False
                for pid in pids:
                    seconds = read_process_cpu_seconds(pid, self._proc)
                    if seconds is None:
                        continue
                    previous = self._last_pid_cpu.get(pid)
                    self._last_pid_cpu[pid] = seconds
                    if previous is not None:
                        used += max(0.0, seconds - previous)
                        known = True
                groups[name] = used / elapsed if known else None  # cores in use over the interval
            self._last_time = now
            meminfo = read_meminfo(self._proc / "meminfo")
            try:
                load = [round(value, 2) for value in os.getloadavg()]
            except OSError:
                load = None
            try:
                disk = free_disk_mib(self._disk_path)
            except OSError:
                disk = None
            record: dict[str, Any] = {
                "t": round(now - self._started, 3),
                "wall": datetime.now(timezone.utc).isoformat(),
                "phase": self._phase,
                "cpu_total_util": None if cpu_util is None else round(cpu_util, 4),
                "cpu_cores": {name: None if value is None else round(value, 4) for name, value in groups.items()},
                "loadavg": load,
                "ram_total_mib": meminfo["total_mib"] or None,
                "ram_available_mib": meminfo["available_mib"] or None,
                "disk_free_mib": disk,
                **((self._gpu.latest or {}) if self._gpu is not None else {}),
                **dict(self._snapshot() if self._snapshot else {}),
                "drive_lag_s": self._drive_lag() if self._drive_lag else None,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            self.samples_written += 1
            return record

    def start(self) -> ResourceSampler:
        if self._gpu is not None:
            self._gpu.start()
        self._thread = threading.Thread(target=self._run, name="tidemark-resource-sampler", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except Exception:  # a failed sample must never stop a run; the gap shows in the file
                continue

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5 + self.interval)
        if self._gpu is not None:
            self._gpu.stop()

    def __enter__(self) -> ResourceSampler:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def read_samples(path: str | Path) -> list[dict[str, Any]]:
    """The samples in a ``resource_samples.jsonl`` (a torn last line is skipped)."""
    samples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            samples.append(json.loads(line))
        except ValueError:
            continue
    return samples


# Window classification and summaries

SETUP_PHASES = frozenset({"setup", "drain"})


def classify_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Which work was ready in one sample, and whether a ready resource sat idle.

    Model side: ``model_ready`` when busy + waiting calls could fill every usable
    slot; ``sequential`` when some model work existed but not enough to fill them
    (a dependency limit, not a scheduler failure); ``model_idle`` otherwise.
    CPU side likewise with running + waiting CPU tasks against CPU capacity.
    Setup and drain phases are labeled as such.
    """
    phase = sample.get("phase")
    busy, usable = int(sample.get("slots_busy") or 0), int(sample.get("slots_usable") or 0)
    waiting = int(sample.get("model_waiting") or 0)
    running, queued = int(sample.get("cpu_running") or 0), int(sample.get("cpu_waiting") or 0)
    capacity = int(sample.get("cpu_capacity") or 0)
    demand = busy + waiting
    model = "model_ready" if usable and demand >= usable else "sequential" if demand else "model_idle"
    cpu_demand = running + queued
    cpu = "cpu_ready" if capacity and cpu_demand >= capacity else "cpu_partial" if cpu_demand else "cpu_idle"
    return {
        "phase": phase,
        "window": phase if phase in SETUP_PHASES else model,
        "model": model,
        "cpu": cpu,
        "model_ready_but_idle": waiting > 0 and busy < usable,
        "cpu_ready_but_idle": queued > 0 and running < capacity,
    }


def _runs(flags: Sequence[bool], times: Sequence[float], minimum: int) -> list[dict[str, float]]:
    runs, start = [], None
    for index, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= minimum:
                runs.append({"start_t": times[start], "end_t": times[index - 1], "samples": index - start})
            start = None
    return runs


def summarize_utilization(
    samples: Sequence[Mapping[str, Any]],
    *,
    warmup_s: float = 10.0,
    sustained_samples: int = 3,
    phases: Sequence[str] = ("autotune", "scored", "smoke"),
) -> dict[str, Any]:
    """Occupancy and CPU utilization where work was ready, sustained idle windows, and likely bottlenecks."""
    measured = [s for s in samples if s.get("phase") in phases and float(s.get("t") or 0) >= warmup_s]
    classes = [classify_sample(s) for s in measured]
    occupancy = [
        int(s.get("slots_busy") or 0) / int(s["slots_usable"])
        for s, c in zip(measured, classes) if c["model"] == "model_ready" and s.get("slots_usable")
    ]
    cpu_values = []
    for s, c in zip(measured, classes):
        cores = (s.get("cpu_cores") or {}).get("cpu_pool")
        if c["cpu"] == "cpu_ready" and cores is not None and s.get("cpu_capacity"):
            cpu_values.append(min(1.0, cores / int(s["cpu_capacity"])))
    times = [float(s.get("t") or 0) for s in measured]
    model_idle_runs = _runs([c["model_ready_but_idle"] for c in classes], times, sustained_samples)
    cpu_idle_runs = _runs([c["cpu_ready_but_idle"] for c in classes], times, sustained_samples)
    counts: dict[str, int] = {}
    for c in classes:
        counts[c["window"]] = counts.get(c["window"], 0) + 1
    gpu_median = statistics.median(occupancy) if occupancy else None
    cpu_median = statistics.median(cpu_values) if cpu_values else None
    explanations = []
    if measured and counts.get("sequential", 0) > counts.get("model_ready", 0):
        explanations.append("sequential_dependencies: most samples had fewer ready model calls than slots")
    if any(c["model_ready_but_idle"] and c["cpu"] == "cpu_ready" for c in classes):
        explanations.append("cpu_starvation: model calls waited while the CPU pool was saturated")
    gpu_utils = [s.get("gpu_util") for s in measured if s.get("gpu_util") is not None]
    if gpu_utils and statistics.median(gpu_utils) >= 95:
        explanations.append("gpu_saturation: median GPU utilization was at least 95%")
    ram = [
        1 - s["ram_available_mib"] / s["ram_total_mib"]
        for s in measured if s.get("ram_available_mib") and s.get("ram_total_mib")
    ]
    if ram and max(ram) >= 0.9:
        explanations.append("memory_pressure: host RAM use reached 90%")
    if any(int(s.get("io_waiting") or 0) > 0 for s in measured):
        explanations.append("io: I/O work waited for a thread in some samples")
    if model_idle_runs or cpu_idle_runs:
        explanations.append("unexplained_ready_but_idle: a ready resource sat idle for a sustained window")
    return {
        "samples": len(samples),
        "measured_samples": len(measured),
        "window_counts": dict(sorted(counts.items())),
        "gpu_slot_occupancy_median": gpu_median,
        "gpu_slot_occupancy_samples": len(occupancy),
        "cpu_ready_utilization_median": cpu_median,
        "cpu_ready_samples": len(cpu_values),
        "model_ready_but_idle_windows": model_idle_runs,
        "cpu_ready_but_idle_windows": cpu_idle_runs,
        "meets_gpu_target": None if gpu_median is None else gpu_median >= GPU_OCCUPANCY_TARGET,
        "meets_cpu_target": None if cpu_median is None else cpu_median >= CPU_UTILIZATION_TARGET,
        "targets": {"gpu_slot_occupancy": GPU_OCCUPANCY_TARGET, "cpu_ready_utilization": CPU_UTILIZATION_TARGET},
        "explanations": explanations,
    }


def envelope_sha256(envelope: HardwareEnvelope) -> str:
    return canonical_hash(envelope.hardware_key())


__all__ = [
    "CPU_UTILIZATION_TARGET",
    "GPU_OCCUPANCY_TARGET",
    "GpuSampler",
    "HardwareEnvelope",
    "LlamaCapabilities",
    "ResourceBudget",
    "ResourceSampler",
    "classify_sample",
    "envelope_sha256",
    "read_cpu_times",
    "read_meminfo",
    "read_process_cpu_seconds",
    "read_samples",
    "summarize_utilization",
]

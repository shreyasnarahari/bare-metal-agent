"""T4/L4 capacity plans, measured fitting, and autotune-then-freeze for Tidemark runs.

A ``CapacityPlan`` is a complete serving and scheduling shape: model-derived KV
bytes per token, weights and runtime overhead, per-slot context, total ``-c``,
agent and compactor slots, KV/flash settings, batch sizes, llama.cpp threads,
CPU/IO/embedding workers, RAM and disk ceilings, and prefetch depth.

The GPU family only picks a **starting candidate** (T4: 2 slots at 16K wide or
1 at 32K tall; L4: 4 at 16K or 2 at 32K). Estimates then reject candidates that
cannot fit the measured free VRAM (with the existing 1 GiB margin), RAM, or
disk *before* launch, and ``fit_capacity`` confirms the rest with the real
server through ``steadyhand.eval.serving.fit_server``. Out of memory costs one
agent slot at the same per-slot context. When one slot does not fit, the
result is ``did_not_fit``; per-slot context is never lowered to make a task fit.

Two phases: ``autotune`` measures candidates on an unscored workload and picks
the best correct, end-to-end tasks/minute; ``FrozenConfig`` then fixes the
whole shape for scored runs. ``ShapeGuard`` starts a new, incompatible segment
when anything about the shape changes mid-run, so timings never pair across
shapes. Plans are cached by hardware, build, model, and config hash, but every
Colab session reruns a short confirmation.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from steadyhand.eval.serving import FittedServer, ServerPreset, fit_server, server_extra_args
from toolsmith.evaluation.models import canonical_hash
from toolsmith.evaluation.runner import atomic_write_json

from ..runtime import HardwareEnvelope, ResourceBudget, ResourceSampler, read_samples, summarize_utilization
from ..scheduler import Scheduler, SchedulerPlan

MIB = 1024 * 1024
# Bytes per cached element. q8_0 and q4_0 store 32 values in 34 and 18 bytes.
KV_ELEMENT_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
# Starting candidates, not measurements: (per-slot context, agent slots).
STARTING_CANDIDATES: dict[tuple[str, str], tuple[int, int]] = {
    ("T4", "wide"): (16384, 2),
    ("T4", "tall"): (32768, 1),
    ("L4", "wide"): (16384, 4),
    ("L4", "tall"): (32768, 2),
}
L4_VRAM_MIB = 22 * 1024
T4_VRAM_MIB = 14 * 1024


@dataclass(frozen=True)
class ModelSpec:
    """Architecture numbers that decide KV size, plus the weights' size on the GPU."""

    name: str
    profile: str
    n_layer: int
    n_kv_head: int
    head_dim: int
    weights_mib: int
    runtime_overhead_mib: int = 1536  # CUDA context and compute buffers; the fit probe corrects it
    host_ram_mib: int = 2048  # llama-server's own host memory with the weights offloaded

    def kv_bytes_per_token(self, kv_cache: str) -> int:
        """K and V, every layer, every KV head: 2 * layers * kv_heads * head_dim * bytes."""
        try:
            element = KV_ELEMENT_BYTES[kv_cache]
        except KeyError:
            raise ValueError(f"Unknown KV cache type {kv_cache!r}") from None
        return int(round(2 * self.n_layer * self.n_kv_head * self.head_dim * element))

    def with_gguf_size(self, size_bytes: int) -> ModelSpec:
        return replace(self, weights_mib=-(-size_bytes // MIB))


# Qwen3-8B: 36 layers, 8 KV heads of 128 (config.json); Q4_K_M weights are about 4.7 GiB.
QWEN3_8B = ModelSpec("Qwen3-8B-Q4_K_M", "qwen3", n_layer=36, n_kv_head=8, head_dim=128, weights_mib=4800)


@dataclass(frozen=True)
class CapacityPlan:
    """One complete serving and scheduling shape (see the module docstring)."""

    gpu_family: str | None
    model: str
    profile: str
    shape: str
    kv_bytes_per_token: int
    weights_mib: int
    runtime_overhead_mib: int
    per_slot_context: int
    agent_slots: int
    compactor_slots: int
    kv_cache: str
    flash_attn: bool
    batch_size: int
    ubatch_size: int
    llama_threads: int
    cpu_process_workers: int
    io_threads: int
    embedding_workers: int
    embedding_threads: int
    episode_workers: int
    affinity: str
    prefetch_depth: int
    ram_ceiling_mib: int
    disk_ceiling_mib: int
    vram_margin_mib: int
    host_ram_mib: int = 2048
    worker_ram_mib: int = 1024

    @property
    def total_slots(self) -> int:
        return self.agent_slots + self.compactor_slots

    @property
    def total_context(self) -> int:
        """llama-server ``-c``: llama.cpp splits it evenly over ``--parallel`` slots."""
        return self.per_slot_context * self.total_slots

    @property
    def kv_mib(self) -> int:
        return -(-self.kv_bytes_per_token * self.total_context // MIB)

    @property
    def vram_estimate_mib(self) -> int:
        return self.weights_mib + self.runtime_overhead_mib + self.kv_mib

    @property
    def ram_estimate_mib(self) -> int:
        return self.host_ram_mib + self.cpu_process_workers * self.worker_ram_mib + self.embedding_workers * 1024

    def shrink(self) -> CapacityPlan:
        """One agent slot fewer at the same per-slot context (the response to low VRAM or OOM)."""
        if self.agent_slots <= 1:
            raise ValueError("Already at one agent slot")
        agent = self.agent_slots - 1
        return replace(self, agent_slots=agent, episode_workers=agent + max(1, agent // 2))

    def preset(self) -> ServerPreset:
        """The ``ServerPreset`` that ``fit_server`` starts (``parallel`` counts compactor slots too)."""
        return ServerPreset(self.gpu_family or "other", self.profile, self.shape, self.total_context, self.total_slots, "tidemark")

    def with_fit(self, fitted: FittedServer) -> CapacityPlan:
        """The plan the server actually runs after fitting (rung and slot count from the probe)."""
        agent = max(1, fitted.preset.parallel - self.compactor_slots)
        if fitted.preset.per_slot_context != self.per_slot_context:
            raise ValueError("Fitting changed the per-slot context; that would change the tasks")
        return replace(
            self, agent_slots=agent, kv_cache=fitted.rung["kv_cache"], flash_attn=fitted.rung["flash_attn"],
            episode_workers=agent + max(1, agent // 2),
        )

    def server_args(self, help_text: str) -> list[str]:
        return server_extra_args(
            help_text, cpu_count=self.llama_threads, kv_cache=self.kv_cache, flash_attn=self.flash_attn,
            batch_size=self.batch_size, ubatch_size=self.ubatch_size,
        )

    def scheduler_plan(self) -> SchedulerPlan:
        return SchedulerPlan(
            agent_slots=self.agent_slots, compactor_slots=self.compactor_slots, affinity=self.affinity,
            episode_workers=self.episode_workers, cpu_workers=self.cpu_process_workers, io_threads=self.io_threads,
            prefetch_depth=self.prefetch_depth,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "total_slots": self.total_slots,
            "total_context": self.total_context,
            "kv_mib": self.kv_mib,
            "vram_estimate_mib": self.vram_estimate_mib,
            "ram_estimate_mib": self.ram_estimate_mib,
        }

    def sha256(self) -> str:
        return canonical_hash(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapacityPlan:
        names = {item for item in cls.__dataclass_fields__}
        return cls(**{key: item for key, item in value.items() if key in names})


@dataclass(frozen=True)
class CapacityDecision:
    """A candidate that fits the estimates (``plan``), or ``did_not_fit`` with the reasons."""

    status: str  # "fits" or "did_not_fit"
    plan: CapacityPlan | None
    reasons: tuple[str, ...] = ()
    shrinks: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "plan": self.plan.as_dict() if self.plan is not None else None,
            "reasons": list(self.reasons),
            "shrinks": self.shrinks,
        }


def starting_candidate(envelope: HardwareEnvelope, shape: str) -> tuple[int, int]:
    """The (per-slot context, agent slots) to start from; another card borrows T4 or L4 rows by VRAM."""
    family = envelope.family
    if family is None:
        family = "L4" if envelope.vram_total_mib >= L4_VRAM_MIB else "T4" if envelope.vram_total_mib >= T4_VRAM_MIB else None
    if family is None or (family, shape) not in STARTING_CANDIDATES:
        raise LookupError(f"No starting candidate for {envelope.gpu_name!r} ({envelope.vram_total_mib} MiB), shape {shape!r}")
    return STARTING_CANDIDATES[(family, shape)]


def plan_capacity(
    envelope: HardwareEnvelope,
    model: ModelSpec,
    *,
    shape: str = "wide",
    kv_cache: str = "q8_0",
    flash_attn: bool = True,
    compactor: str = "shared",
    affinity: str = "pinned",
    batch_size: int = 2048,
    ubatch_size: int = 512,
    prefetch_depth: int = 2,
    budget: ResourceBudget | None = None,
    per_slot_context: int | None = None,
    agent_slots: int | None = None,
) -> CapacityDecision:
    """Fit a starting candidate to the envelope's estimates; shrink slots before anything else."""
    if compactor not in ("shared", "dedicated"):
        raise ValueError("compactor must be 'shared' or 'dedicated'")
    budget = budget or ResourceBudget.from_envelope(envelope)
    start_context, start_slots = starting_candidate(envelope, shape)
    context = per_slot_context or start_context
    slots = agent_slots or start_slots
    plan = CapacityPlan(
        gpu_family=envelope.family, model=model.name, profile=model.profile, shape=shape,
        kv_bytes_per_token=model.kv_bytes_per_token(kv_cache), weights_mib=model.weights_mib,
        runtime_overhead_mib=model.runtime_overhead_mib, per_slot_context=context, agent_slots=slots,
        compactor_slots=1 if compactor == "dedicated" else 0, kv_cache=kv_cache, flash_attn=flash_attn,
        batch_size=batch_size, ubatch_size=ubatch_size, llama_threads=budget.llama_threads,
        cpu_process_workers=budget.live_cpu_workers, io_threads=budget.io_threads,
        embedding_workers=budget.embedding_workers, embedding_threads=budget.embedding_threads,
        episode_workers=slots + max(1, slots // 2), affinity=affinity, prefetch_depth=prefetch_depth,
        ram_ceiling_mib=budget.ram_budget_mib, disk_ceiling_mib=budget.disk_budget_mib,
        vram_margin_mib=budget.vram_margin_mib, host_ram_mib=model.host_ram_mib, worker_ram_mib=budget.worker_ram_mib,
    )
    if kv_cache != "f16" and not flash_attn:
        return CapacityDecision("did_not_fit", None, ("a quantized V cache needs flash attention",))
    reasons: list[str] = []
    shrinks = 0
    while plan.vram_estimate_mib > budget.usable_vram_mib:
        if plan.agent_slots <= 1:
            return CapacityDecision("did_not_fit", None, (
                f"one {context}-token slot needs about {plan.vram_estimate_mib} MiB; "
                f"{budget.usable_vram_mib} MiB is usable after the {budget.vram_margin_mib} MiB margin",
            ), shrinks)
        plan = plan.shrink()
        shrinks += 1
        reasons.append(f"shrank to {plan.agent_slots} agent slots for VRAM")
    while plan.ram_estimate_mib > plan.ram_ceiling_mib and plan.cpu_process_workers > 1:
        plan = replace(plan, cpu_process_workers=plan.cpu_process_workers - 1)
        reasons.append(f"reduced CPU workers to {plan.cpu_process_workers} for RAM")
    if plan.ram_estimate_mib > plan.ram_ceiling_mib:
        return CapacityDecision("did_not_fit", None, (
            f"needs about {plan.ram_estimate_mib} MiB of host RAM; the budget is {plan.ram_ceiling_mib} MiB",
        ), shrinks)
    if plan.disk_ceiling_mib <= 0:
        return CapacityDecision("did_not_fit", None, ("no local disk left after the reserve",), shrinks)
    return CapacityDecision("fits", plan, tuple(reasons), shrinks)


def capacity_candidates(
    envelope: HardwareEnvelope,
    model: ModelSpec,
    rungs: Sequence[Mapping[str, Any]],
    *,
    shapes: Sequence[str] = ("wide",),
    compactors: Sequence[str] = ("shared", "dedicated"),
    affinities: Sequence[str] = ("pinned",),
    batches: Sequence[tuple[int, int]] = ((2048, 512),),
) -> list[CapacityDecision]:
    """Every (shape, KV rung, compactor policy, affinity, batch) decision, did-not-fit ones included."""
    decisions = []
    for shape in shapes:
        for rung in rungs:
            for compactor in compactors:
                for affinity in affinities:
                    for batch, ubatch in batches:
                        decisions.append(plan_capacity(
                            envelope, model, shape=shape, kv_cache=rung["kv_cache"], flash_attn=rung["flash_attn"],
                            compactor=compactor, affinity=affinity, batch_size=batch, ubatch_size=ubatch,
                        ))
    return decisions


def fit_capacity(
    plan: CapacityPlan,
    *,
    help_text: str,
    start: Callable[[ServerPreset, list[str]], Any],
    query_gpu: Callable[[], Mapping[str, Any]],
    probe: Callable[[Any], float | None],
    log: Callable[[str], None] = print,
) -> tuple[FittedServer, CapacityPlan]:
    """Start the plan with ``fit_server`` (its KV/flash ladder from this rung down, slots shrunk on OOM)."""
    from steadyhand.eval.serving import kv_ladder

    ladder = kv_ladder(help_text)
    rung = {"kv_cache": plan.kv_cache, "flash_attn": plan.flash_attn}
    rungs = ladder[ladder.index(rung):] if rung in ladder else ladder
    fitted = fit_server(
        plan.preset(), rungs, start=start,
        extra_args_for=lambda candidate: replace(plan, kv_cache=candidate["kv_cache"], flash_attn=candidate["flash_attn"]).server_args(help_text),
        query_gpu=query_gpu, probe=probe, min_free_mib=plan.vram_margin_mib, log=log,
    )
    return fitted, plan.with_fit(fitted)


@dataclass(frozen=True)
class TuneMeasurement:
    """One candidate measured on the unscored autotune workload."""

    plan_sha256: str
    fits: bool
    correct: bool
    tasks_per_minute: float | None
    prompt_tps: float | None = None
    decode_tps: float | None = None
    queue_delay_s: float | None = None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "notes": list(self.notes)}


@dataclass(frozen=True)
class FrozenConfig:
    """The selected complete configuration. Scored runs record its hash and never change it."""

    plan: CapacityPlan
    hardware_key: dict[str, Any]
    build: str | None
    measurements: tuple[TuneMeasurement, ...] = ()
    selected_by: str = "autotune"

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "plan_sha256": self.plan.sha256(),
            "hardware_key": dict(self.hardware_key),
            "build": self.build,
            "selected_by": self.selected_by,
            "measurements": [m.as_dict() for m in self.measurements],
        }

    def sha256(self) -> str:
        return canonical_hash({"plan": self.plan.sha256(), "hardware_key": self.hardware_key, "build": self.build})


def select_candidate(plans: Sequence[CapacityPlan], measurements: Sequence[TuneMeasurement]) -> CapacityPlan:
    """The fitting, correct candidate with the most tasks/minute (ties: decode speed, then hash)."""
    by_hash = {plan.sha256(): plan for plan in plans}
    eligible = [m for m in measurements if m.fits and m.correct and m.tasks_per_minute is not None and m.plan_sha256 in by_hash]
    if not eligible:
        raise RuntimeError("No autotune candidate fitted and completed the workload correctly")
    best = max(eligible, key=lambda m: (m.tasks_per_minute, m.decode_tps or 0.0, m.plan_sha256))
    return by_hash[best.plan_sha256]


def autotune(
    plans: Sequence[CapacityPlan],
    evaluate: Callable[[CapacityPlan], TuneMeasurement],
    *,
    hardware_key: Mapping[str, Any],
    build: str | None,
) -> FrozenConfig:
    """Measure every candidate (unscored), then freeze the best one."""
    measurements = tuple(evaluate(plan) for plan in plans)
    selected = select_candidate(plans, measurements)
    return FrozenConfig(selected, dict(hardware_key), build, measurements)


@dataclass
class ShapeGuard:
    """Keeps scored records in segments of one frozen shape. A change opens a new, incompatible segment."""

    frozen_sha256: str
    segments: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.segments:
            self.segments.append({"segment": 0, "shape_sha256": self.frozen_sha256, "reason": "frozen"})

    @property
    def current(self) -> dict[str, Any]:
        return self.segments[-1]

    def observe(self, shape_sha256: str, *, reason: str = "shape_change") -> int:
        """The segment index for records made under ``shape_sha256``."""
        if shape_sha256 != self.current["shape_sha256"]:
            self.segments.append({"segment": len(self.segments), "shape_sha256": shape_sha256, "reason": reason})
        return self.current["segment"]

    @staticmethod
    def comparable(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        """Timing pairs are valid only within one segment and one shape."""
        return left.get("segment") == right.get("segment") and left.get("shape_sha256") == right.get("shape_sha256")


class CapacityCache:
    """Capacity decisions on disk, keyed by hardware, build, model, and config hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def key(hardware_key: Mapping[str, Any], build: str | None, model: ModelSpec, config: Mapping[str, Any]) -> str:
        return canonical_hash({"hardware": dict(hardware_key), "build": build, "model": asdict(model), "config": dict(config)})

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> CapacityPlan | None:
        value = self._load().get(key)
        return CapacityPlan.from_dict(value) if value is not None else None

    def put(self, key: str, plan: CapacityPlan) -> None:
        entries = self._load()
        entries[key] = asdict(plan)
        atomic_write_json(self.path, entries)


def utilization_smoke(
    run: Callable[[], Sequence[Any]],
    *,
    scheduler: Scheduler,
    sampler: ResourceSampler,
    warmup_s: float = 5.0,
    sustained_samples: int = 3,
) -> dict[str, Any]:
    """Run ``run()`` (episodes through ``scheduler``) under the ``smoke`` phase and summarize the samples.

    The summary counts only windows with ready work; it never adds artificial load.
    """
    sampler.set_phase("smoke")
    started = time.monotonic()
    try:
        results = list(run())
    finally:
        elapsed = time.monotonic() - started
        sampler.sample()
        sampler.set_phase("drain")
    completed = [r for r in results if r is not None]
    return {
        "elapsed_s": round(elapsed, 3),
        "episodes": len(completed),
        "episodes_per_minute": len(completed) / (elapsed / 60) if elapsed > 0 else None,
        "scheduler": scheduler.plan.as_dict(),
        "slot_stats": scheduler.slots.stats(),
        "utilization": summarize_utilization(
            read_samples(sampler.path), warmup_s=warmup_s, sustained_samples=sustained_samples, phases=("smoke",),
        ),
    }


__all__ = [
    "CapacityCache",
    "CapacityDecision",
    "CapacityPlan",
    "FrozenConfig",
    "KV_ELEMENT_BYTES",
    "ModelSpec",
    "QWEN3_8B",
    "STARTING_CANDIDATES",
    "ShapeGuard",
    "TuneMeasurement",
    "autotune",
    "capacity_candidates",
    "fit_capacity",
    "plan_capacity",
    "select_candidate",
    "starting_candidate",
    "utilization_smoke",
]

"""Capacity plans for T4 and L4, measured fitting, autotune-then-freeze, shape segments, and the plan cache."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from tidemark.eval.serving import (
    QWEN3_8B,
    CapacityCache,
    ShapeGuard,
    TuneMeasurement,
    autotune,
    capacity_candidates,
    fit_capacity,
    plan_capacity,
    select_candidate,
    starting_candidate,
    utilization_smoke,
)
from tidemark.runtime import HardwareEnvelope, ResourceBudget, ResourceSampler
from tidemark.scheduler import Scheduler
from toolsmith.hardware import Hardware

HELP = "--cache-type-k TYPE (f16, q8_0, q4_0)\n--flash-attn FA  on|off|auto\n"


def envelope(name="Tesla T4", vram=15360, free=None, cpus=2, ram=12_982, ram_free=11_500, disk=80_000):
    hardware = Hardware(gpu_name=name, compute_cap="75", vram_mib=vram, cpu_count=cpus, ram_mib=ram, disk_free_mib=disk)
    return HardwareEnvelope.from_hardware(hardware, vram_free=free if free is not None else vram - 400, ram_free=ram_free)


T4 = envelope()
L4 = envelope("NVIDIA L4", vram=23_034, cpus=8, ram=54_000, ram_free=50_000, disk=150_000)


def test_kv_bytes_come_from_the_model_architecture():
    assert QWEN3_8B.kv_bytes_per_token("f16") == 2 * 36 * 8 * 128 * 2 == 147_456
    assert QWEN3_8B.kv_bytes_per_token("q8_0") == 78_336
    with pytest.raises(ValueError):
        QWEN3_8B.kv_bytes_per_token("q2")
    assert QWEN3_8B.with_gguf_size(5_027_782_336).weights_mib == 4795


@pytest.mark.parametrize(
    ("hardware", "shape", "context", "slots"),
    [(T4, "wide", 16384, 2), (T4, "tall", 32768, 1), (L4, "wide", 16384, 4), (L4, "tall", 32768, 2)],
)
def test_starting_candidates_fit_both_gpus(hardware, shape, context, slots):
    decision = plan_capacity(hardware, QWEN3_8B, shape=shape, kv_cache="f16")
    assert decision.status == "fits" and decision.shrinks == 0
    plan = decision.plan
    assert (plan.per_slot_context, plan.agent_slots) == (context, slots)
    assert plan.total_context == context * slots and plan.vram_estimate_mib <= hardware.vram_free_mib - 1024
    assert plan.episode_workers == slots + max(1, slots // 2)
    preset = plan.preset()
    assert preset.context_size == plan.total_context and preset.parallel == slots and preset.per_slot_context == context


def test_live_cpu_pools_are_derived_not_copied_between_gpus():
    small, large = plan_capacity(T4, QWEN3_8B).plan, plan_capacity(L4, QWEN3_8B).plan
    assert (small.cpu_process_workers, small.io_threads, small.llama_threads) == (1, 2, 1)
    assert (large.cpu_process_workers, large.io_threads, large.llama_threads) == (5, 4, 2)
    assert small.scheduler_plan().cpu_workers == 1 and large.scheduler_plan().agent_slots == 4


def test_low_vram_drops_slots_but_never_per_slot_context():
    crowded = envelope(free=10_000)  # one 16K f16 slot needs ~8640 MiB, two ~10944; 8976 usable
    decision = plan_capacity(crowded, QWEN3_8B, kv_cache="f16")
    assert decision.status == "fits" and decision.plan.agent_slots == 1 and decision.plan.per_slot_context == 16384
    assert decision.shrinks == 1 and "shrank" in decision.reasons[0]
    with pytest.raises(ValueError):
        decision.plan.shrink()


def test_a_shape_that_does_not_fit_one_slot_is_did_not_fit():
    decision = plan_capacity(envelope(free=7_000), QWEN3_8B, shape="tall", kv_cache="f16")
    assert decision.status == "did_not_fit" and decision.plan is None and "32768-token slot" in decision.reasons[0]
    assert plan_capacity(T4, QWEN3_8B, kv_cache="q8_0", flash_attn=False).status == "did_not_fit"


def test_ram_and_disk_ceilings_reject_unsafe_candidates_before_launch():
    tight = plan_capacity(L4, QWEN3_8B, budget=replace(ResourceBudget.from_envelope(L4), ram_budget_mib=4_200))
    assert tight.status == "fits" and tight.plan.cpu_process_workers == 1 and "RAM" in tight.reasons[-1]
    starved = plan_capacity(L4, QWEN3_8B, budget=replace(ResourceBudget.from_envelope(L4), ram_budget_mib=2_000))
    assert starved.status == "did_not_fit" and "host RAM" in starved.reasons[0]
    full_disk = plan_capacity(envelope(disk=1_000), QWEN3_8B)
    assert full_disk.status == "did_not_fit" and "disk" in full_disk.reasons[0]


def test_other_cards_borrow_rows_by_vram_and_tiny_cards_have_none():
    assert starting_candidate(envelope("NVIDIA A100-SXM4-40GB", vram=40_960), "wide") == (16384, 4)
    assert starting_candidate(envelope("Tesla V100-SXM2-16GB", vram=16_160), "tall") == (32768, 1)
    with pytest.raises(LookupError):
        starting_candidate(envelope("GTX 1060", vram=6_144), "wide")


def test_the_candidate_grid_includes_dedicated_compactors_and_misses():
    rungs = [{"kv_cache": "q8_0", "flash_attn": True}, {"kv_cache": "f16", "flash_attn": False}]
    decisions = capacity_candidates(T4, QWEN3_8B, rungs, shapes=("wide", "tall"))
    assert len(decisions) == 8
    dedicated = [d.plan for d in decisions if d.plan is not None and d.plan.compactor_slots == 1]
    assert dedicated and all(plan.total_slots == plan.agent_slots + 1 for plan in dedicated)


def test_fitting_shrinks_on_oom_at_the_same_per_slot_context():
    plan = plan_capacity(L4, QWEN3_8B, kv_cache="q8_0").plan
    started = []

    class Server:
        def __init__(self, preset):
            self.preset = preset

        def stop(self, timeout=None):
            pass

    def start(preset, extra_args):
        started.append((preset.parallel, preset.per_slot_context, tuple(extra_args)))
        if preset.parallel > 3:
            raise RuntimeError("CUDA error: out of memory")
        return Server(preset)

    fitted, fitted_plan = fit_capacity(
        plan, help_text=HELP, start=start, log=lambda message: None,
        query_gpu=lambda: {"memory_total_mib": 23_034, "memory_used_mib": 15_000}, probe=lambda server: 40.0,
    )
    assert [s[:2] for s in started] == [(4, 16384), (3, 16384)]
    assert "--cache-type-k" in started[0][2] and "q8_0" in started[0][2]
    assert fitted_plan.agent_slots == 3 and fitted_plan.per_slot_context == 16384 and fitted.decode_tps == 40.0


def test_autotune_picks_the_best_correct_fitting_candidate_and_freezes_it():
    plans = [d.plan for d in capacity_candidates(L4, QWEN3_8B, [{"kv_cache": "q8_0", "flash_attn": True}],
                                                 affinities=("pinned", "work_conserving")) if d.plan]
    speeds = {plan.sha256(): 10.0 + index for index, plan in enumerate(plans)}
    fastest = max(plans, key=lambda plan: speeds[plan.sha256()])

    def evaluate(plan):
        wrong = plan is fastest  # the fastest one gave wrong answers: it must lose
        return TuneMeasurement(plan.sha256(), fits=True, correct=not wrong, tasks_per_minute=speeds[plan.sha256()])

    frozen = autotune(plans, evaluate, hardware_key=L4.hardware_key(), build="b5401")
    assert frozen.plan is not fastest and frozen.plan.sha256() in speeds and len(frozen.measurements) == len(plans)
    expected = max((p for p in plans if p is not fastest), key=lambda p: speeds[p.sha256()])
    assert frozen.plan == expected and frozen.as_dict()["plan_sha256"] == expected.sha256()
    with pytest.raises(RuntimeError):
        select_candidate(plans, [TuneMeasurement(plans[0].sha256(), fits=False, correct=True, tasks_per_minute=99.0)])


def test_a_mid_run_shape_change_opens_an_incompatible_segment():
    plan = plan_capacity(T4, QWEN3_8B).plan
    guard = ShapeGuard(plan.sha256())
    assert guard.observe(plan.sha256()) == 0
    shrunk = plan.shrink()
    assert guard.observe(shrunk.sha256(), reason="oom_restart") == 1
    assert guard.observe(shrunk.sha256()) == 1 and [s["reason"] for s in guard.segments] == ["frozen", "oom_restart"]
    first = {"segment": 0, "shape_sha256": plan.sha256()}
    later = {"segment": 1, "shape_sha256": shrunk.sha256()}
    assert ShapeGuard.comparable(first, dict(first)) and not ShapeGuard.comparable(first, later)


def test_capacity_plans_are_cached_by_hardware_build_model_and_config(tmp_path):
    cache = CapacityCache(tmp_path / "plans.json")
    plan = plan_capacity(T4, QWEN3_8B).plan
    key = CapacityCache.key(T4.hardware_key(), "b5401", QWEN3_8B, {"shape": "wide"})
    assert cache.get(key) is None
    cache.put(key, plan)
    assert cache.get(key) == plan
    assert CapacityCache.key(T4.hardware_key(), "b8233", QWEN3_8B, {"shape": "wide"}) != key
    assert CapacityCache.key(L4.hardware_key(), "b5401", QWEN3_8B, {"shape": "wide"}) != key


def test_the_utilization_smoke_summarizes_only_its_own_phase(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "stat").write_text("cpu  1 0 1 10 0 0 0 0\n")
    (proc / "meminfo").write_text("MemTotal: 1000 kB\nMemAvailable: 500 kB\n")
    with Scheduler(plan_capacity(T4, QWEN3_8B).plan.scheduler_plan(), cpu_executor=lambda: ThreadPoolExecutor(1)) as scheduler:
        sampler = ResourceSampler(tmp_path / "resource_samples.jsonl", proc=proc, snapshot=scheduler.snapshot, disk_path=tmp_path)
        sampler.set_phase("setup")
        sampler.sample()

        def run():
            def work(unit, owner):
                with scheduler.slots.lease(owner):
                    sampler.sample()
                return unit

            return scheduler.run_episodes(range(4), work, key=str)

        result = utilization_smoke(run, scheduler=scheduler, sampler=sampler, warmup_s=0)
    assert result["episodes"] == 4 and result["slot_stats"]["leases"] == 4
    assert result["utilization"]["measured_samples"] >= 4 and result["scheduler"]["agent_slots"] == 2
    assert result["utilization"]["window_counts"].get("setup") is None  # setup samples are not measured

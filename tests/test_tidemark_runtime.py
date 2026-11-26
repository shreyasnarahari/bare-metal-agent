"""Hardware envelopes, resource budgets, the /proc and nvidia-smi sampler, and utilization windows."""

import json

import pytest

from tidemark.runtime import (
    GpuSampler,
    HardwareEnvelope,
    LlamaCapabilities,
    ResourceBudget,
    ResourceSampler,
    classify_sample,
    envelope_sha256,
    read_cpu_times,
    read_meminfo,
    read_process_cpu_seconds,
    read_samples,
    summarize_utilization,
)
from toolsmith.hardware import Hardware

HELP = "--cache-type-k TYPE  KV cache data type for K (f16, q8_0, q4_0)\n--flash-attn FA  on|off|auto\n--slot-save-path PATH\n"


def t4(**overrides):
    values = dict(gpu_name="Tesla T4", compute_cap="75", vram_mib=15360, cpu_count=2, ram_mib=12_982, disk_free_mib=80_000)
    return Hardware(**{**values, **overrides})


def l4(**overrides):
    values = dict(gpu_name="NVIDIA L4", compute_cap="89", vram_mib=23_034, cpu_count=8, ram_mib=54_000, disk_free_mib=150_000)
    return Hardware(**{**values, **overrides})


def test_envelopes_keep_free_readings_out_of_the_cache_key():
    a = HardwareEnvelope.from_hardware(t4(), vram_free=14_900, ram_free=11_000)
    b = HardwareEnvelope.from_hardware(t4(), vram_free=13_000, ram_free=9_000)
    assert a.family == "T4" and a.vram_free_mib == 14_900
    assert envelope_sha256(a) == envelope_sha256(b) and "vram_free_mib" not in a.hardware_key()
    assert a.as_dict()["family"] == "T4"


def test_llama_capabilities_come_from_help_and_probes():
    caps = LlamaCapabilities.from_help(HELP, build="b5401")
    assert caps.flash_attn == "value" and caps.kv_cache_types == ("f16", "q8_0", "q4_0") and caps.slot_save
    assert LlamaCapabilities.from_help("", build=None).kv_cache_types == ()
    probed = caps.with_probe(slots_endpoint=True, id_slot=True)
    assert probed.slots_endpoint and probed.id_slot and probed.build == "b5401"


def test_t4_and_l4_budgets_reserve_room_for_the_server_and_orchestration():
    small = ResourceBudget.from_envelope(HardwareEnvelope.from_hardware(t4(), vram_free=14_900, ram_free=11_500))
    large = ResourceBudget.from_envelope(HardwareEnvelope.from_hardware(l4(), vram_free=22_400, ram_free=50_000))
    assert small.setup_cpu_workers == 2 and small.live_cpu_workers == 1 and small.llama_threads == 1
    assert large.setup_cpu_workers == 8 and large.live_cpu_workers == 5 and large.llama_threads == 2
    assert small.usable_vram_mib == 14_900 - 1024 and large.io_threads == 4 and small.io_threads == 2
    assert small.embedding_threads == 1 and large.embedding_threads == 2


def test_low_ram_bounds_the_worker_counts():
    tight = ResourceBudget.from_envelope(HardwareEnvelope.from_hardware(l4(), ram_free=4_500))
    assert tight.ram_budget_mib == 4_500 - 2_048 and tight.setup_cpu_workers == 2 and tight.live_cpu_workers == 2


@pytest.fixture
def proc(tmp_path):
    root = tmp_path / "proc"
    (root / "4242").mkdir(parents=True)
    (root / "stat").write_text("cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 1 2 3 4\n")
    (root / "meminfo").write_text("MemTotal:       13000000 kB\nMemFree: 1 kB\nMemAvailable:    6500000 kB\n")
    # pid (comm with spaces) state ppid ... utime stime at fields 14 and 15.
    fields = ["S"] + ["0"] * 10 + ["300", "200"] + ["0"] * 30
    (root / "4242" / "stat").write_text(f"4242 (llama server) {' '.join(fields)}\n")
    return root


def test_proc_readers(proc):
    assert read_cpu_times(proc / "stat") == (200, 1000)
    assert read_meminfo(proc / "meminfo") == {"total_mib": 12695, "available_mib": 6347}
    assert read_process_cpu_seconds(4242, proc) == pytest.approx(500 / __import__("os").sysconf("SC_CLK_TCK"))
    assert read_process_cpu_seconds(1, proc) is None and read_cpu_times(proc / "missing") is None


def test_the_sampler_writes_labeled_samples_with_scheduler_and_gpu_counts(proc, tmp_path):
    clock = iter([0.0, 1.0, 2.0, 3.0])
    gpu = GpuSampler(lines=["87, 9000, 15360\n", "garbage\n"]).start()
    gpu._thread.join(timeout=5)
    snapshot = {"slots_busy": 2, "slots_usable": 2, "model_waiting": 1, "cpu_running": 1, "cpu_waiting": 0, "cpu_capacity": 1}
    sampler = ResourceSampler(tmp_path / "resource_samples.jsonl", proc=proc, clock=lambda: next(clock),
                              snapshot=lambda: snapshot, gpu=gpu, pid_groups=lambda: {"llama_server": [4242]},
                              drive_lag=lambda: 1.5, disk_path=tmp_path)
    sampler.set_phase("smoke")
    first = sampler.sample()
    (proc / "stat").write_text("cpu  150 0 150 750 100 0 0 0 0 0\n")
    fields = ["S"] + ["0"] * 10 + ["400", "200"] + ["0"] * 30
    (proc / "4242" / "stat").write_text(f"4242 (llama server) {' '.join(fields)}\n")
    second = sampler.sample()
    assert first["phase"] == "smoke" and first["gpu_util"] == 87 and first["slots_busy"] == 2 and first["drive_lag_s"] == 1.5
    assert first["cpu_cores"] == {"llama_server": None}  # the first reading has no interval yet
    assert second["cpu_total_util"] == pytest.approx(100 / 150, abs=1e-3)
    assert second["cpu_cores"]["llama_server"] > 0 and second["ram_available_mib"] == 6347
    samples = read_samples(sampler.path)
    assert len(samples) == 2 and all("environ" not in json.dumps(s) and "cmdline" not in json.dumps(s) for s in samples)


def test_the_sampler_thread_starts_and_stops(tmp_path, proc):
    sampler = ResourceSampler(tmp_path / "samples.jsonl", interval=0.01, proc=proc, disk_path=tmp_path)
    with sampler:
        while sampler.samples_written < 2:
            pass
    assert len(read_samples(tmp_path / "samples.jsonl")) >= 2
    with pytest.raises(ValueError):
        ResourceSampler(tmp_path / "x.jsonl", interval=0)


def sample(t, phase="smoke", busy=2, usable=2, waiting=0, cpu_running=0, cpu_waiting=0, capacity=2, pool=None, **extra):
    return {"t": t, "phase": phase, "slots_busy": busy, "slots_usable": usable, "model_waiting": waiting,
            "cpu_running": cpu_running, "cpu_waiting": cpu_waiting, "cpu_capacity": capacity,
            "cpu_cores": {"cpu_pool": pool}, **extra}


def test_windows_are_classified_by_ready_work():
    assert classify_sample(sample(1, busy=2, waiting=3))["model"] == "model_ready"
    assert classify_sample(sample(1, busy=1, waiting=0))["model"] == "sequential"
    assert classify_sample(sample(1, busy=0))["model"] == "model_idle"
    assert classify_sample(sample(1, busy=1, waiting=1))["model_ready_but_idle"]
    assert classify_sample(sample(1, phase="setup", busy=0))["window"] == "setup"
    assert classify_sample(sample(1, cpu_running=1, cpu_waiting=2))["cpu_ready_but_idle"]
    assert classify_sample(sample(1, cpu_running=2, cpu_waiting=2))["cpu"] == "cpu_ready"


def test_utilization_counts_only_ready_windows_and_finds_sustained_idle():
    samples = [sample(t, busy=2, waiting=1, cpu_running=2, pool=1.8) for t in range(10, 20)]
    samples += [sample(t, busy=1, waiting=0) for t in range(20, 30)]  # sequential: not a failure
    samples += [sample(t, phase="drain", busy=0) for t in range(30, 35)]
    summary = summarize_utilization(samples, warmup_s=5)
    assert summary["gpu_slot_occupancy_median"] == 1.0 and summary["meets_gpu_target"] is True
    assert summary["cpu_ready_utilization_median"] == pytest.approx(0.9) and summary["meets_cpu_target"] is True
    assert summary["model_ready_but_idle_windows"] == [] and summary["window_counts"] == {"model_ready": 10, "sequential": 10}
    assert "unexplained_ready_but_idle" not in " ".join(summary["explanations"])


def test_a_sustained_free_slot_with_waiting_calls_is_reported():
    samples = [sample(t, busy=1, waiting=2) for t in range(10, 16)] + [sample(16, busy=2, waiting=2)]
    summary = summarize_utilization(samples, warmup_s=0, sustained_samples=3)
    assert summary["model_ready_but_idle_windows"] == [{"start_t": 10, "end_t": 15, "samples": 6}]
    assert summary["gpu_slot_occupancy_median"] == 0.5 and summary["meets_gpu_target"] is False
    assert any(item.startswith("unexplained_ready_but_idle") for item in summary["explanations"])


def test_warmup_and_unmeasured_phases_are_excluded_and_bottlenecks_explained():
    samples = [sample(1, busy=0, waiting=4)]  # before warmup ends
    samples += [sample(t, busy=1, waiting=0, gpu_util=99, ram_total_mib=100, ram_available_mib=5, io_waiting=1)
                for t in range(10, 14)]
    summary = summarize_utilization(samples, warmup_s=5)
    assert summary["measured_samples"] == 4 and summary["gpu_slot_occupancy_median"] is None
    text = " ".join(summary["explanations"])
    assert "sequential_dependencies" in text and "gpu_saturation" in text and "memory_pressure" in text and "io:" in text
    assert summarize_utilization([], warmup_s=0)["meets_gpu_target"] is None

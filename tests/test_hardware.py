import pytest

import toolsmith.hardware as hardware
from toolsmith.hardware import (
    Hardware,
    UnsupportedRuntime,
    build_jobs_for,
    detect_hardware,
    gpu_family,
    parse_gpu_rows,
    require_supported,
    runtime_stamp,
    total_ram_mib,
)

# Colab's T4 runtime: 2 vCPUs, about 12.7 GB RAM; the L4 runtime: 8 vCPUs and far more RAM.
T4 = Hardware("Tesla T4", "75", 15360, 2, 13000, 80_000)
L4 = Hardware("NVIDIA L4", "89", 23034, 8, 54000, 200_000)


def test_parse_gpu_rows_reads_name_capability_and_memory():
    rows = parse_gpu_rows("Tesla T4, 7.5, 15360\nNVIDIA L4, 8.9, 23034\n")

    assert rows == [
        {"name": "Tesla T4", "compute_cap": "75", "vram_mib": 15360},
        {"name": "NVIDIA L4", "compute_cap": "89", "vram_mib": 23034},
    ]
    assert parse_gpu_rows("") == []


def test_gpu_family_and_records_slug():
    assert gpu_family("Tesla T4") == "T4" and gpu_family("NVIDIA L4") == "L4"
    assert gpu_family("NVIDIA L40S") is None and gpu_family("") is None
    assert T4.records_slug == "t4" and L4.records_slug == "l4"
    assert Hardware("NVIDIA A100-SXM4-40GB", "80", 40960, 12, 85000, 1).records_slug == "a100-sxm4-40gb"
    assert Hardware("", "", 0, 2, 13000, 1).records_slug == "unknown-gpu"


def test_settings_at_t4_and_l4_sizes():
    assert (T4.build_jobs(), T4.cpu_workers(), T4.day10_slots(), T4.hf_processes(), T4.peer_workers()) == (2, 2, 4, 2, 1)
    assert (L4.build_jobs(), L4.cpu_workers(), L4.day10_slots(), L4.hf_processes(), L4.peer_workers()) == (8, 8, 8, 6, 7)


def test_build_jobs_are_limited_by_memory_not_only_cpus():
    assert build_jobs_for(8, 8 * 1024) == 4  # 8 vCPUs but 8 GiB: four nvcc jobs at most
    assert build_jobs_for(2, 1024) == 1


def test_unknown_gpus_fall_back_to_vram():
    big = Hardware("NVIDIA A100-SXM4-40GB", "80", 40960, 12, 85000, 1)
    small = Hardware("Quadro RTX 5000", "75", 16384, 4, 30000, 1)

    assert big.day10_slots() == 8 and small.day10_slots() == 4
    assert big.cpu_workers() == 8  # capped at eight processes


def test_require_supported_explains_how_to_switch_runtimes():
    require_supported(T4)
    require_supported(L4)
    for bad, words in (
        (Hardware("", "", 0, 2, 13000, 1), "No NVIDIA GPU"),
        (Hardware("Tesla P4", "61", 7680, 2, 13000, 1), "compute capability 61"),
        (Hardware("Small GPU", "86", 8192, 2, 13000, 1), "8192 MiB"),
    ):
        with pytest.raises(UnsupportedRuntime, match=words) as caught:
            require_supported(bad)
        assert "Change runtime type" in str(caught.value)


def test_detect_hardware_uses_overrides_and_survives_no_gpu(tmp_path):
    detected = detect_hardware(gpu_output="Tesla T4, 7.5, 15360\n", cpu_count=2, ram_mib=13000, disk_path=tmp_path)
    assert (detected.gpu_name, detected.compute_cap, detected.vram_mib) == ("Tesla T4", "75", 15360)
    assert detected.disk_free_mib > 0

    missing = detect_hardware(gpu_output="", cpu_count=2, ram_mib=13000, disk_path=tmp_path / "not" / "there")
    assert missing.gpu_name == "" and missing.family is None


def test_visible_cpus_respects_affinity(monkeypatch):
    monkeypatch.setattr(hardware.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    assert hardware.visible_cpus() == 2


def test_total_ram_reads_meminfo(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       13289344 kB\nMemFree: 1 kB\n")
    assert total_ram_mib(meminfo) == 12977


class FakeJobs:
    def active(self):
        return ["qwen35-server"]


def test_runtime_stamp_carries_identity_load_and_background_jobs():
    stamp = runtime_stamp(T4, FakeJobs())["runtime"]

    assert stamp["gpu_family"] == "T4" and stamp["compute_cap"] == "75" and stamp["cpu_count"] == 2
    assert stamp["background_jobs"] == ["qwen35-server"]
    assert stamp["loadavg"] is None or len(stamp["loadavg"]) == 3
    assert runtime_stamp(L4)["runtime"]["background_jobs"] == []


def test_summary_names_the_derived_settings():
    text = T4.summary()
    assert "Tesla T4 (sm_75" in text and "2 vCPUs" in text and "Day 10 slots 4" in text


def test_runtime_stamp_carries_the_session_id():
    assert runtime_stamp(T4, session_id="abc123")["runtime"]["session_id"] == "abc123"
    assert runtime_stamp(T4)["runtime"]["session_id"] is None

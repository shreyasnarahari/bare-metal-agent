import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import toolsmith.evaluation.hf_jsonschema_runner as hf_runner
import toolsmith.evaluation.schema_coverage as coverage
from toolsmith.evaluation.hf_jsonschema_runner import HfUnit, output_path, plan_units, run_hf_units
from toolsmith.evaluation.maskbench_runner import PEER_FLAGS, run_peer_engines
from toolsmith.evaluation.runner import IdentityMismatchError
from toolsmith.hardware import vram_free_mib

# Schema compile coverage


def test_parallel_compile_coverage_keeps_input_order_and_bounds_concurrency(monkeypatch):
    running, peak, lock = [0], [0], threading.Lock()

    def fake_isolated(schema, unique_id, **limits):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05 if int(unique_id) % 2 else 0.01)
        with lock:
            running[0] -= 1
        return {"unique_id": unique_id, "status": "compiled", "limits": limits}

    monkeypatch.setattr(coverage, "isolated_compile_coverage", fake_isolated)
    items = [({"type": "string"}, str(index)) for index in range(12)]

    results = list(coverage.parallel_compile_coverage(items, workers=3, wall_seconds=4))

    assert [result["unique_id"] for result in results] == [str(index) for index in range(12)]
    assert 1 < peak[0] <= 3
    assert results[0]["limits"] == {"wall_seconds": 4, "memory_mb": 512, "cpu_seconds": 5}
    with pytest.raises(ValueError):
        list(coverage.parallel_compile_coverage(items, workers=0))


def test_parallel_compile_coverage_runs_real_isolated_processes():
    items = [
        ({"type": "string"}, "ok"),
        ({"type": "string", "pattern": "^a+$"}, "pattern"),
    ]

    serial = [coverage.isolated_compile_coverage(schema, name) for schema, name in items]
    parallel = list(coverage.parallel_compile_coverage(items, workers=2))

    assert [r["status"] for r in parallel] == [r["status"] for r in serial]
    assert parallel[0]["status"] == "compiled" and parallel[1]["status"] != "compiled"


# MaskBench peers


def fake_peer_command(log):
    script = (
        "import sys, time, pathlib; log = pathlib.Path(sys.argv[1]);"
        "log.open('a').write(f'start {time.monotonic()}\\n'); time.sleep(0.4);"
        "log.open('a').write(f'end {time.monotonic()}\\n'); (pathlib.Path(sys.argv[2]) / 'done.json').write_text('{}')"
    )

    def command(python, flag, tokenizer, output, files):
        return [sys.executable, "-c", script, str(log), str(output)]

    return command


def peak_overlap(log):
    events = sorted(
        (float(stamp), 1 if kind == "start" else -1)
        for kind, stamp in (line.split() for line in log.read_text().splitlines())
    )
    running = peak = 0
    for _, change in events:
        running += change
        peak = max(peak, running)
    return peak


def test_peer_engines_run_concurrently_alongside_the_kernel_engine(tmp_path):
    log = tmp_path / "events.log"
    seen = {}

    def toolsmith_engine():
        seen["alongside"] = True
        return ["toolsmith-results"]

    outputs, extra, concurrency = run_peer_engines(
        PEER_FLAGS[:4], python="python", tokenizer="tok", files=["a.json"], output_root=tmp_path / "maskbench",
        env={}, workers=2, alongside=toolsmith_engine, stamp={"runtime": {"gpu_family": "L4"}},
        command=fake_peer_command(log),
    )

    assert extra == ["toolsmith-results"] and seen["alongside"]
    assert sorted(outputs) == sorted(PEER_FLAGS[:4])
    assert peak_overlap(log) == 2  # bounded by workers, and more than one at once
    run = json.loads((tmp_path / "maskbench" / "llg" / "run.json").read_text())
    assert run["concurrency"] == concurrency == {
        "peer_processes_at_once": 2, "peer_threads_each": 1,
        "toolsmith_in_kernel_concurrently": True, "engines_total": 5,
    }
    assert run["runtime"]["gpu_family"] == "L4"
    assert all((path / "done.json").is_file() and (path / "runner.log").is_file() for path in outputs.values())


def test_a_failing_peer_is_reported(tmp_path):
    def failing(python, flag, tokenizer, output, files):
        return [sys.executable, "-c", "print('engine crashed'); raise SystemExit(2)"]

    with pytest.raises(RuntimeError, match="engine crashed"):
        run_peer_engines(["--xgr"], python="p", tokenizer="t", files=[], output_root=tmp_path, env={}, workers=1,
                         command=failing)


# HF Qwen3-0.6B generation


def fake_initializer(model_name, token):
    def generate(model, tokenizer, vocabulary, prompt, *controller, max_new_tokens, seed):
        label = "fixed" if controller else "prompt"
        return SimpleNamespace(text=f"{label}:{prompt}", token_ids=(1, 2), stop_reason="eos")

    hf_runner._STATE.update(
        model="model", tokenizer="tokenizer", vocabulary="vocabulary",
        make_controller=lambda vocabulary, schema: ("controller", schema["type"]),
        constrained=generate, unconstrained=generate,
    )


def test_plan_units_renders_once_per_item_and_makes_both_modes():
    rendered = []
    rows = {"Github_easy": [{"unique_id": "a", "json_schema": '{"type": "string"}'}]}

    units = plan_units(rows, lambda split, schema: rendered.append((split, schema)) or f"prompt for {split}")

    assert [unit.mode for unit in units] == ["prompt", "fixed"]
    assert rendered == [("Github_easy", {"type": "string"})]
    assert units[0].schema == {"type": "string"} and units[0].prompt == "prompt for Github_easy"
    with pytest.raises(ValueError):
        HfUnit("s", "x", "beam", "p", {})


def test_hf_units_write_identified_outputs_and_resume(tmp_path):
    units = [HfUnit("Github_easy", "a", mode, "P", {"type": "string"}) for mode in ("prompt", "fixed")]
    identity = {"gpu_family": "T4", "model": "Qwen/Qwen3-0.6B"}

    counts = run_hf_units(units, directory=tmp_path, identity=identity, workers=1, model_name="m", token=None,
                          initializer=fake_initializer)

    assert counts == {"skipped": 0, "written": 2}
    fixed = json.loads(output_path(tmp_path, units[1]).read_text())
    assert fixed["text"] == "fixed:P" and fixed["identity"] == identity
    assert run_hf_units(units, directory=tmp_path, identity=identity, workers=1, model_name="m", token=None,
                        initializer=lambda *a: pytest.fail("nothing left to load")) == {"skipped": 2, "written": 0}
    with pytest.raises(IdentityMismatchError, match="gpu_family"):
        run_hf_units(units, directory=tmp_path, identity={**identity, "gpu_family": "L4"}, workers=1,
                     model_name="m", token=None, initializer=fake_initializer)


def test_hf_units_spread_over_a_spawn_process_pool(tmp_path, monkeypatch):
    created = {}

    class InlinePool:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            created.update(max_workers=max_workers, method=mp_context.get_start_method(), initargs=initargs)
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, items, chunksize):
            created["chunksize"] = chunksize
            return [fn(item) for item in items]

    monkeypatch.setattr(hf_runner, "ProcessPoolExecutor", InlinePool)
    units = [HfUnit("s", str(index), "prompt", "P", {"type": "string"}) for index in range(5)]

    counts = run_hf_units(units, directory=tmp_path, identity={"k": 1}, workers=6, model_name="Qwen/Qwen3-0.6B",
                          token="hf-secret", initializer=fake_initializer)

    assert counts["written"] == 5
    assert created == {"max_workers": 5, "method": "spawn", "initargs": ("Qwen/Qwen3-0.6B", "hf-secret"), "chunksize": 1}


def test_vram_free_reads_total_minus_used():
    assert vram_free_mib("15360, 9000\n") == 6360
    assert vram_free_mib("garbage") is None

"""Structure, shared cells, the CPU gate, and the part 1 live cells of the Tidemark notebook."""

import ast
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "04_tidemark.ipynb"
TOOLSMITH_NOTEBOOK = ROOT / "notebooks" / "01_toolsmith.ipynb"
STEADYHAND_NOTEBOOK = ROOT / "notebooks" / "02_steadyhand.ipynb"
SHARED_SETUP_CELLS = ("environment", "drive-and-secrets", "install-project", "hardware")
LIVE_CELLS = (
    "tidemark-capacity-probe", "tidemark-tokenizer-probe", "tidemark-utilization-smoke",
    "tidemark-baseline", "tidemark-baseline-report", "tidemark-prefix-smoke",
)


def load(path=NOTEBOOK_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_source(cell_id, path=NOTEBOOK_PATH):
    return "".join(next(cell for cell in load(path)["cells"] if cell["id"] == cell_id)["source"])


def test_the_cells_are_valid_ordered_and_have_no_outputs():
    notebook = load()
    ids = [cell["id"] for cell in notebook["cells"]]

    assert ids == [
        "introduction",
        "environment",
        "drive-and-secrets",
        "install-project",
        "hardware",
        "tidemark-plan",
        "tidemark-cpu-gate",
        "tidemark-prefix-gate",
        "tidemark-runtime-notes",
        "tidemark-settings",
        "steadyhand-runtime",
        *LIVE_CELLS,
        "tidemark-live-plan",
        "cleanup",
    ]
    assert notebook["metadata"]["accelerator"] == "GPU"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            source = "".join(cell["source"])
            assert source.startswith("# "), cell["id"]
            ast.parse(source, filename=f"{NOTEBOOK_PATH}:{cell['id']}")


def test_setup_cells_match_toolsmith_and_runtime_and_cleanup_match_steadyhand():
    for cell_id in SHARED_SETUP_CELLS:
        assert notebook_source(cell_id) == notebook_source(cell_id, TOOLSMITH_NOTEBOOK), cell_id
    for cell_id in ("steadyhand-runtime", "cleanup"):
        assert notebook_source(cell_id) == notebook_source(cell_id, STEADYHAND_NOTEBOOK), cell_id


def test_the_introduction_names_the_other_notebooks_plan_and_evidence_boundary():
    intro = notebook_source("introduction")
    assert all(name in intro for name in ("01_toolsmith.ipynb", "02_steadyhand.ipynb", "03_bulkhead.ipynb"))
    assert "Tidemark/" in intro and "tests/test_notebook_04.py" in intro
    assert "Parts 1–2 checkpoint" in intro and "later parts are not implemented yet" in intro
    assert "Nothing above the live cells runs a model" in intro and "both a T4 and an L4" in intro


def test_the_live_plan_requires_both_gpus_and_measured_useful_utilization():
    plan = notebook_source("tidemark-live-plan")
    assert "both T4 and L4" in plan and "both a T4 and an L4" in plan
    assert "Autotune, then freeze" in plan and "95%" in plan and "85%" in plan
    assert "work_conserving" in plan and "pinned" in plan and "artificial work" in plan


def test_the_plan_requires_tested_phase_commits_and_a_canonical_handoff():
    plan = notebook_source("tidemark-plan")
    assert "Tidemark/Handoff.md" in plan and "implementation commit" in plan
    assert "pushes both to `origin/main`" in plan and "remote-SHA verification" in plan
    assert "fresh session" in plan and "Unrelated dirty files" in plan


def test_no_secrets_or_destructive_commands_in_the_source():
    source = "\n".join("".join(cell["source"]) for cell in load()["cells"])

    assert "reset', '--hard" not in source and "fuser" not in source and "sys.modules" not in source
    assert "hf_" not in source.replace("hf_source", "").replace("hf-cache", "")
    assert "userdata.get(name)" in source


def test_functions_declare_the_module_state_they_reassign():
    module_state, functions = set(), []
    for cell in load()["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                module_state.update(t.id for t in node.targets if isinstance(t, ast.Name) and t.id.isupper())
        functions.extend((cell["id"], node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    undeclared = []
    for cell_id, function in functions:
        declared = {name for node in ast.walk(function) if isinstance(node, ast.Global) for name in node.names}
        assigned = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        undeclared += [f"{cell_id}:{function.name}:{name}" for name in sorted((assigned & module_state) - declared)]
    assert undeclared == []


def test_the_cpu_gate_replays_goldens_and_validates_the_seed_set(tmp_path):
    namespace = {
        "PROJECT_DIR": ROOT,
        "DRIVE_RECORDS": tmp_path,
        "CPU_COUNT": 1,
        "RUNTIME_STAMP": lambda: {"runtime": {"gpu_family": "T4", "background_jobs": []}},
        "os": os,
        "Path": Path,
    }

    exec(compile(notebook_source("tidemark-cpu-gate"), str(NOTEBOOK_PATH), "exec"), namespace)

    status = namespace["TIDEMARK_STATUS"]
    assert status["passed"] is True and status["golden_replays"] == 3
    assert status["failed_replays"] == [] and status["live_results"] == "not_run"
    assert status["implementation_present"] is True and status["implementation_status"] == "part_1"
    assert status["seed_dataset"]["tasks"] == 12 and status["seed_dataset"]["distances"] == [5, 10, 20, 40]
    saved = json.loads((tmp_path / "tidemark" / "cpu-gate.json").read_text(encoding="utf-8"))
    assert saved == status


class FakeHardware:
    records_slug = "t4"


def test_the_settings_cell_pins_qwen3_and_derives_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("STEADYHAND_PHASE", raising=False)
    monkeypatch.setenv("TIDEMARK_LOCAL", str(tmp_path / "local"))
    namespace = {"os": os, "Path": Path, "json": json, "DRIVE_RECORDS": tmp_path / "drive", "HARDWARE": FakeHardware()}
    exec(compile(notebook_source("tidemark-settings"), str(NOTEBOOK_PATH), "exec"), namespace)
    assert os.environ["STEADYHAND_PHASE"] == "qwen3/wide" and namespace["TIDEMARK_SHAPE"] == "wide"
    assert namespace["TIDEMARK_GENERATION_PATH"] == "buffered_stream" and namespace["TIDEMARK_AFFINITY"] == "pinned"
    assert namespace["TIDEMARK_RUN_ID"] == "baseline-v1-t4-wide" and (tmp_path / "drive" / "tidemark").is_dir()
    monkeypatch.setenv("STEADYHAND_PHASE", "qwen35/wide")
    try:
        exec(compile(notebook_source("tidemark-settings"), str(NOTEBOOK_PATH), "exec"), dict(namespace))
    except ValueError as error:
        assert "Qwen3-8B on both GPUs" in str(error)
    else:
        raise AssertionError("a non-Qwen3 phase must be refused")


def test_the_live_cells_follow_the_part_1_contract():
    capacity, parity, smoke, baseline, report, prefix = (notebook_source(cell_id) for cell_id in LIVE_CELLS)
    # The shared cleanup cell stops the Tidemark server through this name.
    assert "def stop_steadyhand_runtime():" in capacity and "global TIDEMARK_ACTIVE" in capacity
    assert "fit_capacity(" in capacity and "plan_capacity(" in capacity and "did_not_fit" in capacity
    assert "HardwareEnvelope.detect(" in capacity and "ResourceSampler(" in capacity and "GpuSampler()" in capacity
    assert "run_parity(" in parity and "raise RuntimeError(f'Tokenizer parity failed" in parity
    assert "probe_cache_semantics(" in parity and "gguf_sha256" in parity
    assert "autotune(" in smoke and "utilization_smoke(" in smoke and "ForcedPrefixClient(" in smoke
    assert "generation_path=TIDEMARK_GENERATION_PATH" in smoke and "TIDEMARK_FROZEN" in smoke
    assert "set_phase('scored')" in baseline and "summarize_utilization(" in baseline and "tidemark_mirror(" in baseline
    assert "write_report(" in report and "('t4', 'l4')" in report
    assert "tidemark_production != tidemark_assembled.spans.text" in prefix and "planted_suite(" in prefix
    assert "'timestamp_in_frozen'" in prefix and "'recent_append'" in prefix and "id_slot=0" in prefix
    assert "strategy='prefix_stable'" in prefix and "'warm_reuse_target': 0.90" in prefix
    assert "grammar_cache=TIDEMARK_GRAMMARS" in prefix and "tidemark_lint.get('violation')" in prefix
    for source in (capacity, parity, smoke, baseline, report, prefix):
        # Secrets are passed to clients and redaction, never printed or written into a record.
        assert not [line for line in source.splitlines() if ("print(" in line or "atomic_write_json(" in line)
                    and ("LOCAL_KEY" in line or "HF_TOKEN" in line)]


def test_the_runtime_notes_and_live_plan_describe_the_evidence_rules():
    notes, plan = notebook_source("tidemark-runtime-notes"), notebook_source("tidemark-live-plan")
    assert "once on a T4 and once on an L4" in notes and "untested" in notes and "never filled with artificial work" in notes
    assert plan.startswith("## Later parts") and "Part 1's cells above" in plan


def test_the_prefix_gate_runs_the_planted_suite_on_cpu(tmp_path):
    namespace = {"DRIVE_RECORDS": tmp_path}
    exec(compile(notebook_source("tidemark-prefix-gate"), str(NOTEBOOK_PATH), "exec"), namespace)
    gate = namespace["TIDEMARK_PREFIX_GATE"]
    assert gate["passed"] is True and set(gate) == {"qwen3", "qwen35", "passed"}
    assert all(result["deterministic"] and len(result["cases"]) == 10 for name, result in gate.items() if name != "passed")
    assert json.loads((tmp_path / "tidemark" / "prefix-gate.json").read_text())["passed"] is True

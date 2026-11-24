"""The Bulkhead notebook (``notebooks/03_bulkhead.ipynb``): structure, shared cells, and the CPU-runnable cells.

Bulkhead's cells moved here from the Steadyhand notebook after part 8. The setup
cells stay identical to the Toolsmith notebook's, and the model runtime, server,
stop, and cleanup cells to the Steadyhand notebook's; the tests below hold every
copy to that.
"""

import ast
import json
from pathlib import Path

import pytest

from toolsmith.hardware import Hardware

ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "03_bulkhead.ipynb"
STEADYHAND_NOTEBOOK = ROOT / "notebooks" / "02_steadyhand.ipynb"
TOOLSMITH_NOTEBOOK = ROOT / "notebooks" / "01_toolsmith.ipynb"
T4 = Hardware("Tesla T4", "75", 15360, 2, 13000, 80_000)
SHARED_SETUP_CELLS = ("environment", "drive-and-secrets", "install-project", "hardware")
SHARED_RUNTIME_CELLS = ("steadyhand-runtime", "steadyhand-server", "steadyhand-server-stop", "cleanup")


def load(path=NOTEBOOK_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_source(cell_id, path=NOTEBOOK_PATH):
    return "".join(next(cell for cell in load(path)["cells"] if cell["id"] == cell_id)["source"])


# Structure


def test_the_cells_are_valid_ordered_and_have_no_outputs():
    notebook = load()
    ids = [cell["id"] for cell in notebook["cells"]]

    assert ids == [
        "introduction", "environment", "drive-and-secrets", "install-project", "hardware",
        # The model provisions in the background first, so the CPU sections run while it downloads.
        "model-runtime-notes", "steadyhand-runtime",
        "bulkhead-probe-notes", "bulkhead-probe",  # part 1, CPU
        "bulkhead-attack-suite-intro", "bulkhead-attack-suite",  # part 7, CPU
        "bulkhead-agentdojo-notes", "bulkhead-agentdojo-gate",  # part 8, CPU
        "bulkhead-live-notes", "steadyhand-server", "bulkhead-agentdojo-sweep",  # part 8, live
        "steadyhand-server-stop", "cleanup",
    ]
    assert notebook["metadata"]["accelerator"] == "GPU"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK_PATH}:{cell['id']}")


def test_setup_cells_match_the_toolsmith_notebook():
    for cell_id in SHARED_SETUP_CELLS:
        assert notebook_source(cell_id) == notebook_source(cell_id, TOOLSMITH_NOTEBOOK), cell_id


def test_runtime_and_server_cells_match_the_steadyhand_notebook():
    for cell_id in SHARED_RUNTIME_CELLS:
        assert notebook_source(cell_id) == notebook_source(cell_id, STEADYHAND_NOTEBOOK), cell_id


def test_the_introduction_names_the_other_notebooks_and_the_shared_cells():
    intro = notebook_source("introduction")
    assert "02_steadyhand.ipynb" in intro and "01_toolsmith.ipynb" in intro and "tests/test_notebook_03.py" in intro


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


def test_no_secrets_or_destructive_commands_in_the_source():
    source = "\n".join("".join(cell["source"]) for cell in load()["cells"])

    assert "reset', '--hard" not in source and "fuser" not in source and "sys.modules" not in source
    assert "hf_" not in source.replace("hf_source", "").replace("hf-cache", "")
    assert "userdata.get(name)" in source  # secrets come from Colab Secrets or the Drive .env file


# Bulkhead probe (part 1)


def bulkhead_report(scenario):
    """A report built from the Bulkhead test fixture (no probe runs, so the result is host-independent)."""
    from bulkhead.types import Capability, HostInfo, ProbeReport

    fixture = json.loads((ROOT / "tests" / "fixtures" / "bulkhead_probe_reports.json").read_text())
    overrides = next(item for item in fixture["scenarios"] if item["name"] == scenario)["overrides"]
    records = [
        Capability(name, overrides[name]["status"], overrides[name]["code"], "fixture")
        if name in overrides
        else Capability(name, "ok", "ok", "fixture", facts)
        for name, facts in fixture["baseline_facts"].items()
    ]
    return ProbeReport.build(
        created_at="2026-11-09T12:00:00Z",
        host=HostInfo("Linux", "6.6.0", "x86_64", "3.12.0", wsl=False, container=True, root=True),
        python="/usr/bin/python3",
        helper="/content/bare-metal-agent/src/bulkhead/_probe_child.py",
        bwrap_path=None,
        capabilities=records,
        cgroup_mount="/sys/fs/cgroup",
        cgroup_parent=None,
    )


def run_bulkhead_cell(tmp_path, monkeypatch, scenario):
    import bulkhead.probe

    calls = []

    def fake_run_probe(**options):
        calls.append(options)
        return bulkhead_report(scenario)

    monkeypatch.setattr(bulkhead.probe, "run_probe", fake_run_probe)
    namespace = {"DRIVE_RECORDS": tmp_path, "HF_TOKEN": "hf-test-secret", "LOCAL_KEY": "local-test-secret"}
    exec(compile(notebook_source("bulkhead-probe"), str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace, calls


def test_the_bulkhead_cell_records_the_probe_under_the_gpu_records(tmp_path, monkeypatch, capsys):
    from bulkhead.probe import load_report

    monkeypatch.delenv("BULKHEAD_REQUIRE_TIER", raising=False)
    monkeypatch.setenv("BULKHEAD_CGROUP_PARENT", "/sys/fs/cgroup/delegated")

    namespace, calls = run_bulkhead_cell(tmp_path, monkeypatch, "seccomp_failure")

    saved = tmp_path / "bulkhead" / "probe.colab.json"
    assert namespace["bulkhead_probe_path"] == saved and load_report(saved) == namespace["BULKHEAD_PROBE"]
    assert calls == [{"cgroup_parent": "/sys/fs/cgroup/delegated"}]
    assert "Selected tier: T0 ... NOT a sandbox" in capsys.readouterr().out
    assert "test-secret" not in saved.read_text()


def test_the_bulkhead_cell_fails_when_the_required_tier_is_not_proven(tmp_path, monkeypatch):
    from bulkhead.errors import UnsupportedTierError

    monkeypatch.delenv("BULKHEAD_CGROUP_PARENT", raising=False)
    monkeypatch.setenv("BULKHEAD_REQUIRE_TIER", "T2")

    with pytest.raises(UnsupportedTierError, match="T2"):
        run_bulkhead_cell(tmp_path, monkeypatch, "wsl2_dev_host")
    assert (tmp_path / "bulkhead" / "probe.colab.json").is_file()  # the evidence is written before the check

    monkeypatch.setenv("BULKHEAD_REQUIRE_TIER", "T1")
    namespace, calls = run_bulkhead_cell(tmp_path, monkeypatch, "wsl2_dev_host")
    assert calls == [{"cgroup_parent": None}] and namespace["BULKHEAD_PROBE"].selected_tier.value == "T1"


def test_the_attack_suite_cell_records_untested_profiles_reports_and_resumes_from_drive(tmp_path, monkeypatch, capsys):
    import bulkhead.eval.attacks as attacks
    import bulkhead.eval.usability as usability
    from bulkhead.eval.attacks import Target

    monkeypatch.delenv("BULKHEAD_CGROUP_PARENT", raising=False)
    monkeypatch.delenv("BULKHEAD_OVERHEAD_SAMPLES", raising=False)
    monkeypatch.setattr(attacks, "discover_targets", lambda *a, **k: [Target("T0"), Target("landlock-t1", None, "the probe did not prove landlock.fs")])
    tasks = [task for task in usability.load_tasks() if task["task_id"] in ("bh_b_order_status", "bh_m_refund_injection")]
    monkeypatch.setattr(usability, "load_tasks", lambda *a, **k: tasks)
    runs = []
    real = attacks.run_end_to_end
    monkeypatch.setattr(attacks, "run_end_to_end", lambda *a, **k: runs.append(a[1].attack_id) or real(*a, **k))

    def execute():
        namespace = {"PROJECT_DIR": str(tmp_path / "repo"), "DRIVE_RECORDS": tmp_path / "drive", "BULKHEAD_PROBE": bulkhead_report("wsl2_dev_host"), "HARDWARE": T4}
        exec(compile(notebook_source("bulkhead-attack-suite"), str(NOTEBOOK_PATH), "exec"), namespace)
        return namespace

    first = execute()
    out = capsys.readouterr().out
    assert "landlock-t1            untested: the probe did not prove landlock.fs" in out
    assert sorted(runs) == ["deputy.denied_call_no_recovery", "deputy.recovery_switch", "deputy.shared_dir_handoff"]  # T0 runs only policy-level scenarios
    report = (tmp_path / "drive" / "bulkhead" / "RESULTS.md").read_text()
    applicable = sum(1 for attack in attacks.load_dataset().attacks if attack.expectation("landlock-t1")["verdict"] != "not_applicable")
    assert f"**landlock-t1**: {applicable} cell(s) (attacks and controls) untested" in report and "No escape" in report
    mirrored = tmp_path / "drive" / "bulkhead" / "attacks" / first["bulkhead_attack_run"].run_id / "results.jsonl"
    assert mirrored.read_text() == (first["bulkhead_attack_run"].directory / "results.jsonl").read_text()

    import shutil

    shutil.rmtree(tmp_path / "repo" / "results" / "bulkhead" / "attacks")  # a fresh runtime: only Drive has the records
    runs.clear()
    second = execute()
    assert runs == [] and second["bulkhead_attack_run"].run_id == first["bulkhead_attack_run"].run_id


# Bulkhead on AgentDojo (part 8)


def run_agentdojo_gate(tmp_path, monkeypatch, pytest_tail, *, installed=None):
    import subprocess as subprocess_module

    import bulkhead.eval.agentdojo as adapter

    calls = []
    pin = {"version": "0.1.35", "commit": adapter.AGENTDOJO_COMMIT}

    def fake_run(command, **options):
        calls.append(command)
        return subprocess_module.CompletedProcess(command, 0 if "failed" not in pytest_tail else 1, stdout=f"....\n{pytest_tail}\n", stderr="")

    monkeypatch.setattr(subprocess_module, "run", fake_run)
    monkeypatch.setattr(adapter, "installed_pin", lambda: installed)
    monkeypatch.setattr(adapter, "require_pinned", lambda: pin)
    namespace = {"PROJECT_DIR": str(ROOT), "DRIVE_RECORDS": tmp_path, "time": __import__("time"), "json": json, "Path": Path}
    exec(compile(notebook_source("bulkhead-agentdojo-gate"), str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace, calls


def test_the_agentdojo_gate_installs_the_pin_runs_the_real_tests_and_records_the_result(tmp_path, monkeypatch):
    from bulkhead.eval.agentdojo import AGENTDOJO_REQUIREMENT

    namespace, calls = run_agentdojo_gate(tmp_path, monkeypatch, "57 passed in 30.1s")
    assert calls[0][-1] == AGENTDOJO_REQUIREMENT and calls[0][1:4] == ["-m", "pip", "install"]
    assert calls[1][1:3] == ["-m", "pytest"] and "tests/test_bulkhead_agentdojo.py" in calls[1]
    saved = json.loads((tmp_path / "bulkhead" / "agentdojo" / "cpu_gate.json").read_text())
    assert saved == namespace["BULKHEAD_AGENTDOJO_GATE"] and saved["passed"] and saved["summary"] == "57 passed in 30.1s"
    # Already pinned: nothing is installed.
    _, calls = run_agentdojo_gate(tmp_path, monkeypatch, "57 passed in 30.1s", installed={"version": "0.1.35", "commit": None})
    assert len(calls) == 1 and calls[0][1:3] == ["-m", "pytest"]


@pytest.mark.parametrize("tail", ["44 passed, 13 skipped in 18.0s", "1 failed, 56 passed in 30.0s"])
def test_the_agentdojo_gate_fails_on_skipped_or_failed_tests(tmp_path, monkeypatch, tail):
    with pytest.raises(RuntimeError, match="CPU gate failed"):
        run_agentdojo_gate(tmp_path, monkeypatch, tail)
    assert json.loads((tmp_path / "bulkhead" / "agentdojo" / "cpu_gate.json").read_text())["passed"] is False


def test_the_agentdojo_sweep_cell_runs_smoke_then_selection_paired_and_resumable(tmp_path, monkeypatch, capsys):
    from dataclasses import dataclass

    import steadyhand
    from agentdojo_support import CLASSIFICATION, stand_in_api

    import bulkhead.eval.agentdojo as adapter
    import bulkhead.eval.runner as runner
    from bulkhead.eval.runner import ScriptedModel

    @dataclass
    class Preset:
        parallel: int = 2
        per_slot_context: int = 8192

        def as_dict(self):
            return {"parallel": self.parallel, "per_slot_context": self.per_slot_context}

    monkeypatch.setattr(adapter, "agentdojo_api", stand_in_api)
    monkeypatch.setattr(runner, "load_classification", lambda *a, **k: adapter.load_classification(CLASSIFICATION))
    monkeypatch.setattr(steadyhand, "worker_service", lambda *a: ScriptedModel([{"calls": [["get_balance", {}]]}, {"text": "Your balance is 1810.0."}]))
    monkeypatch.setenv("BULKHEAD_AGENTDOJO_LOCAL", str(tmp_path / "local"))
    monkeypatch.setenv("BULKHEAD_AGENTDOJO_SMOKE", "2")
    monkeypatch.setenv("BULKHEAD_AGENTDOJO_CASES", "all")
    active = {"client": None, "renderer": None, "profile": None, "preset": Preset(), "gpu": {"name": "NVIDIA L4"}, "kv_cache": "f16", "flash_attn": True, "extra_args": []}
    synced = []
    namespace = {
        "PROJECT_DIR": str(tmp_path / "repo"), "DRIVE_RECORDS": tmp_path / "drive", "STEADYHAND_ACTIVE": active,
        "BULKHEAD_AGENTDOJO_GATE": {"passed": True}, "STEADYHAND_RUNTIME": {"llama_cpp_ref": "b8233", "model_repo": "r", "model_file": "q4_k_m.gguf", "model_revision": "v"},
        "ensure_drive": lambda: None, "sync_drive": lambda *a, **k: synced.append(k), "BACKGROUND": None, "start_steadyhand_runtime": lambda: active,
        "os": __import__("os"), "time": __import__("time"), "json": json, "Path": Path,
    }
    exec(compile(notebook_source("bulkhead-agentdojo-sweep"), str(NOTEBOOK_PATH), "exec"), namespace)
    smoke, sweep = namespace["bulkhead_runs"]
    out = capsys.readouterr().out
    assert "6 eligible cases, 3 ineligible; withheld: mini_bank.tag_transaction (/properties/tags/additionalProperties)" in out
    for run, cases in ((smoke, 2), (sweep, 6)):
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["kind"] == "live" and len(manifest["case_ids"]) == cases and manifest["approval_mode"] == "deny_all"
        assert manifest["server"]["model_file"] == "q4_k_m.gguf" and manifest["server"]["parallel"] == 2
        mirrored = tmp_path / "drive" / "bulkhead" / "agentdojo" / run.name
        assert (mirrored / "records.jsonl").read_text() == (run / "records.jsonl").read_text()
        assert {p.name for p in mirrored.iterdir()} >= {"manifest.json", "records.jsonl", "summary.json", "replay.json", "eligibility.json", "journals"}
        assert json.loads((run / "replay.json").read_text())["mismatches"] == []
    report = (tmp_path / "drive" / "bulkhead" / "RESULTS.md").read_text()
    assert f"### Run `{sweep.name}` (live, label `sweep`)" in report and synced == [{"stage": "bulkhead-agentdojo"}]
    # A fresh runtime resumes from Drive: nothing runs twice.
    import shutil

    shutil.rmtree(tmp_path / "local")
    lines = (tmp_path / "drive" / "bulkhead" / "agentdojo" / sweep.name / "records.jsonl").read_text()
    exec(compile(notebook_source("bulkhead-agentdojo-sweep"), str(NOTEBOOK_PATH), "exec"), namespace)
    assert (namespace["bulkhead_runs"][1] / "records.jsonl").read_text() == lines


def test_the_agentdojo_sweep_never_auto_approves():
    sweep = notebook_source("bulkhead-agentdojo-sweep")
    assert "approval_mode" not in sweep and "scripted" not in sweep  # deny_all, the declared primary mode
    assert "workers=bulkhead_state['active']['preset'].parallel" in sweep and "seed=0" in sweep
    assert "start_steadyhand_runtime()" in sweep and "JournalCopier(" in sweep

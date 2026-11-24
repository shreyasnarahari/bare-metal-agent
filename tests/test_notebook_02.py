import ast
import json
import threading
from pathlib import Path

import pytest

import steadyhand
from steadyhand.journal import manifest_path_for
from steadyhand.manifest import load_manifest
from toolsmith.json_utils import strict_json_loads
from toolsmith.prompting import ChatPromptRenderer
from toolsmith.hardware import Hardware
from toolsmith.protocols import QWEN35_PROFILE

ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "02_steadyhand.ipynb"
TOOLSMITH_NOTEBOOK = ROOT / "notebooks" / "01_toolsmith.ipynb"
L4 = Hardware("NVIDIA L4", "89", 23034, 8, 54000, 200_000)
T4 = Hardware("Tesla T4", "75", 15360, 2, 13000, 80_000)
SHARED_SETUP_CELLS = ("environment", "drive-and-secrets", "install-project", "hardware")


def load(path=NOTEBOOK_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_source(cell_id, path=NOTEBOOK_PATH):
    return "".join(next(cell for cell in load(path)["cells"] if cell["id"] == cell_id)["source"])


def first_code_line(source):
    """The first line that is not blank or a comment (cells open with a header comment)."""
    return next(line for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def journal_records(path):
    return [strict_json_loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


# Structure


def test_the_cells_are_valid_ordered_and_have_no_outputs():
    notebook = load()
    ids = [cell["id"] for cell in notebook["cells"]]

    assert ids == [
        "introduction", "environment", "drive-and-secrets", "install-project", "hardware",
        # Provisioning starts in the background first, so the CPU gate runs while it downloads.
        "steadyhand-runtime-notes", "steadyhand-runtime", "steadyhand-sweep-notes", "steadyhand-cpu-gate",
        # Bulkhead's CPU-only probe also runs while the runtime provisions.
        "bulkhead-probe-notes", "bulkhead-probe",
        # Part 7's attack corpus, usability benchmark, and report are CPU-only too.
        "bulkhead-attack-suite-intro", "bulkhead-attack-suite",
        # Part 8's AgentDojo CPU gate needs no model either; its live sweep follows the Steadyhand sweep.
        "bulkhead-agentdojo-notes", "bulkhead-agentdojo-gate",
        "steadyhand-server", "steadyhand-notes", "steadyhand-bare-loop", "steadyhand-smoke",
        "steadyhand-sweep", "bulkhead-agentdojo-sweep", "steadyhand-server-stop", "cleanup",
    ]
    assert notebook["metadata"]["accelerator"] == "GPU"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK_PATH}:{cell['id']}")


def test_setup_cells_match_the_toolsmith_notebook():
    for cell_id in SHARED_SETUP_CELLS:
        assert notebook_source(cell_id) == notebook_source(cell_id, TOOLSMITH_NOTEBOOK), cell_id


def test_the_toolsmith_notebook_no_longer_holds_steadyhand_cells():
    toolsmith_ids = [cell["id"] for cell in load(TOOLSMITH_NOTEBOOK)["cells"]]
    toolsmith_source = "\n".join("".join(cell["source"]) for cell in load(TOOLSMITH_NOTEBOOK)["cells"])

    assert not [cell_id for cell_id in toolsmith_ids if cell_id.startswith("steadyhand")]
    assert "stop_steadyhand_runtime" not in toolsmith_source
    assert "02_steadyhand.ipynb" in notebook_source("introduction", TOOLSMITH_NOTEBOOK)


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


def test_the_runtime_cell_provisions_through_the_shared_module():
    runtime = notebook_source("steadyhand-runtime")

    assert "ensure_llama_server, steadyhand_llama_runtime" in runtime  # submitted as a background job
    assert "ensure_gguf(" in runtime and "resolve_gguf(" in runtime and "jobs=HARDWARE.build_jobs()" in runtime
    assert "'llama_cpp_ref': 'b5401'" in runtime and "'llama_cpp_ref': 'b8233'" in runtime
    assert "select=q4_k_m_language_file" in runtime
    assert "cache_dir_for(DRIVE_BUILD_CACHE" in runtime  # the same Drive cache as the Toolsmith notebook


# CPU gate


def test_the_cpu_gate_runs_and_records_a_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("STEADYHAND_CPU_GATE_TASKS", "4")
    monkeypatch.setenv("STEADYHAND_CPU_GATE_WORKERS", "1")
    namespace = {
        "PROJECT_DIR": ROOT,
        "DRIVE_RECORDS": tmp_path,
        "HF_TOKEN": "hf-test-secret",
        "LOCAL_KEY": "local-test-secret",
    }

    exec(compile(notebook_source("steadyhand-cpu-gate"), str(NOTEBOOK_PATH), "exec"), namespace)

    record = namespace["steadyhand_gate_record"]
    assert record["passed"] is True and record["tasks_run"] == 4 and record["golden_replays"] == 3
    saved = json.loads((tmp_path / "steadyhand" / "cpu-gate.jsonl").read_text())
    assert saved["passed"] is True and "local-test-secret" not in json.dumps(saved)


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
        "PROJECT_DIR": str(tmp_path / "repo"), "DRIVE_RECORDS": tmp_path / "drive", "STEADYHAND_ACTIVE": active, "STEADYHAND_SMOKE": {"ok": True},
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


# Live cells (they need a GPU; these checks pin the plan's invariants)


def test_the_live_cells_keep_one_model_and_fill_the_slots():
    server, sweep, smoke = (notebook_source(cell) for cell in ("steadyhand-server", "steadyhand-sweep", "steadyhand-smoke"))

    assert "os.environ['TOOLSMITH_BASE_URL'] = steadyhand_server.base_url" in server
    assert "stop_steadyhand_runtime()" in server and "free_port('127.0.0.1', 8090)" in server
    assert "fit_server(" in server and "kv_ladder(steadyhand_help)" in server and "server_extra_args(" in server
    assert "probe=probe_decode_speed" in server and "vram_mib=steadyhand_gpu['memory_total_mib']" in server
    assert "STEADYHAND_RUNTIME = collect_steadyhand_runtime()" in server
    assert "server_environment(steadyhand_runtime['build'])" in server
    assert "workers=steadyhand_preset.parallel" in sweep and "JournalCopier(" in sweep
    assert "mirror_dir=steadyhand_drive / 'sweeps'" in sweep and "steadyhand_active['kv_cache']" in sweep
    assert "'q8_0'" not in sweep  # the KV type comes from the rung the server cell kept
    assert "render_report(" in sweep and "slot_context=steadyhand_preset.per_slot_context" in sweep
    assert "max_workers=steadyhand_parallel" in smoke and "/slots" in smoke
    assert first_code_line(notebook_source("cleanup")) == "if 'stop_steadyhand_runtime' in globals():"


# Overlap check (part 1)


class PromptEchoRenderer:
    def render(self, messages, tools=(), *, enable_thinking=None):
        return json.dumps({"messages": messages, "enable_thinking": enable_thinking})


class ScriptedLlama:
    """Fake LlamaCppClient: a native tool call first, then text once a tool result exists."""

    def __init__(self, *, barrier=None):
        self.barrier = barrier
        self.lock = threading.Lock()
        self.calls = []
        self.closed = False

    def complete(self, prompt, **settings):
        payload = json.loads(prompt)
        with self.lock:
            self.calls.append((payload, settings))
        messages = payload["messages"]
        if messages[-1]["role"] == "tool":
            return {"content": "Answered from the tool result.", "tokens_evaluated": 60, "tokens_predicted": 9}
        if self.barrier is not None:
            self.barrier.wait()
        order = "ord_1003" if "ord_1003" in messages[-1]["content"] else "ord_1001"
        name = "list_payments" if order == "ord_1003" else "get_order"
        call = {"name": name, "arguments": {"order_id": order}}
        return {"content": f"<tool_call>{json.dumps(call)}</tool_call>", "tokens_evaluated": 40, "tokens_predicted": 12}

    def close(self):
        self.closed = True


def run_overlap_cell(tmp_path, **extra):
    namespace = {
        "DRIVE_RECORDS": tmp_path / "drive",
        "HF_TOKEN": "hf-test-secret",
        "LOCAL_KEY": "local-test-secret",
        **extra,
    }
    exec(compile(notebook_source("steadyhand-bare-loop"), str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("base_url", "reason"),
    [(None, "TOOLSMITH_BASE_URL is not set"), ("http://10.0.0.5:8080", "localhost")],
)
def test_the_overlap_check_skips_with_a_recorded_reason(tmp_path, monkeypatch, base_url, reason):
    monkeypatch.delenv("TOOLSMITH_BASE_URL", raising=False)
    if base_url is not None:
        monkeypatch.setenv("TOOLSMITH_BASE_URL", base_url)
    monkeypatch.setenv("STEADYHAND_LOCAL_DIR", str(tmp_path / "local"))

    namespace = run_overlap_cell(tmp_path)

    record = namespace["steadyhand_record"]
    assert record["status"] == "skipped" and reason in record["reason"]
    assert journal_records(tmp_path / "drive" / "steadyhand" / "part1-smoke.jsonl")[-1]["status"] == "skipped"
    assert not (tmp_path / "local").exists()


def live_environment(monkeypatch, tmp_path, llama):
    built = {}

    def fake_build_llama_client(settings, *, connections, timeout):
        built.update(base_url=settings.base_url, api_key=settings.api_key, connections=connections)
        return llama

    monkeypatch.setenv("TOOLSMITH_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("STEADYHAND_LOCAL_DIR", str(tmp_path / "local"))
    monkeypatch.delenv("STEADYHAND_PROFILE", raising=False)
    monkeypatch.delenv("TOOLSMITH_PARALLEL", raising=False)
    monkeypatch.setattr(steadyhand, "build_llama_client", fake_build_llama_client)
    return built


def test_the_overlap_check_runs_two_overlapping_episodes_and_copies_journals(tmp_path, monkeypatch):
    llama = ScriptedLlama(barrier=threading.Barrier(2, timeout=10))
    built = live_environment(monkeypatch, tmp_path, llama)
    monkeypatch.setattr(ChatPromptRenderer, "from_pretrained", classmethod(lambda cls, *a, **k: PromptEchoRenderer()))

    record = run_overlap_cell(tmp_path)["steadyhand_record"]

    assert record["status"] == "ran" and record["completions_overlapped"] is True
    assert record["copy_errors"] == [] and record["overlap_seconds"] > 0
    assert built == {"base_url": "http://127.0.0.1:8080", "api_key": "local-test-secret", "connections": 4}
    assert llama.closed
    assert [run["status"] for run in record["episodes"]] == ["done", "done"]
    for run in record["episodes"]:
        local, copy = Path(run["journal"]), Path(run["drive_copy"])
        assert local.parent == tmp_path / "local" and copy.parent == tmp_path / "drive" / "steadyhand"
        assert local.read_bytes() == copy.read_bytes()
        assert load_manifest(manifest_path_for(copy)).completed is True
        assert "local-test-secret" not in local.read_text(encoding="utf-8")
    assert all(settings["grammar"] for _, settings in llama.calls)


class Qwen35Llama(ScriptedLlama):
    """The same script in the Qwen3.5 native format."""

    def complete(self, prompt, **settings):
        response = super().complete(prompt, **settings)
        if response["content"].startswith("<tool_call>{"):
            call = json.loads(response["content"][len("<tool_call>"):-len("</tool_call>")])
            order = call["arguments"]["order_id"]
            response["content"] = (
                f"<tool_call>\n<function={call['name']}>\n<parameter=order_id>\n{order}\n"
                "</parameter>\n</function>\n</tool_call>"
            )
        return response


def test_the_overlap_check_uses_the_server_cells_profile_and_renderer(tmp_path, monkeypatch):
    llama = Qwen35Llama(barrier=threading.Barrier(2, timeout=10))
    live_environment(monkeypatch, tmp_path, llama)
    monkeypatch.setattr(ChatPromptRenderer, "from_pretrained", classmethod(lambda cls, *a, **k: pytest.fail("no download")))
    active = {"profile": QWEN35_PROFILE, "renderer": PromptEchoRenderer()}

    record = run_overlap_cell(tmp_path, STEADYHAND_ACTIVE=active)["steadyhand_record"]

    assert record["status"] == "ran"
    assert record["profile"] == "qwen35" and record["model"] == QWEN35_PROFILE.model
    assert [run["status"] for run in record["episodes"]] == ["done", "done"]


@pytest.mark.parametrize(("phase", "ref", "drive_name"), [
    ("qwen35/wide", "b8233", "0123456789ab-Qwen3.5-9B-Q4_K_M.gguf"),
    ("qwen3/tall", "b5401", "Qwen3-8B-Q4_K_M.gguf"),
])
def test_the_runtime_cell_provisions_the_phase_model_with_shared_cache_names(tmp_path, monkeypatch, sync_jobs, phase, ref, drive_name):
    import os

    import toolsmith.provision as provision

    calls = []
    monkeypatch.setenv("STEADYHAND_PHASE", phase)
    monkeypatch.setattr(provision, "ensure_llama_server", lambda runtime, **options: calls.append(runtime) or runtime.server_bin)
    monkeypatch.setattr(
        provision, "resolve_gguf",
        lambda repo, *, token, filename=None, select=None, api=None: provision.GgufFile(
            repo, filename or "Qwen3.5-9B-Q4_K_M.gguf", "0123456789abcdef", 1234,
        ),
    )
    monkeypatch.setattr(provision, "ensure_gguf", lambda gguf, **options: calls.append(options) or options["local_path"])
    monkeypatch.setattr(ChatPromptRenderer, "from_pretrained", classmethod(lambda cls, *a, **k: PromptEchoRenderer()))
    namespace = {
        "Path": Path, "os": os, "json": json, "HF_TOKEN": "hf-test", "HARDWARE": L4, "BACKGROUND": sync_jobs,
        "DRIVE_BUILD_CACHE": tmp_path / "builds", "DRIVE_MODEL_CACHE": tmp_path / "models",
    }

    exec(compile(notebook_source("steadyhand-runtime"), str(NOTEBOOK_PATH), "exec"), namespace)

    # The cell only starts the jobs; the server cell collects them after the CPU gate.
    assert namespace["STEADYHAND_RUNTIME"] is None
    assert [name.split("-")[1] for name in sync_jobs.submitted] == ["llama.cpp", "gguf", "renderer"]
    runtime = namespace["collect_steadyhand_runtime"]()
    assert runtime["profile_name"] == phase.split("/")[0] and runtime["llama_cpp_ref"] == ref
    assert calls[0].cache == tmp_path / "builds" / f"llama.cpp-{ref}-cuda-sm89-runtime-v2"
    assert calls[1]["drive_path"] == tmp_path / "models" / drive_name
    assert runtime["server_bin"] == calls[0].server_bin and runtime["model"] == calls[1]["local_path"]
    assert isinstance(runtime["renderer"], PromptEchoRenderer) and runtime["gpu"] == "NVIDIA L4"


@pytest.mark.parametrize(("output", "family"), [("Tesla T4, 7.5, 15360\n", "T4"), ("NVIDIA L4, 8.9, 23034\n", "L4")])
def test_the_hardware_cell_detects_the_runtime(tmp_path, monkeypatch, background_logs, output, family):
    import toolsmith.hardware as hardware

    monkeypatch.setattr(hardware, "query_gpus", lambda: output)
    namespace = {"DRIVE_ROOT": tmp_path / "toolsmith"}

    exec(compile(notebook_source("hardware"), str(NOTEBOOK_PATH), "exec"), namespace)

    assert namespace["GPU_FAMILY"] == family and namespace["HARDWARE"].family == family
    assert namespace["CPU_COUNT"] == namespace["HARDWARE"].cpu_count
    jobs = namespace["BACKGROUND"]
    assert jobs.log_dir == background_logs
    exec(compile(notebook_source("hardware"), str(NOTEBOOK_PATH), "exec"), namespace)
    assert namespace["BACKGROUND"] is jobs  # a rerun keeps the running jobs
    assert namespace["DRIVE_RECORDS"] == tmp_path / "toolsmith" / "records" / family.lower()
    assert namespace["DRIVE_RECORDS"].is_dir()
    stamp = namespace["RUNTIME_STAMP"]()["runtime"]
    assert stamp["gpu_family"] == family and stamp["background_jobs"] == []


def test_the_hardware_cell_fails_fast_without_a_gpu(monkeypatch, background_logs):
    import toolsmith.hardware as hardware

    monkeypatch.setattr(hardware, "query_gpus", lambda: "")
    with pytest.raises(hardware.UnsupportedRuntime, match="Change runtime type"):
        exec(compile(notebook_source("hardware"), str(NOTEBOOK_PATH), "exec"), {"DRIVE_ROOT": Path("/nonexistent")})


def test_only_the_hardware_cell_defines_the_records_folder():
    for path in (NOTEBOOK_PATH, TOOLSMITH_NOTEBOOK):
        for cell in load(path)["cells"]:
            if cell["cell_type"] == "code" and cell["id"] != "hardware":
                assert "DRIVE_RECORDS =" not in "".join(cell["source"]), (path.name, cell["id"])
    assert "DRIVE_ROOT / 'records' / HARDWARE.records_slug" in notebook_source("hardware")


def test_the_sweep_and_cleanup_flush_drive():
    assert "sync_drive(BACKGROUND, stage='steadyhand-sweep')" in notebook_source("steadyhand-sweep")
    assert "sync_drive(BACKGROUND, stage='cleanup', remount=False)" in notebook_source("cleanup")
    assert "from toolsmith.drive import ensure_drive, sync_drive" in notebook_source("hardware")


def test_a_sweep_interrupted_by_a_dead_server_restarts_it_once():
    server, sweep = notebook_source("steadyhand-server"), notebook_source("steadyhand-sweep")

    assert "def start_steadyhand_runtime():" in server and "global STEADYHAND_ACTIVE" in server
    assert "steadyhand_active = start_steadyhand_runtime()" in server
    assert "except SweepInterrupted as steadyhand_error:" in sweep and "for steadyhand_attempt in (1, 2):" in sweep
    assert "steadyhand_active = start_steadyhand_runtime()" in sweep and "ensure_drive()" in sweep


def test_records_carry_a_session_id_and_the_intro_explains_resuming(tmp_path, monkeypatch, background_logs):
    import toolsmith.hardware as hardware

    monkeypatch.setattr(hardware, "query_gpus", lambda: "Tesla T4, 7.5, 15360\n")
    namespace = {"DRIVE_ROOT": tmp_path / "toolsmith"}
    exec(compile(notebook_source("hardware"), str(NOTEBOOK_PATH), "exec"), namespace)
    session = namespace["SESSION_ID"]
    exec(compile(notebook_source("hardware"), str(NOTEBOOK_PATH), "exec"), namespace)

    assert namespace["SESSION_ID"] == session  # stable for this kernel
    assert namespace["RUNTIME_STAMP"]()["runtime"]["session_id"] == session
    assert "run all cells from the top" in notebook_source("introduction")


def test_every_code_cell_opens_with_a_header_comment():
    for path in (NOTEBOOK_PATH, TOOLSMITH_NOTEBOOK):
        for cell in load(path)["cells"]:
            if cell["cell_type"] == "code":
                assert "".join(cell["source"]).startswith("# "), (path.name, cell["id"])

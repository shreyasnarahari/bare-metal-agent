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
        "steadyhand-server", "steadyhand-notes", "steadyhand-bare-loop", "steadyhand-smoke",
        "steadyhand-sweep", "steadyhand-server-stop", "cleanup",
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
    assert notebook_source("cleanup").startswith("if 'stop_steadyhand_runtime' in globals():")


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

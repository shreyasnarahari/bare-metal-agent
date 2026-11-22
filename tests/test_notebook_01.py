import ast
import json
from pathlib import Path

from toolsmith.recording import JsonlRecorder


NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "01_toolsmith.ipynb"


def notebook_cell(cell_id):
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    return next(cell for cell in notebook["cells"] if cell["id"] == cell_id)


def test_notebook_has_no_destructive_sync_or_stale_error_outputs():
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "reset', '--hard" not in source
    assert 'reset", "--hard' not in source
    assert "fuser" not in source
    assert "sys.modules" not in source
    assert "ensure_gguf(" in source and "ensure_llama_server(" in source
    assert "EXPECTED_MODEL_SIZE" in source
    assert "QWEN35_LLAMA_REF = 'b8233'" in source
    assert "LLAMA_CPP_REF = 'b5401'" in source
    assert "create_app(service, api_key=LOCAL_KEY)" in source
    assert "QWEN35_CACHED_BUILD" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert not any(output.get("output_type") == "error" for output in cell.get("outputs", []))


def test_notebook_streaming_replay_cell_records_valid_fixtures(tmp_path):
    namespace = {
        "DRIVE_RECORDS": tmp_path,
        "HF_TOKEN": "test-secret",
        "LOCAL_KEY": "test-secret",
        "JsonlRecorder": JsonlRecorder,
    }
    source = "".join(notebook_cell("streaming-offline-replay")["source"])

    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

    path = tmp_path / "days-03-04-parser-fixtures.jsonl"
    record = json.loads(path.read_text())
    assert record["event"] == "parser_fixture_replay"
    assert [fixture["name"] for fixture in record["fixtures"]] == [
        "plain",
        "nested",
        "parallel",
    ]
    assert len(record["fixtures"][2]["tool_calls"]) == 2
    assert "test-secret" not in path.read_text()


def test_notebook_grammar_offline_cell_records_artifacts(tmp_path):
    records = tmp_path / "records"
    namespace = {
        "PROJECT_DIR": NOTEBOOK_PATH.parent.parent,
        "DRIVE_ROOT": tmp_path,
        "DRIVE_RECORDS": records,
        "HF_TOKEN": "test-secret",
        "LOCAL_KEY": "test-secret",
    }
    source = "".join(notebook_cell("grammar-offline-corpus")["source"])

    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

    assert len(namespace["grammar_artifacts"]) == 12
    assert len(namespace["unsupported_results"]) == 12
    path = records / "days-04-05-offline-corpus.jsonl"
    record = json.loads(path.read_text())
    assert record["event"] == "grammar_offline_corpus"
    assert len(record["artifacts"]) == 12
    assert all(Path(artifact["path"]).is_file() for artifact in record["artifacts"])
    assert "test-secret" not in path.read_text()


def test_notebook_sampler_offline_cell_records_differential_results(tmp_path):
    records = tmp_path / "records"
    namespace = {
        "PROJECT_DIR": NOTEBOOK_PATH.parent.parent,
        "DRIVE_ROOT": tmp_path,
        "DRIVE_RECORDS": records,
        "HF_TOKEN": "test-secret",
        "LOCAL_KEY": "test-secret",
    }
    for cell_id in ("grammar-offline-corpus", "sampler-offline-differential"):
        source = "".join(notebook_cell(cell_id)["source"])
        exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

    assert namespace["sampler_offline_record"]["artifact_count"] == 12
    assert namespace["sampler_offline_record"]["case_count"] == 90
    assert namespace["synthetic_prefixes"] > 50
    path = records / "days-06-08-automaton.jsonl"
    record = json.loads(path.read_text())
    assert record["event"] == "sampler_offline_differential"
    assert record["synthetic_mask_prefixes"] > 50
    assert "test-secret" not in path.read_text()


def test_notebook_cell_ids_and_python_sources_are_valid():
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    ids = [cell["id"] for cell in notebook["cells"]]

    assert len(ids) == len(set(ids))
    assert ids[-1] == "cleanup"
    assert {
        "grammar-offline-corpus",
        "grammar-validator-comparison",
        "grammar-tool-choice-sdk",
        "grammar-ablation",
        "sampler-offline-differential",
        "sampler-vocabulary",
        "sampler-model-setup",
        "sampler-fixed-generation",
        "sampler-trigger-generation",
        "sampler-benchmarks",
        "day09-notes",
        "day09-offline-replay",
        "day09-qwen3-api",
        "day09-qwen35-tokenizer",
        "day09-qwen35-runtime",
        "day09-qwen35-api",
        "day09-results",
        "day10-notes",
        "day10-offline-validation",
        "day10-eval-setup",
        "day10-runtime-helpers",
        "day10-own-dataset",
        "day10-own-qwen3",
        "day10-own-qwen35",
        "day10-bfcl-pilot",
        "day10-bfcl-qwen3",
        "day10-bfcl-qwen35",
        "day10-bfcl-score",
        "day10-schema-coverage",
        "day10-jsonschema-generation",
        "day10-hf-jsonschema",
        "day10-maskbench",
        "day10-report",
    } <= set(ids)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK_PATH}:{cell['id']}")


def test_notebook_functions_declare_module_state_they_reassign():
    # Assigning an ALL_CAPS notebook global inside a function without `global`
    # makes it local for the whole function, so earlier reads raise UnboundLocalError.
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    module_state = set()
    functions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else []
            module_state.update(
                target.id for target in targets
                if isinstance(target, ast.Name) and target.id.isupper()
            )
        functions.extend(
            (cell["id"], node) for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    undeclared = []
    for cell_id, function in functions:
        declared = {
            name for node in ast.walk(function)
            if isinstance(node, (ast.Global, ast.Nonlocal)) for name in node.names
        }
        assigned = {
            node.id for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        undeclared.extend(
            f"{cell_id}:{function.name}:{name}"
            for name in sorted((assigned & module_state) - declared)
        )
    assert not undeclared


def test_notebook_bfcl_pilot_checkpoints_do_not_collide_with_full_runs():
    source = "".join(notebook_cell("day10-bfcl-pilot")["source"])

    assert "suffix = '' if limit is None else f'-pilot{limit}'" in source
    assert "{mode.value}{suffix}.jsonl'" in source
    assert "-repair{suffix}.jsonl'" in source


def test_notebook_day09_offline_replay_records_protocol_masks_and_repair(tmp_path):
    records = tmp_path / "records"
    namespace = {
        "PROJECT_DIR": NOTEBOOK_PATH.parent.parent,
        "DRIVE_ROOT": tmp_path,
        "DRIVE_RECORDS": records,
        "HF_TOKEN": "test-secret",
        "LOCAL_KEY": "test-secret",
    }
    source = "".join(notebook_cell("day09-offline-replay")["source"])

    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

    assert namespace["day09_offline_record"]["repair_attempts"] == 2
    assert namespace["day09_offline_record"]["mask_checks"] > 10
    record = json.loads((records / "day-09-offline.jsonl").read_text())
    assert record["event"] == "day09_offline_replay"
    assert len(record["protocol_cases"]) == 3
    assert "test-secret" not in (records / "day-09-offline.jsonl").read_text()


def test_notebook_day10_offline_validation_checks_gold_dataset(tmp_path):
    namespace = {
        "PROJECT_DIR": NOTEBOOK_PATH.parent.parent,
        "DRIVE_RECORDS": tmp_path,
        "Path": Path,
    }
    source = "".join(notebook_cell("day10-offline-validation")["source"])

    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

    summary = json.loads((tmp_path / "day-10-offline-summary.json").read_text())
    assert summary["dataset"]["cases"] == 200
    assert summary["dataset"]["tools"] == 10
    assert len(summary["smoke_case_ids"]) == 12
    assert namespace["day10_dataset_summary"]["categories"]["no-tool"] == 25


def run_cell_with_stubbed_provisioning(cell_id, namespace, monkeypatch):
    import toolsmith.provision as provision

    calls = []
    monkeypatch.setattr(provision, "gpu_query", lambda field: {"name": "NVIDIA L4", "compute_cap": "8.9"}[field])
    monkeypatch.setattr(provision, "ensure_llama_server", lambda runtime: calls.append(runtime) or runtime.server_bin)
    monkeypatch.setattr(
        provision, "resolve_gguf",
        lambda repo, *, token, filename=None, select=None, api=None: provision.GgufFile(
            repo, filename or "Qwen3.5-9B-Q4_K_M.gguf", "0123456789abcdef", 1234,
        ),
    )
    monkeypatch.setattr(provision, "ensure_gguf", lambda gguf, **options: calls.append(options) or options["local_path"])
    source = "".join(notebook_cell(cell_id)["source"])
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
    return calls


def test_setup_cells_use_the_shared_provisioning_module(tmp_path, monkeypatch):
    import os

    namespace = {
        "Path": Path, "os": os, "json": json,
        "DRIVE_BUILD_CACHE": tmp_path / "builds", "DRIVE_MODEL_CACHE": tmp_path / "models",
        "HF_TOKEN": "hf-test", "QWEN35_GGUF_REPO": "unsloth/Qwen3.5-9B-GGUF",
    }

    build = run_cell_with_stubbed_provisioning("build-llama-cpp", namespace, monkeypatch)
    download = run_cell_with_stubbed_provisioning("download-model", namespace, monkeypatch)
    qwen35 = run_cell_with_stubbed_provisioning("day09-qwen35-runtime", namespace, monkeypatch)

    assert namespace["GPU_COMPUTE_CAP"] == "89"
    assert namespace["CACHED_BUILD"] == tmp_path / "builds" / "llama.cpp-b5401-cuda-sm89-runtime-v2"
    assert build[0].ref == "b5401" and build[0].build == namespace["LOCAL_BUILD"]
    assert namespace["SERVER_BIN"] == namespace["LOCAL_BUILD"] / "bin" / "llama-server"
    assert callable(namespace["server_environment"]) and callable(namespace["runtime_matches_gpu"])
    assert callable(namespace["hf_hub_download"])  # day09-qwen35-tokenizer still relies on it
    assert (namespace["MODEL_REVISION"], namespace["EXPECTED_MODEL_SIZE"]) == ("0123456789abcdef", 1234)
    assert download[0]["drive_path"] == tmp_path / "models" / "Qwen3-8B-Q4_K_M.gguf"
    assert qwen35[0].ref == "b8233" and qwen35[0].staging == Path("/content/llama.cpp-qwen35-runtime-staging")
    assert namespace["QWEN35_SERVER_BIN"] == namespace["QWEN35_LOCAL_BUILD"] / "bin" / "llama-server"
    assert namespace["QWEN35_DRIVE_MODEL_FILE"] == tmp_path / "models" / "0123456789ab-Qwen3.5-9B-Q4_K_M.gguf"
    assert namespace["qwen35_runtime_metadata"]["model_size"] == 1234

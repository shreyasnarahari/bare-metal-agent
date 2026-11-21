import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from toolsmith.recording import JsonlRecorder


NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "01_toolsmith.ipynb"


def notebook_cell(cell_id):
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    return next(cell for cell in notebook["cells"] if cell["id"] == cell_id)


def load_build_functions(namespace):
    source = "".join(notebook_cell("build-llama-cpp")["source"])
    tree = ast.parse(source)
    names = {
        "remove_directory",
        "server_environment",
        "server_works",
        "restore_cached_build",
        "cache_local_build",
    }
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def write_executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)


def test_notebook_runtime_cache_restores_over_partial_build(tmp_path, monkeypatch):
    local_build = tmp_path / "local-build"
    cached_build = tmp_path / "cached-build"
    server_bin = local_build / "bin" / "llama-server"
    write_executable(server_bin, "print('fake llama.cpp b5401')\n")
    (local_build / "lib").mkdir()
    (local_build / "lib" / "libllama.so").write_bytes(b"library")
    (local_build / "bin" / "libggml.so.0").write_bytes(b"library")
    (local_build / "bin" / "libggml.so").symlink_to("libggml.so.0")
    (local_build / "CMakeCache.txt").write_text("not a runtime artifact")
    monkeypatch.setenv("TOOLSMITH_RUNTIME_STAGING", str(tmp_path / "restore-staging"))

    namespace = load_build_functions(
        {
            "os": os,
            "shutil": shutil,
            "subprocess": subprocess,
            "Path": Path,
            "LLAMA_CPP_REF": "b5401",
            "LOCAL_BUILD": local_build,
            "CACHED_BUILD": cached_build,
            "SERVER_BIN": server_bin,
        }
    )

    assert namespace["server_works"](local_build)
    namespace["cache_local_build"]()
    assert (cached_build / ".complete").read_text() == "b5401\n"
    assert not (cached_build / "CMakeCache.txt").exists()
    assert namespace["server_works"](cached_build)

    write_executable(server_bin, "raise SystemExit(1)\n")
    shutil.rmtree(local_build / "lib")
    assert not namespace["server_works"](local_build)
    assert namespace["restore_cached_build"]()
    assert namespace["server_works"](local_build)
    assert (local_build / "lib" / "libllama.so").is_file()
    assert (local_build / "bin" / "libggml.so").is_symlink()


def test_notebook_runtime_cache_restores_without_completion_marker(tmp_path, monkeypatch):
    builds = tmp_path / "builds"
    local_build = tmp_path / "local-build"
    cached_build = builds / "llama.cpp-b5401-cuda-runtime-v2"
    server_bin = cached_build / "bin" / "llama-server"
    write_executable(server_bin, "print('fake llama.cpp b5401')\n")
    (cached_build / "bin" / "libggml.so").write_bytes(b"library")
    monkeypatch.setenv("TOOLSMITH_RUNTIME_STAGING", str(tmp_path / "restore-staging"))

    namespace = load_build_functions(
        {
            "os": os,
            "shutil": shutil,
            "subprocess": subprocess,
            "Path": Path,
            "LLAMA_CPP_REF": "b5401",
            "LOCAL_BUILD": local_build,
            "CACHED_BUILD": cached_build,
            "SERVER_BIN": local_build / "bin" / "llama-server",
        }
    )

    assert namespace["restore_cached_build"]()
    assert namespace["server_works"](local_build)


def test_notebook_runtime_cache_rejects_a_different_gpu(tmp_path, monkeypatch):
    builds = tmp_path / "builds"
    local_build = tmp_path / "local-build"
    cached_build = builds / "llama.cpp-b5401-cuda-sm89-runtime-v2"
    write_executable(cached_build / "bin" / "llama-server", "print('fake llama.cpp b5401')\n")
    (cached_build / ".compute-cap").write_text("75\n")
    monkeypatch.setenv("TOOLSMITH_RUNTIME_STAGING", str(tmp_path / "restore-staging"))

    namespace = load_build_functions(
        {
            "os": os,
            "shutil": shutil,
            "subprocess": subprocess,
            "Path": Path,
            "LLAMA_CPP_REF": "b5401",
            "GPU_COMPUTE_CAP": "89",
            "LOCAL_BUILD": local_build,
            "CACHED_BUILD": cached_build,
            "SERVER_BIN": local_build / "bin" / "llama-server",
        }
    )

    assert namespace["restore_cached_build"]() is False
    assert not (local_build / "bin" / "llama-server").exists()

    (cached_build / ".compute-cap").write_text("89\n")
    assert namespace["restore_cached_build"]() is True
    assert namespace["server_works"](local_build)


def test_notebook_runtime_cache_ignores_binary_from_another_cache_directory(tmp_path, monkeypatch):
    builds = tmp_path / "builds"
    local_build = tmp_path / "local-build"
    other_gpu = builds / "llama.cpp-b5401-cuda"
    write_executable(other_gpu / "bin" / "llama-server", "print('fake llama.cpp b5401')\n")
    (other_gpu / "CMakeCache.txt").write_text("full cmake tree")
    monkeypatch.setenv("TOOLSMITH_RUNTIME_STAGING", str(tmp_path / "restore-staging"))

    namespace = load_build_functions(
        {
            "os": os,
            "shutil": shutil,
            "subprocess": subprocess,
            "Path": Path,
            "LLAMA_CPP_REF": "b5401",
            "LOCAL_BUILD": local_build,
            "CACHED_BUILD": builds / "llama.cpp-b5401-cuda-sm89-runtime-v2",
            "SERVER_BIN": local_build / "bin" / "llama-server",
        }
    )

    assert namespace["restore_cached_build"]() is False
    assert not (local_build / "bin" / "llama-server").exists()


def test_notebook_has_no_destructive_sync_or_stale_error_outputs():
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "reset', '--hard" not in source
    assert 'reset", "--hard' not in source
    assert "fuser" not in source
    assert "sys.modules" not in source
    assert "force_download=True" in source
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

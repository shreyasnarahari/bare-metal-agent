import ast
import json
from pathlib import Path

from toolsmith.hardware import Hardware
from toolsmith.recording import JsonlRecorder


NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "01_toolsmith.ipynb"


def STAMP():
    return {"runtime": {"gpu_family": "T4", "loadavg": [0.5, 0.5, 0.5], "background_jobs": []}}


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
        "RUNTIME_STAMP": STAMP,
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
    assert record["runtime"]["gpu_family"] == "T4"  # every record carries the runtime stamp


def test_notebook_grammar_offline_cell_records_artifacts(tmp_path):
    records = tmp_path / "records"
    namespace = {
        "PROJECT_DIR": NOTEBOOK_PATH.parent.parent,
        "DRIVE_ROOT": tmp_path,
        "DRIVE_RECORDS": records,
        "HF_TOKEN": "test-secret",
        "LOCAL_KEY": "test-secret",
        "RUNTIME_STAMP": STAMP,
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
        "RUNTIME_STAMP": STAMP,
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
        "RUNTIME_STAMP": STAMP,
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


def stub_provisioning(monkeypatch):
    import toolsmith.artifacts as artifacts
    import toolsmith.provision as provision

    calls = []
    monkeypatch.setattr(provision, "ensure_llama_server", lambda runtime, **options: calls.append((runtime, options)) or runtime.server_bin)
    monkeypatch.setattr(
        provision, "resolve_gguf",
        lambda repo, *, token, filename=None, select=None, api=None: provision.GgufFile(
            repo, filename or "Qwen3.5-9B-Q4_K_M.gguf", "0123456789abcdef", 1234,
        ),
    )
    monkeypatch.setattr(provision, "ensure_gguf", lambda gguf, **options: calls.append(options) or options["local_path"])
    monkeypatch.setattr(artifacts, "is_gguf", lambda path, size: True)
    return calls


def run_cell(cell_id, namespace):
    source = "".join(notebook_cell(cell_id)["source"])
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)


def test_setup_cells_use_the_shared_provisioning_module(tmp_path, monkeypatch, sync_jobs):
    import os

    calls = stub_provisioning(monkeypatch)
    namespace = {
        "Path": Path, "os": os, "json": json,
        "DRIVE_BUILD_CACHE": tmp_path / "builds", "DRIVE_MODEL_CACHE": tmp_path / "models",
        "HF_TOKEN": "hf-test", "HARDWARE": Hardware("NVIDIA L4", "89", 23034, 8, 54000, 200_000),
        "BACKGROUND": sync_jobs,
    }

    run_cell("build-llama-cpp", namespace)
    qwen3_gguf, qwen35_gguf, (b5401, b5401_options), (b8233, b8233_options) = calls

    # The downloads are submitted before the foreground build, and b8233 only after it.
    assert sync_jobs.submitted[:2] == ["qwen3-gguf", "qwen35-gguf"] and sync_jobs.submitted[-1] == "qwen35-server"
    assert namespace["GPU_COMPUTE_CAP"] == "89"
    assert namespace["CACHED_BUILD"] == tmp_path / "builds" / "llama.cpp-b5401-cuda-sm89-runtime-v2"
    assert b5401.ref == "b5401" and b5401.build == namespace["LOCAL_BUILD"] and b5401_options["jobs"] == 8
    assert b8233.ref == "b8233" and b8233.staging == Path("/content/llama.cpp-qwen35-runtime-staging")
    assert b8233_options["nice"] == 10 and b8233_options["defer"] == sync_jobs.submit
    assert namespace["SERVER_BIN"] == namespace["LOCAL_BUILD"] / "bin" / "llama-server"
    assert callable(namespace["server_environment"]) and callable(namespace["runtime_matches_gpu"])
    assert qwen3_gguf["drive_path"] == tmp_path / "models" / "Qwen3-8B-Q4_K_M.gguf"
    assert qwen35_gguf["drive_path"] == tmp_path / "models" / "0123456789ab-Qwen3.5-9B-Q4_K_M.gguf"

    run_cell("download-model", namespace)
    assert callable(namespace["hf_hub_download"])  # day09-qwen35-tokenizer still relies on it
    assert (namespace["MODEL_REVISION"], namespace["EXPECTED_MODEL_SIZE"]) == ("0123456789abcdef", 1234)

    run_cell("day09-qwen35-runtime", namespace)
    assert namespace["QWEN35_SERVER_BIN"] == namespace["QWEN35_LOCAL_BUILD"] / "bin" / "llama-server"
    assert namespace["QWEN35_LOCAL_MODEL"] == Path("/content/qwen35-Qwen3.5-9B-Q4_K_M.gguf")
    assert namespace["QWEN35_DRIVE_MODEL_FILE"] == tmp_path / "models" / "0123456789ab-Qwen3.5-9B-Q4_K_M.gguf"
    assert namespace["qwen35_runtime_metadata"]["model_size"] == 1234
    assert len(calls) == 4  # waiting provisioned nothing new


def test_day10_cells_use_every_slot_and_vcpu():
    helpers = "".join(notebook_cell("day10-runtime-helpers")["source"])
    setup = "".join(notebook_cell("day10-eval-setup")["source"])
    coverage = "".join(notebook_cell("day10-schema-coverage")["source"])
    generation = "".join(notebook_cell("day10-jsonschema-generation")["source"])
    hf = "".join(notebook_cell("day10-hf-jsonschema")["source"])
    maskbench = "".join(notebook_cell("day10-maskbench")["source"])

    assert "DAY10_SLOTS = HARDWARE.day10_slots()" in helpers and "parallel=DAY10_SLOTS" in helpers
    assert "context_size=DAY10_SLOT_CONTEXT * DAY10_SLOTS" in helpers and "vram_free_mib()" in helpers
    assert "parallel=4" not in helpers and "context_size=16384" not in helpers
    assert "BACKGROUND.submit('maskbench-peers', install_maskbench_peers)" in setup and "nice=10" in setup
    assert "clone_pool.submit(checkout_pinned" in setup
    assert "parallel_compile_coverage(" in coverage and "workers=coverage_workers" in coverage
    assert "MirroredFile(" in coverage and "DAY10_RECORDS / 'schema-coverage.jsonl'" in coverage
    assert "recorded_coverage_ids(coverage_store.local)" in coverage and "if file.name not in coverage_done" in coverage
    assert "coverage_store.sync(force=True)" in coverage
    assert generation.count("load_dataset(") == 1 and "workers=DAY10_SLOTS" in generation
    assert "run_with_restart(profile, lambda: generate_jsonschema(profile))" in generation
    assert "read_identified_json(report_path, identity)" in generation
    assert "workers=HARDWARE.hf_processes()" in hf and "run_hf_units(" in hf
    assert "BACKGROUND.wait('maskbench-peers')" in maskbench and "workers=HARDWARE.peer_workers()" in maskbench
    assert "'concurrency': maskbench_concurrency" in maskbench


def test_every_recorder_carries_the_runtime_stamp():
    import re

    notebook = json.loads(NOTEBOOK_PATH.read_text())
    unstamped = []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        for match in re.finditer(r"JsonlRecorder\(", source):
            depth, end = 0, match.end() - 1
            for end in range(match.end() - 1, len(source)):
                depth += {"(": 1, ")": -1}.get(source[end], 0)
                if depth == 0:
                    break
            if "context=RUNTIME_STAMP" not in source[match.start():end]:
                unstamped.append(cell["id"])
    assert unstamped == []


def test_day10_manifests_include_the_runtime_identity_and_size_workers_to_slots():
    helpers = "".join(notebook_cell("day10-runtime-helpers")["source"])
    pilot = "".join(notebook_cell("day10-bfcl-pilot")["source"])

    assert "'runtime': day10_identity(profile)" in helpers and "'identity': day10_identity(profile)" in helpers
    for key in ("'compute_cap'", "'llama_cpp_ref'", "'context_size'", "'parallel'", "'per_slot_context'"):
        assert key in helpers
    assert helpers.count("manifest=manifest, workers=workers") == 1 and "manifest=repair_manifest, workers=workers" in helpers
    assert "manifest=manifest, workers=workers" in pilot and "manifest=repair_manifest, workers=workers" in pilot
    assert "canonical_hash({'config'" not in helpers + pilot


def test_day10_results_are_local_first_and_flushed_to_drive():
    helpers = "".join(notebook_cell("day10-runtime-helpers")["source"])
    pilot = "".join(notebook_cell("day10-bfcl-pilot")["source"])
    setup = "".join(notebook_cell("day10-eval-setup")["source"])
    cleanup = "".join(notebook_cell("cleanup")["source"])

    assert "MirroredFile(DAY10_LOCAL / name, DAY10_RECORDS / name" in helpers
    assert "DAY10_RECORDS / f'own-" not in helpers and "DAY10_RECORDS / f'bfcl-" not in pilot
    assert "day10_checkpoint(f'own-{profile.name}-repair.jsonl')" in helpers
    assert "log_path=DAY10_LOGS / f'{profile.name}-server.log'" in helpers  # no open file on Drive
    assert "copy_file_atomic(server_log, DAY10_RECORDS / server_log.name)" in helpers
    assert "os.environ['HF_DATASETS_CACHE'] = '/content/hf-datasets'" in setup
    for cell_id in ("day10-own-qwen3", "day10-own-qwen35", "day10-bfcl-qwen3", "day10-bfcl-qwen35",
                    "day10-schema-coverage", "day10-jsonschema-generation", "day10-hf-jsonschema", "day10-maskbench"):
        assert f"sync_drive(BACKGROUND, stage='{cell_id}')" in "".join(notebook_cell(cell_id)["source"]), cell_id
    assert "sync_drive(BACKGROUND, stage='cleanup', remount=False)" in cleanup


def test_day10_runs_restart_a_failed_server_once_and_start_it_only_when_needed():
    helpers = "".join(notebook_cell("day10-runtime-helpers")["source"])
    pilot = "".join(notebook_cell("day10-bfcl-pilot")["source"])

    assert "except (InfrastructureFailure, InfrastructureError) as error:" in helpers
    assert "for attempt in (1, 2):" in helpers and "server.log_tail(40)" in helpers
    assert "outputs = run_with_restart(profile, run)" in helpers and "outputs = run_with_restart(profile, run)" in pilot
    # The runtime starts on the first request, not before: a finished run never loads the model.
    assert "Day10Client(profile, 'own', mode)" in helpers and "Day10Client(profile, 'bfcl', mode)" in pilot
    assert "active = start_day10_runtime(profile)" not in helpers + pilot
    assert "if DAY10_ACTIVE.get('profile') != self.profile:" in helpers  # a dead server is not silently restarted


def test_long_cells_check_drive_and_network_calls_cannot_hang():
    setup = "".join(notebook_cell("day10-eval-setup")["source"])
    hardware = "".join(notebook_cell("hardware")["source"])

    for cell_id in ("day10-own-qwen3", "day10-bfcl-qwen35", "day10-schema-coverage", "day10-maskbench"):
        assert "".join(notebook_cell(cell_id)["source"]).startswith("ensure_drive()"), cell_id
    assert setup.count("check=True, timeout=1800)") == 2
    assert "'HF_HUB_DOWNLOAD_TIMEOUT': '60'" in hardware and "'GIT_HTTP_LOW_SPEED_TIME': '60'" in hardware

import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import toolsmith.provision as provision
from toolsmith.provision import (
    GgufFile,
    LlamaCppRuntime,
    cache_dir_for,
    cache_local_build,
    ensure_gguf,
    ensure_llama_server,
    free_port,
    q4_k_m_language_file,
    read_env_file,
    resolve_gguf,
    restore_cached_build,
    server_works,
)

# The runtime-cache cases below moved here from tests/test_notebook_01.py with the
# notebook logic they cover; the scenarios are unchanged.


def write_executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)


def test_runtime_cache_restores_over_partial_build(tmp_path, monkeypatch):
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
    runtime = LlamaCppRuntime("b5401", tmp_path / "source", local_build, cached_build)

    assert server_works(local_build)
    cache_local_build(runtime)
    assert (cached_build / ".complete").read_text() == "b5401\n"
    assert not (cached_build / "CMakeCache.txt").exists()
    assert server_works(cached_build)

    write_executable(server_bin, "raise SystemExit(1)\n")
    (local_build / "lib" / "libllama.so").unlink()
    (local_build / "lib").rmdir()
    assert not server_works(local_build)
    assert restore_cached_build(runtime)
    assert server_works(local_build)
    assert (local_build / "lib" / "libllama.so").is_file()
    assert (local_build / "bin" / "libggml.so").is_symlink()


def test_runtime_cache_restores_without_completion_marker(tmp_path, monkeypatch):
    cached_build = tmp_path / "builds" / "llama.cpp-b5401-cuda-runtime-v2"
    write_executable(cached_build / "bin" / "llama-server", "print('fake llama.cpp b5401')\n")
    (cached_build / "bin" / "libggml.so").write_bytes(b"library")
    monkeypatch.setenv("TOOLSMITH_RUNTIME_STAGING", str(tmp_path / "restore-staging"))
    runtime = LlamaCppRuntime("b5401", tmp_path / "source", tmp_path / "local-build", cached_build)

    assert restore_cached_build(runtime)
    assert server_works(runtime.build)


def test_runtime_cache_rejects_a_different_gpu(tmp_path, monkeypatch):
    cached_build = tmp_path / "builds" / "llama.cpp-b5401-cuda-sm89-runtime-v2"
    write_executable(cached_build / "bin" / "llama-server", "print('fake llama.cpp b5401')\n")
    (cached_build / ".compute-cap").write_text("75\n")
    monkeypatch.setenv("TOOLSMITH_RUNTIME_STAGING", str(tmp_path / "restore-staging"))
    runtime = LlamaCppRuntime("b5401", tmp_path / "source", tmp_path / "local-build", cached_build, "89")

    assert restore_cached_build(runtime) is False
    assert not runtime.server_bin.exists()

    (cached_build / ".compute-cap").write_text("89\n")
    assert restore_cached_build(runtime) is True
    assert server_works(runtime.build)


def test_runtime_cache_ignores_binary_from_another_cache_directory(tmp_path, monkeypatch):
    other_gpu = tmp_path / "builds" / "llama.cpp-b5401-cuda"
    write_executable(other_gpu / "bin" / "llama-server", "print('fake llama.cpp b5401')\n")
    (other_gpu / "CMakeCache.txt").write_text("full cmake tree")
    monkeypatch.setenv("TOOLSMITH_RUNTIME_STAGING", str(tmp_path / "restore-staging"))
    runtime = LlamaCppRuntime(
        "b5401", tmp_path / "source", tmp_path / "local-build",
        tmp_path / "builds" / "llama.cpp-b5401-cuda-sm89-runtime-v2",
    )

    assert restore_cached_build(runtime) is False
    assert not runtime.server_bin.exists()


def test_cache_directory_names_the_tag_and_gpu(tmp_path):
    assert cache_dir_for(tmp_path, "b8233", "89") == tmp_path / "llama.cpp-b8233-cuda-sm89-runtime-v2"


# ensure_llama_server: reuse, restore, or build


def test_ensure_llama_server_reuses_a_valid_local_build(tmp_path, monkeypatch):
    local_build = tmp_path / "local-build"
    write_executable(local_build / "bin" / "llama-server", "print('ok')\n")
    (local_build / ".compute-cap").write_text("89\n")
    monkeypatch.setattr(provision, "build_from_source", lambda runtime, **options: pytest.fail("must not build"))
    runtime = LlamaCppRuntime("b5401", tmp_path / "source", local_build, tmp_path / "cache", "89")

    assert ensure_llama_server(runtime) == local_build / "bin" / "llama-server"


def test_ensure_llama_server_builds_and_caches_when_nothing_is_cached(tmp_path, monkeypatch):
    runtime = LlamaCppRuntime("b8233", tmp_path / "source", tmp_path / "local-build", tmp_path / "cache", "89",
                              staging=tmp_path / "staging")
    built = []

    def fake_build(target, **options):
        built.append(target.ref)
        write_executable(target.server_bin, "print('built')\n")

    monkeypatch.setattr(provision, "build_from_source", fake_build)

    assert ensure_llama_server(runtime) == runtime.server_bin
    assert built == ["b8233"]
    assert (runtime.cache / ".complete").read_text() == "b8233\n"
    assert (runtime.cache / ".compute-cap").read_text() == "89\n"


def test_ensure_llama_server_refuses_a_build_for_another_gpu(tmp_path, monkeypatch):
    runtime = LlamaCppRuntime("b8233", tmp_path / "source", tmp_path / "local-build", tmp_path / "cache", "89",
                              staging=tmp_path / "staging")
    monkeypatch.setattr(provision, "build_from_source", lambda target, **options: write_executable(target.server_bin, "raise SystemExit(1)\n"))

    with pytest.raises(RuntimeError, match="failed runtime validation"):
        ensure_llama_server(runtime)


# GGUF weights


def entry(name, size):
    return SimpleNamespace(rfilename=name, size=size)


class FakeApi:
    def __init__(self, siblings, sha="abc123def4567890"):
        self.info = SimpleNamespace(siblings=siblings, sha=sha)

    def model_info(self, repo, files_metadata):
        assert files_metadata is True
        return self.info


def test_resolve_gguf_pins_revision_and_size_by_name_or_selector():
    api = FakeApi([entry("Qwen3-8B-Q4_K_M.gguf", 10), entry("Qwen3-8B-Q8_0.gguf", 20)])
    by_name = resolve_gguf("Qwen/Qwen3-8B-GGUF", token="t", filename="Qwen3-8B-Q4_K_M.gguf", api=api)
    assert by_name == GgufFile("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", "abc123def4567890", 10)

    qwen35 = FakeApi([entry("Qwen3.5-9B-Q4_K_M.gguf", 30), entry("mmproj-Q4_K_M.gguf", 5), entry("Qwen3.5-9B-Q8_0.gguf", 40)])
    assert resolve_gguf("unsloth/Qwen3.5-9B-GGUF", token="t", select=q4_k_m_language_file, api=qwen35).size == 30

    two = FakeApi([entry("a-Q4_K_M.gguf", 1), entry("b-Q4_K_M.gguf", 2)])
    with pytest.raises(RuntimeError, match="exactly one"):
        resolve_gguf("repo", token="t", select=q4_k_m_language_file, api=two)
    with pytest.raises(ValueError):
        resolve_gguf("repo", token="t", api=two)


def write_gguf(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * (size - 4))
    return path


def test_ensure_gguf_redownloads_once_when_the_cached_download_is_corrupt(tmp_path):
    gguf = GgufFile("repo", "model.gguf", "rev", 64)
    calls = []

    def download(**options):
        calls.append(options.get("force_download", False))
        target = tmp_path / "hf" / ("fresh.gguf" if options.get("force_download") else "stale.gguf")
        if options.get("force_download"):
            return str(write_gguf(target, 64))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not a gguf")
        return str(target)

    local = ensure_gguf(gguf, local_path=tmp_path / "local.gguf", drive_path=tmp_path / "drive" / "model.gguf",
                        token="t", cache_dir=str(tmp_path / "hf"), download=download)

    assert calls == [False, True]
    assert local.read_bytes()[:4] == b"GGUF"
    assert (tmp_path / "drive" / "model.gguf").stat().st_size == 64  # cached for the next runtime


def test_ensure_gguf_prefers_the_drive_copy_and_never_downloads(tmp_path):
    gguf = GgufFile("repo", "model.gguf", "rev", 32)
    write_gguf(tmp_path / "drive" / "model.gguf", 32)

    local = ensure_gguf(gguf, local_path=tmp_path / "local.gguf", drive_path=tmp_path / "drive" / "model.gguf",
                        token="t", cache_dir="unused", download=lambda **options: pytest.fail("must not download"))

    assert local.stat().st_size == 32


def test_ensure_gguf_fails_when_both_downloads_are_corrupt(tmp_path):
    def download(**options):
        target = tmp_path / "bad.gguf"
        target.write_bytes(b"nope")
        return str(target)

    with pytest.raises(RuntimeError, match="failed GGUF validation"):
        ensure_gguf(GgufFile("repo", "m.gguf", "rev", 64), local_path=tmp_path / "l.gguf",
                    drive_path=tmp_path / "d" / "m.gguf", token=None, cache_dir="c", download=download)


# Small helpers


def test_read_env_file_handles_exports_quotes_and_comments(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# secrets\nexport HF_TOKEN='hf-test'\nLOCAL_KEY=\"local-test\"\nBROKEN\n\nEMPTY=\n")

    assert read_env_file(path) == {"HF_TOKEN": "hf-test", "LOCAL_KEY": "local-test", "EMPTY": ""}
    assert read_env_file(tmp_path / "missing.env") == {}


def test_free_port_skips_a_port_in_use():
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        taken = held.getsockname()[1]
        assert free_port("127.0.0.1", taken, span=5) != taken


def test_provision_imports_only_the_standard_library_up_front():
    import toolsmith.background as background
    import toolsmith.hardware as hardware

    allowed_local = {"from .artifacts", "from .background", "from .hardware"}
    for module in (provision, background, hardware):
        with open(module.__file__, encoding="utf-8") as handle:
            top = [line for line in handle if line.startswith(("import ", "from "))]
        assert not any("huggingface_hub" in line or "httpx" in line for line in top), module.__name__
        local = {" ".join(line.split()[:2]) for line in top if line.startswith("from .")}
        assert local <= allowed_local, module.__name__


# T4-safe builds and background write-back


def test_build_uses_the_given_jobs_nice_level_and_a_log(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(provision, "run_logged", lambda command, **options: commands.append((command, options)))
    (tmp_path / "source" / ".git").mkdir(parents=True)
    runtime = LlamaCppRuntime("b8233", tmp_path / "source", tmp_path / "build", tmp_path / "cache", "75")

    provision.build_from_source(runtime, jobs=2, nice=10)

    configure, compile_ = commands[-2], commands[-1]
    assert "-DCMAKE_CUDA_ARCHITECTURES=75" in configure[0]
    assert compile_[0][-2:] == ["-j", "2"] and compile_[1]["nice"] == 10
    assert all(options["log_path"] == tmp_path / "build.log" for _, options in commands)
    assert not any(command[1] == "clone" for command, _ in commands)  # the source checkout is reused
    assert all(options["stall_seconds"] for _, options in commands)  # nothing can hang forever
    fetch = next(options for command, options in commands if "fetch" in command)
    assert fetch["env"]["GIT_HTTP_LOW_SPEED_LIMIT"] == "1000" and fetch["env"]["GIT_HTTP_LOW_SPEED_TIME"] == "60"


def test_default_build_jobs_follow_cpus_and_memory(monkeypatch):
    monkeypatch.setattr(provision, "visible_cpus", lambda: 2)
    monkeypatch.setattr(provision, "total_ram_mib", lambda: 13000)
    assert provision.default_build_jobs() == 2
    monkeypatch.setattr(provision, "visible_cpus", lambda: 8)
    monkeypatch.setattr(provision, "total_ram_mib", lambda: 6 * 1024)
    assert provision.default_build_jobs() == 3


def test_a_fresh_build_is_cached_through_defer(tmp_path, monkeypatch):
    runtime = LlamaCppRuntime("b8233", tmp_path / "source", tmp_path / "local-build", tmp_path / "cache", "89",
                              staging=tmp_path / "staging")
    deferred = []
    monkeypatch.setattr(provision, "build_from_source",
                        lambda target, **options: write_executable(target.server_bin, "print('built')\n"))

    assert ensure_llama_server(runtime, defer=lambda name, fn, *args: deferred.append((name, fn, args))) == runtime.server_bin

    assert not runtime.cache.exists()  # nothing written to Drive yet
    assert (runtime.build / ".compute-cap").read_text() == "89\n"
    name, fn, args = deferred[0]
    assert name == "cache llama.cpp b8233 to Drive" and fn is cache_local_build
    fn(*args)
    assert (runtime.cache / ".complete").read_text() == "b8233\n"


def test_ensure_gguf_downloads_into_place_without_a_second_copy(tmp_path):
    gguf = GgufFile("repo", "model.gguf", "rev", 64)
    seen = {}

    def download(**options):
        seen.update(options)
        return str(write_gguf(Path(options["local_dir"]) / options["filename"], 64))

    deferred = []
    local = ensure_gguf(gguf, local_path=tmp_path / "local.gguf", drive_path=tmp_path / "drive" / "model.gguf",
                        token="t", cache_dir=str(tmp_path / "download"), download=download,
                        defer=lambda name, fn, *args: deferred.append((name, fn, args)))

    assert seen["local_dir"] == str(tmp_path / "download") and "cache_dir" not in seen
    assert local.stat().st_size == 64 and not (tmp_path / "download" / "model.gguf").exists()
    assert not (tmp_path / "drive" / "model.gguf").exists()  # the Drive copy was deferred
    name, fn, args = deferred[0]
    assert name == "cache model.gguf to Drive"
    fn(*args)
    assert (tmp_path / "drive" / "model.gguf").stat().st_size == 64


def test_ensure_gguf_refuses_a_download_that_cannot_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(provision, "_free_bytes", lambda path: 1024**3)

    with pytest.raises(RuntimeError, match="Not enough disk"):
        ensure_gguf(GgufFile("repo", "m.gguf", "rev", 5 * 1024**3), local_path=tmp_path / "l.gguf",
                    drive_path=tmp_path / "d" / "m.gguf", token=None, cache_dir="c",
                    download=lambda **options: pytest.fail("must not download"))

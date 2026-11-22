"""Provision a Colab-style runtime: a validated llama.cpp server build and GGUF weights.

Shared by ``notebooks/01_toolsmith.ipynb`` and ``notebooks/02_steadyhand.ipynb``
so the two notebooks restore, build, and cache the same way. Builds and models
are cached on Drive and copied to local disk before use; every executable is
checked with ``--version`` and every GGUF by magic bytes and exact size.

Every function may run inside a :class:`toolsmith.background.BackgroundJobs`
job: messages go through ``say`` and commands through ``run_logged``, so a
background build or download writes to its job log instead of the notebook.
Pass ``defer=BACKGROUND.submit`` to move the slow Drive write-back off the
critical path once the local copy is validated.

This module imports only the standard library and other stdlib-only toolsmith
modules at import time, so a notebook can use ``read_env_file`` before
``pip install``. ``huggingface_hub`` is imported only by the functions that
need it.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import copy_file_atomic, is_gguf
from .background import last_progress, run_logged, say
from .hardware import build_jobs_for, total_ram_mib, visible_cpus

LLAMA_CPP_GIT = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_STAGING = "/content/llama.cpp-runtime-staging"
DISK_HEADROOM = 2 * 1024**3
CLONE_STALL_SECONDS = 600  # a clone that prints nothing for 10 minutes is stuck
BUILD_STALL_SECONDS = 1200  # one CUDA object can take minutes on 2 vCPUs, never 20
# git aborts a transfer that stays under 1 KB/s for a minute instead of hanging on a dead connection.
GIT_ENVIRONMENT = {"GIT_HTTP_LOW_SPEED_LIMIT": "1000", "GIT_HTTP_LOW_SPEED_TIME": "60"}

Defer = Callable[..., Any]  # BackgroundJobs.submit(name, fn)


def read_env_file(path: str | Path) -> dict[str, str]:
    """``KEY=value`` lines from a dotenv-style file; missing file means no values."""
    values: dict[str, str] = {}
    path = Path(path)
    if not path.is_file():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def gpu_query(field: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[0].strip()


def free_port(host: str = "127.0.0.1", preferred: int = 8080, span: int = 20) -> int:
    """The first port from ``preferred`` that binds. Never kills a process."""
    for port in range(preferred, preferred + span):
        with socket.socket() as probe:
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"No free port on {host} from {preferred} through {preferred + span - 1}. No process was killed."
    )


def remove_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def server_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    paths = [str(root / "lib"), str(root / "bin")]
    existing = environment.get("LD_LIBRARY_PATH")
    if existing:
        paths.append(existing)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    return environment


def server_works(root: Path) -> bool:
    binary = root / "bin" / "llama-server"
    if not binary.is_file():
        return False
    try:
        binary.chmod(binary.stat().st_mode | 0o755)
        result = subprocess.run(
            [str(binary), "--version"],
            cwd=str(binary.parent),
            env=server_environment(root),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        say(f"Server validation failed for {binary}: {error}")
        return False
    if result.returncode != 0:
        say(f"Server validation failed for {binary}: {result.stderr.strip()}")
        return False
    say(result.stdout.strip() or result.stderr.strip())
    return True


def runtime_matches_gpu(root: Path, compute_cap: str) -> bool:
    marker = root / ".compute-cap"
    recorded = marker.read_text().strip() if marker.is_file() else ""
    if recorded == compute_cap:
        return True
    say(f"{root} was built for CUDA architecture {recorded or 'unknown'}, but this GPU is {compute_cap}.")
    return False


def cache_dir_for(drive_build_cache: Path, ref: str, compute_cap: str) -> Path:
    return Path(drive_build_cache) / f"llama.cpp-{ref}-cuda-sm{compute_cap}-runtime-v2"


@dataclass(frozen=True)
class LlamaCppRuntime:
    """One llama.cpp tag: where its source, local build, and Drive cache live."""

    ref: str
    source: Path
    build: Path
    cache: Path
    compute_cap: str | None = None  # None skips the CUDA-architecture check
    staging: Path | None = None  # default: $TOOLSMITH_RUNTIME_STAGING or /content/llama.cpp-runtime-staging

    @property
    def server_bin(self) -> Path:
        return self.build / "bin" / "llama-server"

    def staging_dir(self) -> Path:
        if self.staging is not None:
            return Path(self.staging)
        return Path(os.environ.get("TOOLSMITH_RUNTIME_STAGING", DEFAULT_STAGING))


def _list_directory(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except FileNotFoundError:
        return []
    except OSError as error:
        say(f"Could not list {path}: {error}")
        return []


def _usable_cache(runtime: LlamaCppRuntime) -> bool:
    candidate = runtime.cache
    # A fresh Drive mount can hide children from stat until the directory is listed.
    _list_directory(candidate)
    _list_directory(candidate / "bin")
    marker = candidate / ".complete"
    if marker.is_file() and marker.read_text() != runtime.ref + "\n":
        say(f"Ignoring {candidate}: completion marker does not match {runtime.ref}")
        return False
    binary = candidate / "bin" / "llama-server"
    if not binary.is_file():
        say(f"No cached llama-server at {binary}")
        return False
    if runtime.compute_cap:
        cap_marker = candidate / ".compute-cap"
        recorded = cap_marker.read_text().strip() if cap_marker.is_file() else ""
        if recorded != runtime.compute_cap:
            say(
                f"Ignoring {candidate}: CUDA architecture {recorded or 'unknown'} "
                f"does not match this GPU ({runtime.compute_cap})."
            )
            return False
    return True


def _copy_cached_runtime(source: Path, destination: Path) -> None:
    # An older notebook stored the whole CMake tree without a completion marker.
    if (source / "CMakeCache.txt").is_file() and not (source / ".complete").is_file():
        (destination / "bin").mkdir(parents=True)
        shutil.copy2(source / "bin" / "llama-server", destination / "bin" / "llama-server")
        for folder_name in ("bin", "lib"):
            folder = source / folder_name
            if not folder.is_dir():
                continue
            target_folder = destination / folder_name
            target_folder.mkdir(parents=True, exist_ok=True)
            for artifact in folder.iterdir():
                if artifact.name == "llama-server":
                    continue
                if not (artifact.name.startswith("lib") and ".so" in artifact.name):
                    continue
                target = target_folder / artifact.name
                if artifact.is_symlink():
                    target.symlink_to(os.readlink(artifact))
                else:
                    shutil.copy2(artifact, target)
        return
    shutil.copytree(source, destination, symlinks=True)


def restore_cached_build(runtime: LlamaCppRuntime) -> bool:
    """Copy a validated runtime from the Drive cache to the local build directory."""
    _list_directory(runtime.cache.parent)
    if not _usable_cache(runtime):
        return False
    staging = runtime.staging_dir()
    remove_directory(staging)
    say(f"Restoring runtime from {runtime.cache}")
    _copy_cached_runtime(runtime.cache, staging)
    if not server_works(staging):
        remove_directory(staging)
        say("Cached runtime is invalid; rebuilding locally.")
        return False
    remove_directory(runtime.build)
    staging.rename(runtime.build)
    return True


def cache_local_build(runtime: LlamaCppRuntime) -> None:
    """Store only the runtime artifacts of a validated local build on Drive."""
    staging = runtime.cache.with_name(runtime.cache.name + "-staging")
    remove_directory(staging)
    (staging / "bin").mkdir(parents=True)
    shutil.copy2(runtime.server_bin, staging / "bin" / "llama-server")
    for artifact in (runtime.build / "bin").iterdir():
        if artifact.name.startswith("lib") and ".so" in artifact.name:
            if artifact.is_symlink():
                (staging / "bin" / artifact.name).symlink_to(os.readlink(artifact))
            else:
                shutil.copy2(artifact, staging / "bin" / artifact.name)
    if (runtime.build / "lib").is_dir():
        shutil.copytree(runtime.build / "lib", staging / "lib", symlinks=True)
    if runtime.compute_cap:
        (runtime.build / ".compute-cap").write_text(runtime.compute_cap + "\n")
        (staging / ".compute-cap").write_text(runtime.compute_cap + "\n")
    (staging / ".complete").write_text(runtime.ref + "\n")
    remove_directory(runtime.cache)
    staging.rename(runtime.cache)
    say(f"Cached validated runtime at {runtime.cache}")


def default_build_jobs() -> int:
    return build_jobs_for(visible_cpus(), total_ram_mib())


def build_log_path(runtime: LlamaCppRuntime) -> Path:
    return runtime.build.with_name(runtime.build.name + ".log")


def build_from_source(
    runtime: LlamaCppRuntime,
    *,
    jobs: int | None = None,
    nice: int = 0,
    log_path: Path | None = None,
) -> None:
    """Clone, configure, and compile ``llama-server`` for this GPU's architecture only.

    ``jobs`` defaults to one per vCPU capped at one per 2 GiB of RAM, so a 2-vCPU,
    13 GB T4 runtime compiles with two nvcc jobs instead of running out of memory.
    Output goes to ``log_path``; progress is reported every 30 seconds.
    """
    jobs = jobs or default_build_jobs()
    log_path = Path(log_path or build_log_path(runtime))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cuda_arch = [f"-DCMAKE_CUDA_ARCHITECTURES={runtime.compute_cap}"] if runtime.compute_cap else []
    flags = ["-DGGML_CUDA=ON", *cuda_arch, "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=ON"]
    config = {"ref": runtime.ref, "compute_cap": runtime.compute_cap, "flags": flags}
    marker = runtime.build / ".build-config"
    # After a kernel restart /content survives: an interrupted build with the same configuration
    # resumes where make stopped instead of recompiling every CUDA object.
    resumable = (
        marker.is_file() and (runtime.build / "CMakeCache.txt").is_file()
        and json.loads(marker.read_text()) == config
    )
    if resumable:
        say(f"Resuming the interrupted llama.cpp {runtime.ref} build in {runtime.build}")
    else:
        log_path.write_text("")
        remove_directory(runtime.build)

    def progress(path: Path) -> None:
        say(f"[llama.cpp {runtime.ref}] building with {jobs} jobs: {last_progress(path) or 'starting'}")

    say(f"Building llama.cpp {runtime.ref} for sm_{runtime.compute_cap or 'native'} with {jobs} jobs; log: {log_path}")
    git = {"log_path": log_path, "env": {**os.environ, **GIT_ENVIRONMENT}, "stall_seconds": CLONE_STALL_SECONDS}
    if not (runtime.source / ".git").exists():
        run_logged(["git", "clone", "--progress", "--filter=blob:none", LLAMA_CPP_GIT, runtime.source], **git)
    run_logged(["git", "-C", runtime.source, "fetch", "--progress", "--depth", "1", "origin", "tag", runtime.ref], **git)
    run_logged(["git", "-C", runtime.source, "checkout", "--detach", runtime.ref], **git)
    run_logged(
        ["cmake", "-S", runtime.source, "-B", runtime.build, *flags],
        log_path=log_path, nice=nice, progress=progress, stall_seconds=BUILD_STALL_SECONDS,
    )
    runtime.build.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(config))
    run_logged(
        ["cmake", "--build", runtime.build, "--target", "llama-server", "-j", str(jobs)],
        log_path=log_path, nice=nice, progress=progress, stall_seconds=BUILD_STALL_SECONDS,
    )


def _matches(runtime: LlamaCppRuntime, root: Path) -> bool:
    return runtime.compute_cap is None or runtime_matches_gpu(root, runtime.compute_cap)


def ensure_llama_server(
    runtime: LlamaCppRuntime,
    *,
    jobs: int | None = None,
    nice: int = 0,
    defer: Defer | None = None,
) -> Path:
    """A validated local llama-server: reuse it, else restore from Drive, else build and cache it.

    With ``defer`` (``BACKGROUND.submit``), a fresh build is cached to Drive in
    the background and this returns as soon as the local build validates.
    """
    if server_works(runtime.build) and _matches(runtime, runtime.build):
        say(f"Using validated local build: {runtime.server_bin}")
    elif not restore_cached_build(runtime):
        build_from_source(runtime, jobs=jobs, nice=nice)
        if not server_works(runtime.build):
            raise RuntimeError(
                "The local build finished but llama-server failed runtime validation. "
                f"Build log: {build_log_path(runtime)}"
            )
        if runtime.compute_cap:
            (runtime.build / ".compute-cap").write_text(runtime.compute_cap + "\n")
        if defer is None:
            cache_local_build(runtime)
        else:
            defer(f"cache llama.cpp {runtime.ref} to Drive", cache_local_build, runtime)
    if not (server_works(runtime.build) and _matches(runtime, runtime.build)):
        raise RuntimeError(f"llama-server is not usable for this GPU at {runtime.server_bin}")
    return runtime.server_bin


@dataclass(frozen=True)
class GgufFile:
    repo: str
    filename: str
    revision: str
    size: int


def q4_k_m_language_file(siblings: Sequence[Any]) -> list[Any]:
    """Q4_K_M language GGUFs in a repo listing (the multimodal projector excluded)."""
    return [
        entry
        for entry in siblings
        if entry.rfilename.endswith(".gguf")
        and "Q4_K_M" in entry.rfilename
        and "mmproj" not in entry.rfilename.lower()
    ]


def resolve_gguf(
    repo: str,
    *,
    token: str | None,
    filename: str | None = None,
    select: Callable[[Sequence[Any]], list[Any]] | None = None,
    api: Any = None,
) -> GgufFile:
    """Pin a GGUF to the repo's current revision and exact size, by name or by a selector."""
    if (filename is None) == (select is None):
        raise ValueError("Give exactly one of filename or select")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    info = api.model_info(repo, files_metadata=True)
    if filename is not None:
        candidates = [entry for entry in info.siblings if entry.rfilename == filename]
    else:
        candidates = select(info.siblings)
    if len(candidates) != 1 or candidates[0].size is None:
        raise RuntimeError(
            f"Expected exactly one sized GGUF in {repo}, found: {[entry.rfilename for entry in candidates]}"
        )
    return GgufFile(repo=repo, filename=candidates[0].rfilename, revision=info.sha, size=candidates[0].size)


def _copy_with_progress(source: Path, destination: Path, expected_size: int) -> Path:
    state = {"last_step": -1}

    def report(copied: int, total: int) -> None:
        percent = int(copied * 100 / total) if total else 100
        step = percent // 20
        if step != state["last_step"]:
            say(f"{destination.name}: {percent}% ({copied:,}/{total:,} bytes)")
            state["last_step"] = step

    return copy_file_atomic(source, destination, expected_size=expected_size, progress=report)


def _free_bytes(path: Path) -> int:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free


def require_disk(path: Path, needed: int) -> None:
    """Fail before a copy or download that cannot fit, instead of halfway through it."""
    free = _free_bytes(path)
    if free < needed + DISK_HEADROOM:
        raise RuntimeError(
            f"Not enough disk at {path}: {free / 1024**3:.1f} GiB free, {(needed + DISK_HEADROOM) / 1024**3:.1f} GiB "
            "needed. Delete unused files under /content or use a runtime with a larger disk."
        )


def _move_into_place(source: Path, destination: Path, expected_size: int) -> None:
    """Rename when possible (same filesystem, no second copy); copy and delete otherwise."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, destination)
    except OSError:
        _copy_with_progress(source, destination, expected_size)
        source.unlink(missing_ok=True)


def ensure_gguf(
    gguf: GgufFile,
    *,
    local_path: Path,
    drive_path: Path,
    token: str | None,
    cache_dir: str,
    download: Callable[..., str] | None = None,
    defer: Defer | None = None,
) -> Path:
    """A validated local GGUF: copied from the Drive cache, or downloaded (re-downloaded once if corrupt).

    A download lands in ``cache_dir`` (``local_dir`` mode) and is renamed onto
    ``local_path``, so the 5-6 GB file is never stored twice on local disk.
    With ``defer``, the copy back to Drive runs in the background.
    """
    if download is None:
        from huggingface_hub import hf_hub_download as download
    local_path, drive_path = Path(local_path), Path(drive_path)
    try:
        list(drive_path.parent.iterdir())  # a fresh Drive mount can hide files until listed
    except OSError as error:
        say(f"Could not list {drive_path.parent}: {error}")
    if not is_gguf(local_path, gguf.size):
        require_disk(local_path.parent, gguf.size)
        if is_gguf(drive_path, gguf.size):
            say(f"Copying cached GGUF from {drive_path}")
            _copy_with_progress(drive_path, local_path, gguf.size)
        else:
            say(f"Downloading {gguf.filename} at revision {gguf.revision}")
            options = {
                "repo_id": gguf.repo,
                "filename": gguf.filename,
                "revision": gguf.revision,
                "local_dir": cache_dir,
                "token": token,
            }
            downloaded = Path(download(**options))
            if not is_gguf(downloaded, gguf.size):
                downloaded = Path(download(**options, force_download=True))
            if not is_gguf(downloaded, gguf.size):
                raise RuntimeError(f"Downloaded model failed GGUF validation: {downloaded}")
            _move_into_place(downloaded, local_path, gguf.size)
    if not is_gguf(local_path, gguf.size):
        raise RuntimeError(f"Local GGUF validation failed: {local_path}")
    if not is_gguf(drive_path, gguf.size):
        say(f"Caching validated GGUF in {drive_path}")
        if defer is None:
            _copy_with_progress(local_path, drive_path, gguf.size)
        else:
            defer(f"cache {drive_path.name} to Drive", _copy_with_progress, local_path, drive_path, gguf.size)
    return local_path


__all__ = [
    "DEFAULT_STAGING",
    "GgufFile",
    "LlamaCppRuntime",
    "build_from_source",
    "build_log_path",
    "cache_dir_for",
    "cache_local_build",
    "default_build_jobs",
    "ensure_gguf",
    "ensure_llama_server",
    "free_port",
    "gpu_query",
    "q4_k_m_language_file",
    "read_env_file",
    "remove_directory",
    "require_disk",
    "resolve_gguf",
    "restore_cached_build",
    "runtime_matches_gpu",
    "server_environment",
    "server_works",
]

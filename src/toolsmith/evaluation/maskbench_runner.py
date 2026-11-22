from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..background import run_logged
from .metrics import percentile
from .runner import atomic_write_json

PEER_FLAGS = ("--llg", "--xgr", "--llamacpp", "--outlines", "--grid")


def run_maskbench_files(
    engine: Any,
    files: list[str | Path],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    try:
        from maskbench import runner as maskbench_runner
    except ImportError as exc:
        raise ImportError("Install the pinned MaskBench package for mask evaluation") from exc
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    maskbench_runner.output_path = str(destination)
    engine.multi = False
    results = []
    for file in files:
        path = Path(file)
        output = destination / path.name
        if output.exists():
            results.append(json.loads(output.read_text()))
            continue
        result = maskbench_runner.process_file(engine, str(path))
        results.append(result)
    return results


def timing_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    mask_times = [
        value
        for result in results
        for value in result.get("all_mask_us", [])
    ]
    compile_times = [result["ttfm_us"] for result in results if "ttfm_us" in result]
    probabilities = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0)
    return {
        "schemas": len(results),
        "compile_errors": sum("compile_error" in result for result in results),
        "validation_errors": sum(
            "validation_error" in result and "should reject" not in result["validation_error"]
            for result in results
        ),
        "invalidation_errors": sum(
            "validation_error" in result and "should reject" in result["validation_error"]
            for result in results
        ),
        "tokens": sum(result.get("num_tokens", 0) for result in results),
        "ttfm_us": {
            f"p{int(probability * 1000) / 10:g}": percentile(compile_times, probability)
            for probability in probabilities
        },
        "tbm_us": {
            f"p{int(probability * 1000) / 10:g}": percentile(mask_times, probability)
            for probability in probabilities
        },
    }


def peer_command(python: str | Path, flag: str, tokenizer: str, output: Path, files: Sequence[str | Path]) -> list[str]:
    return [
        str(python), "-m", "maskbench.runner", flag, "--num-threads", "1",
        "--tokenizer", tokenizer, "--output", str(output), *map(str, files),
    ]


def run_peer_engines(
    flags: Sequence[str],
    *,
    python: str | Path,
    tokenizer: str,
    files: Sequence[str | Path],
    output_root: str | Path,
    env: Mapping[str, str],
    workers: int,
    alongside: Callable[[], Any] | None = None,
    stamp: Mapping[str, Any] | None = None,
    command: Callable[[str | Path, str, str, Path, Sequence[str | Path]], list[str]] = peer_command,
) -> tuple[dict[str, Path], Any, dict[str, Any]]:
    """Run each peer engine as a subprocess, ``workers`` at a time, while ``alongside()`` runs here.

    Each peer keeps ``--num-threads 1``; running several at once (and the
    in-kernel Toolsmith engine as ``alongside``) uses every vCPU, so every
    output directory gets a ``run.json`` saying how many engines shared the
    machine. Returns the output directories, ``alongside``'s result, and that
    concurrency record.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    output_root = Path(output_root)
    concurrency = {
        "peer_processes_at_once": min(workers, len(flags)),
        "peer_threads_each": 1,
        "toolsmith_in_kernel_concurrently": alongside is not None,
        "engines_total": len(flags) + (alongside is not None),
    }

    def run_one(flag: str) -> Path:
        output = output_root / flag.lstrip("-")
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / "run.json", {"engine": flag, "concurrency": concurrency, **(stamp or {})})
        run_logged(command(python, flag, tokenizer, output, files), log_path=output / "runner.log", env=dict(env))
        return output

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(flags))), thread_name_prefix="maskbench-peer") as pool:
        futures = {flag: pool.submit(run_one, flag) for flag in flags}
        extra = alongside() if alongside is not None else None
        outputs = {flag: future.result() for flag, future in futures.items()}
    return outputs, extra, concurrency


__all__ = ["PEER_FLAGS", "peer_command", "run_maskbench_files", "run_peer_engines", "timing_summary"]

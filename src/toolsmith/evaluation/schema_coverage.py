"""Which JSONSchemaBench schemas Toolsmith's compiler supports, and whether its grammars are exact.

``compile_coverage`` classifies each schema (compiled, explicitly unsupported,
invalid, over a limit, or crashed) and times the compile (TTFM, in
microseconds). Each schema compiles in its own process with memory, CPU, and
wall-clock limits, so a pathological schema cannot take down the notebook.
``replay_schema`` then runs the benchmark's valid and invalid instances through
the automaton: a valid instance rejected, or an invalid one accepted, is a
grammar bug.
"""

from __future__ import annotations

import json
import multiprocessing
import resource
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from ..grammar import GBNFCompiler, GrammarConfigurationError, SchemaNotSupportedError
from ..sampling.automaton import AutomatonLimitError, InvalidGrammarError, PushdownAutomaton


@dataclass(frozen=True)
class CoverageResult:
    """One schema's compile outcome, time to first mask, and (if unsupported) where and why."""

    unique_id: str
    status: str
    ttfm_us: int
    grammar_sha256: str | None = None
    pointer: str | None = None
    keyword: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_coverage(schema: dict[str, Any], unique_id: str = "unknown") -> CoverageResult:
    """Compile one schema to a grammar and an automaton, classifying any failure."""
    started = time.perf_counter()
    try:
        compiled = GBNFCompiler().compile_schema(schema)
        PushdownAutomaton.from_compiled(compiled)
    except SchemaNotSupportedError as exc:
        status = "invalid_schema" if str(exc).startswith("Invalid Draft") else "explicit_unsupported"
        return CoverageResult(
            unique_id,
            status,
            int((time.perf_counter() - started) * 1_000_000),
            pointer=exc.pointer,
            keyword=exc.keyword,
            error=str(exc),
        )
    except (GrammarConfigurationError, InvalidGrammarError, AutomatonLimitError) as exc:
        return CoverageResult(
            unique_id,
            "limit_exceeded",
            int((time.perf_counter() - started) * 1_000_000),
            error=str(exc),
        )
    except Exception as exc:
        return CoverageResult(
            unique_id,
            "compile_error",
            int((time.perf_counter() - started) * 1_000_000),
            error=f"{type(exc).__name__}: {exc}",
        )
    return CoverageResult(
        unique_id,
        "compiled",
        int((time.perf_counter() - started) * 1_000_000),
        grammar_sha256=compiled.sha256,
    )


# Feed each benchmark instance (as compact JSON) through the automaton and compare with its label.
def replay_schema(schema: dict[str, Any], tests: list[dict[str, Any]], unique_id: str = "unknown") -> dict[str, Any]:
    coverage = compile_coverage(schema, unique_id)
    result = coverage.as_dict()
    result.update({"valid_passed": 0, "invalid_passed": 0, "validation_error": None, "invalidation_error": None})
    if coverage.status != "compiled":
        return result
    automaton = PushdownAutomaton.from_compiled(GBNFCompiler().compile_schema(schema))
    for index, test in enumerate(tests):
        text = json.dumps(test["data"], ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        state = automaton.advance_bytes(automaton.initial_state(), text.encode("utf-8"))
        accepted = state is not None and automaton.can_finish(state)
        if accepted and not test["valid"]:
            result["invalidation_error"] = f"test #{index}: should reject but did not"
        elif not accepted and test["valid"]:
            result["validation_error"] = f"test #{index}: should accept but did not"
        elif test["valid"]:
            result["valid_passed"] += 1
        else:
            result["invalid_passed"] += 1
    return result


# Runs in a child process: cap memory and CPU time, then compile and send the result back.
def _isolated_worker(schema, unique_id, memory_mb, cpu_seconds, queue):
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    queue.put(compile_coverage(schema, unique_id).as_dict())


# compile_coverage in a fresh process; a timeout or crash becomes a result instead of an error.
def isolated_compile_coverage(
    schema: dict[str, Any],
    unique_id: str,
    *,
    wall_seconds: float = 10,
    memory_mb: int = 512,
    cpu_seconds: int = 5,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_isolated_worker,
        args=(schema, unique_id, memory_mb, cpu_seconds, queue),
    )
    started = time.perf_counter()
    process.start()
    process.join(wall_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "unique_id": unique_id,
            "status": "timeout",
            "ttfm_us": int(wall_seconds * 1_000_000),
            "error": f"Exceeded {wall_seconds}s wall clock limit",
        }
    if queue.empty():
        return {
            "unique_id": unique_id,
            "status": "crash",
            "ttfm_us": int((time.perf_counter() - started) * 1_000_000),
            "error": f"Worker exited with code {process.exitcode}",
        }
    return queue.get()


def parallel_compile_coverage(
    items: Iterable[tuple[dict[str, Any], str]],
    *,
    workers: int,
    wall_seconds: float = 10,
    memory_mb: int = 512,
    cpu_seconds: int = 5,
) -> Iterator[dict[str, Any]]:
    """``isolated_compile_coverage`` for each ``(schema, unique_id)``, ``workers`` processes at a time.

    Each schema still compiles in its own process under the same limits, and the
    compile time is measured inside that process, so running several at once
    (one per vCPU) leaves TTFM close to the serial value. Results come back in
    input order.
    """
    if workers < 1:
        raise ValueError("workers must be positive")

    def one(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
        schema, unique_id = item
        return isolated_compile_coverage(
            schema, unique_id, wall_seconds=wall_seconds, memory_mb=memory_mb, cpu_seconds=cpu_seconds,
        )

    if workers == 1:
        yield from map(one, items)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="coverage") as pool:
        yield from pool.map(one, items)


def recorded_coverage_ids(path) -> tuple[set[str], set[str]]:
    """Schema ids already in a coverage JSONL: (compile records, instance replays).

    Lets an interrupted coverage pass resume where it stopped instead of
    starting over. A cut-off last line is ignored (repair it before appending).
    """
    from pathlib import Path

    compiled: set[str] = set()
    replayed: set[str] = set()
    path = Path(path)
    if not path.is_file():
        return compiled, replayed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("event") == "instance_replay":
            replayed.add(record["unique_id"])
        elif "event" not in record and "unique_id" in record:
            compiled.add(record["unique_id"])
    return compiled, replayed


def pending_file_is_stale(path, run_id: str) -> bool:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return value.get("pending_file") == 1 and value.get("run_id") == run_id


__all__ = [
    "CoverageResult",
    "compile_coverage",
    "isolated_compile_coverage",
    "parallel_compile_coverage",
    "recorded_coverage_ids",
    "pending_file_is_stale",
    "replay_schema",
]

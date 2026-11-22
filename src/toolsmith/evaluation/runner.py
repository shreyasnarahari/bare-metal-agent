from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .models import CaseResult, RunConfig

# Match the Day 10 llama.cpp slot count. One GPU can decode a few sequences at once.
EVAL_WORKERS = 4


class ManifestMismatchError(ValueError):
    pass


class CheckpointRunner:
    def __init__(self, path: str | Path, manifest_sha256: str) -> None:
        self.path = Path(path)
        self.manifest_sha256 = manifest_sha256
        self.completed: dict[str, CaseResult] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                envelope = json.loads(line)
                if envelope["manifest_sha256"] != manifest_sha256:
                    raise ManifestMismatchError(
                        f"Checkpoint {self.path} belongs to a different run configuration"
                    )
                result = CaseResult.from_dict(envelope["result"])
                if result.case_id in self.completed:
                    raise ValueError(f"Duplicate checkpoint case: {result.case_id}")
                self.completed[result.case_id] = result

    def append(self, result: CaseResult) -> None:
        if result.case_id in self.completed:
            raise ValueError(f"Case is already checkpointed: {result.case_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "manifest_sha256": self.manifest_sha256,
            "result": result.as_dict(),
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(envelope, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        self.completed[result.case_id] = result

    def run(
        self,
        cases: Iterable[dict[str, Any]],
        execute: Callable[[dict[str, Any]], CaseResult],
        *,
        workers: int = EVAL_WORKERS,
    ) -> list[CaseResult]:
        if workers < 1:
            raise ValueError("workers must be positive")
        ordered_ids: list[Any] = []
        pending: list[tuple[Any, dict[str, Any]]] = []
        for case in cases:
            case_id = case.get("case_id", case.get("id"))
            ordered_ids.append(case_id)
            if case_id not in self.completed:
                pending.append((case_id, case))
        if pending:
            self._execute_pending(pending, execute, workers)
        return [self.completed[case_id] for case_id in ordered_ids]

    def _execute_pending(
        self,
        pending: list[tuple[Any, dict[str, Any]]],
        execute: Callable[[dict[str, Any]], CaseResult],
        workers: int,
    ) -> None:
        def run_one(item: tuple[Any, dict[str, Any]]) -> None:
            case_id, case = item
            result = execute(case)
            if result.case_id != case_id:
                raise ValueError(f"Executor returned {result.case_id} for {case_id}")
            with lock:
                self.append(result)

        lock = threading.Lock()
        if workers == 1 or len(pending) == 1:
            for item in pending:
                run_one(item)
            return
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = [pool.submit(run_one, item) for item in pending]
            for future in futures:
                future.result()


def atomic_write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


__all__ = [
    "EVAL_WORKERS",
    "CheckpointRunner",
    "ManifestMismatchError",
    "atomic_write_json",
]

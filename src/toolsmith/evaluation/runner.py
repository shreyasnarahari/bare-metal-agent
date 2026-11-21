from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .models import CaseResult, RunConfig


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
    ) -> list[CaseResult]:
        results = []
        for case in cases:
            case_id = case.get("case_id", case.get("id"))
            if case_id in self.completed:
                results.append(self.completed[case_id])
                continue
            result = execute(case)
            if result.case_id != case_id:
                raise ValueError(f"Executor returned {result.case_id} for {case_id}")
            self.append(result)
            results.append(result)
        return results


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


__all__ = ["CheckpointRunner", "ManifestMismatchError", "atomic_write_json"]

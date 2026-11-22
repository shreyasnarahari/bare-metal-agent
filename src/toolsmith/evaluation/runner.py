from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..artifacts import MirroredFile
from ..recording import trim_partial_line
from .models import CaseResult, RunConfig

# The default when a caller does not size the pool to its llama-server slots.
# The notebook passes workers equal to the slots it started (4 on a T4, 8 on an L4).
EVAL_WORKERS = 4


# A case whose requests failed on infrastructure this many times (across resumes) is recorded
# as an error so the run can finish; until then nothing is recorded and a rerun retries it.
INFRASTRUCTURE_RETRIES = 3


class ManifestMismatchError(ValueError):
    pass


class InfrastructureError(RuntimeError):
    """A case could not run because the serving stack failed. It must not become a result.

    ``fallback`` is the error result to record once the case has failed this
    way ``INFRASTRUCTURE_RETRIES`` times.
    """

    def __init__(self, case_id: str, message: str, *, fallback: CaseResult | None = None) -> None:
        super().__init__(f"{case_id}: {message}")
        self.case_id = case_id
        self.message = message
        self.fallback = fallback


class InfrastructureFailure(RuntimeError):
    """A run stopped because the serving stack failed; the affected cases were not recorded."""

    def __init__(self, path: Path, failures: list[tuple[str, str]]) -> None:
        first_case, first_message = failures[0]
        super().__init__(
            f"{len(failures)} case(s) in {path.name} hit infrastructure errors (first: {first_case}: "
            f"{first_message[:300]}). Nothing was recorded for them, so a rerun retries them."
        )
        self.path = path
        self.failures = failures


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix[:-1]: value}


def manifest_differences(recorded: Mapping[str, Any], current: Mapping[str, Any], limit: int = 12) -> list[str]:
    """``field: recorded X, current Y`` for each differing leaf, dotted paths, at most ``limit``."""
    left, right = _flatten(recorded), _flatten(current)
    lines = []
    for key in sorted(set(left) | set(right)):
        if left.get(key, "<missing>") != right.get(key, "<missing>"):
            lines.append(f"{key}: recorded {left.get(key, '<missing>')!r}, current {right.get(key, '<missing>')!r}")
    if len(lines) > limit:
        lines = [*lines[:limit], f"... and {len(lines) - limit} more"]
    return lines


def manifest_sidecar(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".manifest.json")


class CheckpointRunner:
    """Resumable results: one JSON envelope per finished case, stamped with the run's manifest hash.

    With ``manifest`` (the dict that was hashed), the first run also writes it
    beside the checkpoint, and a later mismatch names the fields that changed
    (for example ``runtime.parallel: recorded 4, current 8``).

    ``path`` may be a :class:`~toolsmith.artifacts.MirroredFile`: the checkpoint
    is then written on local disk, restored from its durable copy on a fresh
    runtime, mirrored at most every ``every_seconds`` while cases finish, and
    synced at the end of every run. A final line cut off mid-write is repaired
    before anything is appended.
    """

    def __init__(
        self,
        path: str | Path | MirroredFile,
        manifest_sha256: str,
        *,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.store = path if isinstance(path, MirroredFile) else MirroredFile(path)
        self.path = self.store.local
        self._sidecar = self.store.sibling(".manifest.json")
        self._infra = self.store.sibling(".infra.jsonl")
        for store in (self.store, self._sidecar, self._infra):
            store.restore()
        trim_partial_line(self.path)
        trim_partial_line(self._infra.local)
        self.infrastructure_attempts: dict[str, int] = {}
        if self._infra.local.exists():
            for line in self._infra.local.read_text().splitlines():
                case_id = json.loads(line)["case_id"]
                self.infrastructure_attempts[case_id] = self.infrastructure_attempts.get(case_id, 0) + 1
        self.manifest_sha256 = manifest_sha256
        self.manifest = manifest
        self.completed: dict[str, CaseResult] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                envelope = json.loads(line)
                if envelope["manifest_sha256"] != manifest_sha256:
                    raise ManifestMismatchError(self._mismatch_message())
                result = CaseResult.from_dict(envelope["result"])
                if result.case_id in self.completed:
                    raise ValueError(f"Duplicate checkpoint case: {result.case_id}")
                self.completed[result.case_id] = result

    def _mismatch_message(self) -> str:
        message = f"Checkpoint {self.path} belongs to a different run configuration"
        sidecar = manifest_sidecar(self.path)
        if self.manifest is not None and sidecar.is_file():
            differences = manifest_differences(json.loads(sidecar.read_text()), self.manifest)
            if differences:
                message += ":\n  " + "\n  ".join(differences)
        return message + ". Use another checkpoint path or delete the old file deliberately."

    def append(self, result: CaseResult) -> None:
        if result.case_id in self.completed:
            raise ValueError(f"Case is already checkpointed: {result.case_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.manifest is not None and not self._sidecar.local.exists():
            atomic_write_json(self._sidecar.local, {**self.manifest, "manifest_sha256": self.manifest_sha256})
            self._sidecar.sync(force=True)
        envelope = {
            "manifest_sha256": self.manifest_sha256,
            "result": result.as_dict(),
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(envelope, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        self.completed[result.case_id] = result
        self.store.sync()

    def sync(self) -> None:
        """Copy the checkpoint, its manifest, and its infrastructure log to durable storage now."""
        for store in (self.store, self._sidecar, self._infra):
            store.sync(force=True)

    def _record_infrastructure(self, error: InfrastructureError) -> bool:
        """Log an infrastructure failure; True once the case has used up its retries."""
        self._infra.local.parent.mkdir(parents=True, exist_ok=True)
        with self._infra.local.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"case_id": error.case_id, "message": error.message[:2000]}) + "\n")
        attempts = self.infrastructure_attempts.get(error.case_id, 0) + 1
        self.infrastructure_attempts[error.case_id] = attempts
        return attempts >= INFRASTRUCTURE_RETRIES and error.fallback is not None

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
        try:
            if pending:
                self._execute_pending(pending, execute, workers)
        finally:
            if pending:
                self.sync()
        return [self.completed[case_id] for case_id in ordered_ids]

    def _execute_pending(
        self,
        pending: list[tuple[Any, dict[str, Any]]],
        execute: Callable[[dict[str, Any]], CaseResult],
        workers: int,
    ) -> None:
        """Run the pending cases. An infrastructure error stops new cases from starting; the
        ones in flight finish, and the run raises ``InfrastructureFailure`` without recording
        anything for the failed cases (a rerun retries them)."""
        lock = threading.Lock()
        stop = threading.Event()
        failures: list[tuple[str, str]] = []

        def run_one(item: tuple[Any, dict[str, Any]]) -> None:
            case_id, case = item
            if stop.is_set():
                return
            try:
                result = execute(case)
            except InfrastructureError as error:
                with lock:
                    if self._record_infrastructure(error):
                        self.append(error.fallback)  # retries used up: recorded, visibly, as an error
                    else:
                        failures.append((str(case_id), error.message))
                        stop.set()
                return
            if result.case_id != case_id:
                raise ValueError(f"Executor returned {result.case_id} for {case_id}")
            with lock:
                self.append(result)

        if workers == 1 or len(pending) == 1:
            for item in pending:
                run_one(item)
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
                futures = [pool.submit(run_one, item) for item in pending]
                for future in futures:
                    future.result()
        if failures:
            raise InfrastructureFailure(self.path, failures)


def completed_results(
    path: str | Path | MirroredFile,
    manifest_sha256: str,
    ids: Iterable[Any],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> list[CaseResult] | None:
    """A finished run's results, in ``ids`` order, read from its checkpoint without running anything.

    ``None`` when any case is missing, so a report cell never mistakes a
    partial run for a complete one.
    """
    runner = CheckpointRunner(path, manifest_sha256, manifest=manifest)
    ids = list(ids)
    if any(case_id not in runner.completed for case_id in ids):
        return None
    return [runner.completed[case_id] for case_id in ids]


class IdentityMismatchError(ManifestMismatchError):
    pass


def read_identified_json(path: str | Path, identity: Mapping[str, Any]) -> Any | None:
    """A previously written output if its ``identity`` matches; ``None`` if absent; raises if it differs."""
    path = Path(path)
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    recorded = value.get("identity") if isinstance(value, dict) else None
    if recorded != dict(identity):
        differences = manifest_differences(recorded or {}, identity)
        raise IdentityMismatchError(
            f"{path} was written by a different runtime or configuration:\n  " + "\n  ".join(differences)
            + "\nUse another output path or delete the old file deliberately."
        )
    return value


def write_identified_json(path: str | Path, value: Mapping[str, Any], identity: Mapping[str, Any]) -> Path:
    return atomic_write_json(path, {**value, "identity": dict(identity)})


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
    "INFRASTRUCTURE_RETRIES",
    "CheckpointRunner",
    "InfrastructureError",
    "InfrastructureFailure",
    "IdentityMismatchError",
    "completed_results",
    "manifest_differences",
    "manifest_sidecar",
    "read_identified_json",
    "write_identified_json",
    "ManifestMismatchError",
    "atomic_write_json",
]

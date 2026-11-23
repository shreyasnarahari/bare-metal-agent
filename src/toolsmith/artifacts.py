"""Safe file handling for big artifacts and growing records.

- ``is_gguf`` checks a model file by magic bytes and exact size, so a truncated
  download or copy is never loaded.
- ``copy_file_atomic`` copies through a temporary file and renames it into
  place, so an interrupted copy never leaves a half-written destination.
- ``copy_snapshot`` and ``MirroredFile`` keep durable copies of files that are
  still being appended to (checkpoints, scores), always ending on a whole line.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any


def is_gguf(path: str | Path, expected_size: int | None = None) -> bool:
    """True for a file that starts with the GGUF magic bytes (and has exactly ``expected_size`` bytes)."""
    candidate = Path(path)
    try:
        if not candidate.is_file():
            return False
        size = candidate.stat().st_size
        if size < 4 or (expected_size is not None and size != expected_size):
            return False
        with candidate.open("rb") as source:
            return source.read(4) == b"GGUF"
    except OSError:
        return False


def copy_file_atomic(
    source: str | Path,
    destination: str | Path,
    *,
    expected_size: int | None = None,
    chunk_size: int = 8 * 1024 * 1024,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Copy ``source`` to ``destination`` atomically, optionally checking size and reporting progress."""
    source_path = Path(source)
    destination_path = Path(destination)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not source_path.is_file():
        raise FileNotFoundError(f"Copy source does not exist: {source_path}")
    if source_path.absolute() == destination_path.absolute():
        raise ValueError("Copy source and destination must be different")

    total = source_path.stat().st_size
    if expected_size is not None and total != expected_size:
        raise ValueError(
            f"Source size is {total} bytes; expected {expected_size}: {source_path}"
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(
        f".{destination_path.name}.{uuid.uuid4().hex}.tmp"
    )
    copied = 0
    try:
        with source_path.open("rb") as input_file, temporary.open("wb") as output_file:
            while chunk := input_file.read(chunk_size):
                output_file.write(chunk)
                copied += len(chunk)
                if progress is not None:
                    progress(copied, total)
        # A source that changed size while copying is an error; growing files use copy_snapshot.
        if copied != total:
            raise OSError(f"Copied {copied} of {total} bytes from {source_path}")
        os.replace(temporary, destination_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination_path


def copy_snapshot(source: str | Path, destination: str | Path) -> Path:
    """Atomically copy a file that may still be growing, as it was when the copy started.

    Only whole lines are copied (up to the last newline), so the copy of a JSONL
    file being appended never ends in half a record. A file with no newline at
    all is copied as is.
    """
    source_path, destination_path = Path(source), Path(destination)
    with source_path.open("rb") as input_file:
        total = os.fstat(input_file.fileno()).st_size
        data = input_file.read(total)
    cut = data.rfind(b"\n")
    if cut != -1:
        data = data[: cut + 1]
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, destination_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination_path


class MirroredFile:
    """A file written on fast local disk, with a whole-file copy kept on durable storage (Drive).

    ``restore`` brings the copy back when a fresh runtime has no local file.
    ``sync`` copies the local file over at most every ``every_seconds`` (or at
    once with ``force``). Copies are atomic snapshots of whole lines, so the
    durable copy is always complete, if a little behind. With ``submit``
    (``BACKGROUND.submit`` or ``JournalCopier.submit_call``), throttled syncs
    run off the caller's thread; a forced sync always copies on this thread.
    Copies of one file never overlap and each reads the file when it starts,
    so a later copy is never older than an earlier one.
    """

    def __init__(
        self,
        local: str | Path,
        mirror: str | Path | None = None,
        *,
        every_seconds: float = 60.0,
        submit: Callable[..., Any] | None = None,
    ) -> None:
        self.local = Path(local)
        self.mirror = Path(mirror) if mirror is not None else None
        self.every_seconds = every_seconds
        self.submit = submit
        self.errors: list[str] = []
        self._last = float("-inf")
        self._lock = threading.Lock()
        self._copying = threading.Lock()

    def sibling(self, suffix: str) -> MirroredFile:
        """The companion file ``<name><suffix>`` beside both copies (a manifest, an error log)."""
        mirror = self.mirror.with_name(self.mirror.name + suffix) if self.mirror is not None else None
        return MirroredFile(
            self.local.with_name(self.local.name + suffix), mirror,
            every_seconds=self.every_seconds, submit=self.submit,
        )

    def restore(self) -> bool:
        """Copy the durable file to local disk if only the durable one exists."""
        if self.mirror is None or self.local.exists():
            return False
        try:
            list(self.mirror.parent.iterdir())  # a fresh Drive mount can hide files until listed
        except OSError:
            return False
        if not self.mirror.is_file():
            return False
        copy_snapshot(self.mirror, self.local)
        return True

    def sync(self, *, force: bool = False) -> bool:
        """Copy the local file to the durable one if due (or ``force``). Returns whether a copy started."""
        if self.mirror is None or not self.local.exists():
            return False
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last < self.every_seconds:
                return False
            self._last = now
        if self.submit is not None and not force:
            self.submit(f"mirror {self.local.name}", self._copy)
            return True
        return self._copy()

    def _copy(self) -> bool:
        with self._copying:
            try:
                copy_snapshot(self.local, self.mirror)
            except OSError as error:  # Drive hiccups must not kill the run; the local file is intact
                self.errors.append(f"{self.local} -> {self.mirror}: {type(error).__name__}: {error}")
                return False
        return True

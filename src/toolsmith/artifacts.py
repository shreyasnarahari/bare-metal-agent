from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path


def is_gguf(path: str | Path, expected_size: int | None = None) -> bool:
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
        if copied != total:
            raise OSError(f"Copied {copied} of {total} bytes from {source_path}")
        os.replace(temporary, destination_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination_path

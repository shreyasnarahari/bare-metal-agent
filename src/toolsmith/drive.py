"""Keep Google Drive usable and durable from a Colab notebook.

Colab's Drive mount caches writes on the VM and uploads them in the background,
so a runtime that disconnects can lose the last writes; only
``drive.flush_and_unmount()`` forces them out. A mount can also hang or drop
("Transport endpoint is not connected").

- :func:`ensure_drive` probes the mount in a subprocess (which can be killed if
  the mount hangs) and remounts it with ``force_remount=True`` when needed.
- :func:`sync_drive` waits for background jobs that write to Drive, flushes
  every pending write with ``flush_and_unmount``, and mounts again. The
  notebooks call it at the end of each long cell and in cleanup.

``google.colab`` is imported only when no ``colab_drive`` is passed, so tests
inject a fake. Standard library only otherwise.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .background import say

MOUNT_POINT = "/content/drive"
DRIVE_WRITE_JOBS = ("cache ", "mirror ")  # BackgroundJobs names that write to Drive
_PROBE = (
    "import pathlib, sys, uuid\n"
    "folder = pathlib.Path(sys.argv[1])\n"
    "probe = folder / f'.toolsmith-probe-{uuid.uuid4().hex}'\n"
    "probe.write_text('ok')\n"
    "assert probe.read_text() == 'ok'\n"
    "probe.unlink()\n"
)


def _colab_drive() -> Any:
    from google.colab import drive

    return drive


def drive_responsive(folder: str | Path, *, timeout: float = 30.0) -> bool:
    """True when a small write, read, and delete in ``folder`` finish within ``timeout`` seconds."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE, str(folder)],
            capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def ensure_drive(
    *,
    mount_point: str = MOUNT_POINT,
    probe_folder: str | Path | None = None,
    colab_drive: Any = None,
    timeout: float = 30.0,
    log: Callable[[str], None] = say,
) -> bool:
    """Make sure Drive answers; remount it if it is missing or hung. Returns whether it remounted."""
    folder = Path(probe_folder or Path(mount_point) / "MyDrive")
    if drive_responsive(folder, timeout=timeout):
        return False
    log(f"Google Drive did not answer within {timeout:g}s at {folder}; remounting it.")
    (colab_drive or _colab_drive()).mount(mount_point, force_remount=True)
    if not drive_responsive(folder, timeout=timeout):
        raise RuntimeError(
            f"Google Drive still does not answer at {folder} after a remount. Results so far are on local "
            "disk under /content; disconnect and reconnect the runtime, then run all cells to resume."
        )
    return True


def sync_drive(
    jobs: Any = None,
    *,
    stage: str = "cell",
    remount: bool = True,
    mount_point: str = MOUNT_POINT,
    colab_drive: Any = None,
    drain_timeout: float = 600.0,
    policy: str | None = None,
    log: Callable[[str], None] = say,
) -> dict[str, str]:
    """Push every pending Drive write to Google's servers now.

    Waits up to ``drain_timeout`` for background jobs whose names mark them as
    Drive writers, then ``flush_and_unmount`` and (with ``remount``) mounts
    again. ``TOOLSMITH_DRIVE_SYNC=cleanup`` limits this to the cleanup stage,
    for a session where remounting asks to authorize again. Returns the Drive
    jobs that had not finished.
    """
    policy = (policy or os.environ.get("TOOLSMITH_DRIVE_SYNC") or "always").lower()
    if policy == "off" or (policy == "cleanup" and stage != "cleanup"):
        return {}
    unfinished: dict[str, str] = {}
    if jobs is not None:
        unfinished = jobs.drain(drain_timeout, match=lambda name: name.startswith(DRIVE_WRITE_JOBS))
        if unfinished:
            log(f"Drive writes still pending after {drain_timeout:g}s: {unfinished}")
    drive = colab_drive or _colab_drive()
    drive.flush_and_unmount()
    if remount:
        drive.mount(mount_point)
    log(f"Drive writes flushed ({stage}).")
    return unfinished


__all__ = ["DRIVE_WRITE_JOBS", "MOUNT_POINT", "drive_responsive", "ensure_drive", "sync_drive"]

"""Run manifests: exactly what produced a set of results.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

from .models import RunConfig, canonical_hash


# The config, dataset summary, environment, git commit (and whether the tree had local edits),
# Python and platform, plus a hash of all of it.
def build_manifest(
    config: RunConfig,
    *,
    project_dir: str | Path,
    dataset: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    root = Path(project_dir)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip())
    value = {
        "config": config.as_dict(),
        "dataset": dataset,
        "environment": environment,
        "repository": {"commit": commit, "dirty": dirty},
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    value["manifest_sha256"] = canonical_hash(value)
    return value


__all__ = ["build_manifest"]

"""Append-only JSONL evidence records, with secrets redacted.

Every evaluation and notebook cell writes what it did as one JSON object per
line. Before writing, values under sensitive keys (``api_key``, ``token``, ...)
are replaced, and every known secret string (such as ``HF_TOKEN`` and
``LOCAL_KEY``) is masked wherever it appears, so records can be shared.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REDACTED = "[redacted]"
# Values under these keys (compared case-insensitively, "-" as "_") are always redacted.
_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "hf_token",
    "local_key",
    "password",
    "refresh_token",
    "secret",
    "token",
}


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    # Walk nested dicts and lists; secrets are replaced longest first so overlaps cannot leak.
    if isinstance(value, Mapping):
        redacted = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            redacted[key] = _REDACTED if normalized in _SENSITIVE_KEYS else _redact(item, secrets)
        return redacted
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, _REDACTED)
        return value
    return value


class JsonlRecorder:
    """Append redacted JSON records, one per line.

    ``context`` (a mapping, or a zero-argument callable evaluated per record)
    adds fields such as the runtime stamp to every record. A record's own
    fields win over context fields with the same name.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        secrets: Iterable[str] = (),
        context: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.path = Path(path)
        self._secrets = tuple(sorted({secret for secret in secrets if secret}, key=len, reverse=True))
        self._context = context

    def snapshot(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """The redacted, timestamped record that ``append`` would write, without writing it."""
        merged = dict(record)
        if self._context is not None:
            context = self._context() if callable(self._context) else self._context
            for key, value in context.items():
                merged.setdefault(key, value)
        snapshot = _redact(merged, self._secrets)
        snapshot.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        return snapshot

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = self.snapshot(record)
        serialized = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(serialized + "\n")
        return snapshot


def trim_partial_line(path: str | Path) -> bool:
    """Repair a JSONL file whose last line was cut off (a crash or copy mid-write).

    A complete record that only lacks its newline gets one; anything else after
    the last newline is dropped, so the next append starts on a fresh line.
    Returns whether the file changed.
    """
    path = Path(path)
    if not path.is_file():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    start = data.rfind(b"\n") + 1
    try:
        json.loads(data[start:])
    except ValueError:
        with path.open("r+b") as handle:
            handle.truncate(start)
        return True
    with path.open("ab") as handle:
        handle.write(b"\n")
    return True

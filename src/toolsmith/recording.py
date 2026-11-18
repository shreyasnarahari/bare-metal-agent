from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REDACTED = "[redacted]"
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
    def __init__(self, path: str | Path, *, secrets: Iterable[str] = ()) -> None:
        self.path = Path(path)
        self._secrets = tuple(sorted({secret for secret in secrets if secret}, key=len, reverse=True))

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = _redact(dict(record), self._secrets)
        snapshot.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        serialized = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(serialized + "\n")
        return snapshot

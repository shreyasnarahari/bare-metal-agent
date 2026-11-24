"""The output gate: bounded, sanitized, deterministically labeled tool output before the model or journal sees it.

Raw bytes are already capped while they are read (``bulkhead.supervisor``):
only bounded prefixes and counts exist. This module turns a prefix into safe text:

1. A multibyte UTF-8 sequence cut off by truncation is dropped, not counted
   as invalid. Other invalid bytes become U+FFFD, and ``invalid_utf8`` is set.
2. ANSI escape sequences are removed (CSI, OSC including terminal titles and
   hyperlinks, and other ESC sequences), as are NUL and every other C0
   control except newline and tab, DEL, C1 controls, and bidirectional
   overrides, embeddings, isolates, and marks. ``\\r\\n`` becomes ``\\n``; a
   lone ``\\r`` (used to overwrite a line on a terminal) is removed. Nothing
   else is changed: whitespace and content are left as the program wrote them.
3. A line that could forge the wrapper (``[BEGIN UNTRUSTED OUTPUT`` or
   ``[END UNTRUSTED OUTPUT``) is escaped with a leading backslash.

``render`` wraps the result for the model:

    {"trust": "untrusted",
     "content": "[BEGIN UNTRUSTED OUTPUT <id>]\\n...\\n[END UNTRUSTED OUTPUT <id>]",
     "truncated": false}

The id derives from the request hash and the stream name, so replay produces
identical text. ``render`` also redacts configured secret values as a final
defense; a secret reaching raw output is still a failure of the filesystem or
environment boundary, whatever the redaction hides.

The wrapper and the ``trust`` label explain the data to the model; they
enforce nothing. Enforcement is the episode's taint, already updated by the
adapter, and the policy engine.

``bound_json`` bounds structured outside content as valid JSON (strings and
lists shortened, then a truncation object), never by cutting serialized text.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

BEGIN = "[BEGIN UNTRUSTED OUTPUT {id}]"
END = "[END UNTRUSTED OUTPUT {id}]"
REDACTED = "[REDACTED]"
_ANSI = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC (titles, hyperlinks), terminated by BEL or ST
    r"|\x1b[PX^_][^\x1b]*(?:\x1b\\)?"  # DCS, SOS, PM, APC
    r"|\x1b[ -/]*[0-~]"  # other two-character escapes
    r"|\x9b[0-?]*[ -/]*[@-~]"  # 8-bit CSI
)
_CONTROLS = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f؜‎‏‪-‮⁦-⁩]")
_FORGERY = re.compile(r"^(\s*)(\[(?:BEGIN|END) UNTRUSTED OUTPUT)", re.MULTILINE)


@dataclass(frozen=True)
class GatedText:
    """Sanitized text from one stream, with what the gate did to it."""

    text: str
    raw_bytes: int  # bytes the program wrote
    kept_bytes: int  # bytes retained (the bounded prefix) before sanitizing
    invalid_utf8: bool
    stripped: int  # control, escape, and bidi characters removed
    neutralized: int  # forged wrapper lines escaped
    truncated: bool

    def stats(self) -> dict[str, Any]:
        return {
            "raw_bytes": self.raw_bytes, "kept_bytes": self.kept_bytes, "invalid_utf8": self.invalid_utf8,
            "stripped": self.stripped, "neutralized": self.neutralized, "truncated": self.truncated,
        }


def _drop_partial_tail(data: bytes) -> bytes:
    """Remove a UTF-8 sequence that truncation cut off at the end (at most 3 bytes)."""
    for back in range(1, min(4, len(data)) + 1):
        byte = data[-back]
        if byte & 0xC0 != 0x80:  # a lead or ASCII byte
            needed = 2 if byte & 0xE0 == 0xC0 else 3 if byte & 0xF0 == 0xE0 else 4 if byte & 0xF8 == 0xF0 else 1
            return data[:-back] if needed > back else data
    return data


def sanitize(data: bytes, *, raw_bytes: int | None = None, truncated: bool = False) -> GatedText:
    """Make a bounded byte prefix safe to show (see the module docstring)."""
    kept = len(data)
    if truncated:
        data = _drop_partial_tail(data)
    try:
        text = data.decode("utf-8")
        invalid = False
    except UnicodeDecodeError:
        text = data.decode("utf-8", "replace")
        invalid = True
    before = len(text)
    text = _ANSI.sub("", text)
    stripped = before - len(text)
    text = text.replace("\r\n", "\n").replace("\r", "\x00")  # a lone CR would overwrite the line on a terminal
    text, controls = _CONTROLS.subn("", text)
    text, neutralized = _FORGERY.subn(r"\1\\\2", text)
    return GatedText(text, kept if raw_bytes is None else raw_bytes, kept, invalid, stripped + controls, neutralized, truncated)


def delimiter_id(request_sha256: str, stream: str) -> str:
    """A short id for one stream of one request: the same on every replay."""
    return hashlib.sha256(f"{request_sha256}:{stream}".encode()).hexdigest()[:16]


def redact(text: str, secrets: Iterable[str]) -> tuple[str, int]:
    """Replace every configured secret value (four characters or longer) with ``[REDACTED]``."""
    count = 0
    for secret in sorted({value for value in secrets if isinstance(value, str) and len(value) >= 4}, key=len, reverse=True):
        occurrences = text.count(secret)
        if occurrences:
            text = text.replace(secret, REDACTED)
            count += occurrences
    return text, count


def render(gated: GatedText, *, request_sha256: str, stream: str, secrets: Iterable[str] = ()) -> tuple[dict[str, Any], int]:
    """The model-facing object for one stream, and how many secret values were redacted."""
    text, redacted = redact(gated.text, secrets)
    if not text:
        content = ""
    else:
        ident = delimiter_id(request_sha256, stream)
        body = text if text.endswith("\n") else text + "\n"
        content = f"{BEGIN.format(id=ident)}\n{body}{END.format(id=ident)}"
    return {"trust": "untrusted", "content": content, "truncated": gated.truncated}, redacted


def bound_json(value: Any, max_bytes: int) -> tuple[Any, bool]:
    """``value`` shrunk until its JSON is at most ``max_bytes``, always valid JSON. Returns (value, truncated)."""

    def size(item: Any) -> int:
        return len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode())

    if size(value) <= max_bytes:
        return value, False

    def shrink(item: Any, budget: int) -> Any:
        if isinstance(item, str):
            return item if len(item) <= budget else item[: max(0, budget)] + "…"
        if isinstance(item, list):
            kept = [shrink(element, budget) for element in item[: max(1, budget // 64)]]
            return kept + ([{"omitted_items": len(item) - len(kept)}] if len(kept) < len(item) else [])
        if isinstance(item, Mapping):
            keys = list(item)[: max(1, budget // 64)]
            kept = {str(key): shrink(item[key], budget) for key in keys}
            if len(keys) < len(item):
                kept["omitted_keys"] = len(item) - len(keys)
            return kept
        return item

    budget = max_bytes
    while budget >= 16:
        candidate = shrink(value, budget)
        if size(candidate) <= max_bytes:
            return candidate, True
        budget //= 2
    fallback = {"truncated": True, "bytes": size(value)}
    return (fallback if size(fallback) <= max_bytes else {}), True


__all__ = ["BEGIN", "END", "REDACTED", "GatedText", "bound_json", "delimiter_id", "redact", "render", "sanitize"]

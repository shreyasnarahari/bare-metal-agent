from __future__ import annotations

import codecs
import json
from typing import Any


class SSEDecodeError(ValueError):
    pass


class SSEJSONDecoder:
    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._buffer = ""
        self._data_lines: list[str] = []
        self.done_received = False

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        if not isinstance(chunk, bytes):
            raise TypeError("SSE input must be bytes")
        try:
            self._buffer += self._utf8.decode(chunk)
        except UnicodeDecodeError as exc:
            raise SSEDecodeError("SSE stream contains invalid UTF-8") from exc
        return self._decode_lines(final=False)

    def finish(self) -> list[dict[str, Any]]:
        try:
            self._buffer += self._utf8.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise SSEDecodeError("SSE stream ends inside a UTF-8 character") from exc
        return self._decode_lines(final=True)

    def _decode_lines(self, *, final: bool) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            event = self._process_line(line.removesuffix("\r"))
            if event is not None:
                events.append(event)
        if final:
            if self._buffer:
                event = self._process_line(self._buffer.removesuffix("\r"))
                if event is not None:
                    events.append(event)
            event = self._dispatch()
            if event is not None:
                events.append(event)
        return events

    def _process_line(self, line: str) -> dict[str, Any] | None:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if not separator:
            field, value = line, ""
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data_lines.append(value)
        return None

    def _dispatch(self) -> dict[str, Any] | None:
        if not self._data_lines:
            return None
        payload = "\n".join(self._data_lines)
        self._data_lines = []
        if payload == "[DONE]":
            self.done_received = True
            return None
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SSEDecodeError(f"SSE event contains invalid JSON: {exc.msg}") from exc
        if not isinstance(event, dict):
            raise SSEDecodeError("SSE event must contain a JSON object")
        return event

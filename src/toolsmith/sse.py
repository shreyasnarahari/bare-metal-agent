"""Incremental decoder for server-sent events (SSE) carrying JSON payloads.

llama-server and the shim stream responses as ``data: {...}`` lines separated by
blank lines, ending with ``data: [DONE]``. Network chunks can split a line, or
even a UTF-8 character, anywhere, so bytes are buffered until whole lines and
whole events are available.
"""

from __future__ import annotations

import codecs
import json
from typing import Any


class SSEDecodeError(ValueError):
    """The stream is not valid UTF-8 SSE with JSON-object payloads."""


class SSEJSONDecoder:
    """Feed raw bytes; get back each complete JSON event. ``done_received`` marks ``[DONE]``."""

    def __init__(self) -> None:
        # An incremental decoder keeps a multi-byte character split across chunks until it is whole.
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
        # End of stream: flush the decoder and dispatch any event that lacked a final blank line.
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
        # SSE rules: a blank line ends an event, ":" starts a comment, and only data fields matter here.
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
        # One event may span several data lines; they are joined with newlines, per the SSE spec.
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

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .json_utils import StrictJSONError, strict_json_loads


class ToolCallParseError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_tool_call") -> None:
        super().__init__(message)
        self.code = code


def _argument_json(text: str) -> Any:
    try:
        return strict_json_loads(text)
    except StrictJSONError as exc:
        raise ToolCallParseError(
            f"Tool-call arguments are not strict JSON: {exc}",
            code="invalid_tool_call_json",
        ) from exc


@dataclass(frozen=True)
class ParsedToolCall:
    id: str
    name: str
    arguments: str

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class ParsedResponse:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ParsedToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ContentDelta:
    text: str


@dataclass(frozen=True)
class ToolCallStart:
    index: int
    id: str
    name: str


@dataclass(frozen=True)
class ToolCallArgumentsDelta:
    index: int
    text: str


@dataclass(frozen=True)
class ToolCallEnd:
    index: int


ParserEvent: TypeAlias = (
    ReasoningDelta | ContentDelta | ToolCallStart | ToolCallArgumentsDelta | ToolCallEnd
)


class _TrimmedText:
    """Keep streamed text identical to the stripped final message."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._held = ""
        self._started = False

    def push(self, text: str) -> str | None:
        if not text:
            return None
        if not self._started:
            text = text.lstrip()
            if not text:
                return None
        body = text.rstrip()
        held = text[len(body):]
        delta = None
        if body:
            delta = self._held + body
            self._held = ""
            self._started = True
            self._parts.append(delta)
        if held:
            self._held += held
            self._started = True
        return delta

    def value(self) -> str | None:
        return "".join(self._parts) or None


@dataclass
class _CallBuilder:
    index: int
    id: str
    name: str
    argument_parts: list[str] = field(default_factory=list)


class _JSONValueScanner:
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.in_string = False
        self._escaped = False
        self.complete = False

    def scan(self, text: str, position: int) -> int:
        while position < len(text) and not self.complete:
            character = text[position]
            position += 1
            if self.in_string:
                if self._escaped:
                    self._escaped = False
                elif character == "\\":
                    self._escaped = True
                elif character == '"':
                    self.in_string = False
                continue

            if character == '"':
                self.in_string = True
            elif character in "{[":
                self.stack.append(character)
            elif character in "}]":
                expected = "{" if character == "}" else "["
                if not self.stack or self.stack[-1] != expected:
                    raise ToolCallParseError(
                        "Mismatched JSON brackets",
                        code="invalid_tool_call_json",
                    )
                self.stack.pop()
                if not self.stack:
                    self.complete = True
        return position


class _EnvelopeParser:
    def __init__(self, owner: Qwen3IncrementalParser) -> None:
        self.owner = owner
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.position = 0
        self.state = "start_object"
        self.current_key: str | None = None
        self.keys: set[str] = set()
        self.name: str | None = None
        self.value_scanner: _JSONValueScanner | None = None

    def feed(self, text: str, events: list[ParserEvent]) -> None:
        self.buffer += text
        self._process(events)

    def finish(self, events: list[ParserEvent]) -> None:
        self._process(events)
        if self.state != "done" or self._non_whitespace(self.position) != len(self.buffer):
            raise ToolCallParseError(
                "Tool call contains incomplete or malformed JSON",
                code="invalid_tool_call_json",
            )
        try:
            decoded = json.loads(self.buffer)
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(
                f"Tool call contains invalid JSON: {exc.msg}",
                code="invalid_tool_call_json",
            ) from exc
        if not isinstance(decoded, dict):
            raise ToolCallParseError("Tool call must be a JSON object")
        if not self.name or self.keys != {"name", "arguments"}:
            raise ToolCallParseError("Tool call needs exactly name and arguments")

    def _process(self, events: list[ParserEvent]) -> None:
        while True:
            if self.state == "done":
                if self._non_whitespace(self.position) != len(self.buffer):
                    raise ToolCallParseError(
                        "Unexpected content after the tool-call object",
                        code="invalid_tool_call_json",
                    )
                return
            if self.state == "start_object":
                position = self._non_whitespace(self.position)
                if position == len(self.buffer):
                    return
                if self.buffer[position] != "{":
                    raise ToolCallParseError("Tool call must be a JSON object")
                self.position = position + 1
                self.state = "key_or_end"
            elif self.state == "key_or_end":
                position = self._non_whitespace(self.position)
                if position == len(self.buffer):
                    return
                if self.buffer[position] == "}":
                    self.position = position + 1
                    self.state = "done"
                    continue
                if self.buffer[position] != '"':
                    raise ToolCallParseError(
                        "Tool-call object keys must be JSON strings",
                        code="invalid_tool_call_json",
                    )
                decoded = self._decode(position)
                if decoded is None:
                    return
                key, end = decoded
                if key in self.keys:
                    raise ToolCallParseError(f"Duplicate tool-call field: {key}")
                if key not in {"name", "arguments"}:
                    raise ToolCallParseError(f"Unsupported tool-call field: {key}")
                self.keys.add(key)
                self.current_key = key
                self.position = end
                self.state = "colon"
            elif self.state == "colon":
                position = self._non_whitespace(self.position)
                if position == len(self.buffer):
                    return
                if self.buffer[position] != ":":
                    raise ToolCallParseError(
                        "Expected a colon after the tool-call field",
                        code="invalid_tool_call_json",
                    )
                self.position = position + 1
                self.state = "value"
            elif self.state == "value":
                position = (
                    self.position
                    if self.value_scanner is not None
                    else self._non_whitespace(self.position)
                )
                if position == len(self.buffer):
                    return
                if self.current_key == "name":
                    decoded = self._decode(position)
                    if decoded is None:
                        return
                    name, end = decoded
                    if not isinstance(name, str) or not name.strip():
                        raise ToolCallParseError("Tool call needs a non-empty string name")
                    self.name = name
                    self.position = end
                    self.state = "comma_or_end"
                    self.owner._on_name(name, events)
                else:
                    if self.value_scanner is None:
                        if self.buffer[position] != "{":
                            raise ToolCallParseError("Tool-call arguments must be a JSON object")
                        self.value_scanner = _JSONValueScanner()
                    end = self.value_scanner.scan(self.buffer, position)
                    self.owner._on_arguments(self.buffer[position:end], events)
                    self.position = end
                    if not self.value_scanner.complete:
                        return
                    self.value_scanner = None
                    self.state = "comma_or_end"
            elif self.state == "comma_or_end":
                position = self._non_whitespace(self.position)
                if position == len(self.buffer):
                    return
                if self.buffer[position] == ",":
                    self.position = position + 1
                    self.state = "key_or_end"
                    continue
                if self.buffer[position] == "}":
                    self.position = position + 1
                    self.state = "done"
                    continue
                raise ToolCallParseError(
                    "Expected a comma or closing brace in the tool-call object",
                    code="invalid_tool_call_json",
                )
            else:
                raise AssertionError(self.state)

    def _decode(self, position: int) -> tuple[Any, int] | None:
        try:
            return self.decoder.raw_decode(self.buffer, position)
        except json.JSONDecodeError:
            return None

    def _non_whitespace(self, position: int) -> int:
        while position < len(self.buffer) and self.buffer[position] in " \t\r\n":
            position += 1
        return position


class Qwen3IncrementalParser:
    def __init__(
        self,
        id_factory: Callable[[], str] | None = None,
        *,
        expect_thinking: bool | None = None,
    ) -> None:
        self._id_factory = id_factory or (lambda: f"call_{uuid.uuid4().hex[:24]}")
        self._expect_thinking = expect_thinking
        self._state = "start"
        self._buffer = ""
        self._failed = False
        self._finished = False
        self._response: ParsedResponse | None = None
        self._content = _TrimmedText()
        self._reasoning = _TrimmedText()
        self._calls: list[_CallBuilder] = []
        self._used_ids: set[str] = set()
        self._envelope: _EnvelopeParser | None = None
        self._current_call: _CallBuilder | None = None
        self._pending_arguments: list[str] = []
        self._in_string = False
        self._escaped = False

    @property
    def response(self) -> ParsedResponse:
        if self._response is None:
            raise RuntimeError("Parser has not finished")
        return self._response

    def feed(self, text: str) -> list[ParserEvent]:
        self._ensure_open()
        if not isinstance(text, str):
            raise TypeError("Parser input must be a string")
        self._buffer += text
        events: list[ParserEvent] = []
        try:
            self._drain(events)
        except ToolCallParseError:
            self._failed = True
            raise
        return events

    def finish(self) -> list[ParserEvent]:
        self._ensure_open()
        events: list[ParserEvent] = []
        try:
            if self._state == "start":
                self._state = "text"
            if self._state == "thinking" or self._state == "omitted_thinking":
                raise ToolCallParseError(
                    "Thinking block is missing its closing tag",
                    code="unclosed_thinking_block",
                )
            if self._state == "tool_call":
                raise ToolCallParseError(
                    "Tool call is missing its closing tag",
                    code="unclosed_tool_call",
                )
            if self._buffer:
                self._emit_content(self._buffer, events)
                self._buffer = ""
            self._finished = True
            self._response = ParsedResponse(
                content=self._content.value(),
                reasoning_content=self._reasoning.value(),
                tool_calls=[
                    ParsedToolCall(
                        id=call.id,
                        name=call.name,
                        arguments=json.dumps(
                            _argument_json("".join(call.argument_parts)),
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        ),
                    )
                    for call in self._calls
                ],
            )
        except ToolCallParseError:
            self._failed = True
            raise
        return events

    def _ensure_open(self) -> None:
        if self._finished:
            raise RuntimeError("Parser is already finished")
        if self._failed:
            raise RuntimeError("Parser is in a failed state")

    def _drain(self, events: list[ParserEvent]) -> None:
        while True:
            if self._state == "start":
                if not self._process_start(events):
                    return
            elif self._state in {"thinking", "omitted_thinking"}:
                if not self._process_thinking(events):
                    return
            elif self._state == "text":
                if not self._process_text(events):
                    return
            elif self._state == "tool_call":
                if not self._process_tool_call(events):
                    return
            else:
                raise AssertionError(self._state)

    def _process_start(self, events: list[ParserEvent]) -> bool:
        if not self._buffer:
            return False
        explicit = self._buffer.find("<think>")
        omitted = self._buffer.find("</think>")
        tool_open = self._buffer.find("<tool_call>")
        tool_close = self._buffer.find("</tool_call>")

        if explicit != -1 and (omitted == -1 or explicit < omitted):
            if explicit:
                self._emit_content(self._buffer[:explicit], events)
            self._buffer = self._buffer[explicit + len("<think>"):]
            self._state = "thinking"
            return True
        if omitted != -1:
            self._emit_reasoning(self._buffer[:omitted], events)
            self._buffer = self._buffer[omitted + len("</think>"):]
            self._state = "text"
            return True
        if self._expect_thinking is True:
            leading = self._buffer.lstrip()
            if not leading or any(
                marker.startswith(leading) for marker in ("<think>", "<tool_call>")
            ):
                return False
            self._state = "text" if tool_open != -1 else "omitted_thinking"
            return True
        if self._expect_thinking is False or tool_open != -1 or tool_close != -1:
            self._state = "text"
            return True
        return False

    def _process_thinking(self, events: list[ParserEvent]) -> bool:
        closing = self._buffer.find("</think>")
        if closing != -1:
            self._emit_reasoning(self._buffer[:closing], events)
            self._buffer = self._buffer[closing + len("</think>"):]
            self._state = "text"
            return True
        safe = self._safe_length(self._buffer, ("</think>",))
        if safe:
            self._emit_reasoning(self._buffer[:safe], events)
            self._buffer = self._buffer[safe:]
        return False

    def _process_text(self, events: list[ParserEvent]) -> bool:
        candidates = [
            (position, marker)
            for marker in ("<tool_call>", "</tool_call>", "<think>", "</think>")
            if (position := self._buffer.find(marker)) != -1
        ]
        if not candidates:
            safe = self._safe_length(
                self._buffer,
                ("<tool_call>", "</tool_call>", "<think>", "</think>"),
            )
            if safe:
                self._emit_content(self._buffer[:safe], events)
                self._buffer = self._buffer[safe:]
            return False

        position, marker = min(candidates, key=lambda candidate: candidate[0])
        self._emit_content(self._buffer[:position], events)
        if marker == "</tool_call>":
            raise ToolCallParseError(
                "Found a tool-call closing tag without an opening tag",
                code="unexpected_tool_call_close",
            )
        if marker == "</think>":
            raise ToolCallParseError(
                "Found a thinking closing tag without an opening tag",
                code="unexpected_thinking_close",
            )
        self._buffer = self._buffer[position + len(marker):]
        if marker == "<think>":
            self._state = "thinking"
        else:
            self._state = "tool_call"
            self._envelope = _EnvelopeParser(self)
            self._current_call = None
            self._pending_arguments = []
            self._in_string = False
            self._escaped = False
        return True

    def _process_tool_call(self, events: list[ParserEvent]) -> bool:
        position = 0
        while position < len(self._buffer):
            character = self._buffer[position]
            if not self._in_string and self._buffer.startswith("</tool_call>", position):
                payload = self._buffer[:position]
                self._buffer = self._buffer[position + len("</tool_call>"):]
                if payload and self._envelope is not None:
                    self._envelope.feed(payload, events)
                if self._envelope is None or self._current_call is None:
                    raise ToolCallParseError("Tool call is missing its function name")
                self._envelope.finish(events)
                events.append(ToolCallEnd(self._current_call.index))
                self._envelope = None
                self._current_call = None
                self._pending_arguments = []
                self._state = "text"
                self._in_string = False
                self._escaped = False
                return True
            if not self._in_string and self._buffer.startswith("<tool_call>", position):
                raise ToolCallParseError(
                    "Nested tool-call opening tags are not valid",
                    code="nested_tool_call",
                )

            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif character == "\\":
                    self._escaped = True
                elif character == '"':
                    self._in_string = False
            elif character == '"':
                self._in_string = True
            position += 1

        safe = len(self._buffer) if self._in_string else self._safe_length(
            self._buffer, ("</tool_call>", "<tool_call>")
        )
        if safe and self._envelope is not None:
            self._envelope.feed(self._buffer[:safe], events)
            self._buffer = self._buffer[safe:]
        return False

    def _on_name(self, name: str, events: list[ParserEvent]) -> None:
        if self._current_call is not None:
            return
        call_id = self._id_factory()
        if not call_id:
            raise ToolCallParseError("Tool-call ID factory returned an empty ID")
        if call_id in self._used_ids:
            raise ToolCallParseError("Duplicate tool-call ID", code="duplicate_tool_call_id")
        self._used_ids.add(call_id)
        call = _CallBuilder(index=len(self._calls), id=call_id, name=name)
        self._calls.append(call)
        self._current_call = call
        events.append(ToolCallStart(call.index, call.id, name))
        for fragment in self._pending_arguments:
            call.argument_parts.append(fragment)
            events.append(ToolCallArgumentsDelta(call.index, fragment))
        self._pending_arguments = []

    def _on_arguments(self, fragment: str, events: list[ParserEvent]) -> None:
        if not fragment:
            return
        if self._current_call is None:
            self._pending_arguments.append(fragment)
            return
        self._current_call.argument_parts.append(fragment)
        events.append(ToolCallArgumentsDelta(self._current_call.index, fragment))

    def _emit_content(self, text: str, events: list[ParserEvent]) -> None:
        delta = self._content.push(text)
        if delta:
            events.append(ContentDelta(delta))

    def _emit_reasoning(self, text: str, events: list[ParserEvent]) -> None:
        delta = self._reasoning.push(text)
        if delta:
            events.append(ReasoningDelta(delta))

    @staticmethod
    def _safe_length(text: str, markers: tuple[str, ...]) -> int:
        hold = 0
        for marker in markers:
            maximum = min(len(marker) - 1, len(text))
            for size in range(1, maximum + 1):
                if text.endswith(marker[:size]):
                    hold = max(hold, size)
        return len(text) - hold

"""Incremental parser for Qwen3.5's native tool-call format.

Qwen3.5 writes each call as XML-like markup instead of JSON::

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Paris
    </parameter>
    </function>
    </tool_call>

Parameter values are bare text, so the parser uses the tool's JSON Schema to
decide what each value means: ``21`` stays a string for a string parameter but
becomes a number for a number parameter. As parameters arrive, the parser emits
them as fragments of one JSON arguments object (``{"city":"Paris"}``), so
callers see the same events as for Qwen3 and the OpenAI API.

The surrounding structure (optional reasoning, text, holding back partial tags)
mirrors ``toolsmith.streaming``; see that module for the general approach.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .grammar import nullable_value_schema
from .registry import Tool
from .streaming import (
    ContentDelta,
    ParsedResponse,
    ParsedToolCall,
    ParserEvent,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallParseError,
    ToolCallStart,
    _TrimmedText,
)
from .validation import strict_json_loads

# Tags that switch the parser between reasoning, text, and tool-call states.
_MARKERS = ("<tool_call>", "</tool_call>", "<think>", "</think>")
_PARAMETER_END = "</parameter>"


def _safe_length(text: str, markers: Sequence[str]) -> int:
    # Emit everything except a tail that could be the start of a marker split across chunks.
    hold = 0
    for marker in markers:
        maximum = min(len(marker) - 1, len(text))
        for size in range(1, maximum + 1):
            if text.endswith(marker[:size]):
                hold = max(hold, size)
    return len(text) - hold


def _trim_framing_newlines(value: str) -> str:
    # The template puts each value on its own line; drop exactly one newline at each end,
    # so a value that really starts or ends with a newline keeps the rest.
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _strict_json(value: str) -> Any:
    return strict_json_loads(value)


def _same_json_value(left: Any, right: Any) -> bool:
    # True == 1 in Python, but they are different JSON values, so the types must match too.
    return type(left) is type(right) and left == right


def _is_nullable(schema: Mapping[str, Any]) -> bool:
    return nullable_value_schema(schema) is not None


class _XMLCallParser:
    """State machine for one ``<function=...>...</function>`` block inside ``<tool_call>``.

    States: function_open -> function_name -> parameter_or_function_end ->
    (parameter_name -> parameter_value -> parameter_or_function_end)* ->
    function_done. Each step waits for more text when the buffer runs out.
    """

    def __init__(self, owner: Qwen35IncrementalParser) -> None:
        self.owner = owner
        self.buffer = ""
        self.state = "function_open"
        self.parameter_name: str | None = None
        self.parameter_names: set[str] = set()

    def feed(self, text: str, events: list[ParserEvent]) -> None:
        self.buffer += text
        self._process(events)

    def finish(self, events: list[ParserEvent]) -> None:
        self._process(events)
        self.buffer = self.buffer.lstrip(" \t\r\n")
        if self.state != "function_done" or self.buffer:
            raise ToolCallParseError(
                "Qwen3.5 tool call is incomplete or malformed",
                code="invalid_tool_call_xml",
            )

    # True: the buffer starts with ``literal`` (consumed). None: it may, once more text arrives.
    # False: it cannot.
    def _literal(self, literal: str) -> bool | None:
        if self.buffer.startswith(literal):
            self.buffer = self.buffer[len(literal):]
            return True
        if literal.startswith(self.buffer):
            return None
        return False

    def _required_literal(self, literal: str, events: list[ParserEvent]) -> bool:
        matched = self._literal(literal)
        if matched is None:
            return False
        if matched is False:
            raise ToolCallParseError(
                f"Expected {literal!r} in Qwen3.5 tool call",
                code="invalid_tool_call_xml",
            )
        return True

    def _process(self, events: list[ParserEvent]) -> None:
        while True:
            if self.state == "function_done":
                if self.buffer.lstrip(" \t\r\n"):
                    raise ToolCallParseError(
                        "Unexpected content after Qwen3.5 function block",
                        code="invalid_tool_call_xml",
                    )
                self.buffer = ""
                return

            if self.state in {"function_open", "parameter_or_function_end"}:
                self.buffer = self.buffer.lstrip(" \t\r\n")
            if not self.buffer:
                return

            if self.state == "function_open":
                if not self._required_literal("<function=", events):
                    return
                self.state = "function_name"
            elif self.state == "function_name":
                end = self.buffer.find(">")
                if end == -1:
                    if any(character in self.buffer for character in "<\r\n"):
                        raise ToolCallParseError(
                            "Invalid Qwen3.5 function name",
                            code="invalid_tool_call_xml",
                        )
                    return
                name = self.buffer[:end]
                if not name or any(character in name for character in "<\r\n"):
                    raise ToolCallParseError(
                        "Qwen3.5 function name must be non-empty text",
                        code="invalid_tool_call_xml",
                    )
                self.buffer = self.buffer[end + 1:]
                self.owner._on_name(name, events)
                self.state = "parameter_or_function_end"
            elif self.state == "parameter_or_function_end":
                parameter = self._literal("<parameter=")
                if parameter is True:
                    self.state = "parameter_name"
                    continue
                if parameter is None:
                    return
                if parameter is False:
                    if not self._required_literal("</function>", events):
                        return
                    self.owner._on_function_end(events)
                    self.state = "function_done"
                    continue
            elif self.state == "parameter_name":
                end = self.buffer.find(">")
                if end == -1:
                    if any(character in self.buffer for character in "<\r\n"):
                        raise ToolCallParseError(
                            "Invalid Qwen3.5 parameter name",
                            code="invalid_tool_call_xml",
                        )
                    return
                name = self.buffer[:end]
                if not name or any(character in name for character in "<\r\n"):
                    raise ToolCallParseError(
                        "Qwen3.5 parameter name must be non-empty text",
                        code="invalid_tool_call_xml",
                    )
                if name in self.parameter_names:
                    raise ToolCallParseError(
                        f"Duplicate Qwen3.5 parameter: {name!r}",
                        code="duplicate_tool_call_parameter",
                    )
                self.parameter_names.add(name)
                self.parameter_name = name
                self.buffer = self.buffer[end + 1:]
                self.state = "parameter_value"
            elif self.state == "parameter_value":
                # A value runs to </parameter>. Structural tags before that mean the model broke the format.
                end = self.buffer.find(_PARAMETER_END)
                reserved = [
                    position
                    for marker in ("<parameter=", "</function>")
                    if (position := self.buffer.find(marker)) != -1
                ]
                if reserved and (end == -1 or min(reserved) < end):
                    raise ToolCallParseError(
                        "Parameter value contains reserved markup or lacks its closing tag",
                        code="invalid_tool_call_xml",
                    )
                if end == -1:
                    return
                value = _trim_framing_newlines(self.buffer[:end])
                self.buffer = self.buffer[end + len(_PARAMETER_END):]
                if self.parameter_name is None:
                    raise AssertionError("Missing parameter name")
                self.owner._on_parameter(self.parameter_name, value, events)
                self.parameter_name = None
                self.state = "parameter_or_function_end"
            else:
                raise AssertionError(self.state)


class Qwen35IncrementalParser:
    """Feed text chunks with ``feed``, then call ``finish``; ``response`` holds the result.

    ``tools`` supplies each parameter's schema for value conversion. ``expect_thinking``
    works as in the Qwen3 parser, except that a ``<tool_call>`` also ends reasoning
    that was never closed (Qwen3.5 sometimes skips ``</think>`` before a call).
    """

    def __init__(
        self,
        tools: Sequence[Tool],
        id_factory: Callable[[], str] | None = None,
        *,
        expect_thinking: bool | None = None,
    ) -> None:
        self._tools = tuple(tools)
        self._id_factory = id_factory or (lambda: f"call_{uuid.uuid4().hex[:24]}")
        self._expect_thinking = expect_thinking
        self._state = "start"
        self._buffer = ""
        self._failed = False
        self._finished = False
        self._response: ParsedResponse | None = None
        self._content = _TrimmedText()
        self._reasoning = _TrimmedText()
        self._calls: list[dict[str, Any]] = []
        self._used_ids: set[str] = set()
        self._envelope: _XMLCallParser | None = None
        self._current_call: dict[str, Any] | None = None

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
            if self._state == "thinking":
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
            # Re-parse each call's assembled JSON strictly and re-serialize it compactly.
            calls = []
            for call in self._calls:
                raw_arguments = "".join(call["argument_parts"])
                arguments = _strict_json(raw_arguments)
                calls.append(
                    ParsedToolCall(
                        id=call["id"],
                        name=call["name"],
                        arguments=json.dumps(
                            arguments,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        ),
                    )
                )
            self._response = ParsedResponse(
                content=self._content.value(),
                reasoning_content=self._reasoning.value(),
                tool_calls=calls,
            )
        except (ToolCallParseError, ValueError) as exc:
            self._failed = True
            if isinstance(exc, ToolCallParseError):
                raise
            raise ToolCallParseError(
                "Could not reconstruct Qwen3.5 arguments",
                code="invalid_tool_call_json",
            ) from exc
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
            elif self._state == "thinking":
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

    # Decide how the response begins (explicit <think>, a bare </think>, a tool call, or text).
    def _process_start(self, events: list[ParserEvent]) -> bool:
        if not self._buffer:
            return False
        explicit = self._buffer.find("<think>")
        omitted = self._buffer.find("</think>")
        tool_open = self._buffer.find("<tool_call>")
        if explicit != -1 and (omitted == -1 or explicit < omitted):
            if explicit:
                self._emit_content(self._buffer[:explicit], events)
            self._buffer = self._buffer[explicit + len("<think>"):]
            self._state = "thinking"
            return True
        if omitted != -1 and (tool_open == -1 or omitted < tool_open):
            self._emit_reasoning(self._buffer[:omitted], events)
            self._buffer = self._buffer[omitted + len("</think>"):]
            self._state = "text"
            return True
        if self._expect_thinking is True and tool_open != -1:
            self._emit_reasoning(self._buffer[:tool_open], events)
            self._buffer = self._buffer[tool_open:]
            self._state = "text"
            return True
        if self._expect_thinking is True:
            leading = self._buffer.lstrip()
            if not leading or any(marker.startswith(leading) for marker in _MARKERS):
                return False
            self._state = "thinking"
            return True
        if self._expect_thinking is False or tool_open != -1 or "</tool_call>" in self._buffer:
            self._state = "text"
            return True
        return False

    # Stream reasoning until </think> or, if the model skipped it, until a <tool_call>.
    def _process_thinking(self, events: list[ParserEvent]) -> bool:
        closing = self._buffer.find("</think>")
        tool_open = self._buffer.find("<tool_call>")
        if tool_open != -1 and (closing == -1 or tool_open < closing):
            self._emit_reasoning(self._buffer[:tool_open], events)
            self._buffer = self._buffer[tool_open:]
            self._state = "text"
            return True
        if closing != -1:
            self._emit_reasoning(self._buffer[:closing], events)
            self._buffer = self._buffer[closing + len("</think>"):]
            self._state = "text"
            return True
        safe = _safe_length(self._buffer, ("</think>", "<tool_call>"))
        if safe:
            self._emit_reasoning(self._buffer[:safe], events)
            self._buffer = self._buffer[safe:]
        return False

    def _process_text(self, events: list[ParserEvent]) -> bool:
        candidates = [
            (position, marker)
            for marker in _MARKERS
            if (position := self._buffer.find(marker)) != -1
        ]
        if not candidates:
            safe = _safe_length(self._buffer, _MARKERS)
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
            self._envelope = _XMLCallParser(self)
            self._current_call = None
        return True

    # Pass the markup to the call parser until </tool_call>; a nested <tool_call> is an error.
    def _process_tool_call(self, events: list[ParserEvent]) -> bool:
        closing = self._buffer.find("</tool_call>")
        nested = self._buffer.find("<tool_call>")
        if nested != -1 and (closing == -1 or nested < closing):
            raise ToolCallParseError(
                "Nested or reserved tool-call markup is not valid",
                code="nested_tool_call",
            )
        if closing != -1:
            payload = self._buffer[:closing]
            self._buffer = self._buffer[closing + len("</tool_call>"):]
            if self._envelope is None:
                raise ToolCallParseError("Tool call parser is not initialized")
            self._envelope.feed(payload, events)
            if self._current_call is None:
                raise ToolCallParseError("Tool call is missing its function name")
            self._envelope.finish(events)
            events.append(ToolCallEnd(self._current_call["index"]))
            self._envelope = None
            self._current_call = None
            self._state = "text"
            return True
        safe = _safe_length(self._buffer, ("</tool_call>", "<tool_call>"))
        if safe and self._envelope is not None:
            self._envelope.feed(self._buffer[:safe], events)
            self._buffer = self._buffer[safe:]
        return False

    def _on_name(self, name: str, events: list[ParserEvent]) -> None:
        if self._current_call is not None:
            raise ToolCallParseError("Nested function blocks are not valid")
        call_id = self._id_factory()
        if not call_id:
            raise ToolCallParseError("Tool-call ID factory returned an empty ID")
        if call_id in self._used_ids:
            raise ToolCallParseError("Duplicate tool-call ID", code="duplicate_tool_call_id")
        self._used_ids.add(call_id)
        # The JSON arguments object opens now and is filled one parameter at a time.
        call = {
            "index": len(self._calls),
            "id": call_id,
            "name": name,
            "argument_parts": ["{"],
            "parameters": 0,
        }
        self._calls.append(call)
        self._current_call = call
        events.append(ToolCallStart(call["index"], call_id, name))
        events.append(ToolCallArgumentsDelta(call["index"], "{"))

    # The JSON Schema of one parameter, or None for an unknown tool or parameter.
    def _schema_for(self, function_name: str, parameter_name: str) -> Mapping[str, Any] | None:
        for tool in self._tools:
            if tool.name != function_name:
                continue
            properties = tool.parameters.get("properties", {})
            if not isinstance(properties, Mapping):
                return None
            schema = properties.get(parameter_name)
            return schema if isinstance(schema, Mapping) else None
        return None

    def _convert_value(self, function_name: str, name: str, raw: str) -> Any:
        # Interpret bare parameter text by the schema: strings stay text; numbers, booleans, null,
        # arrays, and objects are decoded as JSON. A value that does not decode stays text, so
        # validation (not the parser) reports the type error.
        schema = self._schema_for(function_name, name)
        if schema is None:
            return raw
        if _is_nullable(schema):
            try:
                return _strict_json(raw)
            except (ValueError, json.JSONDecodeError):
                return raw
        # An enum may mix types: an exact string member wins, otherwise try JSON.
        enum = schema.get("enum")
        if isinstance(enum, list):
            for value in enum:
                if isinstance(value, str) and raw == value:
                    return raw
            try:
                decoded = _strict_json(raw)
            except (ValueError, json.JSONDecodeError):
                return raw
            if any(_same_json_value(decoded, value) for value in enum):
                return decoded
            return decoded
        kind = schema.get("type")
        if kind == "string" or kind is None:
            return raw
        if kind in {"integer", "number", "boolean", "null", "array", "object"}:
            try:
                return _strict_json(raw)
            except (ValueError, json.JSONDecodeError):
                return raw
        return raw

    # Append one "name":value pair to the call's JSON arguments and stream it as a delta.
    def _on_parameter(self, name: str, raw: str, events: list[ParserEvent]) -> None:
        if self._current_call is None:
            raise ToolCallParseError("Parameter arrived before a function name")
        value = self._convert_value(self._current_call["name"], name, raw)
        prefix = "," if self._current_call["parameters"] else ""
        fragment = (
            prefix
            + json.dumps(name, ensure_ascii=False)
            + ":"
            + json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        )
        self._current_call["parameters"] += 1
        self._current_call["argument_parts"].append(fragment)
        events.append(ToolCallArgumentsDelta(self._current_call["index"], fragment))

    # </function>: close the JSON arguments object.
    def _on_function_end(self, events: list[ParserEvent]) -> None:
        if self._current_call is None:
            raise ToolCallParseError("Function closing tag arrived before a name")
        self._current_call["argument_parts"].append("}")
        events.append(ToolCallArgumentsDelta(self._current_call["index"], "}"))

    def _emit_content(self, text: str, events: list[ParserEvent]) -> None:
        delta = self._content.push(text)
        if delta:
            events.append(ContentDelta(delta))

    def _emit_reasoning(self, text: str, events: list[ParserEvent]) -> None:
        delta = self._reasoning.push(text)
        if delta:
            events.append(ReasoningDelta(delta))


class Qwen35ToolCallParser:
    """Parse a finished Qwen3.5 response in one call (a wrapper around the incremental parser)."""

    def __init__(
        self,
        tools: Sequence[Tool],
        id_factory: Callable[[], str] | None = None,
        *,
        expect_thinking: bool | None = None,
    ) -> None:
        self._tools = tuple(tools)
        self._id_factory = id_factory
        self._expect_thinking = expect_thinking

    def parse(self, text: str) -> ParsedResponse:
        parser = Qwen35IncrementalParser(
            self._tools,
            self._id_factory,
            expect_thinking=self._expect_thinking,
        )
        parser.feed(text)
        parser.finish()
        return parser.response


__all__ = ["Qwen35IncrementalParser", "Qwen35ToolCallParser"]

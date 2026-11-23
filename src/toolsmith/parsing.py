"""Whole-response Qwen3 tool-call parsing.

``Qwen3ToolCallParser`` is a convenience wrapper around the incremental parser
in ``toolsmith.streaming``: it feeds a complete response in one go and returns
the final ``ParsedResponse``. Streaming code uses the incremental parser
directly. The other names are re-exported so callers can import them from here.
"""

from __future__ import annotations

from collections.abc import Callable

from .streaming import (
    ContentDelta,
    ParsedResponse,
    ParsedToolCall,
    ParserEvent,
    Qwen3IncrementalParser,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallParseError,
    ToolCallStart,
)

__all__ = [
    "ContentDelta",
    "ParsedResponse",
    "ParsedToolCall",
    "ParserEvent",
    "Qwen3IncrementalParser",
    "Qwen3ToolCallParser",
    "ReasoningDelta",
    "ToolCallArgumentsDelta",
    "ToolCallEnd",
    "ToolCallParseError",
    "ToolCallStart",
]


class Qwen3ToolCallParser:
    """Parse a finished Qwen3 response into text, reasoning, and tool calls."""

    def __init__(
        self,
        id_factory: Callable[[], str] | None = None,
        *,
        expect_thinking: bool | None = None,
    ) -> None:
        self._id_factory = id_factory
        self._expect_thinking = expect_thinking

    def parse(self, text: str) -> ParsedResponse:
        # A fresh incremental parser per response: parsing state never leaks between calls.
        parser = Qwen3IncrementalParser(
            self._id_factory,
            expect_thinking=self._expect_thinking,
        )
        parser.feed(text)
        parser.finish()
        return parser.response

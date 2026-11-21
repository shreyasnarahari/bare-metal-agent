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
    def __init__(
        self,
        id_factory: Callable[[], str] | None = None,
        *,
        expect_thinking: bool | None = None,
    ) -> None:
        self._id_factory = id_factory
        self._expect_thinking = expect_thinking

    def parse(self, text: str) -> ParsedResponse:
        parser = Qwen3IncrementalParser(
            self._id_factory,
            expect_thinking=self._expect_thinking,
        )
        parser.feed(text)
        parser.finish()
        return parser.response

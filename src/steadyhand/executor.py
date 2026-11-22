"""Sequential tool execution with one timeout around every handler.

Handlers run after the shim has returned the calls. The FastAPI routes never
call them. Each call runs on a short-lived daemon thread so a handler that
blocks is reported as ``timeout`` and cannot wedge the episode or the test run.

A timed-out thread cannot be killed and keeps running in the background. The
real store handlers are microsecond-scale dict operations, so in practice only
a deliberately blocking stub reaches the timeout. Such stubs must not touch the
store. Part 2 replaces the single timeout with per-tool ``ToolPolicy.timeout_s``.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from toolsmith.evaluation.models import canonical_hash
from toolsmith.json_utils import StrictJSONError, strict_json_loads
from toolsmith.registry import ToolRegistry

from .store import NotFoundError, Store
from .types import ToolCall, ToolErrorType, ToolResult

DEFAULT_TOOL_TIMEOUT_S = 5.0
MAX_MESSAGE_CHARS = 300


class SequentialExecutor:
    """Runs one call at a time against the episode's own store. Stateless, so shareable."""

    def __init__(self, timeout_s: float = DEFAULT_TOOL_TIMEOUT_S) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s

    def run(self, call: ToolCall, registry: ToolRegistry, store: Store) -> ToolResult:
        started = time.monotonic()

        def failure(
            error_type: ToolErrorType,
            message: str,
            arguments_sha256: str | None = None,
        ) -> ToolResult:
            return ToolResult(
                call_id=call.id,
                tool=call.name,
                arguments=call.arguments,
                arguments_sha256=arguments_sha256,
                ok=False,
                error_type=error_type,
                message=one_line(message),
                elapsed_s=time.monotonic() - started,
            )

        try:
            tool = registry.get(call.name)
        except KeyError:
            return failure(ToolErrorType.HANDLER_ERROR, f"No registered tool is named {call.name!r}")
        if tool.handler is None:
            return failure(ToolErrorType.HANDLER_ERROR, f"Tool {call.name!r} has no handler")
        try:
            arguments = strict_json_loads(call.arguments)
        except StrictJSONError as exc:
            return failure(ToolErrorType.HANDLER_ERROR, f"Arguments are not strict JSON: {exc}")
        if not isinstance(arguments, dict):
            return failure(ToolErrorType.HANDLER_ERROR, "Arguments must be a JSON object")
        digest = canonical_hash(arguments)

        finished, value, error = self._call_with_timeout(tool.handler, store, arguments)
        if not finished:
            return failure(
                ToolErrorType.TIMEOUT,
                f"{call.name} did not finish within {self.timeout_s:g} seconds",
                digest,
            )
        if isinstance(error, NotFoundError):
            return failure(ToolErrorType.NOT_FOUND, str(error), digest)
        if error is not None:
            return failure(ToolErrorType.HANDLER_ERROR, f"{type(error).__name__}: {error}", digest)
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            return failure(
                ToolErrorType.HANDLER_ERROR,
                f"{call.name} returned a value that is not JSON: {exc}",
                digest,
            )
        return ToolResult(
            call_id=call.id,
            tool=call.name,
            arguments=call.arguments,
            arguments_sha256=digest,
            ok=True,
            value=value,
            elapsed_s=time.monotonic() - started,
        )

    def _call_with_timeout(
        self,
        handler: Any,
        store: Store,
        arguments: dict[str, Any],
    ) -> tuple[bool, Any, BaseException | None]:
        outcome: dict[str, Any] = {}

        def target() -> None:
            try:
                outcome["value"] = handler(store, **arguments)
            except BaseException as exc:  # reported to the model, never re-raised here
                outcome["error"] = exc

        worker = threading.Thread(target=target, name="steadyhand-tool", daemon=True)
        worker.start()
        worker.join(self.timeout_s)
        if worker.is_alive():
            return False, None, None
        return True, outcome.get("value"), outcome.get("error")


def one_line(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Collapse whitespace and cap length, so no traceback or wall of text reaches the model."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


__all__ = ["DEFAULT_TOOL_TIMEOUT_S", "SequentialExecutor", "one_line"]

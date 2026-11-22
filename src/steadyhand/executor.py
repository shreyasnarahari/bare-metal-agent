"""Sequential tool execution, one policy timeout around every handler.

Handlers run after the shim has returned the calls and after ``validate_call``
accepted them. The FastAPI routes never call them. Each call runs on a
short-lived daemon thread so a handler that blocks is reported as ``timeout``
and cannot wedge the episode or the test run.

A timed-out thread cannot be killed and keeps running in the background. The
real store handlers are microsecond-scale dict operations, so in practice only
a deliberately blocking stub reaches the timeout. Such stubs must not touch the
store.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from toolsmith.evaluation.models import canonical_hash

from .errors import ErrorObservation
from .registry import SteadyhandRegistry
from .store import NotFoundError, Store
from .types import ToolCall, ToolResult
from .validate import (
    decode_arguments,
    handler_error_observation,
    not_found_observation,
    timeout_observation,
    unknown_tool_observation,
)


class SequentialExecutor:
    """Runs one call at a time against the episode's own store. Stateless, so shareable."""

    def run(self, call: ToolCall, registry: SteadyhandRegistry, store: Store) -> ToolResult:
        started = time.monotonic()

        def failure(error: ErrorObservation, digest: str | None, *, handler_called: bool) -> ToolResult:
            return ToolResult(
                call_id=call.id,
                tool=call.name,
                arguments=call.arguments,
                arguments_sha256=digest,
                ok=False,
                error=error,
                handler_called=handler_called,
                elapsed_s=time.monotonic() - started,
            )

        # Defensive: validate_call has normally rejected these already.
        if call.name not in registry:
            return failure(unknown_tool_observation(call.name, registry), None, handler_called=False)
        arguments, error = decode_arguments(call)
        if error is not None:
            return failure(error, None, handler_called=False)
        tool = registry.get(call.name)
        policy = registry.policy(call.name)
        digest = canonical_hash(arguments)
        if tool.handler is None:
            return failure(
                handler_error_observation(call.name, "this tool has no handler"),
                digest,
                handler_called=False,
            )

        finished, value, raised = self._call_with_timeout(tool.handler, store, arguments, policy.timeout_s)
        if not finished:
            return failure(timeout_observation(call.name, policy), digest, handler_called=True)
        if isinstance(raised, NotFoundError):
            return failure(not_found_observation(call.name, raised), digest, handler_called=True)
        if raised is not None:
            return failure(
                handler_error_observation(call.name, f"{type(raised).__name__}: {raised}"),
                digest,
                handler_called=True,
            )
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            return failure(
                handler_error_observation(call.name, f"returned a value that is not JSON: {exc}"),
                digest,
                handler_called=True,
            )
        return ToolResult(
            call_id=call.id,
            tool=call.name,
            arguments=call.arguments,
            arguments_sha256=digest,
            ok=True,
            value=value,
            handler_called=True,
            elapsed_s=time.monotonic() - started,
        )

    @staticmethod
    def _call_with_timeout(
        handler: Any,
        store: Store,
        arguments: dict[str, Any],
        timeout_s: float,
    ) -> tuple[bool, Any, BaseException | None]:
        outcome: dict[str, Any] = {}

        def target() -> None:
            try:
                outcome["value"] = handler(store, **arguments)
            except BaseException as exc:  # reported to the model, never re-raised here
                outcome["error"] = exc

        worker = threading.Thread(target=target, name="steadyhand-tool", daemon=True)
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            return False, None, None
        return True, outcome.get("value"), outcome.get("error")


__all__ = ["SequentialExecutor"]

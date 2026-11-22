"""Sequential tool execution, one policy timeout around every handler.

Handlers run after the shim has returned the calls and after ``validate_call``
and the loop guard accepted them. The FastAPI routes never call them. Each call
runs on a short-lived daemon thread so a handler that blocks is reported as
``timeout`` and cannot wedge the episode or the test run.

With a ``FaultInjector`` the order is ``injector -> handler``: the injector
decides whether the real handler runs, on which store, and what the model is
told. Without one, behavior is unchanged.

A timed-out thread cannot be killed and keeps running in the background. The
real store handlers are microsecond-scale dict operations, so in practice only
a deliberately blocking stub reaches the timeout. Such stubs must not touch the
store.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from toolsmith.evaluation.models import canonical_hash

from .errors import ErrorObservation
from .policy import ToolPolicy
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

if TYPE_CHECKING:
    from .faults import FaultInjector


class SequentialExecutor:
    """Runs one call at a time against the episode's own store. Stateless, so shareable."""

    def run(
        self,
        call: ToolCall,
        registry: SteadyhandRegistry,
        store: Store,
        *,
        injector: FaultInjector | None = None,
        via: str = "model",
    ) -> ToolResult:
        started = time.monotonic()

        def refused(error: ErrorObservation, digest: str | None) -> ToolResult:
            return ToolResult(
                call_id=call.id,
                tool=call.name,
                arguments=call.arguments,
                arguments_sha256=digest,
                ok=False,
                error=error,
                handler_called=False,
                elapsed_s=time.monotonic() - started,
            )

        # Defensive: validate_call has normally rejected these already.
        if call.name not in registry:
            return refused(unknown_tool_observation(call.name, registry), None)
        arguments, error = decode_arguments(call)
        if error is not None:
            return refused(error, None)
        tool = registry.get(call.name)
        policy = registry.policy(call.name)
        digest = canonical_hash(arguments)
        if tool.handler is None:
            return refused(handler_error_observation(call.name, "this tool has no handler"), digest)

        def invoke(target: Store) -> ToolResult:
            return self._invoke(call, tool.handler, policy, arguments, digest, target, started)

        if injector is None:
            return invoke(store)
        result = injector.apply(
            call,
            invoke,
            store=store,
            policy=policy,
            result_schema=registry.result_schema(call.name),
            arguments_sha256=digest,
            via=via,
        )
        return result if result.elapsed_s else replace(result, elapsed_s=time.monotonic() - started)

    def _invoke(
        self,
        call: ToolCall,
        handler: Any,
        policy: ToolPolicy,
        arguments: dict[str, Any],
        digest: str,
        target: Store,
        started: float,
    ) -> ToolResult:
        """Run the real handler on ``target`` and turn the outcome into a ``ToolResult``."""

        def failed(error: ErrorObservation) -> ToolResult:
            return ToolResult(
                call_id=call.id,
                tool=call.name,
                arguments=call.arguments,
                arguments_sha256=digest,
                ok=False,
                error=error,
                handler_called=True,
                elapsed_s=time.monotonic() - started,
            )

        finished, value, raised = self._call_with_timeout(handler, target, arguments, policy.timeout_s)
        if not finished:
            return failed(timeout_observation(call.name, policy))
        if isinstance(raised, NotFoundError):
            return failed(not_found_observation(call.name, raised))
        if raised is not None:
            return failed(handler_error_observation(call.name, f"{type(raised).__name__}: {raised}"))
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            return failed(handler_error_observation(call.name, f"returned a value that is not JSON: {exc}"))
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

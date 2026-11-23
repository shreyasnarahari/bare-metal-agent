"""Sequential tool execution, one policy timeout around every handler.

Handlers run after the shim has returned the calls and after ``validate_call``
and the loop guard accepted them. The FastAPI routes never call them. Each call
runs on a short-lived daemon thread so a handler that blocks is reported as
``timeout`` and cannot wedge the episode or the test run.

With a ``FaultInjector`` the order is ``injector -> handler``: the injector
decides whether the real handler runs, on which store, and what the model is
told. Without one, behavior is unchanged.

With Bulkhead enabled, the episode's executor is a
``steadyhand.security.SecuredExecutor`` wrapped around this one. It makes a
policy decision for every call (model calls, recovery retries, verify probes,
and switches alike) and only hands an allowed call to this executor, so a
denied call never reaches the injector, never consumes a fault index, and
never runs a handler. The thread timeout here is not a security boundary.

A tool that needs a backend (``Effect.EXEC``, a sandboxed spec, or a network
or filesystem sink; see ``bulkhead.replay.requires_backend``) never runs
in-process, whoever calls this executor: its handler is never called and the
thread timeout below never applies to it. Without a ``backend`` runner it is
refused as a ``sandbox_violation`` before the injector. With one, the runner
replaces the handler: a live sandbox backend (``bulkhead.backends.bwrap``
from part 4), or in replay the recorded backend, which serves recorded runs
and never re-executes them. The runner holds no store lock while the external
process runs; exec tools never touch the in-memory store.

A timed-out thread cannot be killed and keeps running in the background. The
real store handlers are microsecond-scale dict operations, so in practice only
a deliberately blocking stub reaches the timeout. Such stubs must not touch the
store.
"""

from __future__ import annotations

import inspect
import json
import threading
import time
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from bulkhead.replay import requires_backend
from toolsmith.evaluation.models import canonical_hash

from .errors import ErrorObservation
from .policy import Effect, ToolPolicy
from .registry import SteadyhandRegistry
from .store import NotFoundError, Store
from .types import ToolCall, ToolResult
from .validate import (
    decode_arguments,
    handler_error_observation,
    not_found_observation,
    sandbox_violation_observation,
    timeout_observation,
    unknown_tool_observation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .faults import FaultInjector

    # Runs a call that needs a backend: (call, arguments, arguments hash, registry, via) -> result.
    BackendRunner = Callable[[ToolCall, dict[str, Any], str, SteadyhandRegistry, str], ToolResult]


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
        idempotency_key: str | None = None,
        call_index: int | None = None,
        backend: BackendRunner | None = None,
    ) -> ToolResult:
        """``idempotency_key`` reaches handlers that accept one (the server side of strategy d).
        ``call_index`` is an injector index reserved earlier, for parallel reads. ``backend`` runs
        calls that must never run in-process; without it they are refused."""
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
        spec = registry.security(call.name)
        needs_backend = requires_backend(spec, exec_effect=policy.effect is Effect.EXEC)
        if needs_backend and backend is None:
            problem = "this tool never runs in-process; it needs a sandbox backend"
            return refused(sandbox_violation_observation(call.name, "bulkhead.backend.required", problem), digest)

        # The key is added server-side only for handlers that take one; the model never supplies it.
        handler_arguments = dict(arguments)
        if idempotency_key is not None and _accepts_key(tool.handler):
            handler_arguments["idempotency_key"] = idempotency_key

        def invoke(target: Store) -> ToolResult:
            if needs_backend:
                assert backend is not None
                return backend(call, dict(arguments), digest, registry, via)
            return self._invoke(call, tool.handler, policy, handler_arguments, digest, target, started)

        # With faults scheduled, the injector decides whether ``invoke`` runs and what comes back.
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
            call_index=call_index,
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
        # A result must be plain JSON: it goes into the conversation and the journal as is.
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
        """``(finished, value, exception)``; ``finished`` is False when the handler outlived its timeout."""
        outcome: dict[str, Any] = {}

        def target() -> None:
            try:
                with store.lock:  # one handler at a time per episode store
                    outcome["value"] = handler(store, **arguments)
            except BaseException as exc:  # reported to the model, never re-raised here
                outcome["error"] = exc

        worker = threading.Thread(target=target, name="steadyhand-tool", daemon=True)
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            return False, None, None
        return True, outcome.get("value"), outcome.get("error")


# Whether a handler has an idempotency_key parameter (cached: signatures never change).
@lru_cache(maxsize=None)
def _accepts_key(handler: Any) -> bool:
    return "idempotency_key" in inspect.signature(handler).parameters


__all__ = ["SequentialExecutor"]

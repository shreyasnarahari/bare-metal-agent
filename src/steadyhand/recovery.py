"""Recovery strategies: what Steadyhand does after a tool failure or a suspicious success.

Each strategy maps a typed observation to a ``RecoveryAction``; ``apply_recovery``
carries the action out. Strategies share the controller, journal, and fault
schedule, and differ only in that mapping and three switches:

=====================  ==========  =======  ===========================================
strategy               keyed       sanity   on failure
=====================  ==========  =======  ===========================================
(a) raw_error          no          no       return a bare error string to the model
(b) typed_hint         no          no       return the typed observation (default)
(c) retry_safe         no          no       retry reads and idempotent writes in place
(d) idempotent_verify  yes         yes      verify unsafe writes, then one same-key retry
(e) switch             no          no       one alternative on permanent_unavailable
(f) escalate           yes         yes      (b)+(c)+(d)+(e), then escalate
=====================  ==========  =======  ===========================================

"Keyed" strategies send every non-idempotent write with a server-side
idempotency key. Policy-layer attempts are tagged ``via="policy"``: they
consume fault-injector indexes but never count toward the repeated-call guard.
Argument errors (``invalid_arguments``, ``unknown_tool``, ``parse_error``)
always go back to the model and are never retried or escalated here.

Waits are capped per wait and per episode, and use ``ctx.sleep`` so tests and
replay never sleep. Jitter comes from a hash of the call id, not ``random``,
so replay reproduces every wait exactly.

With Bulkhead enabled, ``ctx.executor`` is the episode's ``SecuredExecutor``:
every retry, verify probe, and switch gets its own policy decision and audit
record, and a switch to an alternative tool is authorized as that tool. A
security error (``permission_denied``, ``permission_required``,
``sandbox_violation``) is never retried, verified, or switched around, whatever
the strategy; ``permission_required`` escalates. ``Effect.EXEC`` tools are
never retried or verified automatically.

What a sandboxed call can come back as, and what recovery does with it:

- a policy denial or ask (``permission_denied`` / ``permission_required``):
  never retried; a new request is authorized on its own;
- a sandbox that could not be built, or a proxy that could not start
  (``sandbox_violation`` with a ``bulkhead.setup.*`` rule): infrastructure,
  not a denial and not a tool failure; never retried;
- a memory, PID, disk, or CPU ceiling (``resource_exhausted``): never retried,
  since the same program would exhaust it again;
- a deadline (``timeout``): unknown effects; never blindly retried for exec;
- an ordinary result, including a nonzero exit or a signal, and including a
  program whose CONNECT the proxy refused (the program saw the 403; the
  denial is in ``egress_decision``): returned to the model as is.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from toolsmith.evaluation.models import canonical_hash

from .executor import SequentialExecutor
from .faults import FaultInjector
from .idempotency import idempotency_key
from .journal import EpisodeJournal
from .policy import Effect, ToolPolicy
from .registry import SteadyhandRegistry
from .sanity import SANITY_CHECKS
from .security import SECURITY_ERRORS
from .store import Store
from .tools_ledger import ARGUMENT_MAPS, build_alternatives
from .types import ToolCall, ToolResult
from .validate import outcome_unknown_observation, suspicious_result_observation, validate_call

if TYPE_CHECKING:
    from .security import SecurityGate

# Errors only the model can fix (it sent a bad call), and errors that come from the tool's transport.
ARGUMENT_ERRORS = frozenset({"invalid_arguments", "unknown_tool", "parse_error"})
TRANSPORT_ERRORS = frozenset({"timeout", "rate_limit", "server_error"})
# Backoff: 0.5 s doubling per attempt, at most 2 s per wait and 10 s per episode.
BASE_WAIT_S = 0.5
MAX_WAIT_S = 2.0
MAX_TOTAL_WAIT_S = 10.0


class RecoveryAction(str, Enum):
    """What a strategy decides after a result: show it, retry, verify a write, switch tools, or give up."""

    RETURN_TO_MODEL = "return_to_model"
    RETRY = "retry"
    VERIFY = "verify"
    SWITCH = "switch"
    ESCALATE = "escalate"


@dataclass
class RecoveryBudget:
    """Policy waiting allowed in one episode. Waits never hold a GPU slot, but they hold a worker."""

    max_wait_s: float = MAX_WAIT_S
    max_total_wait_s: float = MAX_TOTAL_WAIT_S
    waited_s: float = 0.0


@dataclass
class RecoveryContext:
    """Everything a strategy may read, and the means to make policy-layer calls.

    ``executor`` is the episode's executor: with Bulkhead, a ``SecuredExecutor`` that
    authorizes each policy-layer attempt afresh before the injector or a handler sees it.
    ``security`` is the episode's gate: each attempt's result is applied to the episode's
    taint right after it is journaled, before the next attempt is decided.
    """

    call: ToolCall
    arguments: dict[str, Any] | None  # validated arguments; None when validation failed
    policy: ToolPolicy | None
    registry: SteadyhandRegistry
    store: Store
    executor: SequentialExecutor
    injector: FaultInjector | None
    journal: EpisodeJournal
    step: int
    episode_id: str
    sleep: Callable[[float], None]
    budget: RecoveryBudget
    idempotency_key: str | None = None
    attempts: int = 0
    verified: bool = False
    switched: bool = False
    security: SecurityGate | None = None

    @property
    def validated(self) -> bool:
        """Whether the call passed validation (only valid calls may be retried or verified)."""
        return self.arguments is not None and self.policy is not None

    def backoff(self) -> float | None:
        """The next exponential wait with deterministic jitter, or None when the budget is spent."""
        # Jitter comes from a hash of the call id and attempt, not a random number, so replays match.
        fraction = int(canonical_hash([self.call.id, self.attempts])[:8], 16) / 0xFFFFFFFF
        wait = min(self.budget.max_wait_s, BASE_WAIT_S * (2**self.attempts) * (1 + 0.25 * fraction))
        wait = round(wait, 6)
        if self.budget.waited_s + wait > self.budget.max_total_wait_s:
            return None
        self.budget.waited_s += wait
        return wait

    def attempt(
        self,
        purpose: str,
        call: ToolCall,
        registry: SteadyhandRegistry,
        *,
        key: str | None = None,
        waited_s: float = 0.0,
    ) -> ToolResult:
        """One policy-layer call: journaled, fault-injectable, invisible to the guard."""
        options: dict[str, Any] = {"via": "policy"}
        if self.injector is not None:
            options["injector"] = self.injector
        if key is not None:
            options["idempotency_key"] = key
        result = self.executor.run(call, registry, self.store, **options)
        if result.fault is not None:
            self.journal.injected_fault(step=self.step, fault=result.fault.as_dict())
        record = self.record(purpose, call, result, key=key, waited_s=waited_s)
        if self.security is not None:
            self.security.observe(call_id=call.id, tool=call.name, result=result, source_seq=record["seq"], registry=registry)
        return result

    def record(self, purpose: str, call: ToolCall, result: ToolResult, *, key: str | None, waited_s: float) -> dict[str, Any]:
        """Journal one policy-layer attempt (retry, verify, or switch); returns the record."""
        return self.journal.policy_attempt(
            step=self.step,
            tool_call_id=self.call.id,
            purpose=purpose,
            tool=call.name,
            arguments=call.arguments,
            ok=result.ok,
            error_type=result.error.error_type.value if result.error is not None else None,
            handler_called=result.handler_called,
            idempotency_key=key,
            waited_s=waited_s,
        )


class RecoveryStrategy(Protocol):
    """A recovery policy: flags for what it turns on, and its decisions after a failure or a success."""

    name: str
    keyed_writes: bool  # send non-idempotent writes with a server-side idempotency key
    sanity_checks: bool  # run opt-in invariant checks on successful results
    raw_errors: bool  # show the model a bare error string instead of the typed shape

    # Called after a failed result, with the typed observation.
    def on_observation(self, ctx: RecoveryContext, observation: dict[str, Any]) -> RecoveryAction: ...

    # Called after a successful result (which may have just failed a sanity check).
    def on_success(self, ctx: RecoveryContext, observation: dict[str, Any]) -> RecoveryAction: ...


class TypedHint:
    """(b) The default: hand the typed observation back. No policy-layer calls."""

    name = "typed_hint"
    keyed_writes = False
    sanity_checks = False
    raw_errors = False

    def on_observation(self, ctx: RecoveryContext, observation: dict[str, Any]) -> RecoveryAction:
        return RecoveryAction.RETURN_TO_MODEL

    def on_success(self, ctx: RecoveryContext, observation: dict[str, Any]) -> RecoveryAction:
        if observation.get("error_type") == "suspicious_result" and _unsafe(ctx):
            return RecoveryAction.ESCALATE  # the effect may already have happened; never repeat it
        return RecoveryAction.RETURN_TO_MODEL


class RawError(TypedHint):
    """(a) The weak baseline: ``{"ok": false, "error": "<text>"}`` with no type, details, or hint."""

    name = "raw_error"
    raw_errors = True


class RetrySafe(TypedHint):
    """(c) Retry reads and idempotent writes in place, on the policy's transient errors only."""

    name = "retry_safe"

    def on_observation(self, ctx: RecoveryContext, observation: dict[str, Any]) -> RecoveryAction:
        return RecoveryAction.RETRY if _safe_retry(ctx, observation) else RecoveryAction.RETURN_TO_MODEL


class IdempotentVerify(TypedHint):
    """(d) Key every unsafe write; after a timeout or transport error, verify, then retry once with the same key."""

    name = "idempotent_verify"
    keyed_writes = True
    sanity_checks = True

    def on_observation(self, ctx: RecoveryContext, observation: dict[str, Any]) -> RecoveryAction:
        return RecoveryAction.VERIFY if _needs_verify(ctx, observation) else RecoveryAction.RETURN_TO_MODEL


class Switch(TypedHint):
    """(e) On permanent_unavailable, call the policy's first alternative once. If that fails, escalate."""

    name = "switch"

    def __init__(self, alternatives: SteadyhandRegistry | None = None) -> None:
        self._alternatives = alternatives

    @property
    def alternatives(self) -> SteadyhandRegistry:
        if self._alternatives is None:
            self._alternatives = build_alternatives()
        return self._alternatives

    def on_observation(self, ctx: RecoveryContext, observation: dict[str, Any]) -> RecoveryAction:
        return RecoveryAction.SWITCH if _can_switch(ctx, observation) else RecoveryAction.RETURN_TO_MODEL


class Escalate(Switch):
    """(f) The full loop: typed hints, safe retries, verified unsafe writes, one alternative, then escalate."""

    name = "escalate"
    keyed_writes = True
    sanity_checks = True

    def on_observation(self, ctx: RecoveryContext, observation: dict[str, Any]) -> RecoveryAction:
        error_type = observation.get("error_type")
        if error_type in ARGUMENT_ERRORS or not ctx.validated:
            return RecoveryAction.RETURN_TO_MODEL
        if _safe_retry(ctx, observation):
            return RecoveryAction.RETRY
        if _needs_verify(ctx, observation):
            return RecoveryAction.VERIFY
        if error_type == "permanent_unavailable":
            return RecoveryAction.SWITCH if _can_switch(ctx, observation) else RecoveryAction.ESCALATE
        return RecoveryAction.RETURN_TO_MODEL


# The six strategies compared in the evaluation, weakest first.
STRATEGIES: dict[str, Callable[[], RecoveryStrategy]] = {
    "raw_error": RawError,
    "typed_hint": TypedHint,
    "retry_safe": RetrySafe,
    "idempotent_verify": IdempotentVerify,
    "switch": Switch,
    "escalate": Escalate,
}
DEFAULT_STRATEGY = "typed_hint"


def strategy_for(name: str) -> RecoveryStrategy:
    """A fresh strategy instance by name."""
    try:
        return STRATEGIES[name]()
    except KeyError:
        raise ValueError(f"Unknown recovery policy {name!r}; choose one of {sorted(STRATEGIES)}") from None


# The decision rules the strategies share.


def _writes(ctx: RecoveryContext) -> bool:
    # A non-idempotent write: repeating it would repeat its effect (and a keyed verify can check it).
    return ctx.policy is not None and ctx.policy.effect is Effect.NON_IDEMPOTENT_WRITE


def _unsafe(ctx: RecoveryContext) -> bool:
    # Any call whose repetition could repeat its effect: non-idempotent writes and exec.
    return ctx.policy is not None and ctx.policy.effect in (Effect.NON_IDEMPOTENT_WRITE, Effect.EXEC)


def _safe_retry(ctx: RecoveryContext, observation: Mapping[str, Any]) -> bool:
    # Retry in place only reads and idempotent writes, on errors the policy lists, within its attempts.
    policy = ctx.policy
    return (
        ctx.validated
        and policy is not None
        and policy.effect in (Effect.READ, Effect.IDEMPOTENT_WRITE)
        and policy.retry is not None
        and observation.get("error_type") in policy.retry.on
        and ctx.attempts < policy.retry.max_attempts
    )


def _needs_verify(ctx: RecoveryContext, observation: Mapping[str, Any]) -> bool:
    # An unsafe write that may have happened (timeout or transport error): check before retrying, once.
    error_type = observation.get("error_type")
    return (
        ctx.validated
        and _writes(ctx)
        and not ctx.verified
        and (error_type == "timeout" or (observation.get("retryable") is True and error_type in TRANSPORT_ERRORS))
    )


def _can_switch(ctx: RecoveryContext, observation: Mapping[str, Any]) -> bool:
    # A permanently unavailable tool with an alternative, not yet tried in this recovery.
    return (
        ctx.validated
        and not ctx.switched
        and observation.get("error_type") == "permanent_unavailable"
        and bool(ctx.policy and ctx.policy.alternatives)
    )


# How the verify tool shows that a keyed write already happened.
def _refund_with_key(value: Any, key: str) -> Any:
    return next((refund for refund in value.get("refunds", []) if refund.get("idempotency_key") == key), None)


def _note_with_key(value: Any, key: str) -> Any:
    note = next((note for note in value.get("notes", []) if note.get("idempotency_key") == key), None)
    return None if note is None else {"order_id": value.get("order_id"), "note": note}


# For each verifiable write: the verify tool's arguments, and how to find the keyed row in its result.
VERIFY_PROBES: dict[str, tuple[Callable[[Mapping[str, Any]], dict[str, Any]], Callable[[Any, str], Any]]] = {
    "issue_refund": (lambda arguments: {"order_id": arguments["order_id"]}, _refund_with_key),
    "add_order_note": (lambda arguments: {"order_id": arguments["order_id"]}, _note_with_key),
    # create_ticket can only be verified once a ticket id is known; after a lost
    # reply it is not, so its recovery is the same-key retry the key table makes safe.
}


@dataclass(frozen=True)
class RecoveryOutcome:
    """What the model will see, how it was recovered, and whether the episode must escalate."""

    result: ToolResult
    recovered_by: str | None = None
    escalate_reason: str | None = None
    suspicious: bool = False
    acted: bool = False


def apply_recovery(strategy: RecoveryStrategy, ctx: RecoveryContext, result: ToolResult) -> RecoveryOutcome:
    """Drive ``strategy`` from the first result to what the model should see. Bounded:
    retries by ``RetrySpec.max_attempts`` and the wait budget, one verify, one alternative."""
    recovered_by: str | None = None
    acted = False
    while True:
        # A success: run the strategy's sanity check (if any), then let it accept or escalate.
        if result.ok:
            suspicious = False
            if strategy.sanity_checks and ctx.validated and ctx.call.name in SANITY_CHECKS:
                check, hint = SANITY_CHECKS[ctx.call.name]
                problem = check(ctx.store, ctx.arguments or {}, result.value)
                ctx.journal.sanity_checked(
                    step=ctx.step, tool_call_id=ctx.call.id, tool=ctx.call.name, passed=problem is None, problem=problem,
                )
                if problem is not None:
                    suspicious, acted = True, True
                    result = replace(result, ok=False, value=None, error=suspicious_result_observation(ctx.call.name, hint, problem))
            action = strategy.on_success(ctx, result.error.as_dict() if result.error else {"ok": True})
            if action is RecoveryAction.ESCALATE:
                reason = f"{ctx.call.name} returned a result that failed a consistency check"
                return RecoveryOutcome(result, recovered_by, reason, suspicious, True)
            return RecoveryOutcome(_render(strategy, result), recovered_by, None, suspicious, acted)

        # A failure: ask the strategy what to do, act, and loop on the new result.
        assert result.error is not None
        # Security errors bypass the strategy: no retry, verify, or switch can route around them.
        if result.error.error_type.value in SECURITY_ERRORS:
            reason = None
            if result.error.error_type.value == "permission_required":
                rule = next((detail.got for detail in result.error.details if detail.field == "rule_id"), "unknown")
                reason = f"{result.tool} needs human approval under rule {rule}"
            return RecoveryOutcome(_render(strategy, result), recovered_by, reason, False, acted or reason is not None)
        action = strategy.on_observation(ctx, result.error.as_dict())
        if action is RecoveryAction.RETURN_TO_MODEL:
            return RecoveryOutcome(_render(strategy, result), recovered_by, None, False, acted)
        acted = True
        if action is RecoveryAction.RETRY:
            wait = ctx.backoff()
            if wait is None:
                return RecoveryOutcome(_render(strategy, result), recovered_by, None, False, acted)
            ctx.sleep(wait)
            ctx.attempts += 1
            result = ctx.attempt("retry", ctx.call, ctx.registry, key=ctx.idempotency_key, waited_s=wait)
            recovered_by = "retry" if result.ok else recovered_by
            continue
        if action is RecoveryAction.VERIFY:
            result, recovered_by, reason = _verify(ctx, result)
            if reason is not None:
                return RecoveryOutcome(result, recovered_by, reason, False, acted)
            if not result.ok:
                return RecoveryOutcome(_render(strategy, result), recovered_by, None, False, acted)
            continue  # a verified or retried success still gets its sanity check
        if action is RecoveryAction.SWITCH:
            alternatives = getattr(strategy, "alternatives", None)
            result, recovered_by, reason = _switch(ctx, result, alternatives)
            if reason is not None:
                return RecoveryOutcome(result, recovered_by, reason, False, acted)
            continue
        reason = f"{ctx.call.name} failed with {result.error.error_type.value} and no recovery path remains"
        return RecoveryOutcome(result, recovered_by, reason, False, acted)


# The raw_error baseline shows errors as bare strings; every other strategy shows the typed shape.
def _render(strategy: RecoveryStrategy, result: ToolResult) -> ToolResult:
    if strategy.raw_errors and result.error is not None:
        return replace(result, error=replace(result.error, raw=True))
    return result


# Did the keyed write already happen? Ask the verify tool. Found: report success without writing
# again. Not found: retry once with the same key (safe either way). Probe failed: escalate,
# because the outcome is unknown. Without a key or probe, just retry with the key.
def _verify(ctx: RecoveryContext, failed: ToolResult) -> tuple[ToolResult, str | None, str | None]:
    ctx.verified = True
    key = ctx.idempotency_key
    probe_spec = VERIFY_PROBES.get(ctx.call.name)
    verify_tool = ctx.policy.verify if ctx.policy is not None else None
    if key is None or probe_spec is None or verify_tool is None or verify_tool not in ctx.registry:
        retried = ctx.attempt("retry", ctx.call, ctx.registry, key=key)
        return retried, ("retry" if retried.ok else None), None
    build_arguments, find = probe_spec
    probe_call = ToolCall(f"{ctx.call.id}~verify", verify_tool, json.dumps(build_arguments(ctx.arguments or {})))
    probe = ctx.attempt("verify", probe_call, ctx.registry)
    if not probe.ok:
        assert probe.error is not None
        unknown = replace(
            failed,
            error=outcome_unknown_observation(ctx.call.name, f"{verify_tool} failed with {probe.error.error_type.value}"),
            fault=None,
        )
        reason = f"{ctx.call.name} may or may not have taken effect, and {verify_tool} could not confirm it"
        return unknown, None, reason
    present = find(probe.value, key)
    if present is not None:
        return replace(failed, ok=True, value=present, error=None, fault=None, handler_called=False), "verify", None
    retried = ctx.attempt("retry", ctx.call, ctx.registry, key=key)
    return retried, ("retry" if retried.ok else None), None


# Call the policy's first alternative once, with mapped arguments, validated and keyed like any call.
def _switch(
    ctx: RecoveryContext,
    failed: ToolResult,
    alternatives: SteadyhandRegistry | None,
) -> tuple[ToolResult, str | None, str | None]:
    ctx.switched = True
    alternative = ctx.policy.alternatives[0] if ctx.policy is not None else ""
    mapper = ARGUMENT_MAPS.get((ctx.call.name, alternative))
    if alternatives is None or mapper is None or alternative not in alternatives:
        return failed, None, f"{ctx.call.name} is unavailable and its alternative {alternative} cannot be called"
    alternative_call = ToolCall(f"{ctx.call.id}~switch", alternative, json.dumps(mapper(ctx.arguments or {})))
    verdict = validate_call(alternative_call, alternatives, ctx.store)
    if not verdict.ok or verdict.arguments is None:
        rejected = verdict.rejected(alternative_call)
        ctx.record("switch", alternative_call, rejected, key=None, waited_s=0.0)
        return failed, None, f"{ctx.call.name} is unavailable and the alternative {alternative} rejected the request"
    key = idempotency_key(ctx.episode_id, alternative, verdict.arguments)
    outcome = ctx.attempt("switch", alternative_call, alternatives, key=key)
    if not outcome.ok:
        assert outcome.error is not None
        reason = (
            f"{ctx.call.name} is unavailable and the alternative {alternative} failed "
            f"({outcome.error.error_type.value})"
        )
        return failed, None, reason
    return replace(failed, ok=True, value=outcome.value, error=None, fault=None, handler_called=False), "switch", None


def no_sleep(seconds: float) -> None:
    """Replay never waits; the recorded waits are checked instead."""


__all__ = [
    "DEFAULT_STRATEGY",
    "Escalate",
    "IdempotentVerify",
    "RawError",
    "RecoveryAction",
    "RecoveryBudget",
    "RecoveryContext",
    "RecoveryOutcome",
    "RecoveryStrategy",
    "RetrySafe",
    "STRATEGIES",
    "Switch",
    "TypedHint",
    "apply_recovery",
    "no_sleep",
    "strategy_for",
]

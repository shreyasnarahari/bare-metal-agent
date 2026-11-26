"""The adapter between Bulkhead and Steadyhand: one security path for every call.

With ``run_episode(..., security=SecurityConfig(...))`` every call is
authorized by Bulkhead's pure policy engine before anything runs:

    validate_call -> repeated-call guard -> SecurityGate.authorize
      -> DENY / ASK: a typed observation; no fault injector, no handler
      -> ALLOW: fault injector -> trusted in-process handler, or for exec,
         sandboxed, filesystem, and network tools a sandbox backend

The episode's executor becomes a ``SecuredExecutor``, so the recovery layer's
retries, verify probes, and switches to an alternative tool each get their own
fresh decision and audit record; nothing relies on the model call having been
authorized for a different tool or different arguments. Every decision is a
``policy_decision`` journal record, in call order.

- ``DENY`` gives the model a non-retryable ``permission_denied`` observation;
  it may choose another permitted action.
- ``ASK`` journals ``approval_requested``, answers the call with
  ``permission_required``, and escalates. The finished journal is never
  rewritten. A human (or a declared scripted approver) may then approve; the
  retry is a fresh claimed episode (``run_approved_episode``) linked to its
  parent, carrying one single-use grant for that exact call.

Taint (part 3): the gate holds the episode's ``EpisodeTaintState``. After each
result is journaled (model calls and policy-layer attempts alike, in call
order), an observation that carries outside content from a tool whose output
is not declared trusted makes the episode untrusted and journals a
``taint_transition``. The next decision sees the new taint, so an untrusted
read followed by a sensitive call in the same message still asks. Refusals,
validation errors, and other loop-authored observations carry no outside
content. Taint never decreases, and is never shared between episodes.

Tools that must not run in-process go to an ``ExecutionBackend`` through
``BackendRunner``. It resolves an immutable ``ExecutionRequest`` from the
registered spec (the exec tool's program descriptor, its timeout, its sandbox),
hands it to the backend only after ``ALLOW`` (and after the fault injector,
which may prevent the call but cannot change its spec), converts what the
backend observed into a typed tool result, and journals the run as
``sandbox_execution`` between the ``policy_decision`` and the ``tool_result``.
The live backend is ``bulkhead.backends.bwrap`` (part 4, the namespace/seccomp
phase of tier 2); replay uses the recorded backend.

A backend run's events are journaled in this order, between the call's
``policy_decision`` and its ``tool_result``: ``sandbox_execution`` (the run and
its typed result), one ``egress_decision`` per proxied request (authority and
byte counts only), and ``output_gate`` (what sanitizing did to each stream).
The model and the journal only ever see gated output: sanitized, bounded,
redacted, and wrapped in deterministic untrusted-output delimiters. The
wrapper explains; the taint state and policy enforce.

With ``security=None`` nothing here runs and episodes behave exactly as before.
Without a backend, sandboxed tools are denied by the policy gate (no available
tier) and refused by the executor.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bulkhead.approvals import (
    ApprovalMode,
    ApprovalRequest,
    Approver,
    Grant,
    approve,
    unconsumed,
)
from bulkhead.errors import GrantError, PolicyConfigError
from bulkhead.manifest import PathAccess, SecuritySpec, Sink, Trust, check_keys
from bulkhead.policy import Decision, NarrowingOutcome, PolicyEngine, PolicyRequest, TaskPolicy, ToolRule, Verdict
from bulkhead.backends.base import ExecResult, ProgramError
from bulkhead.replay import BackendIdentity, ExecutionBackend, ExecutionRequest, RecordedExecutionMismatch, requires_backend
from bulkhead.taint import DEFAULT_TAINT_SENSITIVE, EpisodeTaintState, TaintSource, TaintTransition
from bulkhead.types import SandboxTier, canonical_hash, canonical_json
from toolsmith.json_utils import StrictJSONError, strict_json_loads

from .errors import ErrorObservation, ErrorType, one_line
from .executor import SequentialExecutor
from .journal import ReplayMismatch
from .policy import Effect
from .registry import SteadyhandRegistry
from .store import Store
from .tools import HANDLERS
from .types import ToolCall, ToolResult
from .validate import (
    permission_denied_observation,
    permission_required_observation,
    resource_exhausted_observation,
    sandbox_violation_observation,
    timeout_observation,
)

if TYPE_CHECKING:
    from .faults import FaultInjector
    from .journal import EpisodeJournal
    from .recovery import RecoveryStrategy
    from .types import EpisodeResult, ModelClient

CONFIG_VERSION = 1
SECURITY_ERRORS = frozenset({"permission_denied", "permission_required", "sandbox_violation"})
STORE_TOOLS = tuple(handler.__name__ for handler in HANDLERS)


@dataclass(frozen=True)
class SecurityConfig:
    """Everything that decides a secured episode's verdicts: hashed into its manifest identity.

    ``available_tiers`` are the sandbox tiers that are both proven by the part 1
    probe and backed by an implemented backend. It is empty until part 4, so
    every sandboxed tool is denied. ``grants`` are honored only in an approved
    episode whose parent is the grant's episode.
    """

    task_policy: TaskPolicy
    approval_mode: ApprovalMode = ApprovalMode.DENY_ALL
    grants: tuple[Grant, ...] = ()
    available_tiers: tuple[SandboxTier, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.task_policy, TaskPolicy):
            raise PolicyConfigError("task_policy must be a TaskPolicy")
        try:
            object.__setattr__(self, "approval_mode", ApprovalMode(self.approval_mode))
            tiers = tuple(sorted({SandboxTier(tier) for tier in self.available_tiers}, key=lambda tier: tier.rank))
        except ValueError as exc:
            raise PolicyConfigError(f"unknown approval mode or tier: {exc}") from None
        if SandboxTier.T0 in tiers:
            raise PolicyConfigError("T0 is not a sandbox and cannot back sandboxed tools")
        grants = tuple(self.grants)
        if not all(isinstance(grant, Grant) for grant in grants):
            raise PolicyConfigError("grants must be Grant values")
        ids = [grant.approval_id for grant in grants]
        if len(set(ids)) != len(ids):
            raise PolicyConfigError("a grant may appear only once")
        object.__setattr__(self, "available_tiers", tiers)
        object.__setattr__(self, "grants", tuple(sorted(grants, key=lambda grant: grant.approval_id)))

    def with_grants(self, grants: Iterable[Grant]) -> SecurityConfig:
        return replace(self, grants=tuple(grants))

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": CONFIG_VERSION,
            "task_policy": self.task_policy.as_dict(),
            "approval_mode": self.approval_mode.value,
            "grants": [grant.as_dict() for grant in self.grants],
            "available_tiers": [tier.value for tier in self.available_tiers],
        }

    def sha256(self) -> str:
        """The episode's security policy hash; with a grant it is the amended policy hash."""
        return canonical_hash(self.as_dict())

    @classmethod
    def from_dict(cls, value: Any) -> SecurityConfig:
        check_keys(value, ("version", "task_policy", "approval_mode", "grants", "available_tiers"), "security", error=PolicyConfigError)
        if value["version"] != CONFIG_VERSION:
            raise PolicyConfigError(f"security config version {value['version']!r} is not {CONFIG_VERSION}")
        if not isinstance(value["grants"], list) or not isinstance(value["available_tiers"], list):
            raise PolicyConfigError("grants and available_tiers must be lists")
        return cls(
            task_policy=TaskPolicy.from_dict(value["task_policy"]),
            approval_mode=value["approval_mode"],
            grants=tuple(Grant.from_dict(grant) for grant in value["grants"]),
            available_tiers=tuple(value["available_tiers"]),
        )


def store_task_policy(
    *,
    tools: Sequence[str] = STORE_TOOLS,
    deny_tools: Sequence[str] = (),
    ask_tools: Sequence[str] = (),
    rules: Sequence[ToolRule] = (),
    taint_sensitive: Sequence[Sink] = DEFAULT_TAINT_SENSITIVE,
    taint_sensitive_tools: Sequence[str] = (),
    available_tools: Sequence[str] = (),
) -> TaskPolicy:
    """A task policy for mock-store episodes: the ten store tools (plus ``available_tools``), no paths, no network.

    ``taint_sensitive_tools`` names store effects the task treats as sensitive (for example
    ``issue_refund``): once the episode is untrusted, calling them asks.
    """
    return TaskPolicy(
        tools=(*tools, *available_tools),
        deny_tools=tuple(deny_tools),
        ask_tools=tuple(ask_tools),
        rules=tuple(rules),
        taint_sensitive=tuple(taint_sensitive),
        taint_sensitive_tools=tuple(taint_sensitive_tools),
    )


def carries_outside_content(result: ToolResult, *, backend_run: bool) -> bool:
    """Whether an observation holds data a tool produced (and so can carry outside content).

    A success always does. A failure does when the tool itself produced it: a handler
    exception, or any result of a backend run that started. Refusals, validation errors,
    guard blocks, injected faults, and timeouts are short loop-authored metadata.
    """
    if result.ok:
        return True
    if not result.handler_called or result.error is None:
        return False
    return backend_run or result.error.error_type is ErrorType.HANDLER_ERROR


def tool_result_from(call: ToolCall, digest: str, result: Mapping[str, Any]) -> ToolResult:
    """Rebuild a ``ToolResult`` from a backend's typed result (live or recorded)."""
    if result["ok"]:
        return ToolResult(call.id, call.name, call.arguments, digest, True, value=result["value"], handler_called=result["handler_called"])
    return ToolResult(
        call.id, call.name, call.arguments, digest, False,
        error=ErrorObservation.from_dict(result["error"]), handler_called=result["handler_called"],
    )


def typed_result(result: ToolResult) -> dict[str, Any]:
    """A ``ToolResult`` as the typed result a backend reports."""
    body: dict[str, Any] = {"ok": result.ok, "handler_called": result.handler_called}
    if result.ok:
        body["value"] = result.value
    else:
        assert result.error is not None
        body["error"] = result.error.as_dict()
    return body


def decode_for_policy(call: ToolCall) -> tuple[Any, str]:
    """The decoded arguments and their hash; undecodable arguments hash their raw text (and are denied)."""
    try:
        arguments = strict_json_loads(call.arguments)
    except StrictJSONError:
        return call.arguments, canonical_hash({"__unparsed_arguments__": call.arguments})
    return arguments, canonical_hash(arguments)


def grant_summary(requested: Mapping[str, Any] | None) -> str:
    """One line naming what an approval would allow: the call, the rules that asked, any widening."""
    if not requested:
        return "nothing to grant"
    widen = canonical_json(dict(requested.get("widen") or {}))
    text = f"{requested.get('tool')} with arguments {str(requested.get('arguments_sha256'))[:12]}; rules {', '.join(requested.get('rules') or [])}"
    return one_line(text if widen == "{}" else f"{text}; widen {widen}", 240)


@dataclass(frozen=True)
class Authorization:
    """A decision for one exact call. The secured executor runs a call only with an allowed one."""

    decision: Decision
    call_id: str
    tool: str
    arguments_sha256: str
    via: str

    @property
    def allowed(self) -> bool:
        return self.decision.allowed

    @property
    def asks(self) -> bool:
        return self.decision.verdict is Verdict.ASK

    def covers(self, call: ToolCall) -> bool:
        return call.id == self.call_id and call.name == self.tool and decode_for_policy(call)[1] == self.arguments_sha256


class SecurityGate:
    """One episode's Bulkhead adapter: decides, journals, and builds observations. Never shared."""

    def __init__(
        self,
        config: SecurityConfig,
        registry: SteadyhandRegistry,
        *,
        journal: EpisodeJournal,
        episode_id: str,
        parent_episode_id: str | None = None,
        step: Callable[[], int] = lambda: 0,
        engine: PolicyEngine | None = None,
        backend_identity: BackendIdentity | None = None,
    ) -> None:
        if config.grants and parent_episode_id is None:
            raise GrantError("grants are honored only in an approved episode that names its parent")
        self.config = config
        self.policy = config.task_policy
        self.registry = registry
        self.journal = journal
        self.episode_id = episode_id
        self.parent_episode_id = parent_episode_id
        self.engine = engine or PolicyEngine()
        self.taint_state = EpisodeTaintState()  # owned by this episode alone; never shared
        self.consumed: list[str] = []
        self.backend_identity = backend_identity
        self._step = step
        self._available = frozenset(config.available_tiers)

    @property
    def taint(self) -> Trust:
        return self.taint_state.label

    def step(self) -> int:
        return self._step()

    def summary(self) -> dict[str, Any]:
        """The ``bulkhead`` field of ``episode_started``."""
        return {
            "security_config_sha256": self.config.sha256(),
            "security_spec_sha256": self.registry.security_spec_sha256(),
            "task_policy_sha256": self.policy.sha256(),
            "approval_mode": self.config.approval_mode.value,
            "parent_episode_id": self.parent_episode_id,
            "backend": self.backend_identity.as_dict() if self.backend_identity is not None else None,
        }

    def observe(
        self, *, call_id: str, tool: str, result: ToolResult, source_seq: int, registry: SteadyhandRegistry | None = None,
    ) -> TaintTransition | None:
        """Apply one journaled result's taint (``source_seq`` is its record); journal a change of label."""
        registry = registry if registry is not None else self.registry
        spec = self.spec_for(tool, registry)
        backend_run = tool in registry and requires_backend(spec, exec_effect=registry.policy(tool).effect is Effect.EXEC)
        if not carries_outside_content(result, backend_run=backend_run):
            return None
        trust = spec.output_trust if spec is not None else Trust.UNTRUSTED
        self.taint_state, transition = self.taint_state.observe(TaintSource(call_id, tool, source_seq), trust)
        if transition is not None:
            self.journal.taint_transition(
                step=self._step(),
                tool_call_id=call_id,
                source_tool=tool,
                before=transition.before.value,
                after=transition.after.value,
                source_seq=source_seq,
            )
        return transition

    @staticmethod
    def spec_for(name: str, registry: SteadyhandRegistry) -> SecuritySpec | None:
        return registry.security(name) if name in registry else None

    def authorize(self, call: ToolCall, registry: SteadyhandRegistry | None = None, *, via: str = "model") -> Authorization:
        """Decide one call and journal the decision (and, for ASK, the approval request)."""
        registry = registry if registry is not None else self.registry
        arguments, digest = decode_for_policy(call)
        request = PolicyRequest(
            episode_id=self.episode_id,
            call_id=call.id,
            via=via,
            tool=call.name,
            arguments=arguments,
            arguments_sha256=digest,
            policy=self.policy,
            spec=self.spec_for(call.name, registry),
            taint_before=self.taint,
            grants=unconsumed(self.config.grants, self.consumed),
            available_tiers=self._available,
            parent_episode_id=self.parent_episode_id,
        )
        decision = self.engine.decide(request)
        if decision.grant_id is not None:
            self.consumed.append(decision.grant_id)  # single use: the next identical call asks again
        step = self._step()
        self.journal.policy_decision(
            step=step,
            tool_call_id=call.id,
            via=via,
            tool=call.name,
            arguments_sha256=digest,
            verdict=decision.verdict.value,
            rule_id=decision.rule_id,
            reason_code=decision.reason_code,
            taint_before=self.taint.value,
            policy_sha256=decision.policy_sha256,
            grant_id=decision.grant_id,
            normalized_request_sha256=decision.normalized_request_sha256,
        )
        if decision.verdict is Verdict.ASK:
            assert decision.requested_grant is not None
            self.journal.approval_requested(
                step=step,
                tool_call_id=call.id,
                rule_id=decision.rule_id,
                request_sha256=decision.normalized_request_sha256,
                normalized_grant=decision.requested_grant,
                parent_episode_id=self.parent_episode_id,
                approval_mode=self.config.approval_mode.value,
            )
        return Authorization(decision, call.id, call.name, digest, via)

    def refusal(self, call: ToolCall, authorization: Authorization) -> ToolResult:
        """The tool result for a denied or asked call. The handler did not run."""
        decision = authorization.decision
        if decision.verdict is Verdict.DENY:
            error = permission_denied_observation(call.name, decision.rule_id, decision.reason_code)
        elif decision.verdict is Verdict.ASK:
            error = permission_required_observation(
                call.name, decision.rule_id, decision.reason_code, grant_summary(decision.requested_grant)
            )
        else:
            raise ValueError("an allowed call is not refused")
        return ToolResult(
            call_id=call.id,
            tool=call.name,
            arguments=call.arguments,
            arguments_sha256=authorization.arguments_sha256,
            ok=False,
            error=error,
            handler_called=False,
        )

    @staticmethod
    def escalation_reason(call: ToolCall, authorization: Authorization) -> str:
        decision = authorization.decision
        return f"{call.name} needs human approval under rule {decision.rule_id} ({grant_summary(decision.requested_grant)})"

    def batchable(self, call: ToolCall, registry: SteadyhandRegistry) -> bool:
        """Whether a read may join a parallel batch decided on one starting taint.

        Only an in-process call with a spec, no sink, no egress, no write-path argument, and
        no taint-sensitive listing qualifies; anything else runs sequentially, so it is decided
        after every earlier result's taint has been applied.
        """
        spec = self.spec_for(call.name, registry)
        if spec is None or spec.sinks or spec.sandbox is not None:
            return False
        if call.name in self.policy.taint_sensitive_tools:
            return False
        rule = self.policy.rules_by_tool().get(call.name)
        return rule is None or all(access is not PathAccess.WRITE for access in rule.paths.values())

    def narrow(self, candidate: TaskPolicy) -> NarrowingOutcome:
        """Apply a narrower policy; a widening leaves the policy unchanged and returns the ASK it needs."""
        outcome = self.engine.narrow(self.policy, candidate)
        if outcome.verdict is Verdict.ALLOW:
            self.policy = outcome.policy
        return outcome


def process_tool_result(
    call: ToolCall, digest: str, process: ExecResult, policy: Any, *, request_sha256: str = "0" * 64, secrets: Sequence[str] = ()
) -> tuple[ToolResult, list[dict[str, Any]] | None]:
    """Turn what a live backend observed into the tool result the model sees.

    A sandbox that never started is a ``sandbox_violation`` (the program did
    not run; ``handler_called`` is false). A run stopped by a memory, PID,
    disk, or CPU ceiling is ``resource_exhausted`` (never retried: it would
    exhaust the ceiling again). A run killed at its deadline is a ``timeout``
    with unknown effects, never retried automatically. Anything
    else, including a nonzero exit or a signal, is a valid result whose value
    is the bounded ``ExecResult`` (exit code, signal, stdout, stderr,
    truncation flags). Its output is untrusted (the exec specs declare it so).

    A completed run's streams pass through the output gate: already sanitized
    by the backend, they are redacted (``secrets``) and wrapped in
    deterministic untrusted-output delimiters derived from ``request_sha256``.
    Returns the result and the gate's stream records (None when no output is shown).
    """
    if not process.started:
        rule = process.setup_error or "bulkhead.setup.failed"
        error = sandbox_violation_observation(call.name, rule, "the sandbox could not be set up, so the program did not run")
        return ToolResult(call.id, call.name, call.arguments, digest, False, error=error, handler_called=False), None
    if process.exhausted is not None:
        error = resource_exhausted_observation(call.name, process.exhausted)
        return ToolResult(call.id, call.name, call.arguments, digest, False, error=error, handler_called=True), None
    if process.timed_out:
        return ToolResult(call.id, call.name, call.arguments, digest, False, error=timeout_observation(call.name, policy), handler_called=True), None
    value, streams = process.render(request_sha256, secrets)
    return ToolResult(call.id, call.name, call.arguments, digest, True, value=value, handler_called=True), streams


class BackendRunner:
    """Runs calls that must not run in-process through the episode's backend and journals each run.

    The backend must run at the task's required tier. The request carries the
    exec tool's program (from its registered descriptor) and the policy
    timeout; program text never reaches the journal. A recorded backend's
    mismatch (replay asked for a run the recording does not hold) becomes a
    ``ReplayMismatch`` naming the recorded record and field.
    """

    def __init__(self, backend: ExecutionBackend, gate: SecurityGate) -> None:
        self.backend = backend
        self.gate = gate

    def __call__(self, call: ToolCall, arguments: dict[str, Any], digest: str, registry: SteadyhandRegistry, via: str) -> ToolResult:
        spec = registry.security(call.name)
        policy = registry.policy(call.name)

        def violation(rule: str, problem: str) -> ToolResult:
            return ToolResult(call.id, call.name, call.arguments, digest, False, error=sandbox_violation_observation(call.name, rule, problem))

        if spec is None or not spec.sandboxed:
            return violation("bulkhead.backend.no_sandbox", "the tool has no sandbox specification to run under")
        if self.backend.identity.tier is not self.gate.policy.required_tier:
            return violation("bulkhead.backend.tier", "the backend's tier is not the tier the task requires")
        descriptor = registry.program(call.name)
        program = None
        if descriptor is not None:
            try:
                program = descriptor.program(arguments)
            except ProgramError as exc:
                return violation(exc.code, str(exc))
        request = ExecutionRequest(self.gate.episode_id, call.id, via, call.name, digest, spec, program=program, timeout_s=policy.timeout_s)
        try:
            outcome = self.backend.execute(request, arguments)
        except RecordedExecutionMismatch as exc:
            raise ReplayMismatch(str(exc), seq=exc.seq, field=exc.field) from None
        if outcome.result is not None:
            result = tool_result_from(call, digest, outcome.result)
            streams = list(outcome.gate) if outcome.gate is not None else None
        else:
            assert outcome.process is not None
            result, streams = process_tool_result(
                call, digest, outcome.process, policy, request_sha256=request.sha256, secrets=getattr(self.gate.journal, "secrets", ()),
            )
        step = self.gate.step()
        journal = self.gate.journal
        journal.sandbox_execution(step=step, tool_call_id=call.id, via=via, tool=call.name, result=typed_result(result), **outcome.audit)
        for record in outcome.egress:
            journal.egress_decision(step=step, tool_call_id=call.id, **record)
        if streams is not None:
            journal.output_gate(step=step, tool_call_id=call.id, via=via, tool=call.name, streams=streams)
        return result


class SecuredExecutor:
    """Wraps the episode's executor: every call is authorized before the injector or handler sees it.

    The loop passes the ``Authorization`` it already made (decisions for one
    message are made in call order before anything runs). Policy-layer calls
    from recovery come without one and get a fresh decision here. Calls that
    must not run in-process go to ``runner`` (a ``BackendRunner``), or are
    refused when there is none.
    """

    def __init__(self, inner: SequentialExecutor, gate: SecurityGate, runner: BackendRunner | None = None) -> None:
        self.inner = inner
        self.gate = gate
        self.runner = runner

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
        authorization: Authorization | None = None,
        execution_context: Any = None,
    ) -> ToolResult:
        if authorization is None:
            authorization = self.gate.authorize(call, registry, via=via)
        elif not authorization.covers(call):
            raise RuntimeError(f"an authorization for {authorization.tool} was presented for {call.name}")
        if not authorization.allowed:
            return self.gate.refusal(call, authorization)
        options: dict[str, Any] = {"via": via, "backend": self.runner}
        if injector is not None:
            options["injector"] = injector
        if idempotency_key is not None:
            options["idempotency_key"] = idempotency_key
        if call_index is not None:
            options["call_index"] = call_index
        if execution_context is not None:
            options["execution_context"] = execution_context  # trusted resources, only after ALLOW
        return self.inner.run(call, registry, store, **options)


# Approvals: from an escalated journal to a linked approved episode


def approval_requests(journal_path: str | Path) -> list[ApprovalRequest]:
    """Every approval request an episode journaled, rebuilt with the decision that asked."""
    from .journal import JournalFormatError, read_journal

    records = read_journal(journal_path)
    episode_id = records[0]["episode_id"]
    requests = []
    for index, record in enumerate(records):
        if record["event"] != "approval_requested":
            continue
        decision = records[index - 1] if index else {}
        if decision.get("event") != "policy_decision" or decision.get("tool_call_id") != record["tool_call_id"] or decision.get("verdict") != Verdict.ASK.value:
            raise JournalFormatError("approval_requested must follow its ASK policy_decision", seq=record["seq"], field="event")
        requests.append(
            ApprovalRequest(
                episode_id=episode_id,
                tool_call_id=record["tool_call_id"],
                tool=decision["tool"],
                arguments_sha256=decision["arguments_sha256"],
                policy_sha256=decision["policy_sha256"],
                rule_id=record["rule_id"],
                request_sha256=record["request_sha256"],
                normalized_grant=record["normalized_grant"],
            )
        )
    return requests


@dataclass(frozen=True)
class ApprovalLineage:
    """How an approved retry links to the episode that asked: parent id, run label, and its one grant."""

    episode_id: str
    parent_episode_id: str
    run_label: str
    grant: Grant


def approval_lineage(journal_path: str | Path) -> ApprovalLineage | None:
    """The parent/child link recorded in an episode's manifest, or None for an episode that is not a retry.

    Checks the link is self-consistent: one grant, bound to the parent, named by the run label.
    """
    from .journal import JournalFormatError, manifest_path_for
    from .manifest import load_manifest

    manifest = load_manifest(manifest_path_for(journal_path))
    if manifest.parent_episode_id is None:
        return None
    config = SecurityConfig.from_dict(manifest.security_config)
    if len(config.grants) != 1:
        raise JournalFormatError("an approved retry carries exactly one grant", field="security_config")
    grant = config.grants[0]
    if grant.episode_id != manifest.parent_episode_id or manifest.run_label != f"approved-{grant.approval_id}":
        raise JournalFormatError("the approval link is inconsistent", field="parent_episode_id")
    return ApprovalLineage(manifest.episode_id, manifest.parent_episode_id, manifest.run_label, grant)


def resolve_approvals(journal_path: str | Path, approver: Approver, *, now: datetime, ttl_s: float = 3600.0) -> list[Grant]:
    """Ask ``approver`` about each request; one grant per approved request, nothing for a decline."""
    return [
        approve(request, approver=approver.name, now=now, ttl_s=ttl_s)
        for request in approval_requests(journal_path)
        if approver.decide(request)
    ]


def run_approved_episode(
    directory: str | Path,
    parent_journal: str | Path,
    grant: Grant,
    client: ModelClient,
    registry: SteadyhandRegistry,
    store: Store,
    *,
    now: datetime,
    secrets: Sequence[str] = (),
    executor: SequentialExecutor | None = None,
    recovery: RecoveryStrategy | None = None,
    clock: Callable[[], float] = time.monotonic,
    cancel: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    backend: ExecutionBackend | None = None,
) -> EpisodeResult | None:
    """Retry an escalated episode as a fresh claimed episode carrying one single-use grant.

    The retry starts from the parent's initial messages on an unmodified copy of
    the same fixture; nothing from the parent's run is replayed. Its run label
    is ``approved-<approval id>``, its manifest names the parent and holds the
    amended security configuration, and the parent's files are never touched.
    Returns ``None`` when the approved episode is already claimed or complete.
    """
    from .controller import Budgets
    from .faults import FaultSchedule
    from .journal import manifest_path_for, read_journal
    from .manifest import load_manifest
    from .recovery import strategy_for
    from .replay import run_claimed_episode

    parent_path = Path(parent_journal)
    records = read_journal(parent_path)
    if records[-1]["event"] != "episode_finished" or records[-1]["status"] != "escalated":
        raise GrantError("only a finished, escalated episode can be retried with an approval")
    manifest = load_manifest(manifest_path_for(parent_path))
    if not manifest.secured or manifest.security_config is None:
        raise GrantError("the parent episode did not run with Bulkhead")
    if grant.episode_id != manifest.episode_id:
        raise GrantError("the grant belongs to another episode")
    if grant.expired(now):
        raise GrantError("the grant has expired")
    answered = [
        request
        for request in approval_requests(parent_path)
        if request.tool == grant.tool
        and request.arguments_sha256 == grant.arguments_sha256
        and request.policy_sha256 == grant.policy_sha256
        and canonical_json(dict(request.normalized_grant)) == canonical_json(dict(grant.normalized_grant))
    ]
    if not answered:
        raise GrantError("the grant answers no approval request of the parent episode")
    if store.snapshot_sha256() != manifest.fixture_sha256:
        raise GrantError("an approved retry must start from the parent's unmodified fixture")
    config = SecurityConfig.from_dict(manifest.security_config).with_grants((grant,))
    started = records[0]
    faults = FaultSchedule.from_dict(manifest.fault_schedule) if manifest.fault_schedule is not None else None
    return run_claimed_episode(
        directory,
        client,
        registry,
        store,
        started["messages"],
        budgets=Budgets(**started["budgets"]),
        guard_limit=started["guard_limit"],
        paraphrase_final=started["paraphrase_final"],
        secrets=secrets,
        run_label=f"approved-{grant.approval_id}",
        server_parallel=manifest.server_parallel,
        context_size=manifest.context_size,
        executor=executor,
        clock=clock,
        cancel=cancel,
        faults=faults,
        recovery=recovery or strategy_for(manifest.recovery_policy),
        system_prompt_variant=manifest.system_prompt_variant,
        sleep=sleep,
        security=config,
        parent_episode_id=manifest.episode_id,
        backend=backend,
    )


__all__ = [
    "CONFIG_VERSION",
    "SECURITY_ERRORS",
    "STORE_TOOLS",
    "ApprovalLineage",
    "Authorization",
    "BackendRunner",
    "process_tool_result",
    "SecuredExecutor",
    "SecurityConfig",
    "SecurityGate",
    "approval_lineage",
    "approval_requests",
    "carries_outside_content",
    "decode_for_policy",
    "grant_summary",
    "resolve_approvals",
    "run_approved_episode",
    "store_task_policy",
    "tool_result_from",
    "typed_result",
]

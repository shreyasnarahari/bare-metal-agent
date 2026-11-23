"""The adapter between Bulkhead and Steadyhand: one security path for every call.

With ``run_episode(..., security=SecurityConfig(...))`` every call is
authorized by Bulkhead's pure policy engine before anything runs:

    validate_call -> repeated-call guard -> SecurityGate.authorize
      -> DENY / ASK: a typed observation; no fault injector, no handler
      -> ALLOW: fault injector -> trusted in-process handler (or, from part 4,
         a sandbox backend)

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

With ``security=None`` nothing here runs and episodes behave exactly as before.
No operating-system sandbox exists yet: sandboxed tools are denied until part
4 adds a backend and part 1's probe proves its tier.
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
from bulkhead.manifest import SecuritySpec, Sink, Trust, check_keys
from bulkhead.policy import Decision, NarrowingOutcome, PolicyEngine, PolicyRequest, TaskPolicy, ToolRule, Verdict
from bulkhead.types import SandboxTier, canonical_hash, canonical_json
from toolsmith.json_utils import StrictJSONError, strict_json_loads

from .errors import one_line
from .executor import SequentialExecutor
from .registry import SteadyhandRegistry
from .store import Store
from .tools import HANDLERS
from .types import ToolCall, ToolResult
from .validate import permission_denied_observation, permission_required_observation

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
    taint_sensitive: Sequence[Sink] = (Sink.FILESYSTEM, Sink.NETWORK),
) -> TaskPolicy:
    """A task policy for mock-store episodes: the ten store tools, no paths, no network."""
    return TaskPolicy(
        tools=tuple(tools),
        deny_tools=tuple(deny_tools),
        ask_tools=tuple(ask_tools),
        rules=tuple(rules),
        taint_sensitive=tuple(taint_sensitive),
    )


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
        # Part 3 moves this to UNTRUSTED when outside content arrives; until then it stays TRUSTED.
        self.taint = Trust.TRUSTED
        self.consumed: list[str] = []
        self._step = step
        self._available = frozenset(config.available_tiers)

    def summary(self) -> dict[str, Any]:
        """The ``bulkhead`` field of ``episode_started``."""
        return {
            "security_config_sha256": self.config.sha256(),
            "security_spec_sha256": self.registry.security_spec_sha256(),
            "task_policy_sha256": self.policy.sha256(),
            "approval_mode": self.config.approval_mode.value,
            "parent_episode_id": self.parent_episode_id,
        }

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
        """Whether a read may join a parallel batch: it has a spec, no sink, and no egress."""
        spec = self.spec_for(call.name, registry)
        if spec is None or spec.sinks:
            return False
        return spec.sandbox is None or not spec.sandbox.net.enabled

    def narrow(self, candidate: TaskPolicy) -> NarrowingOutcome:
        """Apply a narrower policy; a widening leaves the policy unchanged and returns the ASK it needs."""
        outcome = self.engine.narrow(self.policy, candidate)
        if outcome.verdict is Verdict.ALLOW:
            self.policy = outcome.policy
        return outcome


class SecuredExecutor:
    """Wraps the episode's executor: every call is authorized before the injector or handler sees it.

    The loop passes the ``Authorization`` it already made (decisions for one
    message are made in call order before anything runs). Policy-layer calls
    from recovery come without one and get a fresh decision here.
    """

    def __init__(self, inner: SequentialExecutor, gate: SecurityGate) -> None:
        self.inner = inner
        self.gate = gate

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
    ) -> ToolResult:
        if authorization is None:
            authorization = self.gate.authorize(call, registry, via=via)
        elif not authorization.covers(call):
            raise RuntimeError(f"an authorization for {authorization.tool} was presented for {call.name}")
        if not authorization.allowed:
            return self.gate.refusal(call, authorization)
        options: dict[str, Any] = {"via": via}
        if injector is not None:
            options["injector"] = injector
        if idempotency_key is not None:
            options["idempotency_key"] = idempotency_key
        if call_index is not None:
            options["call_index"] = call_index
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
    )


__all__ = [
    "CONFIG_VERSION",
    "SECURITY_ERRORS",
    "STORE_TOOLS",
    "Authorization",
    "SecuredExecutor",
    "SecurityConfig",
    "SecurityGate",
    "approval_requests",
    "decode_for_policy",
    "grant_summary",
    "resolve_approvals",
    "run_approved_episode",
    "store_task_policy",
]

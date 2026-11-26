"""Bulkhead part 2 in Steadyhand: one security path for model calls, recovery, faults, and approvals."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import steadyhand.loop as loop_module
from bulkhead.approvals import ApprovalMode, DenyAll, ScriptedApproval, ScriptedApprover
from bulkhead.errors import GrantError
from bulkhead.manifest import Bounds, Egress, Sandbox, SecuritySpec, Sink
from bulkhead.policy import ToolRule
from bulkhead.types import SandboxTier
from steadyhand import (
    Effect,
    EpisodeJournal,
    EpisodeStatus,
    SequentialExecutor,
    SteadyhandClient,
    SteadyhandRegistry,
    ToolCall,
    ToolPolicy,
    build_registry,
    initial_messages,
    load_fixture,
    run_episode,
    strategy_for,
)
from steadyhand.faults import FaultInjector, FaultMode, FaultSpec, build_schedule
from steadyhand.journal import ReplayMismatch, check_record, manifest_path_for
from steadyhand.manifest import CONTEXT_FIELDS, SECURITY_FIELDS, EpisodeManifest, build_manifest, load_manifest, require_match
from steadyhand.replay import replay_episode, run_claimed_episode
from steadyhand.security import (
    SecuredExecutor,
    SecurityConfig,
    SecurityGate,
    approval_requests,
    resolve_approvals,
    run_approved_episode,
    store_task_policy,
)
from steadyhand.store import Store
from steadyhand.tools import TOOL_SECURITY
from steadyhand.tools_exec import ExecDisabledError, run_python, run_shell
from toolsmith.grammar import GBNFCompiler
from toolsmith.json_utils import strict_json_loads
from toolsmith.protocols import QWEN3_PROFILE, QWEN35_PROFILE

ROOT = Path(__file__).parent.parent
GOLDEN = ROOT / "tests" / "fixtures" / "steadyhand_episodes"
REFUND = {"order_id": "ord_1001", "amount_inr": 120, "reason": "Dented lid"}
GOAL = "Refund 120 INR on ord_1001 for the dented lid."
NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
SMALL_REFUNDS = ToolRule("issue_refund", bounds={"amount_inr": Bounds(1, 100)})


class FakeService:
    profile = QWEN3_PROFILE
    use_grammars = True

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def completion(content=None, calls=()):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            for call_id, name, arguments in calls
        ]
    return {
        "choices": [{"index": 0, "message": message}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 6},
        "toolsmith": {"repair_attempts": 0},
    }


def pins(*specs, seed=0):
    return build_schedule(seed, [FaultSpec(*spec) for spec in specs])


def config(**policy):
    return SecurityConfig(task_policy=store_task_policy(**policy))


def episode(tmp_path, *turns, security=None, strategy="typed_hint", faults=None, registry=None, store=None, goal=GOAL):
    store = store if store is not None else load_fixture().clone()
    service = FakeService(*turns)
    result = run_episode(
        SteadyhandClient(service),
        registry or build_registry(),
        store,
        initial_messages(goal),
        journal=EpisodeJournal.create(tmp_path),
        faults=faults,
        recovery=strategy_for(strategy),
        sleep=lambda seconds: None,
        security=security,
    )
    return result, service, store


def records(path, event=None):
    lines = [strict_json_loads(line) for line in Path(path).read_text().splitlines()]
    return [line for line in lines if event is None or line["event"] == event]


def seen(service, request=1, message=-1):
    return json.loads(service.requests[request].messages[message]["content"])


@pytest.fixture
def execution_log(monkeypatch):
    """Every policy decision and every real handler run, in the order they happen.

    Handlers only ever run inside ``SequentialExecutor._call_with_timeout``, so
    logging there sees first attempts, retries, verify probes, switches, and
    implicit-fault runs on a store clone alike, whether or not they are journaled.
    """
    log = []
    original_append = EpisodeJournal.append
    original_call = SequentialExecutor._call_with_timeout

    def append(self, event, **fields):
        if event == "policy_decision":
            log.append(("decision", fields["tool"], fields["verdict"]))
        return original_append(self, event, **fields)

    def call(handler, store, arguments, timeout_s):
        log.append(("run", handler.__name__, None))
        return original_call(handler, store, arguments, timeout_s)

    monkeypatch.setattr(EpisodeJournal, "append", append)
    monkeypatch.setattr(SequentialExecutor, "_call_with_timeout", staticmethod(call))
    return log


def assert_runs_follow_allows(log):
    """Each handler run consumes a fresh ALLOW for its tool; no run happens without one."""
    pending: dict[str, str] = {}
    for kind, tool, verdict in log:
        if kind == "decision":
            pending[tool] = verdict
        else:
            assert pending.pop(tool, None) == "allow", f"{tool} ran without a current ALLOW: {log}"
    log.clear()


# Disabled mode changes nothing


def test_without_bulkhead_journals_and_manifests_are_unchanged(tmp_path):
    result, _, _ = episode(tmp_path, completion(calls=[("c1", "get_order", {"order_id": "ord_1001"})]), completion("Delivered."))
    lines = records(result.journal_path)

    assert not [line for line in lines if line["event"] in ("policy_decision", "approval_requested")]
    assert "bulkhead" not in lines[0]
    legacy = build_manifest(
        SteadyhandClient(FakeService()).settings(), build_registry(), load_fixture().clone(), initial_messages(GOAL),
        budgets={"max_steps": 8}, guard_limit=3,
    )
    # Bulkhead and Tidemark fields are both left out of a manifest that uses neither.
    assert set(legacy.as_dict()) == {field.name for field in fields(EpisodeManifest)} - set(SECURITY_FIELDS) - set(CONTEXT_FIELDS)
    for manifest_file in sorted(GOLDEN.glob("*.manifest.json")):
        manifest = load_manifest(manifest_file)  # golden manifests predate Bulkhead and still load
        assert not manifest.secured and not set(SECURITY_FIELDS) & set(manifest.as_dict())


def test_golden_episodes_still_replay_with_bulkhead_off():
    for path in sorted(GOLDEN.glob("ep_*.jsonl")):
        replay_episode(path)


# Allowed, denied, and asked calls in an episode


def test_allowed_calls_behave_exactly_as_without_bulkhead(tmp_path, execution_log):
    turns = (completion(calls=[("c1", "issue_refund", REFUND)]), completion("Refunded."))
    plain, plain_service, plain_store = episode(tmp_path / "plain", *turns)
    execution_log.clear()  # the unsecured run makes no decisions
    secured, secured_service, secured_store = episode(tmp_path / "secured", *turns, security=config())
    assert_runs_follow_allows(execution_log)

    assert secured.status is plain.status is EpisodeStatus.DONE
    assert seen(secured_service) == seen(plain_service)
    assert secured_store.snapshot_sha256() == plain_store.snapshot_sha256()
    lines = records(secured.journal_path)
    events = [line["event"] for line in lines]
    assert events.index("policy_decision") < events.index("tool_result")
    [decision] = records(secured.journal_path, "policy_decision")
    assert (decision["verdict"], decision["rule_id"], decision["via"], decision["tool_call_id"]) == ("allow", "task.allow", "model", "c1")
    assert decision["taint_before"] == "trusted" and decision["grant_id"] is None
    assert lines[0]["bulkhead"]["approval_mode"] == "deny_all"


def test_a_denied_call_gets_a_typed_observation_and_the_episode_continues(tmp_path):
    result, service, store = episode(
        tmp_path,
        completion(calls=[("c1", "issue_refund", REFUND)]),
        completion("I am not allowed to refund; I opened a ticket instead."),
        security=config(deny_tools=("issue_refund",), tools=("get_order", "create_ticket")),
    )

    observation = seen(service)
    assert observation["error_type"] == "permission_denied" and observation["retryable"] is False
    assert observation["details"] == [{"field": "rule_id", "problem": "tool_denied", "got": "task.deny.tool"}]
    assert "task.deny.tool" in observation["hint"]
    assert result.status is EpisodeStatus.DONE
    [tool_result] = records(result.journal_path, "tool_result")
    assert tool_result["handler_called"] is False
    assert store.snapshot_sha256() == store.fixture_sha256


def test_an_ask_escalates_with_its_rule_id_and_changes_nothing(tmp_path):
    result, service, store = episode(
        tmp_path,
        completion(calls=[("c1", "issue_refund", REFUND)]),
        security=config(rules=(SMALL_REFUNDS,)),
        faults=pins(("issue_refund", 0, FaultMode.LOST_ACKNOWLEDGMENT)),
    )

    assert result.status is EpisodeStatus.ESCALATED and len(service.requests) == 1
    assert store.snapshot_sha256() == store.fixture_sha256
    lines = records(result.journal_path)
    assert [line["event"] for line in lines[-5:]] == ["policy_decision", "approval_requested", "tool_result", "escalated", "episode_finished"]
    _, request, tool_result, escalated, _ = lines[-5:]
    assert request["rule_id"] == "task.widen.bound" and request["approval_mode"] == "deny_all"
    assert request["normalized_grant"]["widen"] == {"bounds": {"issue_refund": {"amount_inr": {"minimum": 1, "maximum": 120}}}}
    assert tool_result["handler_called"] is False and tool_result["error"]["error_type"] == "permission_required"
    assert "task.widen.bound" in escalated["reason"] and "task.widen.bound" in result.final_content
    assert "issue_refund (permission_required x1)" in result.final_content
    assert not records(result.journal_path, "injected_fault")  # the pinned fault never saw the call


def test_a_model_resend_of_a_denied_call_gets_a_fresh_decision(tmp_path):
    result, _, store = episode(
        tmp_path,
        completion(calls=[("c1", "issue_refund", REFUND)]),
        completion(calls=[("c2", "issue_refund", REFUND)]),
        completion("Not allowed."),
        security=config(deny_tools=("issue_refund",), tools=("get_order",)),
    )

    decisions = records(result.journal_path, "policy_decision")
    assert [(d["tool_call_id"], d["verdict"]) for d in decisions] == [("c1", "deny"), ("c2", "deny")]
    assert store.refunds_for("ord_1001") == []


# Recovery and faults go through the same gate


def test_a_recovery_retry_gets_its_own_decision(tmp_path, execution_log):
    result, service, _ = episode(
        tmp_path,
        completion(calls=[("c1", "get_order", {"order_id": "ord_1001"})]),
        completion("Delivered."),
        security=config(),
        strategy="retry_safe",
        faults=pins(("get_order", 0, FaultMode.EXPLICIT_TRANSIENT, "timeout")),
    )

    decisions = records(result.journal_path, "policy_decision")
    assert [(d["via"], d["verdict"]) for d in decisions] == [("model", "allow"), ("policy", "allow")]
    assert seen(service)["ok"] is True
    assert [kind for kind, _, _ in execution_log] == ["decision", "decision", "run"]  # the faulted first try never ran
    assert_runs_follow_allows(execution_log)


def test_a_verify_probe_is_authorized_and_cannot_run_when_not_permitted(tmp_path, execution_log):
    tools = tuple(name for name in store_task_policy().tools if name != "list_payments")
    result, _, store = episode(
        tmp_path,
        completion(calls=[("c1", "issue_refund", REFUND)]),
        security=config(tools=tools),
        strategy="idempotent_verify",
        faults=pins(("issue_refund", 0, FaultMode.LOST_ACKNOWLEDGMENT)),
    )

    decisions = records(result.journal_path, "policy_decision")
    assert [(d["tool"], d["via"], d["verdict"], d["rule_id"]) for d in decisions] == [
        ("issue_refund", "model", "allow", "task.allow"),
        ("list_payments", "policy", "ask", "task.widen.tool"),
    ]
    [attempt] = records(result.journal_path, "policy_attempt")
    assert (attempt["purpose"], attempt["handler_called"], attempt["error_type"]) == ("verify", False, "permission_required")
    assert result.status is EpisodeStatus.ESCALATED
    assert len(store.refunds_for("ord_1001")) == 1  # the lost-ack refund, never a second one
    assert ("run", "list_payments", None) not in execution_log
    assert_runs_follow_allows(execution_log)


def test_switch_cannot_route_around_the_policy_to_an_alternative(tmp_path, execution_log):
    faults = pins(("issue_refund", 0, FaultMode.EXPLICIT_PERMANENT))
    blocked, _, blocked_store = episode(
        tmp_path / "blocked", completion(calls=[("c1", "issue_refund", REFUND)]),
        security=config(), strategy="switch", faults=faults,
    )
    assert not [entry for entry in execution_log if entry[0] == "run"]
    assert_runs_follow_allows(execution_log)
    allowed, allowed_service, allowed_store = episode(
        tmp_path / "allowed", completion(calls=[("c1", "issue_refund", REFUND)]), completion("Refunded via ledger."),
        security=config(tools=(*store_task_policy().tools, "refund_via_ledger")), strategy="switch", faults=faults,
    )

    switch_decision = records(blocked.journal_path, "policy_decision")[-1]
    assert (switch_decision["tool"], switch_decision["via"], switch_decision["verdict"]) == ("refund_via_ledger", "policy", "ask")
    assert blocked.status is EpisodeStatus.ESCALATED and blocked_store.ledger_for("ord_1001") == []
    assert allowed.status is EpisodeStatus.DONE and len(allowed_store.ledger_for("ord_1001")) == 1
    assert seen(allowed_service)["ok"] is True
    assert ("run", "refund_via_ledger", None) in execution_log
    assert_runs_follow_allows(execution_log)


def test_a_denied_model_call_never_reaches_recovery(tmp_path):
    result, _, store = episode(
        tmp_path, completion(calls=[("c1", "issue_refund", REFUND)]), completion("Not allowed."),
        security=config(deny_tools=("issue_refund",), tools=("get_order", "refund_via_ledger")),
        strategy="escalate", faults=pins(("issue_refund", 0, FaultMode.EXPLICIT_PERMANENT)),
    )

    assert records(result.journal_path, "policy_attempt") == [] and records(result.journal_path, "recovery") == []
    assert store.ledger_for("ord_1001") == [] and result.status is EpisodeStatus.DONE


def test_fault_injection_cannot_see_or_index_a_refused_call():
    registry = build_registry()
    store = load_fixture().clone()
    call = ToolCall("c1", "issue_refund", json.dumps(REFUND))
    injector = FaultInjector(pins(("issue_refund", 0, FaultMode.EXPLICIT_TRANSIENT, "timeout")))
    denied_gate = SecurityGate(
        config(deny_tools=("issue_refund",), tools=("get_order",)), registry, journal=RecordingJournal(), episode_id="ep_test"
    )
    denied = SecuredExecutor(SequentialExecutor(), denied_gate)
    allowed = SecuredExecutor(SequentialExecutor(), SecurityGate(config(), registry, journal=RecordingJournal(), episode_id="ep_test"))

    refused = denied.run(call, registry, store, injector=injector)
    assert refused.error.error_type.value == "permission_denied" and refused.fault is None
    assert injector.next_index("issue_refund") == 0  # no index consumed
    assert [name for name, _ in denied_gate.journal.records] == ["policy_decision"]
    faulted = allowed.run(call, registry, store, injector=injector)
    assert faulted.fault is not None and faulted.fault.call_index == 0


class RecordingJournal:
    """Stands in for EpisodeJournal in gate-level tests: records each event method call."""

    def __init__(self):
        self.records = []

    def __getattr__(self, name):
        return lambda **fields: self.records.append((name, fields))


def test_an_authorization_cannot_be_reused_for_another_call():
    registry = build_registry()
    gate = SecurityGate(config(), registry, journal=RecordingJournal(), episode_id="ep_test")
    executor = SecuredExecutor(SequentialExecutor(), gate)
    authorization = gate.authorize(ToolCall("c1", "get_order", '{"order_id": "ord_1001"}'), registry)

    with pytest.raises(RuntimeError, match="presented"):
        executor.run(ToolCall("c1", "issue_refund", json.dumps(REFUND)), registry, load_fixture().clone(), authorization=authorization)
    with pytest.raises(RuntimeError):
        executor.run(ToolCall("c1", "get_order", '{"order_id": "ord_1002"}'), registry, load_fixture().clone(), authorization=authorization)


# Parallel reads and sinks


def test_parallel_reads_are_decided_and_journaled_in_call_order(tmp_path):
    calls = [("c1", "get_order", {"order_id": "ord_1001"}), ("c2", "get_customer", {"customer_id": "cus_101"}),
             ("c3", "list_payments", {"order_id": "ord_1001"})]
    result, _, _ = episode(tmp_path, completion(calls=calls), completion("Done."), security=config())

    lines = records(result.journal_path)
    decisions = [line for line in lines if line["event"] == "policy_decision"]
    results = [line for line in lines if line["event"] == "tool_result"]
    assert [d["tool_call_id"] for d in decisions] == ["c1", "c2", "c3"] == [r["tool_call_id"] for r in results]
    assert max(d["seq"] for d in decisions) < min(r["seq"] for r in results)  # every decision before any result


def test_an_ask_inside_a_parallel_batch_stops_planning_at_that_call(tmp_path):
    calls = [("c1", "get_order", {"order_id": "ord_1001"}), ("c2", "get_customer", {"customer_id": "cus_101"}),
             ("c3", "list_payments", {"order_id": "ord_1001"})]
    result, _, _ = episode(tmp_path, completion(calls=calls), security=config(ask_tools=("get_customer",)))

    assert [d["tool_call_id"] for d in records(result.journal_path, "policy_decision")] == ["c1", "c2"]
    assert [(r["tool_call_id"], r["handler_called"]) for r in records(result.journal_path, "tool_result")] == [("c1", True), ("c2", False)]
    assert result.status is EpisodeStatus.ESCALATED


def fetch_page(store: Store, url: str) -> dict:
    """Fetch a page (a stand-in network read)."""
    raise AssertionError("never run: no sandbox backend")


def test_calls_with_a_sink_or_egress_are_never_batched(tmp_path, monkeypatch):
    registry = SteadyhandRegistry()
    for handler, policy, semantic in __import__("steadyhand.tools", fromlist=["TOOL_SPECS"]).TOOL_SPECS:
        registry.register(handler, policy, semantic, security=TOOL_SECURITY[handler.__name__])
    network = SecuritySpec(sandbox=Sandbox(net=Egress(allow=("pypi.org:443",))), sinks=(Sink.NETWORK,))
    registry.register(fetch_page, ToolPolicy(Effect.READ, 5.0, idempotent=True), security=network)
    registry.freeze()

    def no_pool(*args, **kwargs):
        raise AssertionError("a call with egress was batched")

    monkeypatch.setattr(loop_module, "ThreadPoolExecutor", no_pool)
    calls = [("c1", "get_order", {"order_id": "ord_1001"}), ("c2", "fetch_page", {"url": "https://pypi.org/"})]
    result, _, _ = episode(tmp_path, completion(calls=calls), completion("Done."), security=config(tools=("get_order", "fetch_page")), registry=registry)

    decisions = records(result.journal_path, "policy_decision")
    assert [(d["tool"], d["verdict"], d["rule_id"]) for d in decisions] == [
        ("get_order", "allow", "task.allow"), ("fetch_page", "deny", "bulkhead.tier.unavailable"),
    ]


# Exec tools: registered, compiled, and never run


def test_exec_tools_are_denied_and_never_run_in_process(tmp_path):
    registry = build_registry(exec_tools=True)
    calls = [("c1", "run_shell", {"command": "cat /etc/passwd"})]
    secured, secured_service, _ = episode(
        tmp_path / "secured", completion(calls=calls), completion("Cannot run commands."),
        security=config(tools=(*store_task_policy().tools, "run_shell")), registry=registry,
    )
    legacy, legacy_service, _ = episode(tmp_path / "legacy", completion(calls=calls), completion("Cannot."), registry=registry)

    assert seen(secured_service)["error_type"] == "permission_denied"
    [decision] = records(secured.journal_path, "policy_decision")
    assert decision["rule_id"] == "bulkhead.tier.unavailable"
    assert seen(legacy_service)["error_type"] == "sandbox_violation"  # the executor refuses exec even without Bulkhead
    for result in (secured, legacy):
        assert records(result.journal_path, "tool_result")[0]["handler_called"] is False
    with pytest.raises(ExecDisabledError):
        run_shell(load_fixture().clone(), "ls")
    with pytest.raises(ExecDisabledError):
        run_python(load_fixture().clone(), "print(1)")


@pytest.mark.parametrize("profile", [QWEN3_PROFILE, QWEN35_PROFILE])
def test_exec_tool_schemas_compile_through_both_qwen_grammar_profiles(profile):
    registry = build_registry(exec_tools=True)
    exec_tools = [registry.get("run_python"), registry.get("run_shell")]

    for choice in ("auto", "required", {"type": "function", "function": {"name": "run_shell"}}):
        grammar = GBNFCompiler(protocol=profile.protocol).compile_tools(exec_tools, choice)
        assert grammar is not None and "run_shell" in grammar.text
    assert GBNFCompiler(protocol=profile.protocol).compile_tools(list(registry), "auto") is not None
    assert {"code"} == set(registry.get("run_python").parameters["properties"])
    assert {"command"} == set(registry.get("run_shell").parameters["properties"])


def test_recovery_never_retries_an_exec_tool():
    from steadyhand.recovery import RecoveryBudget, RecoveryContext, RetrySafe, _safe_retry, _unsafe

    registry = build_registry(exec_tools=True)
    context = RecoveryContext(
        call=ToolCall("c", "run_shell", '{"command": "ls"}'), arguments={"command": "ls"}, policy=registry.policy("run_shell"),
        registry=registry, store=load_fixture().clone(), executor=SequentialExecutor(), injector=None,
        journal=None, step=0, episode_id="ep", sleep=lambda s: None, budget=RecoveryBudget(),
    )

    assert not _safe_retry(context, {"error_type": "timeout", "retryable": True}) and _unsafe(context)
    assert RetrySafe().on_observation(context, {"error_type": "timeout"}).value == "return_to_model"


# Identity, manifests, and replay


def test_security_policy_is_part_of_identity_and_resume_refuses_a_change():
    registry, store = build_registry(), load_fixture().clone()
    messages = initial_messages(GOAL)
    settings = SteadyhandClient(FakeService()).settings()
    options = {"budgets": {"max_steps": 8}, "guard_limit": 3}
    legacy = build_manifest(settings, registry, store, messages, **options)
    secured = build_manifest(settings, registry, store, messages, security=config(), **options)
    other = build_manifest(settings, registry, store, messages, security=config(rules=(SMALL_REFUNDS,)), **options)
    exec_registry = build_manifest(settings, build_registry(exec_tools=True), store, messages, security=config(), **options)

    assert len({legacy.episode_id, secured.episode_id, other.episode_id, exec_registry.episode_id}) == 4
    assert secured.secured and not legacy.secured
    assert secured.security_config == config().as_dict() and secured.bulkhead_version
    for recorded, current in ((secured, other), (legacy, secured), (secured, exec_registry)):
        with pytest.raises(ReplayMismatch):
            require_match(recorded, current)
    with pytest.raises(ValueError):
        build_manifest(settings, registry, store, messages, parent_episode_id="ep_x", **options)


def run_claimed(directory, *turns, security, store=None, run_label=None):
    service = FakeService(*turns)
    result = run_claimed_episode(
        directory, SteadyhandClient(service), build_registry(), store or load_fixture().clone(), initial_messages(GOAL),
        security=security, run_label=run_label, sleep=lambda seconds: None,
    )
    return result, service


def test_a_secured_episode_replays_exactly_and_a_tampered_config_is_refused(tmp_path):
    result, _ = run_claimed(tmp_path, completion(calls=[("c1", "issue_refund", REFUND)]), security=config(rules=(SMALL_REFUNDS,)))
    path = result.journal_path

    replayed = replay_episode(path)
    assert replayed.status is EpisodeStatus.ESCALATED
    manifest_file = manifest_path_for(path)
    document = json.loads(manifest_file.read_text())
    document["security_config"]["task_policy"]["rules"] = []
    manifest_file.write_text(json.dumps(document))
    with pytest.raises(ReplayMismatch, match="security config"):
        replay_episode(path)


# Approvals: a new linked episode, never a rewrite


def test_an_approved_retry_is_a_new_linked_episode_and_the_parent_never_changes(tmp_path):
    security = config(rules=(SMALL_REFUNDS,))
    parent, _ = run_claimed(tmp_path, completion(calls=[("c1", "issue_refund", REFUND)]), security=security)
    parent_journal = Path(parent.journal_path)
    before = (parent_journal.read_bytes(), manifest_path_for(parent_journal).read_bytes())

    [request] = approval_requests(parent_journal)
    assert (request.tool, request.rule_id, request.episode_id) == ("issue_refund", "task.widen.bound", parent.episode_id)
    assert resolve_approvals(parent_journal, DenyAll(), now=NOW) == []  # a decline creates nothing
    [grant] = resolve_approvals(parent_journal, ScriptedApprover((ScriptedApproval("issue_refund", REFUND),)), now=NOW)

    store = load_fixture().clone()
    child = run_approved_episode(
        tmp_path, parent_journal, grant, SteadyhandClient(FakeService(completion(calls=[("c1", "issue_refund", REFUND)]), completion("Refunded."))),
        build_registry(), store, now=NOW + timedelta(minutes=1), sleep=lambda seconds: None,
    )

    assert child.status is EpisodeStatus.DONE and child.episode_id != parent.episode_id
    assert len(store.refunds_for("ord_1001")) == 1
    manifest = load_manifest(manifest_path_for(child.journal_path))
    assert manifest.parent_episode_id == parent.episode_id and manifest.run_label == f"approved-{grant.approval_id}"
    assert [item["approval_id"] for item in manifest.security_config["grants"]] == [grant.approval_id]
    assert manifest.task_policy_sha256 != load_manifest(manifest_path_for(parent_journal)).task_policy_sha256  # amended
    [decision] = records(child.journal_path, "policy_decision")
    assert (decision["verdict"], decision["rule_id"], decision["grant_id"]) == ("allow", "approval.grant", grant.approval_id)
    assert (parent_journal.read_bytes(), manifest_path_for(parent_journal).read_bytes()) == before
    # The approval cannot be run twice, and both episodes replay exactly.
    again = run_approved_episode(
        tmp_path, parent_journal, grant, SteadyhandClient(FakeService()), build_registry(), load_fixture().clone(), now=NOW,
    )
    assert again is None
    assert replay_episode(child.journal_path).status is EpisodeStatus.DONE
    assert replay_episode(parent_journal).status is EpisodeStatus.ESCALATED


def test_a_grant_is_single_use_inside_the_approved_episode(tmp_path):
    parent, _ = run_claimed(tmp_path, completion(calls=[("c1", "issue_refund", REFUND)]), security=config(rules=(SMALL_REFUNDS,)))
    [grant] = resolve_approvals(parent.journal_path, ScriptedApprover((ScriptedApproval("issue_refund", REFUND),)), now=NOW)
    store = load_fixture().clone()

    child = run_approved_episode(
        tmp_path, parent.journal_path, grant,
        SteadyhandClient(FakeService(
            completion(calls=[("c1", "issue_refund", REFUND)]),
            completion(calls=[("c2", "issue_refund", REFUND)]),
        )),
        build_registry(), store, now=NOW, sleep=lambda seconds: None,
    )

    verdicts = [(d["tool_call_id"], d["verdict"], d["rule_id"]) for d in records(child.journal_path, "policy_decision")]
    assert verdicts == [("c1", "allow", "approval.grant"), ("c2", "ask", "task.widen.bound")]
    assert child.status is EpisodeStatus.ESCALATED and len(store.refunds_for("ord_1001")) == 1


def test_a_changed_call_in_the_approved_episode_is_not_covered(tmp_path):
    parent, _ = run_claimed(tmp_path, completion(calls=[("c1", "issue_refund", REFUND)]), security=config(rules=(SMALL_REFUNDS,)))
    [grant] = resolve_approvals(parent.journal_path, ScriptedApprover((ScriptedApproval("issue_refund", REFUND),)), now=NOW)
    store = load_fixture().clone()

    child = run_approved_episode(
        tmp_path, parent.journal_path, grant,
        SteadyhandClient(FakeService(completion(calls=[("c1", "issue_refund", {**REFUND, "amount_inr": 150})]))),
        build_registry(), store, now=NOW, sleep=lambda seconds: None,
    )

    assert child.status is EpisodeStatus.ESCALATED and store.refunds_for("ord_1001") == []


def test_approved_retries_refuse_bad_grants_and_parents(tmp_path):
    security = config(rules=(SMALL_REFUNDS,))
    parent, _ = run_claimed(tmp_path / "a", completion(calls=[("c1", "issue_refund", REFUND)]), security=security)
    finished, _ = run_claimed(tmp_path / "b", completion("Nothing to do."), security=security)
    [grant] = resolve_approvals(parent.journal_path, ScriptedApprover((ScriptedApproval("issue_refund", REFUND),)), now=NOW)
    client = SteadyhandClient(FakeService())

    def attempt(journal=parent.journal_path, grant=grant, now=NOW, store=None):
        run_approved_episode(tmp_path / "children", journal, grant, client, build_registry(), store or load_fixture().clone(), now=now)

    with pytest.raises(GrantError, match="expired"):
        attempt(now=NOW + timedelta(hours=2))
    with pytest.raises(GrantError, match="escalated"):
        attempt(journal=finished.journal_path)
    with pytest.raises(GrantError, match="another episode"):
        attempt(grant=replace(grant, episode_id="ep_0000000000000000"))
    with pytest.raises(GrantError, match="answers no approval"):
        attempt(grant=replace(grant, policy_sha256="0" * 64))
    tampered = load_fixture().clone()
    tampered.add_note("ord_1001", "already touched")
    with pytest.raises(GrantError, match="unmodified fixture"):
        attempt(store=tampered)
    assert not (tmp_path / "children").exists()


def test_grants_are_honored_only_in_an_episode_that_names_its_parent(tmp_path):
    parent, _ = run_claimed(tmp_path, completion(calls=[("c1", "issue_refund", REFUND)]), security=config(rules=(SMALL_REFUNDS,)))
    [grant] = resolve_approvals(parent.journal_path, ScriptedApprover((ScriptedApproval("issue_refund", REFUND),)), now=NOW)

    with pytest.raises(GrantError, match="names its parent"):
        episode(tmp_path / "x", completion(calls=[("c1", "issue_refund", REFUND)]), security=config(rules=(SMALL_REFUNDS,)).with_grants((grant,)))


# Journal schema


def test_new_journal_events_are_strictly_checked():
    decision = {
        "event": "policy_decision", "episode_id": "ep_x", "seq": 1, "recorded_at": "t", "step": 0, "tool_call_id": "c1",
        "via": "model", "tool": "get_order", "arguments_sha256": "0" * 64, "verdict": "allow", "rule_id": "task.allow",
        "reason_code": "within_task_policy", "taint_before": "trusted", "policy_sha256": "0" * 64, "grant_id": None,
        "normalized_request_sha256": "0" * 64,
    }
    check_record(decision, 1)
    with pytest.raises(ReplayMismatch):
        check_record({**decision, "arguments": "{}"}, 1)
    with pytest.raises(ReplayMismatch):
        check_record({key: value for key, value in decision.items() if key != "rule_id"}, 1)


def test_security_config_round_trips_and_refuses_t0_backends():
    full = SecurityConfig(store_task_policy(rules=(SMALL_REFUNDS,)), ApprovalMode.SCRIPTED, (), (SandboxTier.T2, SandboxTier.T1))

    assert SecurityConfig.from_dict(full.as_dict()) == full and full.available_tiers == (SandboxTier.T1, SandboxTier.T2)
    with pytest.raises(Exception, match="T0"):
        SecurityConfig(store_task_policy(), available_tiers=(SandboxTier.T0,))
    with pytest.raises(Exception):
        SecurityConfig.from_dict({**full.as_dict(), "auto_approve": True})

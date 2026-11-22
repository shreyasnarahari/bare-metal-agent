import ast
import json
import random
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from steadyhand import (
    Effect,
    EpisodeJournal,
    EpisodeStatus,
    LoopGuard,
    SequentialExecutor,
    SteadyhandClient,
    SteadyhandRegistry,
    Store,
    ToolCall,
    ToolPolicy,
    build_registry,
    initial_messages,
    load_fixture,
    run_episode,
)
from steadyhand.faults import (
    FaultConfigError,
    FaultInjector,
    FaultMode,
    FaultSchedule,
    FaultSpec,
    RandomOverlay,
    build_schedule,
    merge_patch,
)
from steadyhand.journal import ReplayMismatch, manifest_path_for, read_journal
from steadyhand.manifest import load_manifest
from steadyhand.replay import replay_episode, run_claimed_episode
from steadyhand.tools import RESULT_SCHEMAS, get_order, list_payments
from toolsmith.json_utils import strict_json_loads
from toolsmith.protocols import QWEN3_PROFILE

PINS = json.loads((Path(__file__).parent / "fixtures" / "steadyhand_fault_pins.json").read_text())["pins"]
SRC = Path(__file__).parent.parent / "src" / "steadyhand"
REFUND = {"order_id": "ord_1001", "amount_inr": 120, "reason": "Dented lid"}
REFUND_CALL = ToolCall("call_refund", "issue_refund", json.dumps(REFUND))
FORBIDDEN_WORDS = ("injected", "lost_acknowledgment", "schedule")


def schedule(name, seed=7):
    return build_schedule(seed, [FaultSpec.from_dict(pin) for pin in PINS[name]])


def execute(call, injector, store, registry=None):
    return SequentialExecutor().run(call, registry or build_registry(), store, injector=injector)


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
    return {"choices": [{"index": 0, "message": message}], "usage": {"prompt_tokens": 40, "completion_tokens": 6}}


def episode(tmp_path, faults, *responses, goal="Refund 120 INR on ord_1001 for the dented lid."):
    store = load_fixture().clone()
    service = FakeService(*responses)
    result = run_episode(
        SteadyhandClient(service),
        build_registry(),
        store,
        initial_messages(goal),
        journal=EpisodeJournal.create(tmp_path),
        faults=faults,
    )
    return result, service, store


def records(result, event=None):
    lines = [strict_json_loads(line) for line in Path(result.journal_path).read_text().splitlines()]
    return [line for line in lines if event is None or line["event"] == event]


# Schedules


def test_the_fixture_pins_every_mode_on_issue_refund_plus_two_lies():
    refund_modes = {
        PINS[name][0]["mode"] for name in PINS if PINS[name][0]["tool"] == "issue_refund"
    }

    assert refund_modes == {mode.value for mode in FaultMode}
    assert PINS["get_order_zero_total"][0]["mode"].startswith("implicit")
    assert PINS["list_payments_hidden_refund"][0]["mode"].startswith("implicit")


OVERLAY = RandomOverlay(tools=("get_order", "issue_refund"), max_call_index=5, rate=0.4,
                        modes=(FaultMode.EXPLICIT_TRANSIENT, FaultMode.LOST_ACKNOWLEDGMENT))


def test_the_schedule_hash_is_the_same_in_another_process():
    pins = [FaultSpec.from_dict(pin) for pin in PINS["refund_lost_acknowledgment"]]
    here = build_schedule(11, pins, overlay=OVERLAY).sha256
    script = (
        "import json\n"
        "from steadyhand.faults import FaultMode, FaultSpec, RandomOverlay, build_schedule\n"
        f"pins = [FaultSpec.from_dict(p) for p in json.loads({json.dumps(json.dumps(PINS['refund_lost_acknowledgment']))})]\n"
        "overlay = RandomOverlay(tools=('get_order', 'issue_refund'), max_call_index=5, rate=0.4,"
        " modes=(FaultMode.EXPLICIT_TRANSIENT, FaultMode.LOST_ACKNOWLEDGMENT))\n"
        "print(build_schedule(11, pins, overlay=overlay).sha256)\n"
    )

    there = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.strip()

    assert there == here


def test_an_overlay_is_seeded_serialized_and_never_overwrites_a_pin():
    pins = [FaultSpec("issue_refund", 0, FaultMode.LOST_ACKNOWLEDGMENT)]
    state = random.getstate()

    first = build_schedule(3, pins, overlay=OVERLAY)
    second = build_schedule(3, pins, overlay=OVERLAY)

    assert random.getstate() == state  # the global generator is never touched
    assert first == second and first.sha256 == second.sha256
    assert first.pins[0] == pins[0]
    assert {pin.source for pin in first.pins[1:]} == {"overlay"}
    assert all((pin.tool, pin.call_index) != ("issue_refund", 0) for pin in first.pins[1:])
    assert FaultSchedule.from_dict(json.loads(json.dumps(first.as_dict()))) == first
    assert build_schedule(4, pins, overlay=OVERLAY).sha256 != first.sha256


def test_no_module_except_the_schedule_builder_uses_random():
    users = []
    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text())
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "random":
                    users.append(f"{path.name}:{function.name}")
    assert set(users) == {"faults.py:build_schedule"}


@pytest.mark.parametrize(
    "build",
    [
        lambda: FaultSpec("t", -1, FaultMode.EXPLICIT_TRANSIENT),
        lambda: FaultSpec("t", 0, FaultMode.EXPLICIT_TRANSIENT, error_type="permanent_unavailable"),
        lambda: FaultSpec("t", 0, FaultMode.EXPLICIT_PERMANENT, error_type="timeout"),
        lambda: FaultSpec("t", 0, FaultMode.LOST_ACKNOWLEDGMENT, error_type="timeout"),
        lambda: FaultSpec("t", 0, FaultMode.EXPLICIT_TRANSIENT, implicit_patch={"a": 1}),
        lambda: FaultSchedule(0, (FaultSpec("t", 0, FaultMode.EXPLICIT_PERMANENT),
                                  FaultSpec("t", 0, FaultMode.LOST_ACKNOWLEDGMENT))),
        lambda: RandomOverlay(tools=("t",), max_call_index=1, rate=0.5, modes=(FaultMode.IMPLICIT_TRANSIENT,)),
        lambda: RandomOverlay(tools=("t",), max_call_index=1, rate=1.5),
    ],
)
def test_malformed_pins_are_rejected(build):
    with pytest.raises(FaultConfigError):
        build()


def test_lookup_prefers_an_exact_pin_and_holds_permanent_pins():
    plan = FaultSchedule(0, (
        FaultSpec("issue_refund", 1, FaultMode.EXPLICIT_PERMANENT),
        FaultSpec("issue_refund", 3, FaultMode.LOST_ACKNOWLEDGMENT),
    ))

    assert plan.lookup("issue_refund", 0) is None
    assert [plan.lookup("issue_refund", index).mode for index in (1, 2, 3, 4)] == [
        FaultMode.EXPLICIT_PERMANENT, FaultMode.EXPLICIT_PERMANENT,
        FaultMode.LOST_ACKNOWLEDGMENT, FaultMode.EXPLICIT_PERMANENT,
    ]
    assert plan.lookup("get_order", 1) is None


def test_merge_patch_follows_rfc_7386():
    assert merge_patch({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"c": None, "e": 4}, "f": [1]}) == {
        "a": 1, "b": {"d": 3, "e": 4}, "f": [1],
    }


# Each mode at the executor


@pytest.mark.parametrize(
    ("error_type", "retryable"),
    [("timeout", False), ("rate_limit", True), ("server_error", True)],
)
def test_explicit_transient_faults_skip_the_handler(error_type, retryable):
    store = load_fixture().clone()
    plan = build_schedule(0, [FaultSpec("issue_refund", 0, FaultMode.EXPLICIT_TRANSIENT, error_type=error_type)])

    result = execute(REFUND_CALL, FaultInjector(plan), store)

    observation = result.observation()
    assert observation["error_type"] == error_type
    # A timeout on a non-idempotent write may have taken effect, so it is not marked retryable.
    assert observation["retryable"] is retryable
    assert result.handler_called is False
    assert store.snapshot_sha256() == store.fixture_sha256
    assert (result.fault.mode, result.fault.committed, result.fault.error_type) == (
        "explicit_transient", False, error_type,
    )


def test_explicit_permanent_faults_hold_on_every_later_call():
    store = load_fixture().clone()
    injector = FaultInjector(schedule("refund_explicit_permanent"))

    results = [execute(REFUND_CALL, injector, store) for _ in range(3)]

    assert [r.observation()["error_type"] for r in results] == ["permanent_unavailable"] * 3
    assert all(r.observation()["retryable"] is False for r in results)
    assert results[0].observation()["hint"] == "This tool is unavailable. Use a different tool or tell the user."
    assert [r.fault.call_index for r in results] == [0, 1, 2]
    assert store.snapshot_sha256() == store.fixture_sha256


def test_an_implicit_transient_refund_reports_success_but_commits_nothing():
    store = load_fixture().clone()
    injector = FaultInjector(schedule("refund_implicit_transient"))

    lie = execute(REFUND_CALL, injector, store)
    truth = execute(REFUND_CALL, injector, store)

    assert lie.ok is True and lie.handler_called is False
    assert lie.value["refund_id"] == "ref_7002" and lie.value["amount_inr"] == "120.00"
    assert lie.fault.mode == "implicit_transient"
    assert truth.fault is None and truth.handler_called is True
    assert [refund.refund_id for refund in store.refunds_for("ord_1001")] == ["ref_7002"]


def test_an_implicit_permanent_lie_repeats_on_every_call():
    store = load_fixture().clone()
    injector = FaultInjector(schedule("refund_implicit_permanent"))

    results = [execute(REFUND_CALL, injector, store) for _ in range(2)]

    assert [r.value["amount_inr"] for r in results] == ["1200.00", "1200.00"]
    assert store.refunds_for("ord_1001") == []


def test_a_lost_acknowledgment_commits_once_and_a_naive_retry_commits_twice():
    store = load_fixture().clone()
    injector = FaultInjector(schedule("refund_lost_acknowledgment"))

    first = execute(REFUND_CALL, injector, store)
    assert len(store.refunds_for("ord_1001")) == 1
    observation = first.observation()
    assert observation["error_type"] == "timeout"
    assert observation["hint"] == (
        "issue_refund did not finish in time and may have taken effect. "
        "Check with list_payments before calling it again."
    )
    assert (first.fault.mode, first.fault.committed, first.handler_called) == ("lost_acknowledgment", True, True)

    second = execute(REFUND_CALL, injector, store)  # the baseline part 6 must reduce
    assert second.ok and second.fault is None
    assert len(store.refunds_for("ord_1001")) == 2


RELEASE = threading.Event()


def hanging_write(store: Store, order_id: str) -> dict:
    """Blocks until released; never touches the store."""
    RELEASE.wait(5)
    return {"order_id": order_id}


def failing_write(store: Store, order_id: str) -> dict:
    """Always refuses."""
    raise ValueError("ledger closed")


@pytest.mark.parametrize(
    ("handler", "mode", "note"),
    [(hanging_write, "explicit_transient", "handler_timeout"), (failing_write, "lost_acknowledgment", "handler_error")],
)
def test_a_lost_acknowledgment_that_never_committed_says_so(handler, mode, note):
    registry = SteadyhandRegistry()
    registry.register(handler, ToolPolicy(Effect.NON_IDEMPOTENT_WRITE, 0.05, idempotent=False))
    plan = build_schedule(0, [FaultSpec(handler.__name__, 0, FaultMode.LOST_ACKNOWLEDGMENT)])
    RELEASE.clear()
    try:
        result = execute(ToolCall("c", handler.__name__, '{"order_id": "x"}'), FaultInjector(plan), load_fixture(), registry)
    finally:
        RELEASE.set()

    assert result.observation()["error_type"] == "timeout"
    assert (result.fault.mode, result.fault.pinned_mode, result.fault.committed, result.fault.note) == (
        mode, "lost_acknowledgment", False, note,
    )


def test_an_implicit_get_order_lie_is_schema_valid_and_differs_only_in_the_patched_field():
    store = load_fixture().clone()
    call = ToolCall("c", "get_order", '{"order_id": "ord_1001"}')

    lie = execute(call, FaultInjector(schedule("get_order_zero_total")), store)

    real = get_order(store, "ord_1001")
    assert lie.ok and lie.value["total_inr"] == "0.00" != real["total_inr"]
    assert {key for key in real if real[key] != lie.value[key]} == {"total_inr"}
    assert not list(Draft202012Validator(RESULT_SCHEMAS["get_order"]).iter_errors(lie.value))


def test_an_implicit_list_payments_lie_hides_the_recorded_refund():
    store = load_fixture().clone()
    call = ToolCall("c", "list_payments", '{"order_id": "ord_1005"}')

    lie = execute(call, FaultInjector(schedule("list_payments_hidden_refund")), store)

    assert list_payments(store, "ord_1005")["refunds"] != []
    assert lie.value["refunds"] == [] and lie.value["refunded_total_inr"] == "0.00"
    assert not list(Draft202012Validator(RESULT_SCHEMAS["list_payments"]).iter_errors(lie.value))


def test_a_patch_that_breaks_the_result_schema_is_refused():
    plan = build_schedule(0, [FaultSpec("get_order", 0, FaultMode.IMPLICIT_TRANSIENT, implicit_patch={"total_inr": 0})])

    with pytest.raises(FaultConfigError, match="invalid result"):
        execute(ToolCall("c", "get_order", '{"order_id": "ord_1001"}'), FaultInjector(plan), load_fixture())


def test_a_lie_on_a_call_that_fails_anyway_reports_the_real_failure():
    plan = build_schedule(0, [FaultSpec.from_dict(pin) for pin in PINS["get_order_zero_total"]])

    result = execute(ToolCall("c", "get_order", '{"order_id": "ord_9999"}'), FaultInjector(plan), load_fixture())

    assert result.observation()["error_type"] == "not_found"
    assert (result.fault.note, result.handler_called) == ("base_call_failed", False)


def test_every_result_schema_accepts_the_real_handler_output():
    store = load_fixture().clone()
    calls = {
        "get_order": {"order_id": "ord_1002"}, "get_customer": {"customer_id": "cus_101"},
        "list_payments": {"order_id": "ord_1005"}, "get_ticket": {"ticket_id": "tkt_9002"},
        "search_orders": {"customer_id": "cus_101"}, "issue_refund": REFUND,
        "create_ticket": {"customer_id": "cus_101", "subject": "Lid", "order_id": "ord_1001"},
        "add_order_note": {"order_id": "ord_1001", "note": "x"},
        "set_ticket_status": {"ticket_id": "tkt_9001", "status": "pending"},
        "cancel_order": {"order_id": "ord_1003"},
    }
    for name, arguments in calls.items():
        result = SequentialExecutor().run(ToolCall("c", name, json.dumps(arguments)), build_registry(), store)
        assert result.ok, name
        assert not list(Draft202012Validator(RESULT_SCHEMAS[name]).iter_errors(result.value)), name


# Episodes


def test_a_validation_failure_does_not_consume_the_fault_index(tmp_path):
    bad = {**REFUND, "amount_inr": -120}
    result, _, store = episode(
        tmp_path,
        schedule("refund_lost_acknowledgment"),
        completion(calls=[("c_bad", "issue_refund", bad)]),
        completion(calls=[("c_good", "issue_refund", REFUND)]),
        completion("Checking."),
    )

    [fault] = records(result, "injected_fault")
    assert (fault["call_index"], fault["tool_call_id"], fault["committed"]) == (0, "c_good", True)
    assert len(store.refunds_for("ord_1001")) == 1


def test_the_guard_ignores_policy_retries():
    guard = LoopGuard(limit=3)

    for _ in range(5):
        decision = guard.observe(REFUND_CALL, via="policy")
    assert (decision.blocked, decision.count) == (False, 0)
    assert guard.observe(REFUND_CALL).count == 1
    with pytest.raises(ValueError):
        guard.observe(REFUND_CALL, via="bystander")


@pytest.mark.parametrize("name", sorted(PINS))
def test_the_model_never_sees_the_fault_class(tmp_path, name):
    tool = PINS[name][0]["tool"]
    arguments = {"issue_refund": REFUND, "get_order": {"order_id": "ord_1001"},
                 "list_payments": {"order_id": "ord_1005"}}[tool]

    result, service, _ = episode(
        tmp_path, schedule(name), completion(calls=[("c1", tool, arguments)]), completion("Done."),
    )

    seen = json.dumps(service.requests[-1].messages).lower()
    assert not any(word in seen for word in FORBIDDEN_WORDS)
    [fault] = records(result, "injected_fault")
    assert fault["mode"] in {mode.value for mode in FaultMode}
    [tool_result] = records(result, "tool_result")
    assert not any(word in json.dumps(tool_result).lower() for word in FORBIDDEN_WORDS)


def test_the_journal_separates_injected_faults_from_model_errors(tmp_path):
    result, _, _ = episode(
        tmp_path,
        schedule("refund_lost_acknowledgment"),
        completion(calls=[("c1", "issue_refund", REFUND)]),
        completion("Checking."),
    )

    lines = records(result)
    fault = next(line for line in lines if line["event"] == "injected_fault")
    tool = next(line for line in lines if line["event"] == "tool_result")
    assert fault["seq"] + 1 == tool["seq"]
    assert (fault["tool_call_id"], fault["mode"], fault["committed"], fault["via"]) == (
        tool["tool_call_id"], "lost_acknowledgment", True, "model",
    )
    assert fault["schedule_sha256"] == schedule("refund_lost_acknowledgment").sha256
    assert tool["error"]["error_type"] == "timeout"
    assert all(line.get("error_code") is None for line in lines if line["event"] == "model_turn")


def test_two_workers_on_the_same_pinned_task_each_start_at_index_zero(tmp_path):
    plan = schedule("refund_lost_acknowledgment")
    outcomes = {}

    def worker(name):
        result, _, store = episode(
            tmp_path / name, plan, completion(calls=[("c1", "issue_refund", REFUND)]), completion("Done."),
        )
        outcomes[name] = (records(result, "injected_fault")[0]["call_index"], len(store.refunds_for("ord_1001")))

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes == {"a": (0, 1), "b": (0, 1)}


# Replay


def record_claimed(directory, faults, *turns):
    return run_claimed_episode(
        directory,
        SteadyhandClient(FakeService(*turns)),
        build_registry(),
        load_fixture().clone(),
        initial_messages("Refund 120 INR on ord_1001 for the dented lid."),
        faults=faults,
    )


def test_a_lost_acknowledgment_episode_replays_to_the_same_store(tmp_path):
    plan = schedule("refund_lost_acknowledgment")

    result = record_claimed(
        tmp_path, plan,
        completion(calls=[("c1", "issue_refund", REFUND)]),
        completion(calls=[("c2", "list_payments", {"order_id": "ord_1001"})]),
        completion("The refund went through once."),
    )

    manifest = load_manifest(manifest_path_for(result.journal_path))
    assert manifest.fault_schedule_sha256 == plan.sha256
    assert FaultSchedule.from_dict(manifest.fault_schedule) == plan
    finished = read_journal(result.journal_path)[-1]
    store = load_fixture().clone()
    store.add_refund("ord_1001", "120", "Dented lid")
    assert finished["store_sha256"] == store.snapshot_sha256()  # the refund is in the store
    assert json.loads(result.messages[3]["content"])["error_type"] == "timeout"  # the model saw a timeout

    replayed = replay_episode(result.journal_path)
    assert replayed.status is EpisodeStatus.DONE
    assert replayed.messages == result.messages


def test_a_pin_altered_without_its_hash_is_refused(tmp_path):
    result = record_claimed(
        tmp_path, schedule("refund_lost_acknowledgment"),
        completion(calls=[("c1", "issue_refund", REFUND)]), completion("Done."),
    )
    manifest_file = manifest_path_for(result.journal_path)
    manifest = json.loads(manifest_file.read_text())
    manifest["fault_schedule"]["pins"][0]["mode"] = "explicit_permanent"
    manifest_file.write_text(json.dumps(manifest))

    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(result.journal_path)

    assert caught.value.field == "fault_schedule_sha256"


def test_a_consistently_altered_pin_changes_the_replayed_events(tmp_path):
    result = record_claimed(
        tmp_path, schedule("refund_lost_acknowledgment"),
        completion(calls=[("c1", "issue_refund", REFUND)]), completion("Done."),
    )
    manifest_file = manifest_path_for(result.journal_path)
    manifest = json.loads(manifest_file.read_text())
    altered = build_schedule(7, [FaultSpec("issue_refund", 0, FaultMode.EXPLICIT_PERMANENT)])
    manifest["fault_schedule"] = altered.as_dict()
    manifest["fault_schedule_sha256"] = altered.sha256
    manifest_file.write_text(json.dumps(manifest))
    fault_seq = next(line["seq"] for line in read_journal(result.journal_path) if line["event"] == "injected_fault")

    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(result.journal_path)

    assert caught.value.seq == fault_seq


def test_a_fault_free_journal_cannot_claim_a_schedule(tmp_path):
    result = record_claimed(tmp_path, None, completion("Hello."))
    manifest_file = manifest_path_for(result.journal_path)
    manifest = json.loads(manifest_file.read_text())
    manifest["fault_schedule_sha256"] = "a" * 64
    manifest_file.write_text(json.dumps(manifest))

    with pytest.raises(ReplayMismatch) as caught:
        replay_episode(result.journal_path)

    assert caught.value.field == "fault_schedule"

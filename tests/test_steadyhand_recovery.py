import json
import threading
import time
from pathlib import Path

import pytest

import steadyhand.loop as loop_module
from steadyhand import (
    Effect,
    EpisodeJournal,
    EpisodeStatus,
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
    strategy_for,
    validate_call,
)
from steadyhand.faults import FaultMode, FaultSpec, build_schedule
from steadyhand.idempotency import idempotency_key
from steadyhand.journal import ReplayMismatch, manifest_path_for
from steadyhand.loop import _answerable
from steadyhand.manifest import build_manifest, load_manifest
from steadyhand.messages import FAILURE_AWARE_ADDENDUM, SYSTEM_PROMPT
from steadyhand.recovery import STRATEGIES, RecoveryBudget, RecoveryContext
from steadyhand.replay import replay_episode, run_claimed_episode
from steadyhand.sanity import LIST_PAYMENTS_HINT
from toolsmith.json_utils import strict_json_loads
from toolsmith.protocols import QWEN3_PROFILE

REFUND = {"order_id": "ord_1001", "amount_inr": 120, "reason": "Dented lid"}
BAD_REFUND = {**REFUND, "amount_inr": -120}
GOAL = "Refund 120 INR on ord_1001 for the dented lid."


class FakeService:
    profile = QWEN3_PROFILE
    use_grammars = True

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class FakeSleep:
    def __init__(self):
        self.waits = []

    def __call__(self, seconds):
        self.waits.append(seconds)


def completion(content=None, calls=(), repair_attempts=0):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            for call_id, name, arguments in calls
        ]
    return {
        "choices": [{"index": 0, "message": message}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 6},
        "toolsmith": {"repair_attempts": repair_attempts},
    }


def pins(*specs, seed=0):
    return build_schedule(seed, [FaultSpec(*spec) if isinstance(spec, tuple) else spec for spec in specs])


def episode(tmp_path, strategy, *turns, faults=None, goal=GOAL, registry=None, store=None, variant="plain"):
    store = store if store is not None else load_fixture().clone()
    service = FakeService(*turns)
    sleep = FakeSleep()
    result = run_episode(
        SteadyhandClient(service),
        registry or build_registry(),
        store,
        initial_messages(goal, variant=variant),
        journal=EpisodeJournal.create(tmp_path),
        faults=faults,
        recovery=strategy_for(strategy),
        sleep=sleep,
    )
    return result, service, store, sleep


def records(result, event=None):
    lines = [strict_json_loads(line) for line in Path(result.journal_path).read_text().splitlines()]
    return [line for line in lines if event is None or line["event"] == event]


def seen(service, request=1, message=-1):
    return json.loads(service.requests[request].messages[message]["content"])


# Selection


def test_every_strategy_is_selectable_by_name():
    assert sorted(STRATEGIES) == [
        "escalate", "idempotent_verify", "raw_error", "retry_safe", "switch", "typed_hint",
    ]
    for name in STRATEGIES:
        assert strategy_for(name).name == name
    with pytest.raises(ValueError, match="Unknown recovery policy"):
        strategy_for("retry_forever")


# (a) and (b)


def test_raw_error_hides_type_details_and_hint_that_typed_hint_shows(tmp_path):
    raw, raw_service, raw_store, _ = episode(
        tmp_path / "a", "raw_error", completion(calls=[("c1", "issue_refund", BAD_REFUND)]), completion("Sorry."),
    )
    typed, typed_service, typed_store, _ = episode(
        tmp_path / "b", "typed_hint", completion(calls=[("c1", "issue_refund", BAD_REFUND)]), completion("Sorry."),
    )

    assert seen(raw_service) == {"ok": False, "error": "amount_inr: must be > 0"}
    typed_observation = seen(typed_service)
    assert {"error_type", "details", "hint"} <= set(typed_observation)
    for store in (raw_store, typed_store):
        assert store.snapshot_sha256() == store.fixture_sha256
    # The journal keeps the typed error for scoring either way.
    [raw_result] = records(raw, "tool_result")
    assert raw_result["error"]["error_type"] == "invalid_arguments"
    assert json.loads(raw_result["content"]) == {"ok": False, "error": "amount_inr: must be > 0"}


# (c) retry_safe


def test_retry_safe_retries_a_failed_read_in_place(tmp_path):
    faults = pins(("get_order", 0, FaultMode.EXPLICIT_TRANSIENT, "timeout"))

    result, service, _, sleep = episode(
        tmp_path, "retry_safe",
        completion(calls=[("c1", "get_order", {"order_id": "ord_1001"})]),
        completion("Delivered."),
        faults=faults,
    )

    assert len(service.requests) == 2  # one model turn saw the result
    assert seen(service)["ok"] is True
    assert len(sleep.waits) == 1 and 0.5 <= sleep.waits[0] <= 0.625
    [fault] = records(result, "injected_fault")
    assert (fault["call_index"], fault["via"]) == (0, "model")
    [attempt] = records(result, "policy_attempt")
    assert (attempt["purpose"], attempt["ok"], attempt["waited_s"]) == ("retry", True, sleep.waits[0])
    [recovery] = records(result, "recovery")
    assert recovery["recovered_by"] == "retry"


def test_retry_safe_stops_at_the_policy_attempt_cap(tmp_path):
    faults = pins(*[("get_order", index, FaultMode.EXPLICIT_TRANSIENT, "server_error") for index in range(3)])

    result, service, _, sleep = episode(
        tmp_path, "retry_safe",
        completion(calls=[("c1", "get_order", {"order_id": "ord_1001"})]),
        completion("The order service is failing."),
        faults=faults,
    )

    assert seen(service)["error_type"] == "server_error"
    assert len(sleep.waits) == 2 and all(wait <= 2.0 for wait in sleep.waits)
    assert sleep.waits[1] > sleep.waits[0]
    assert [f["via"] for f in records(result, "injected_fault")] == ["model", "policy", "policy"]


def test_retry_safe_never_retries_a_non_idempotent_write(tmp_path):
    faults = pins(("issue_refund", 0, FaultMode.EXPLICIT_TRANSIENT, "rate_limit"))

    result, service, store, sleep = episode(
        tmp_path, "retry_safe",
        completion(calls=[("c1", "issue_refund", REFUND)]), completion("Rate limited."),
        faults=faults,
    )

    assert seen(service)["error_type"] == "rate_limit"
    assert records(result, "policy_attempt") == [] and sleep.waits == []
    assert store.refunds_for("ord_1001") == []


def test_policy_waits_stop_when_the_episode_budget_is_spent():
    registry = build_registry()
    context = RecoveryContext(
        call=ToolCall("c", "get_order", "{}"), arguments={}, policy=registry.policy("get_order"),
        registry=registry, store=load_fixture(), executor=SequentialExecutor(), injector=None,
        journal=None, step=0, episode_id="ep_test", sleep=FakeSleep(),
        budget=RecoveryBudget(max_total_wait_s=1.0),
    )

    first = context.backoff()
    context.attempts += 1

    assert first is not None and 0.5 <= first <= 0.625
    assert context.backoff() is None  # 0.5 + about 1.0 would pass the 1.0 s budget


# (d) idempotent_verify


def test_lost_acknowledgment_under_idempotent_verify_makes_one_refund(tmp_path):
    faults = pins(("issue_refund", 0, FaultMode.LOST_ACKNOWLEDGMENT))

    result, service, store, _ = episode(
        tmp_path, "idempotent_verify",
        completion(calls=[("c1", "issue_refund", REFUND)]),
        completion(calls=[("c2", "issue_refund", REFUND)]),  # the model retries anyway
        completion("Refunded once."),
        faults=faults,
    )

    [refund] = store.refunds_for("ord_1001")
    assert refund.idempotency_key == idempotency_key(result.episode_id, "issue_refund", REFUND)
    first = seen(service)
    assert first["ok"] is True and first["value"]["refund_id"] == refund.refund_id
    [recovery] = records(result, "recovery")
    assert recovery["recovered_by"] == "verify"
    assert [a["purpose"] for a in records(result, "policy_attempt")] == ["verify"]


def test_the_same_baseline_without_keys_makes_two_refunds(tmp_path):
    faults = pins(("issue_refund", 0, FaultMode.LOST_ACKNOWLEDGMENT))

    _, _, store, _ = episode(
        tmp_path, "typed_hint",
        completion(calls=[("c1", "issue_refund", REFUND)]),
        completion(calls=[("c2", "issue_refund", REFUND)]),
        completion("Done."),
        faults=faults,
    )

    assert len(store.refunds_for("ord_1001")) == 2


def test_keys_dedupe_within_an_episode_but_not_across_episodes():
    store = load_fixture().clone()
    call = ToolCall("c", "issue_refund", json.dumps(REFUND))
    run = SequentialExecutor().run

    first = run(call, build_registry(), store, idempotency_key=idempotency_key("ep_a", "issue_refund", REFUND))
    again = run(call, build_registry(), store, idempotency_key=idempotency_key("ep_a", "issue_refund", {**REFUND, "amount_inr": 120.0}))
    other = run(call, build_registry(), store, idempotency_key=idempotency_key("ep_b", "issue_refund", REFUND))

    assert first.value == again.value
    assert other.value["refund_id"] != first.value["refund_id"]
    assert len(store.refunds_for("ord_1001")) == 2
    assert store.clone().receipt_for(first.value["idempotency_key"]) == ("refund", first.value["refund_id"])


def test_when_verify_cannot_tell_the_outcome_is_unknown_and_the_episode_escalates(tmp_path):
    faults = pins(
        ("issue_refund", 0, FaultMode.LOST_ACKNOWLEDGMENT),
        ("list_payments", 0, FaultMode.EXPLICIT_PERMANENT),
    )

    result, service, store, _ = episode(
        tmp_path, "idempotent_verify", completion(calls=[("c1", "issue_refund", REFUND)]), faults=faults,
    )

    assert result.status is EpisodeStatus.ESCALATED
    assert len(service.requests) == 1
    assert len(store.refunds_for("ord_1001")) == 1  # only the committed one
    [tool] = records(result, "tool_result")
    assert tool["error"]["error_type"] == "outcome_unknown" and tool["error"]["retryable"] is False
    assert "list_payments could not confirm it" in result.final_content


def test_verify_absent_retries_once_with_the_same_key(tmp_path):
    faults = pins(("issue_refund", 0, FaultMode.EXPLICIT_TRANSIENT, "rate_limit"))

    result, service, store, _ = episode(
        tmp_path, "idempotent_verify",
        completion(calls=[("c1", "issue_refund", REFUND)]), completion("Refunded."),
        faults=faults,
    )

    assert seen(service)["ok"] is True
    assert [a["purpose"] for a in records(result, "policy_attempt")] == ["verify", "retry"]
    retry = records(result, "policy_attempt")[1]
    assert retry["idempotency_key"] == idempotency_key(result.episode_id, "issue_refund", REFUND)
    assert len(store.refunds_for("ord_1001")) == 1


def test_a_lying_verify_probe_cannot_cause_a_second_refund(tmp_path):
    faults = pins(
        FaultSpec("issue_refund", 0, FaultMode.LOST_ACKNOWLEDGMENT),
        FaultSpec("list_payments", 0, FaultMode.IMPLICIT_TRANSIENT,
                  implicit_patch={"refunds": [], "refunded_total_inr": "0.00"}),
    )

    result, service, store, _ = episode(
        tmp_path, "idempotent_verify",
        completion(calls=[("c1", "issue_refund", REFUND)]), completion("Refunded."),
        faults=faults,
    )

    assert [a["purpose"] for a in records(result, "policy_attempt")] == ["verify", "retry"]
    assert len(store.refunds_for("ord_1001")) == 1  # the same-key retry was deduplicated by the store
    assert seen(service)["ok"] is True


# (e) switch


def test_switch_calls_the_ledger_once_when_issue_refund_is_unavailable(tmp_path):
    faults = pins(("issue_refund", 0, FaultMode.EXPLICIT_PERMANENT))

    result, service, store, _ = episode(
        tmp_path, "switch", completion(calls=[("c1", "issue_refund", REFUND)]), completion("Refunded via ledger."),
        faults=faults,
    )

    assert store.refunds_for("ord_1001") == []  # issue_refund's handler never ran
    [entry] = store.ledger_for("ord_1001")
    observation = seen(service)
    assert observation["ok"] is True and observation["value"]["entry_id"] == entry.entry_id
    [attempt] = records(result, "policy_attempt")
    assert (attempt["purpose"], attempt["tool"], attempt["ok"]) == ("switch", "refund_via_ledger", True)
    assert "ledger" in store.to_dict()


# (f) escalate


def test_escalate_returns_argument_errors_to_the_model_and_the_correction_succeeds(tmp_path):
    result, service, store, _ = episode(
        tmp_path, "escalate",
        completion(calls=[("c1", "issue_refund", BAD_REFUND)]),
        completion(calls=[("c2", "issue_refund", REFUND)]),
        completion("Refunded."),
    )

    assert result.status is EpisodeStatus.DONE
    assert seen(service)["error_type"] == "invalid_arguments"
    assert records(result, "policy_attempt") == []
    assert len(store.refunds_for("ord_1001")) == 1


def test_escalate_after_a_dead_alternative_names_both_tools(tmp_path):
    faults = pins(
        ("issue_refund", 0, FaultMode.EXPLICIT_PERMANENT),
        ("refund_via_ledger", 0, FaultMode.EXPLICIT_PERMANENT),
    )

    result, _, store, _ = episode(tmp_path, "escalate", completion(calls=[("c1", "issue_refund", REFUND)]), faults=faults)

    assert result.status is EpisodeStatus.ESCALATED
    assert store.refunds_for("ord_1001") == [] and store.ledger_for("ord_1001") == []
    assert "issue_refund (permanent_unavailable x1)" in result.final_content
    assert "refund_via_ledger (permanent_unavailable x1)" in result.final_content
    assert "alternative refund_via_ledger failed" in result.final_content


def test_escalate_on_a_permanent_failure_with_no_alternative(tmp_path):
    faults = pins(("add_order_note", 0, FaultMode.EXPLICIT_PERMANENT))

    result, _, _, _ = episode(
        tmp_path, "escalate", completion(calls=[("c1", "add_order_note", {"order_id": "ord_1001", "note": "x"})]),
        faults=faults,
    )

    assert result.status is EpisodeStatus.ESCALATED


# Sanity checks


def test_a_list_payments_lie_is_flagged_and_the_refund_in_that_turn_never_runs(tmp_path):
    faults = pins(FaultSpec("list_payments", 0, FaultMode.IMPLICIT_TRANSIENT, implicit_patch={"captured_total_inr": "999.00"}))

    result, service, store, _ = episode(
        tmp_path, "escalate",
        completion(calls=[("c1", "list_payments", {"order_id": "ord_1001"}), ("c2", "issue_refund", REFUND)]),
        completion("Something is off with the payments; I'll stop."),
        faults=faults,
    )

    payments, refund = (json.loads(m["content"]) for m in service.requests[1].messages[-2:])
    assert payments["error_type"] == "suspicious_result" and payments["retryable"] is False
    assert payments["hint"] == LIST_PAYMENTS_HINT
    assert refund["error_type"] == "suspicious_result" and refund["hint"].startswith("Not run")
    assert store.refunds_for("ord_1001") == []
    [check] = records(result, "sanity_checked")
    assert (check["tool"], check["passed"]) == ("list_payments", False)


def test_a_suspicious_refund_receipt_escalates_instead_of_retrying(tmp_path):
    faults = pins(FaultSpec("issue_refund", 0, FaultMode.IMPLICIT_TRANSIENT, implicit_patch={"amount_inr": "0.00"}))

    result, _, store, _ = episode(tmp_path, "escalate", completion(calls=[("c1", "issue_refund", REFUND)]), faults=faults)

    assert result.status is EpisodeStatus.ESCALATED
    assert store.refunds_for("ord_1001") == []
    assert [c["passed"] for c in records(result, "sanity_checked")] == [False]
    assert records(result, "policy_attempt") == []


def test_honest_results_pass_and_a_lie_without_an_invariant_is_missed(tmp_path):
    hidden = pins(FaultSpec("list_payments", 0, FaultMode.IMPLICIT_TRANSIENT,
                            implicit_patch={"refunds": [], "refunded_total_inr": "0.00"}))

    result, service, _, _ = episode(
        tmp_path, "escalate", completion(calls=[("c1", "list_payments", {"order_id": "ord_1005"})]),
        completion("No refunds yet."), faults=hidden,
    )

    assert seen(service)["ok"] is True  # internally consistent lie: allowed to slip through
    assert [c["passed"] for c in records(result, "sanity_checked")] == [True]


# Concurrency inside one episode


class CountingPool(loop_module.ThreadPoolExecutor):
    created = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CountingPool.created.append(kwargs.get("max_workers"))


ORDER = []


def make_read(name, delay):
    def read(store: Store, order_id: str) -> dict:
        time.sleep(delay)
        ORDER.append(name)
        return {"order_id": order_id, "reader": name}

    read.__name__ = name
    read.__doc__ = f"Stub read {name}."
    return read


def test_parallel_reads_answer_in_call_order(tmp_path, monkeypatch):
    CountingPool.created.clear()
    ORDER.clear()
    monkeypatch.setattr(loop_module, "ThreadPoolExecutor", CountingPool)
    registry = SteadyhandRegistry()
    for name, delay in (("read_a", 0.03), ("read_b", 0.0), ("read_c", 0.01)):
        registry.register(make_read(name, delay), ToolPolicy(Effect.READ, 5.0, idempotent=True))
    calls = [(f"c_{n}", n, {"order_id": "ord_1001"}) for n in ("read_a", "read_b", "read_c")]

    result, service, _, _ = episode(tmp_path, "typed_hint", completion(calls=calls), completion("Done."),
                                    registry=registry.freeze())

    assert CountingPool.created == [3]
    tool_messages = service.requests[1].messages[-3:]
    assert [m["tool_call_id"] for m in tool_messages] == ["c_read_a", "c_read_b", "c_read_c"]
    assert [json.loads(m["content"])["value"]["reader"] for m in tool_messages] == ["read_a", "read_b", "read_c"]
    assert [r["tool_call_id"] for r in records(result, "tool_result")] == ["c_read_a", "c_read_b", "c_read_c"]


def test_parallel_reads_take_fault_indexes_in_call_order(tmp_path):
    faults = pins(("get_order", 1, FaultMode.EXPLICIT_TRANSIENT, "server_error"))
    calls = [("c1", "get_order", {"order_id": "ord_1001"}), ("c2", "get_order", {"order_id": "ord_1002"})]

    result, service, _, _ = episode(tmp_path, "typed_hint", completion(calls=calls), completion("Done."), faults=faults)

    first, second = (json.loads(m["content"]) for m in service.requests[1].messages[-2:])
    assert first["ok"] is True and second["error_type"] == "server_error"
    [fault] = records(result, "injected_fault")
    assert (fault["tool_call_id"], fault["call_index"]) == ("c2", 1)


def test_writes_in_one_message_run_in_order_without_the_pool(tmp_path, monkeypatch):
    CountingPool.created.clear()
    monkeypatch.setattr(loop_module, "ThreadPoolExecutor", CountingPool)
    calls = [("c1", "issue_refund", REFUND), ("c2", "add_order_note", {"order_id": "ord_1001", "note": "Refunded"})]

    result, _, store, _ = episode(tmp_path, "typed_hint", completion(calls=calls), completion("Done."))

    assert CountingPool.created == []
    assert [r["tool"] for r in records(result, "tool_result")] == ["issue_refund", "add_order_note"]
    assert len(store.refunds_for("ord_1001")) == 1 and len(store.order("ord_1001").notes) == 1


def test_the_store_lock_survives_cloning_and_pickling():
    import pickle

    store = load_fixture()
    clone = pickle.loads(pickle.dumps(store.clone()))

    assert isinstance(clone.lock, type(threading.RLock()))
    assert clone.lock is not store.lock and clone.snapshot_sha256() == store.snapshot_sha256()


# Manifest, replay, and prompt variants


def record_claimed(directory, strategy, faults, *turns, variant="plain"):
    return run_claimed_episode(
        directory,
        SteadyhandClient(FakeService(*turns)),
        build_registry(),
        load_fixture().clone(),
        initial_messages(GOAL, variant=variant),
        faults=faults,
        recovery=strategy_for(strategy),
        system_prompt_variant=variant,
        sleep=FakeSleep(),
    )


@pytest.mark.parametrize(
    ("strategy", "faults", "turns"),
    [
        ("retry_safe", pins(("get_order", 0, FaultMode.EXPLICIT_TRANSIENT, "timeout")),
         [completion(calls=[("c1", "get_order", {"order_id": "ord_1001"})]), completion("Delivered.")]),
        ("idempotent_verify", pins(("issue_refund", 0, FaultMode.LOST_ACKNOWLEDGMENT)),
         [completion(calls=[("c1", "issue_refund", REFUND)]), completion("Refunded.")]),
        ("switch", pins(("issue_refund", 0, FaultMode.EXPLICIT_PERMANENT)),
         [completion(calls=[("c1", "issue_refund", REFUND)]), completion("Refunded.")]),
        ("escalate", pins(("issue_refund", 0, FaultMode.EXPLICIT_PERMANENT), ("refund_via_ledger", 0, FaultMode.EXPLICIT_PERMANENT)),
         [completion(calls=[("c1", "issue_refund", REFUND)])]),
        ("raw_error", None, [completion(calls=[("c1", "issue_refund", BAD_REFUND)]), completion("Sorry.")]),
    ],
)
def test_each_strategy_records_and_replays(tmp_path, strategy, faults, turns):
    result = record_claimed(tmp_path, strategy, faults, *turns)

    manifest = load_manifest(manifest_path_for(result.journal_path))
    assert manifest.recovery_policy == strategy
    assert records(result, "episode_started")[0]["recovery_policy"] == strategy
    replayed = replay_episode(result.journal_path)
    assert (replayed.status, replayed.final_content) == (result.status, result.final_content)


def test_a_recovery_policy_mismatch_fails_replay(tmp_path):
    result = record_claimed(
        tmp_path, "idempotent_verify", pins(("issue_refund", 0, FaultMode.LOST_ACKNOWLEDGMENT)),
        completion(calls=[("c1", "issue_refund", REFUND)]), completion("Refunded."),
    )

    with pytest.raises(ReplayMismatch) as override:
        replay_episode(result.journal_path, recovery=strategy_for("typed_hint"))
    assert override.value.field == "recovery_policy"

    manifest_file = manifest_path_for(result.journal_path)
    manifest = json.loads(manifest_file.read_text())
    manifest["recovery_policy"] = "escalate"
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ReplayMismatch) as tampered:
        replay_episode(result.journal_path)
    assert tampered.value.field == "recovery_policy"


def test_the_failure_aware_prompt_is_data_hashed_into_the_manifest(tmp_path):
    plain = initial_messages(GOAL)
    aware = initial_messages(GOAL, variant="failure_aware")
    settings = SteadyhandClient(FakeService()).settings()

    def manifest(messages, variant):
        return build_manifest(settings, build_registry(), load_fixture(), messages,
                              budgets={}, guard_limit=3, system_prompt_variant=variant)

    assert aware[0]["content"] == f"{SYSTEM_PROMPT}\n\n{FAILURE_AWARE_ADDENDUM}"
    assert plain[0]["content"] == SYSTEM_PROMPT
    assert manifest(aware, "failure_aware").system_prompt_sha256 != manifest(plain, "plain").system_prompt_sha256
    assert "inject" not in FAILURE_AWARE_ADDENDUM.lower() and "fault" not in FAILURE_AWARE_ADDENDUM.lower()
    with pytest.raises(ValueError):
        manifest(plain, "failure_aware")
    with pytest.raises(ValueError):
        manifest(aware, "plain")
    result = record_claimed(tmp_path, "escalate", None, completion("Hello."), variant="failure_aware")
    assert load_manifest(manifest_path_for(result.journal_path)).system_prompt_variant == "failure_aware"


def test_the_model_cannot_supply_its_own_idempotency_key():
    arguments = {**REFUND, "idempotency_key": "idk_forged"}

    verdict = validate_call(ToolCall("c", "issue_refund", json.dumps(arguments)), build_registry(), load_fixture())

    assert verdict.error.details[0].field == "/idempotency_key"
    assert "idempotency_key" not in json.dumps(build_registry().schemas())


def test_the_ledger_round_trips_and_keyless_rows_serialize_as_before():
    store = load_fixture().clone()
    store.add_ledger_entry("ord_1001", "120", "Dented lid", idempotency_key="idk_x")
    store.add_refund("ord_1002", "10", "Late")

    again = Store.from_dict(store.to_dict())

    assert again.snapshot_sha256() == store.snapshot_sha256()
    assert again.receipt_for("idk_x") == ("ledger", "led_1")
    assert "idempotency_key" not in json.dumps(store.to_dict()["refunds"])
    assert "ledger" not in load_fixture().to_dict()


def test_shim_repair_and_a_typed_tool_error_both_show_in_order(tmp_path):
    result, _, _, _ = episode(
        tmp_path, "escalate",
        completion(calls=[("c1", "issue_refund", BAD_REFUND)], repair_attempts=1),
        completion("Which amount?"),
    )

    lines = [line for line in records(result) if line["event"] in ("model_turn", "tool_result")]
    assert lines[0]["event"] == "model_turn" and lines[0]["repair_attempts"] == 1
    assert lines[1]["event"] == "tool_result" and lines[1]["error"]["error_type"] == "invalid_arguments"


def test_an_unfinished_turn_is_dropped_before_a_paraphrase():
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "{}"},
    ]

    assert _answerable(history) == history[:1]
    assert _answerable(history + [{"role": "tool", "tool_call_id": "b", "content": "{}"}]) == history + [
        {"role": "tool", "tool_call_id": "b", "content": "{}"}
    ]

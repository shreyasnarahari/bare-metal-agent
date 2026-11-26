"""The shared scheduler: slot leases, affinity, episode workers, prioritized pools, and bounded prefetch."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tidemark.scheduler import (
    PREFETCH,
    REQUIRED,
    PriorityPool,
    Scheduler,
    SchedulerPlan,
    SlotLeasedClient,
    SlotPool,
    bounded_prefetch,
)


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.001)


def test_model_calls_never_exceed_the_slots():
    pool = SlotPool(3, affinity="work_conserving")
    active, peak, lock = [0], [0], threading.Lock()

    def call(owner):
        for _ in range(20):
            with pool.lease(owner):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.0005)
                with lock:
                    active[0] -= 1

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(call, [f"ep{i}" for i in range(10)]))
    assert peak[0] == 3 and pool.stats()["max_busy"] == 3 and pool.stats()["leases"] == 200


def test_work_conserving_fills_every_free_slot_in_arrival_order():
    pool = SlotPool(2, affinity="work_conserving")
    first, second = pool.acquire("a"), pool.acquire("b")
    assert {first.slot_id, second.slot_id} == {0, 1}
    order = []
    threads = [threading.Thread(target=lambda o=o: order.append((o, pool.acquire(o)))) for o in ("c", "d")]
    for thread in threads:
        thread.start()
        wait_for(lambda: pool.snapshot()["model_waiting"] == threads.index(thread) + 1)
    assert pool.snapshot() == {"slots_busy": 2, "slots_usable": 2, "slots_pinned_idle": 0, "model_waiting": 2}
    pool.release(first)
    wait_for(lambda: len(order) == 1)
    assert order[0][0] == "c" and order[0][1].slot_id == first.slot_id  # the free slot went to the earliest waiter
    pool.release(second)
    wait_for(lambda: len(order) == 2)
    assert pool.snapshot()["model_waiting"] == 0


def test_work_conserving_prefers_the_previous_slot_and_records_migrations():
    pool = SlotPool(2, affinity="work_conserving")
    a = pool.acquire("a")
    pool.release(a)
    again = pool.acquire("a")
    assert again.slot_id == a.slot_id and not again.migrated  # affinity kept while the slot is free
    pool.release(again)
    hold = pool.acquire("b")  # b takes a's slot while a waits on a tool
    assert hold.slot_id == a.slot_id
    moved = pool.acquire("a")
    assert moved.slot_id != a.slot_id and moved.migrated and pool.stats()["migrations"] == 1


def SlotLease_for(pool, owner):
    from tidemark.scheduler import SlotLease

    slot = next(slot for slot, holder in pool._busy.items() if holder == owner)
    return SlotLease(slot, owner, "agent", 0.0, False)


def test_pinned_affinity_keeps_a_slot_through_tool_waits_until_the_episode_ends():
    pool = SlotPool(1, affinity="pinned")
    lease = pool.acquire("a")
    pool.release(lease)  # a is running a tool; its slot stays pinned
    assert pool.snapshot() == {"slots_busy": 0, "slots_usable": 1, "slots_pinned_idle": 1, "model_waiting": 0}
    got = []
    waiter = threading.Thread(target=lambda: got.append(pool.acquire("b")))
    waiter.start()
    wait_for(lambda: pool.snapshot()["model_waiting"] == 1)
    assert pool.acquire("a").slot_id == lease.slot_id and not got  # a keeps priority on its own slot
    pool.release(SlotLease_for(pool, "a"))
    pool.end_owner("a")
    waiter.join(timeout=5)
    assert got[0].slot_id == lease.slot_id and not got[0].migrated


def test_a_dedicated_compactor_slot_is_only_for_compaction():
    pool = SlotPool(1, compactor_slots=1, affinity="work_conserving")
    agent = pool.acquire("a")
    compactor = pool.acquire("a", kind="compactor")
    assert agent.slot_id == 0 and compactor.slot_id == 1 and pool.usable == 2
    with pytest.raises(TimeoutError):
        pool.acquire("b", timeout=0.05)  # the compactor slot is never lent to an agent call
    shared = SlotPool(1, affinity="work_conserving")
    assert shared.acquire("x", kind="compactor").slot_id == 0
    with pytest.raises(ValueError):
        pool.acquire("a", kind="embedding")


def test_episode_workers_overlap_tools_while_model_calls_stay_within_slots():
    plan = SchedulerPlan(agent_slots=2, affinity="work_conserving", cpu_workers=1, io_threads=1)
    assert plan.workers == 3
    in_model, in_tool, peaks, lock = [0], [0], {"model": 0, "tool": 0, "workers": 0}, threading.Lock()
    running = [0]

    with Scheduler(plan, cpu_executor=lambda: ThreadPoolExecutor(1)) as scheduler:
        def work(unit, owner):
            with lock:
                running[0] += 1
                peaks["workers"] = max(peaks["workers"], running[0])
            for _ in range(3):
                with scheduler.slots.lease(owner):
                    with lock:
                        in_model[0] += 1
                        peaks["model"] = max(peaks["model"], in_model[0])
                    time.sleep(0.002)
                    with lock:
                        in_model[0] -= 1
                with scheduler.tool():
                    with lock:
                        in_tool[0] += 1
                        peaks["tool"] = max(peaks["tool"], in_tool[0])
                    time.sleep(0.004)
                    with lock:
                        in_tool[0] -= 1
            with lock:
                running[0] -= 1
            return unit * 10

        results = scheduler.run_episodes(range(9), work, key=lambda unit: f"ep{unit}")
    assert results == [unit * 10 for unit in range(9)]
    assert peaks["model"] <= 2 and peaks["workers"] == 3 and peaks["tool"] >= 2


def test_the_snapshot_reports_every_queue():
    with Scheduler(SchedulerPlan(agent_slots=2, cpu_workers=2, io_threads=3), cpu_executor=lambda: ThreadPoolExecutor(2)) as scheduler:
        snapshot = scheduler.snapshot()
    assert snapshot["slots_usable"] == 2 and snapshot["cpu_capacity"] == 2 and snapshot["io_capacity"] == 3
    assert {"episodes_active", "episodes_waiting", "tools_active", "cpu_waiting", "io_running"} <= set(snapshot)


def test_required_work_runs_before_prefetch_and_prefetch_is_bounded():
    gate = threading.Event()
    order = []
    pool = PriorityPool("cpu", lambda: ThreadPoolExecutor(1), 1, queue_bound=2)
    blocker = pool.submit(gate.wait)
    wait_for(lambda: pool.counts() == (1, 0))
    pool.submit(order.append, "prefetch-1", priority=PREFETCH)
    pool.submit(order.append, "prefetch-2", priority=PREFETCH)
    blocked = threading.Event()

    def third():
        pool.submit(order.append, "prefetch-3", priority=PREFETCH)
        blocked.set()

    threading.Thread(target=third, daemon=True).start()
    time.sleep(0.05)
    assert not blocked.is_set()  # a full queue makes prefetch wait
    required = pool.submit(order.append, "required", priority=REQUIRED)  # required work never waits for room
    gate.set()
    blocker.result(timeout=5)
    required.result(timeout=5)
    wait_for(lambda: len(order) == 4)
    assert order[0] == "required" and order[1:] == ["prefetch-1", "prefetch-2", "prefetch-3"]
    pool.close()


def test_pool_errors_reach_the_caller():
    pool = PriorityPool("io", lambda: ThreadPoolExecutor(1), 1)
    future = pool.submit(lambda: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        future.result(timeout=5)
    pool.close()
    with pytest.raises(RuntimeError):
        pool.submit(print)


def test_bounded_prefetch_keeps_order_and_limits_work_ahead():
    prepared = []
    consumed = []
    for value in bounded_prefetch(range(10), lambda item: prepared.append(item) or item * 2, depth=2):
        time.sleep(0.005)
        consumed.append(value)
        assert len(prepared) - len(consumed) <= 3  # depth ready items plus the one being prepared
    assert consumed == [item * 2 for item in range(10)]
    with pytest.raises(ValueError):
        list(bounded_prefetch([1], lambda item: item, depth=0))

    def fail(item):
        raise RuntimeError("bad item")

    with pytest.raises(RuntimeError, match="bad item"):
        list(bounded_prefetch([1], fail, depth=1))


def test_the_leased_client_pins_each_call_to_its_slot():
    class Inner:
        def __init__(self):
            self.calls = []

        def settings(self):
            return {"model": "m"}

        def complete(self, messages, tools, *, tool_choice="auto", id_slot=None):
            self.calls.append((tool_choice, id_slot))
            return "turn"

        def render_prompt(self, messages, tools, *, tool_choice="auto"):
            return "prompt"

    pool = SlotPool(2, affinity="pinned")
    inner = Inner()
    client = SlotLeasedClient(inner, pool, "ep1")
    assert client.complete([], [], tool_choice="none") == "turn" and client.settings() == {"model": "m"}
    assert inner.calls == [("none", 0)] and client.leases[0].slot_id == 0 and client.render_prompt([], []) == "prompt"
    plain = SlotLeasedClient(Inner(), pool, "ep2", pass_slot=False)
    plain.complete([], [])
    assert plain.inner.calls == [("auto", None)]


def test_plans_validate_and_hash():
    plan = SchedulerPlan(agent_slots=4, compactor_slots=1, affinity="work_conserving")
    assert plan.workers == 6 and plan.usable_slots == 5 and plan.as_dict()["episode_workers"] == 6
    assert plan.sha256() != SchedulerPlan(agent_slots=4, affinity="work_conserving").sha256()
    with pytest.raises(ValueError):
        SchedulerPlan(agent_slots=0)
    with pytest.raises(ValueError):
        SchedulerPlan(agent_slots=1, affinity="greedy")

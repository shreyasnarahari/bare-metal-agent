"""One work-conserving scheduler per notebook run, shared by every episode and phase.

- **Slot pool**: at most ``usable`` agent, compactor, and judge calls are inside
  llama-server at once, each under an explicit ``id_slot`` lease. There is no
  unbounded HTTP queue. Waiters are served in arrival order.
- **Affinity**: ``pinned`` keeps an episode on one slot from its first call to
  its end (for cache measurements; a pinned slot idles while its episode runs
  tools). ``work_conserving`` leases per call and prefers the episode's previous
  slot when it is free; otherwise it hands over any free slot and records the
  migration. The two policies are never mixed within one paired run.
- **Episode workers** may outnumber slots (default ``slots + max(1, slots // 2)``):
  they run tools, shape observations, and wait on memory, and take a slot only
  for a model call.
- **CPU pool**: one bounded process pool (never nested, never one per episode).
  **I/O pool**: bounded threads for blob reads, journal writes, and Drive copies.
  Both serve ``REQUIRED`` work (needed by a running episode) before
  ``PREFETCH`` work, and only prefetch submissions wait for room in the queue,
  so backpressure bounds memory without starving running episodes.
- **Bounded prefetch** prepares upcoming work ahead of the GPU, in order.

``snapshot()`` gives the resource sampler the counts that decide which work was ready.
"""

from __future__ import annotations

import heapq
import itertools
import multiprocessing
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

from toolsmith.evaluation.models import canonical_hash

T = TypeVar("T")
R = TypeVar("R")
AFFINITIES = ("pinned", "work_conserving")
SLOT_KINDS = ("agent", "compactor", "judge")
REQUIRED = 0
PREFETCH = 1


@dataclass(frozen=True)
class SchedulerPlan:
    """The scheduler's shape; frozen into the manifest for scored runs."""

    agent_slots: int
    compactor_slots: int = 0  # dedicated compactor slots; 0 means compaction shares agent slots
    affinity: str = "pinned"
    episode_workers: int | None = None
    cpu_workers: int = 1
    io_threads: int = 2
    prefetch_depth: int = 2
    cpu_queue_bound: int | None = None
    io_queue_bound: int | None = None

    def __post_init__(self) -> None:
        if self.agent_slots < 1:
            raise ValueError("agent_slots must be at least 1")
        if self.compactor_slots < 0 or self.cpu_workers < 1 or self.io_threads < 1 or self.prefetch_depth < 1:
            raise ValueError("pool sizes must be positive (compactor_slots may be 0)")
        if self.affinity not in AFFINITIES:
            raise ValueError(f"affinity must be one of {AFFINITIES}")

    @property
    def workers(self) -> int:
        return self.episode_workers or self.agent_slots + max(1, self.agent_slots // 2)

    @property
    def usable_slots(self) -> int:
        return self.agent_slots + self.compactor_slots

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "episode_workers": self.workers}

    def sha256(self) -> str:
        return canonical_hash(self.as_dict())


@dataclass
class SlotLease:
    """One model call's slot. ``migrated`` means the owner last used a different slot."""

    slot_id: int
    owner: str
    kind: str
    waited_s: float
    migrated: bool


@dataclass
class _Ticket:
    owner: str
    kind: str
    created: float
    event: threading.Event = field(default_factory=threading.Event)
    lease: SlotLease | None = None


class SlotPool:
    """The model-slot semaphore with explicit slot ids and an affinity policy."""

    def __init__(
        self,
        agent_slots: int,
        *,
        compactor_slots: int = 0,
        affinity: str = "pinned",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if agent_slots < 1 or compactor_slots < 0:
            raise ValueError("Need at least one agent slot")
        if affinity not in AFFINITIES:
            raise ValueError(f"affinity must be one of {AFFINITIES}")
        self.agent_slots = agent_slots
        self.compactor_slots = compactor_slots
        self.affinity = affinity
        self._clock = clock
        self._lock = threading.Lock()
        self._busy: dict[int, str] = {}  # slot -> owner, while a call runs
        self._pinned: dict[str, int] = {}  # owner -> slot (pinned policy)
        self._last: dict[str, int] = {}  # owner -> last slot used
        self._waiters: deque[_Ticket] = deque()
        self.migrations = 0
        self.leases = 0
        self.max_busy = 0
        self.wait_s = 0.0

    @property
    def usable(self) -> int:
        return self.agent_slots + self.compactor_slots

    def _candidates(self, kind: str) -> range:
        if kind == "compactor" and self.compactor_slots:
            return range(self.agent_slots, self.usable)
        return range(self.agent_slots)

    def _choose(self, ticket: _Ticket) -> int | None:
        candidates = self._candidates(ticket.kind)
        if self.affinity == "pinned" and ticket.kind == "agent":
            pinned = self._pinned.get(ticket.owner)
            if pinned is not None:
                return pinned if pinned not in self._busy else None
            taken = set(self._pinned.values())
            free = [slot for slot in candidates if slot not in self._busy and slot not in taken]
            return free[0] if free else None
        taken = set(self._pinned.values()) if self.affinity == "pinned" else set()
        free = [slot for slot in candidates if slot not in self._busy and slot not in taken]
        if not free:
            return None
        previous = self._last.get(ticket.owner)
        return previous if previous in free else free[0]

    def _dispatch(self) -> None:
        # Earlier waiters get the first chance at a slot; a waiter that cannot be served (its pinned
        # slot is busy, or no slot of its kind is free) does not block later ones.
        for ticket in list(self._waiters):
            slot = self._choose(ticket)
            if slot is None:
                continue
            self._waiters.remove(ticket)
            previous = self._last.get(ticket.owner)
            migrated = previous is not None and previous != slot
            self.migrations += migrated
            if self.affinity == "pinned" and ticket.kind == "agent":
                self._pinned[ticket.owner] = slot
            self._busy[slot] = ticket.owner
            self._last[ticket.owner] = slot
            self.leases += 1
            self.max_busy = max(self.max_busy, len(self._busy))
            waited = self._clock() - ticket.created
            self.wait_s += waited
            ticket.lease = SlotLease(slot, ticket.owner, ticket.kind, waited, migrated)
            ticket.event.set()

    def acquire(self, owner: str, kind: str = "agent", timeout: float | None = None) -> SlotLease:
        if kind not in SLOT_KINDS:
            raise ValueError(f"kind must be one of {SLOT_KINDS}")
        ticket = _Ticket(owner, kind, self._clock())
        with self._lock:
            self._waiters.append(ticket)
            self._dispatch()
        if not ticket.event.wait(timeout):
            with self._lock:
                if ticket.lease is None:
                    self._waiters.remove(ticket)
                    raise TimeoutError(f"No {kind} slot for {owner} within {timeout}s")
        assert ticket.lease is not None
        return ticket.lease

    def release(self, lease: SlotLease) -> None:
        with self._lock:
            if self._busy.get(lease.slot_id) != lease.owner:
                raise RuntimeError(f"Slot {lease.slot_id} is not leased to {lease.owner}")
            del self._busy[lease.slot_id]
            self._dispatch()

    @contextmanager
    def lease(self, owner: str, kind: str = "agent") -> Iterator[SlotLease]:
        held = self.acquire(owner, kind)
        try:
            yield held
        finally:
            self.release(held)

    def end_owner(self, owner: str) -> None:
        """The episode finished: free its pinned slot and forget its affinity."""
        with self._lock:
            self._pinned.pop(owner, None)
            self._last.pop(owner, None)
            self._dispatch()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            pinned_idle = sum(1 for slot in self._pinned.values() if slot not in self._busy)
            return {
                "slots_busy": len(self._busy),
                "slots_usable": self.usable,
                "slots_pinned_idle": pinned_idle,
                "model_waiting": len(self._waiters),
            }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "affinity": self.affinity, "usable": self.usable, "leases": self.leases,
                "migrations": self.migrations, "max_busy": self.max_busy, "total_wait_s": round(self.wait_s, 6),
            }


class PriorityPool:
    """A bounded, prioritized front for one executor: at most ``workers`` tasks run at once.

    ``REQUIRED`` tasks are taken before ``PREFETCH`` ones, FIFO within a priority.
    A ``PREFETCH`` submission waits while ``queue_bound`` tasks are already queued.
    """

    def __init__(self, name: str, executor: Callable[[], Executor], workers: int, *, queue_bound: int | None = None) -> None:
        self.name = name
        self.workers = workers
        self.queue_bound = queue_bound or 4 * workers
        self._factory = executor
        self._executor: Executor | None = None
        self._heap: list[tuple[int, int, Callable[..., Any], tuple[Any, ...], Future[Any]]] = []
        self._order = itertools.count()
        self._cond = threading.Condition()
        self._running = 0
        self._closed = False
        self._dispatchers: list[threading.Thread] = []

    def _start(self) -> None:
        if self._executor is None:
            self._executor = self._factory()
            for index in range(self.workers):
                thread = threading.Thread(target=self._dispatch, name=f"tidemark-{self.name}-{index}", daemon=True)
                thread.start()
                self._dispatchers.append(thread)

    def submit(self, fn: Callable[..., R], *args: Any, priority: int = REQUIRED) -> Future[R]:
        future: Future[R] = Future()
        with self._cond:
            if self._closed:
                raise RuntimeError(f"{self.name} pool is closed")
            self._start()
            while priority != REQUIRED and len(self._heap) >= self.queue_bound and not self._closed:
                self._cond.wait()
            heapq.heappush(self._heap, (priority, next(self._order), fn, args, future))
            self._cond.notify_all()
        return future

    def _dispatch(self) -> None:
        while True:
            with self._cond:
                while not self._heap and not self._closed:
                    self._cond.wait()
                if not self._heap:
                    return
                _, _, fn, args, future = heapq.heappop(self._heap)
                self._running += 1
                self._cond.notify_all()
            try:
                if future.set_running_or_notify_cancel():
                    assert self._executor is not None
                    inner = self._executor.submit(fn, *args)
                    try:
                        future.set_result(inner.result())
                    except BaseException as exc:  # handed to the caller through the future
                        future.set_exception(exc)
            finally:
                with self._cond:
                    self._running -= 1
                    self._cond.notify_all()

    def counts(self) -> tuple[int, int]:
        with self._cond:
            return self._running, len(self._heap)

    def close(self, wait: bool = True) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        for thread in self._dispatchers:
            thread.join(timeout=None if wait else 0)
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)


def bounded_prefetch(items: Iterable[T], prepare: Callable[[T], R], depth: int) -> Iterator[R]:
    """``prepare(item)`` for each item on a background thread, at most ``depth`` ready ahead, in order."""
    if depth < 1:
        raise ValueError("depth must be positive")
    ready: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def produce() -> None:
        try:
            for item in items:
                if stop.is_set():
                    return
                value = prepare(item)
                while not stop.is_set():
                    try:
                        ready.put(("item", value), timeout=0.1)
                        break
                    except queue.Full:
                        continue
            ready.put(("done", None))
        except BaseException as exc:
            ready.put(("error", exc))

    thread = threading.Thread(target=produce, name="tidemark-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            kind, value = ready.get()
            if kind == "done":
                return
            if kind == "error":
                raise value
            yield value
    finally:
        stop.set()


class SlotLeasedClient:
    """A ``ModelClient`` whose every call holds a slot lease and is pinned to that slot with ``id_slot``.

    ``inner`` is usually a ``SteadyhandClient``; ``pass_slot=False`` for a client
    without ``id_slot`` support. ``leases`` keeps this owner's lease history.
    """

    def __init__(self, inner: Any, slots: SlotPool, owner: str, *, kind: str = "agent", pass_slot: bool = True) -> None:
        self.inner = inner
        self.slots = slots
        self.owner = owner
        self.kind = kind
        self.pass_slot = pass_slot
        self.leases: list[SlotLease] = []

    def settings(self) -> dict[str, Any]:
        return self.inner.settings()

    def render_prompt(self, *args: Any, **kwargs: Any) -> str:
        return self.inner.render_prompt(*args, **kwargs)

    def complete(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]], *, tool_choice: Any = "auto") -> Any:
        with self.slots.lease(self.owner, self.kind) as lease:
            self.leases.append(lease)
            if self.pass_slot:
                return self.inner.complete(messages, tools, tool_choice=tool_choice, id_slot=lease.slot_id)
            return self.inner.complete(messages, tools, tool_choice=tool_choice)


def _process_pool(workers: int) -> Callable[[], Executor]:
    return lambda: ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def _thread_pool(workers: int, name: str) -> Callable[[], Executor]:
    return lambda: ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)


class Scheduler:
    """Slots, episode workers, CPU and I/O pools, and counters for one run.

    ``cpu_executor`` replaces the spawn-context process pool (tests pass a thread
    pool, since process-pool work must be picklable).
    """

    def __init__(
        self,
        plan: SchedulerPlan,
        *,
        cpu_executor: Callable[[], Executor] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.plan = plan
        self.slots = SlotPool(plan.agent_slots, compactor_slots=plan.compactor_slots, affinity=plan.affinity, clock=clock)
        self.cpu = PriorityPool("cpu", cpu_executor or _process_pool(plan.cpu_workers), plan.cpu_workers,
                                queue_bound=plan.cpu_queue_bound)
        self.io = PriorityPool("io", _thread_pool(plan.io_threads, "tidemark-io"), plan.io_threads,
                               queue_bound=plan.io_queue_bound)
        self._lock = threading.Lock()
        self._episodes_active = 0
        self._episodes_waiting = 0
        self._tools_active = 0

    def client(self, inner: Any, owner: str, *, kind: str = "agent", pass_slot: bool = True) -> SlotLeasedClient:
        return SlotLeasedClient(inner, self.slots, owner, kind=kind, pass_slot=pass_slot)

    @contextmanager
    def tool(self) -> Iterator[None]:
        """Count an active tool call (for the sampler)."""
        with self._lock:
            self._tools_active += 1
        try:
            yield
        finally:
            with self._lock:
                self._tools_active -= 1

    def submit_cpu(self, fn: Callable[..., R], *args: Any, priority: int = REQUIRED) -> Future[R]:
        return self.cpu.submit(fn, *args, priority=priority)

    def submit_io(self, fn: Callable[..., R], *args: Any, priority: int = REQUIRED) -> Future[R]:
        return self.io.submit(fn, *args, priority=priority)

    def run_episodes(
        self,
        units: Iterable[T],
        work: Callable[[T, str], R],
        *,
        key: Callable[[T], str] = str,
        prepare: Callable[[T], T] | None = None,
    ) -> list[R]:
        """Run ``work(unit, owner)`` for every unit on ``plan.workers`` episode workers, in unit order.

        ``prepare`` runs ahead on the prefetch thread (``plan.prefetch_depth`` deep). A pinned slot is
        released when its episode's ``work`` returns.
        """
        items = list(units)
        with self._lock:
            self._episodes_waiting += len(items)
        source: Iterable[T] = bounded_prefetch(items, prepare, self.plan.prefetch_depth) if prepare else items
        results: list[R] = []

        def run(unit: T) -> R:
            owner = key(unit)
            with self._lock:
                self._episodes_waiting -= 1
                self._episodes_active += 1
            try:
                return work(unit, owner)
            finally:
                self.slots.end_owner(owner)
                with self._lock:
                    self._episodes_active -= 1

        with ThreadPoolExecutor(max_workers=self.plan.workers, thread_name_prefix="tidemark-episode") as pool:
            pending: deque[Future[R]] = deque()
            for unit in source:
                # At most `workers` episodes are submitted at once, so prepared work never piles up.
                while len(pending) >= self.plan.workers:
                    results.append(pending.popleft().result())
                pending.append(pool.submit(run, unit))
            while pending:
                results.append(pending.popleft().result())
        return results

    def snapshot(self) -> dict[str, Any]:
        cpu_running, cpu_waiting = self.cpu.counts()
        io_running, io_waiting = self.io.counts()
        with self._lock:
            counts = {
                "episodes_active": self._episodes_active,
                "episodes_waiting": self._episodes_waiting,
                "tools_active": self._tools_active,
            }
        return {
            **self.slots.snapshot(),
            **counts,
            "cpu_running": cpu_running,
            "cpu_waiting": cpu_waiting,
            "cpu_capacity": self.plan.cpu_workers,
            "io_running": io_running,
            "io_waiting": io_waiting,
            "io_capacity": self.plan.io_threads,
        }

    def close(self) -> None:
        self.cpu.close()
        self.io.close()

    def __enter__(self) -> Scheduler:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = [
    "AFFINITIES",
    "PREFETCH",
    "PriorityPool",
    "REQUIRED",
    "SLOT_KINDS",
    "Scheduler",
    "SchedulerPlan",
    "SlotLease",
    "SlotLeasedClient",
    "SlotPool",
    "bounded_prefetch",
]

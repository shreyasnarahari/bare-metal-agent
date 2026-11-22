"""In-memory mock commerce store that Steadyhand episodes act on.

Each episode owns its own ``Store`` (``load_fixture()`` once, then ``clone()``
per episode), so concurrent episodes and later fault schedules never share
mutations. Ids come from counters seeded by the fixture, and every timestamp
comes from the fixture's fixed ``clock``, so replays produce identical stores.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from toolsmith.evaluation.models import canonical_hash

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "evaluation" / "data" / "steadyhand_store_v1.json"
)

OrderStatus = Literal["placed", "shipped", "delivered", "cancelled"]
TicketStatus = Literal["open", "pending", "resolved"]
ORDER_STATUSES: tuple[str, ...] = OrderStatus.__args__
TICKET_STATUSES: tuple[str, ...] = TicketStatus.__args__
CANCELLABLE_STATUSES = frozenset({"placed"})

_ID_PREFIXES = {"note": "note", "refund": "ref", "ticket": "tkt"}
_CENT = Decimal("0.01")


class NotFoundError(LookupError):
    """The store has no row with the requested id."""

    def __init__(self, entity: str, entity_id: str) -> None:
        super().__init__(f"{entity} {entity_id!r} was not found")
        self.entity = entity
        self.entity_id = entity_id


class StoreConflictError(ValueError):
    """The request is well formed but the row's current state refuses it."""


def money(value: Decimal | float | int | str) -> str:
    """Format an INR amount as a two-decimal string, the store's money format."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Amounts must be finite numbers")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Not an amount: {value!r}") from exc
    if not amount.is_finite():
        raise ValueError("Amounts must be finite numbers")
    return str(amount.quantize(_CENT))


@dataclass(frozen=True)
class Customer:
    customer_id: str
    name: str
    email: str
    tier: str


@dataclass(frozen=True)
class LineItem:
    sku: str
    name: str
    quantity: int
    unit_price_inr: str


@dataclass(frozen=True)
class Note:
    note_id: str
    text: str
    created_at: str


@dataclass
class Order:
    order_id: str
    customer_id: str
    status: str
    placed_at: str
    total_inr: str
    items: list[LineItem] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    cancelled_at: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Order:
        values = dict(row)
        values["items"] = [LineItem(**item) for item in values.get("items", [])]
        values["notes"] = [Note(**note) for note in values.get("notes", [])]
        return cls(**values)


@dataclass(frozen=True)
class Payment:
    payment_id: str
    order_id: str
    method: str
    status: str
    amount_inr: str
    created_at: str


@dataclass(frozen=True)
class StatusTransition:
    status: str
    at: str


@dataclass
class Ticket:
    ticket_id: str
    customer_id: str
    order_id: str | None
    subject: str
    status: str
    created_at: str
    transitions: list[StatusTransition] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Ticket:
        values = dict(row)
        values["transitions"] = [
            StatusTransition(**transition) for transition in values.get("transitions", [])
        ]
        return cls(**values)


@dataclass(frozen=True)
class Refund:
    refund_id: str
    order_id: str
    amount_inr: str
    reason: str
    status: str
    created_at: str


class Store:
    """Orders, customers, payments, tickets, and refunds for one episode."""

    def __init__(
        self,
        *,
        version: str,
        clock: str,
        counters: dict[str, int],
        customers: list[Customer],
        orders: list[Order],
        payments: list[Payment],
        tickets: list[Ticket],
        refunds: list[Refund],
        fixture_sha256: str | None = None,
    ) -> None:
        unknown = set(counters) - set(_ID_PREFIXES)
        missing = set(_ID_PREFIXES) - set(counters)
        if unknown or missing:
            raise ValueError(
                f"Store counters must be exactly {sorted(_ID_PREFIXES)}; "
                f"missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
        self.version = version
        self.clock = clock
        self.fixture_sha256 = fixture_sha256
        self._counters = dict(counters)
        self._customers = _index(customers, "customer_id")
        self._orders = _index(orders, "order_id")
        self._payments = _index(payments, "payment_id")
        self._tickets = _index(tickets, "ticket_id")
        self._refunds = _index(refunds, "refund_id")

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, fixture_sha256: str | None = None) -> Store:
        return cls(
            version=data["version"],
            clock=data["clock"],
            counters=data["counters"],
            customers=[Customer(**row) for row in data["customers"]],
            orders=[Order.from_dict(row) for row in data["orders"]],
            payments=[Payment(**row) for row in data["payments"]],
            tickets=[Ticket.from_dict(row) for row in data["tickets"]],
            refunds=[Refund(**row) for row in data["refunds"]],
            fixture_sha256=fixture_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        """The store's public state, in the fixture's own shape."""
        return {
            "version": self.version,
            "clock": self.clock,
            "counters": dict(self._counters),
            "customers": [asdict(row) for row in self._customers.values()],
            "orders": [asdict(row) for row in self._orders.values()],
            "payments": [asdict(row) for row in self._payments.values()],
            "tickets": [asdict(row) for row in self._tickets.values()],
            "refunds": [asdict(row) for row in self._refunds.values()],
        }

    def snapshot_sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def clone(self) -> Store:
        return deepcopy(self)

    # Reads

    def customer(self, customer_id: str) -> Customer:
        return _lookup(self._customers, "customer", customer_id)

    def order(self, order_id: str) -> Order:
        return _lookup(self._orders, "order", order_id)

    def ticket(self, ticket_id: str) -> Ticket:
        return _lookup(self._tickets, "ticket", ticket_id)

    def orders_for(self, customer_id: str, status: str | None = None) -> list[Order]:
        self.customer(customer_id)
        return [
            order
            for order in self._orders.values()
            if order.customer_id == customer_id and (status is None or order.status == status)
        ]

    def payments_for(self, order_id: str) -> list[Payment]:
        self.order(order_id)
        return [payment for payment in self._payments.values() if payment.order_id == order_id]

    def refunds_for(self, order_id: str) -> list[Refund]:
        self.order(order_id)
        return [refund for refund in self._refunds.values() if refund.order_id == order_id]

    @property
    def refunds(self) -> tuple[Refund, ...]:
        return tuple(self._refunds.values())

    @property
    def tickets(self) -> tuple[Ticket, ...]:
        return tuple(self._tickets.values())

    # Writes

    def add_refund(self, order_id: str, amount_inr: Decimal | float | str, reason: str) -> Refund:
        """Append a refund. Non-idempotent: every call creates a new refund row."""
        self.order(order_id)
        amount = Decimal(money(amount_inr))
        if amount <= 0:
            raise ValueError("Refund amount must be greater than zero")
        refund = Refund(
            refund_id=self._next_id("refund"),
            order_id=order_id,
            amount_inr=money(amount),
            reason=reason,
            status="issued",
            created_at=self.clock,
        )
        self._refunds[refund.refund_id] = refund
        return refund

    def add_ticket(self, customer_id: str, subject: str, order_id: str | None = None) -> Ticket:
        """Open a ticket under a newly allocated id. Non-idempotent."""
        self.customer(customer_id)
        if order_id is not None and self.order(order_id).customer_id != customer_id:
            raise StoreConflictError(f"Order {order_id!r} does not belong to {customer_id!r}")
        ticket = Ticket(
            ticket_id=self._next_id("ticket"),
            customer_id=customer_id,
            order_id=order_id,
            subject=subject,
            status="open",
            created_at=self.clock,
            transitions=[StatusTransition(status="open", at=self.clock)],
        )
        self._tickets[ticket.ticket_id] = ticket
        return ticket

    def add_note(self, order_id: str, text: str) -> Note:
        """Append a note to an order. Non-idempotent: repeating it duplicates the note."""
        order = self.order(order_id)
        note = Note(note_id=self._next_id("note"), text=text, created_at=self.clock)
        order.notes.append(note)
        return note

    def set_ticket_status(self, ticket_id: str, status: str) -> Ticket:
        """Move a ticket to ``status``. Idempotent: the same status adds no transition."""
        if status not in TICKET_STATUSES:
            raise ValueError(f"Ticket status must be one of {list(TICKET_STATUSES)}")
        ticket = self.ticket(ticket_id)
        if ticket.status != status:
            ticket.status = status
            ticket.transitions.append(StatusTransition(status=status, at=self.clock))
        return ticket

    def cancel_order(self, order_id: str) -> Order:
        """Cancel a placed order. Idempotent: an already-cancelled order is returned unchanged."""
        order = self.order(order_id)
        if order.status == "cancelled":
            return order
        if order.status not in CANCELLABLE_STATUSES:
            raise StoreConflictError(
                f"Order {order_id!r} is {order.status} and can no longer be cancelled"
            )
        order.status = "cancelled"
        order.cancelled_at = self.clock
        return order

    def _next_id(self, kind: str) -> str:
        number = self._counters[kind]
        self._counters[kind] = number + 1
        return f"{_ID_PREFIXES[kind]}_{number}"


def load_fixture(path: str | Path | None = None) -> Store:
    """Load the deterministic fixture. Clone the result once per episode."""
    fixture_path = Path(path) if path is not None else FIXTURE_PATH
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    digest = canonical_hash(data)
    store = Store.from_dict(data, fixture_sha256=digest)
    if store.snapshot_sha256() != digest:
        raise ValueError(f"Fixture does not round-trip through Store: {fixture_path}")
    return store


def _index(rows: list[Any], key: str) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for row in rows:
        row_id = getattr(row, key)
        if row_id in indexed:
            raise ValueError(f"Duplicate {key}: {row_id!r}")
        indexed[row_id] = row
    return indexed


def _lookup(rows: dict[str, Any], entity: str, entity_id: str) -> Any:
    try:
        return rows[entity_id]
    except KeyError:
        raise NotFoundError(entity, entity_id) from None


__all__ = [
    "CANCELLABLE_STATUSES",
    "Customer",
    "FIXTURE_PATH",
    "LineItem",
    "Note",
    "NotFoundError",
    "ORDER_STATUSES",
    "Order",
    "OrderStatus",
    "Payment",
    "Refund",
    "StatusTransition",
    "Store",
    "StoreConflictError",
    "TICKET_STATUSES",
    "Ticket",
    "TicketStatus",
    "load_fixture",
    "money",
]

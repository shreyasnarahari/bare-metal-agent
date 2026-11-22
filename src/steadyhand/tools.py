"""The ten mock tools: five reads, three non-idempotent writes, two idempotent writes.

Every handler takes the episode store as its first parameter, then typed
arguments. ``Tool.from_function`` builds the model-facing JSON Schema from the
typed arguments only, so ``store`` never appears in the schema. The registry
stores the raw function on ``Tool.handler``, and the executor binds the store:
``handler(store, **arguments)``. Handlers return plain JSON-compatible dicts and
raise ``NotFoundError`` for ids the store does not hold.

``build_registry`` attaches each tool's policy and semantic check. Store rules
that depend on state (ids exist, refund limits) are semantic checks, not
JSON Schema keywords, so the schemas stay inside the grammar subset.
"""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from typing import Annotated, Any, TypedDict

from pydantic import Field, TypeAdapter

from .policy import Effect, RetrySpec, ToolPolicy
from .registry import SemanticCheck, SteadyhandRegistry
from .store import (
    Customer,
    Note,
    Order,
    OrderStatus,
    Payment,
    Refund,
    Store,
    Ticket,
    TicketStatus,
    money,
)
from .validate import check_create_ticket, check_issue_refund, require_existing

OrderId = Annotated[str, Field(description="Order id, for example ord_1001.")]
CustomerId = Annotated[str, Field(description="Customer id, for example cus_101.")]
TicketId = Annotated[str, Field(description="Support ticket id, for example tkt_9001.")]


def get_order(store: Store, order_id: OrderId) -> dict[str, Any]:
    """Look up one order: its customer, status, line items, total in INR, and notes."""
    return asdict(store.order(order_id))


def get_customer(store: Store, customer_id: CustomerId) -> dict[str, Any]:
    """Look up one customer's name, email, and loyalty tier."""
    return asdict(store.customer(customer_id))


def list_payments(store: Store, order_id: OrderId) -> dict[str, Any]:
    """List the payments and refunds recorded for one order, with paid and refunded totals."""
    payments = store.payments_for(order_id)
    refunds = store.refunds_for(order_id)
    captured = sum(
        (Decimal(payment.amount_inr) for payment in payments if payment.status == "captured"),
        start=Decimal(0),
    )
    refunded = sum((Decimal(refund.amount_inr) for refund in refunds), start=Decimal(0))
    return {
        "order_id": order_id,
        "payments": [asdict(payment) for payment in payments],
        "refunds": [asdict(refund) for refund in refunds],
        "captured_total_inr": money(captured),
        "refunded_total_inr": money(refunded),
    }


def get_ticket(store: Store, ticket_id: TicketId) -> dict[str, Any]:
    """Look up one support ticket, including its status history."""
    return asdict(store.ticket(ticket_id))


def search_orders(
    store: Store,
    customer_id: CustomerId,
    status: Annotated[
        OrderStatus | None,
        Field(description="Only return orders in this status. Omit to return every order."),
    ] = None,
) -> dict[str, Any]:
    """Find a customer's orders, optionally filtered by order status."""
    orders = store.orders_for(customer_id, status)
    return {
        "customer_id": customer_id,
        "status": status,
        "orders": [
            {
                "order_id": order.order_id,
                "status": order.status,
                "placed_at": order.placed_at,
                "total_inr": order.total_inr,
            }
            for order in orders
        ],
    }


def issue_refund(
    store: Store,
    order_id: OrderId,
    amount_inr: Annotated[float, Field(description="Refund amount in Indian rupees, for example 120.50.")],
    reason: Annotated[str, Field(description="Short reason recorded on the refund.")],
) -> dict[str, Any]:
    """Issue a refund against an order. Every call creates a new refund."""
    return asdict(store.add_refund(order_id, amount_inr, reason))


def create_ticket(
    store: Store,
    customer_id: CustomerId,
    subject: Annotated[str, Field(description="One-line summary of the customer's problem.")],
    order_id: Annotated[
        str | None,
        Field(description="Order the ticket is about, if any, for example ord_1001."),
    ] = None,
) -> dict[str, Any]:
    """Open a new support ticket for a customer. Every call creates a new ticket."""
    return asdict(store.add_ticket(customer_id, subject, order_id))


def add_order_note(
    store: Store,
    order_id: OrderId,
    note: Annotated[str, Field(description="Text of the note to attach to the order.")],
) -> dict[str, Any]:
    """Attach an internal note to an order. Every call appends another note."""
    created = store.add_note(order_id, note)
    return {"order_id": order_id, "note": asdict(created)}


def set_ticket_status(
    store: Store,
    ticket_id: TicketId,
    status: Annotated[TicketStatus, Field(description="New ticket status.")],
) -> dict[str, Any]:
    """Set a support ticket's status. Setting the current status again changes nothing."""
    return asdict(store.set_ticket_status(ticket_id, status))


def cancel_order(store: Store, order_id: OrderId) -> dict[str, Any]:
    """Cancel an order that has not shipped. Cancelling a cancelled order changes nothing."""
    return asdict(store.cancel_order(order_id))


READ_HANDLERS = (get_order, get_customer, list_payments, get_ticket, search_orders)
NON_IDEMPOTENT_WRITE_HANDLERS = (issue_refund, create_ticket, add_order_note)
IDEMPOTENT_WRITE_HANDLERS = (set_ticket_status, cancel_order)
HANDLERS = READ_HANDLERS + NON_IDEMPOTENT_WRITE_HANDLERS + IDEMPOTENT_WRITE_HANDLERS


TOOL_TIMEOUT_S = 5.0
TRANSIENT_ERRORS = ("timeout", "rate_limit", "server_error")
SAFE_RETRY = RetrySpec(on=TRANSIENT_ERRORS, max_attempts=2, backoff="exp")

READ_POLICY = ToolPolicy(Effect.READ, TOOL_TIMEOUT_S, idempotent=True, retry=SAFE_RETRY)
IDEMPOTENT_WRITE_POLICY = ToolPolicy(
    Effect.IDEMPOTENT_WRITE, TOOL_TIMEOUT_S, idempotent=True, retry=SAFE_RETRY
)


def _non_idempotent(verify: str, alternatives: tuple[str, ...] = ()) -> ToolPolicy:
    return ToolPolicy(
        Effect.NON_IDEMPOTENT_WRITE,
        TOOL_TIMEOUT_S,
        idempotent=False,
        retry=None,
        verify=verify,
        alternatives=alternatives,
    )


class PaymentsResult(TypedDict):
    order_id: str
    payments: list[Payment]
    refunds: list[Refund]
    captured_total_inr: str
    refunded_total_inr: str


class OrderSummary(TypedDict):
    order_id: str
    status: OrderStatus
    placed_at: str
    total_inr: str


class SearchResult(TypedDict):
    customer_id: str
    status: OrderStatus | None
    orders: list[OrderSummary]


class NoteResult(TypedDict):
    order_id: str
    note: Note


def result_schema(result_type: Any) -> dict[str, Any]:
    """A closed JSON Schema for a handler's return value. Used to check fabricated values."""
    return _closed(TypeAdapter(result_type).json_schema())


def _closed(schema: Any) -> Any:
    if isinstance(schema, dict):
        closed = {key: _closed(value) for key, value in schema.items()}
        if "properties" in closed:
            closed["additionalProperties"] = False
        return closed
    if isinstance(schema, list):
        return [_closed(item) for item in schema]
    return schema


RESULT_TYPES: dict[str, Any] = {
    "get_order": Order,
    "get_customer": Customer,
    "list_payments": PaymentsResult,
    "get_ticket": Ticket,
    "search_orders": SearchResult,
    "issue_refund": Refund,
    "create_ticket": Ticket,
    "add_order_note": NoteResult,
    "set_ticket_status": Ticket,
    "cancel_order": Order,
}
RESULT_SCHEMAS: dict[str, dict[str, Any]] = {name: result_schema(kind) for name, kind in RESULT_TYPES.items()}


# (handler, policy, semantic check). Reads come first: a write's verify tool
# must already be registered.
TOOL_SPECS: tuple[tuple[Any, ToolPolicy, SemanticCheck | None], ...] = (
    (get_order, READ_POLICY, None),  # an absent order is a not_found result from the handler
    (get_customer, READ_POLICY, require_existing("customer", "customer_id")),
    (list_payments, READ_POLICY, require_existing("order", "order_id")),
    (get_ticket, READ_POLICY, require_existing("ticket", "ticket_id")),
    (search_orders, READ_POLICY, require_existing("customer", "customer_id")),
    (issue_refund, _non_idempotent("list_payments", ("refund_via_ledger",)), check_issue_refund),
    (create_ticket, _non_idempotent("get_ticket"), check_create_ticket),
    (add_order_note, _non_idempotent("get_order"), require_existing("order", "order_id")),
    (set_ticket_status, IDEMPOTENT_WRITE_POLICY, require_existing("ticket", "ticket_id")),
    (cancel_order, IDEMPOTENT_WRITE_POLICY, require_existing("order", "order_id")),
)


def build_registry() -> SteadyhandRegistry:
    """A frozen registry of the ten tools with their policies. Safe to share across workers."""
    registry = SteadyhandRegistry()
    for handler, policy, semantic in TOOL_SPECS:
        registry.register(handler, policy, semantic, RESULT_SCHEMAS[handler.__name__])
    return registry.freeze()


__all__ = [
    "HANDLERS",
    "IDEMPOTENT_WRITE_HANDLERS",
    "IDEMPOTENT_WRITE_POLICY",
    "NON_IDEMPOTENT_WRITE_HANDLERS",
    "READ_HANDLERS",
    "READ_POLICY",
    "RESULT_SCHEMAS",
    "SAFE_RETRY",
    "TOOL_SPECS",
    "TOOL_TIMEOUT_S",
    "add_order_note",
    "build_registry",
    "cancel_order",
    "create_ticket",
    "get_customer",
    "get_order",
    "get_ticket",
    "issue_refund",
    "list_payments",
    "search_orders",
    "set_ticket_status",
]

"""The ten mock tools: five reads, three non-idempotent writes, two idempotent writes.

Every handler takes the episode store as its first parameter, then typed
arguments. ``Tool.from_function`` builds the model-facing JSON Schema from the
typed arguments only, so ``store`` never appears in the schema. The registry
stores the raw function on ``Tool.handler``, and the executor binds the store:
``handler(store, **arguments)``. Handlers return plain JSON-compatible dicts and
raise ``NotFoundError`` for ids the store does not hold.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import asdict
from decimal import Decimal
from typing import Annotated, Any

from pydantic import Field

from toolsmith.registry import Tool, ToolRegistry

from .store import OrderStatus, Store, TicketStatus, money

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


def tool_from_handler(handler: Callable[..., Any]) -> Tool:
    """Build a ``Tool`` whose schema covers every parameter after ``store``."""
    signature = inspect.signature(handler, eval_str=True)
    parameters = list(signature.parameters.values())
    if not parameters or parameters[0].name != "store":
        raise TypeError(f"{handler.__name__} must take the store as its first parameter")

    def arguments_only(**arguments: Any) -> Any:
        raise NotImplementedError

    arguments_only.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=parameters[1:],
        return_annotation=inspect.Signature.empty,
    )
    schema = Tool.from_function(arguments_only, name=handler.__name__).parameters
    return Tool.from_function(handler, parameters=schema)


def build_registry() -> ToolRegistry:
    """A fresh registry of the ten tools. Safe to share across worker threads."""
    return ToolRegistry(tool_from_handler(handler) for handler in HANDLERS)


__all__ = [
    "HANDLERS",
    "IDEMPOTENT_WRITE_HANDLERS",
    "NON_IDEMPOTENT_WRITE_HANDLERS",
    "READ_HANDLERS",
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
    "tool_from_handler",
]

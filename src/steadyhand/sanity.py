"""Invariant checks on a few successful results, to catch some implicit lies.

Only tools listed in ``SANITY_CHECKS`` opt in. A check returns ``None`` when
the value passes, or a one-line problem. It never invents a corrected value,
and it is allowed to miss lies it has no invariant for.

``list_payments`` in this store records refunds separately: payments keep their
original amounts. So the checks are that the listed rows add up to the
reported totals, that the payments still in force add up to the order total
(zero for a cancelled order), and that refunds never exceed it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from .store import NotFoundError, Store

_PAISA = Decimal("0.01")  # rounding tolerance when comparing INR amounts

# What the model is told when a result fails its check.
LIST_PAYMENTS_HINT = (
    "Payment list does not match the order total. Call get_order and list_payments again before refunding."
)
ISSUE_REFUND_HINT = "The refund receipt failed a consistency check. Do not refund again; tell the user it needs checking."


def check_list_payments(store: Store, arguments: Mapping[str, Any], value: Any) -> str | None:
    """Do the listed payments and refunds add up to the reported totals and the order total?"""
    try:
        order = store.order(arguments["order_id"])
        total = Decimal(order.total_inr)
        payments = value["payments"]
        refunds = [*value["refunds"], *value.get("ledger_entries", [])]
        captured = sum((Decimal(p["amount_inr"]) for p in payments if p["status"] == "captured"), Decimal(0))
        in_force = sum((Decimal(p["amount_inr"]) for p in payments if p["status"] != "voided"), Decimal(0))
        refunded = sum((Decimal(r["amount_inr"]) for r in refunds), Decimal(0))
        reported_captured = Decimal(value["captured_total_inr"])
        reported_refunded = Decimal(value["refunded_total_inr"])
    except (KeyError, TypeError, InvalidOperation, NotFoundError) as exc:
        return f"payment list could not be read: {type(exc).__name__}"
    # A cancelled order should have no payment left in force.
    expected = Decimal(0) if order.status == "cancelled" else total
    if abs(captured - reported_captured) > _PAISA:
        return f"captured payments add up to {captured} INR but the total says {reported_captured} INR"
    if abs(refunded - reported_refunded) > _PAISA:
        return f"listed refunds add up to {refunded} INR but the total says {reported_refunded} INR"
    if abs(in_force - expected) > _PAISA:
        return f"payments add up to {in_force} INR but the order total is {expected} INR"
    if refunded > total + _PAISA:
        return f"refunds of {refunded} INR exceed the order total of {total} INR"
    return None


def check_issue_refund(store: Store, arguments: Mapping[str, Any], value: Any) -> str | None:
    """Is the refund positive, and do all refunds on the order stay within its total?"""
    try:
        amount = Decimal(value["amount_inr"])
        order = store.order(value["order_id"])
    except (KeyError, TypeError, InvalidOperation, NotFoundError) as exc:
        return f"refund receipt could not be read: {type(exc).__name__}"
    if amount <= 0:
        return f"the receipt amount {amount} INR is not positive"
    refunded = store.refunded_total(order.order_id)
    if refunded > Decimal(order.total_inr) + _PAISA:
        return f"recorded refunds of {refunded} INR exceed the order total of {order.total_inr} INR"
    return None


# (store, arguments, result value) -> a problem description, or None if the value passes.
SanityCheck = Callable[[Store, Mapping[str, Any], Any], "str | None"]
SANITY_CHECKS: dict[str, tuple[SanityCheck, str]] = {
    "list_payments": (check_list_payments, LIST_PAYMENTS_HINT),
    "issue_refund": (check_issue_refund, ISSUE_REFUND_HINT),
}


__all__ = [
    "ISSUE_REFUND_HINT",
    "LIST_PAYMENTS_HINT",
    "SANITY_CHECKS",
    "check_issue_refund",
    "check_list_payments",
]

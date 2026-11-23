"""``refund_via_ledger``: the alternative the switch strategy uses when ``issue_refund`` is down.

It lives in its own registry, owned by the recovery layer, not in the
model-facing one: the model never sees or calls it, and the model-facing
schema and policy hashes stay unchanged. ``issue_refund``'s policy has named
it as its alternative since part 2.

It is an idempotent write: every call carries the same kind of key strategy
(d) uses, and a repeat with that key returns the original entry. With Bulkhead
enabled, a switch to it gets its own policy decision: a task that does not
allow ``refund_via_ledger`` cannot reach it through recovery.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field

from .policy import Effect, ToolPolicy
from .registry import SteadyhandRegistry
from .store import LedgerEntry, Store, public_dict
from .tools import STORE_WRITE_SECURITY, TOOL_TIMEOUT_S, OrderId, result_schema
from .validate import check_issue_refund

LEDGER_POLICY = ToolPolicy(Effect.IDEMPOTENT_WRITE, TOOL_TIMEOUT_S, idempotent=True)


def refund_via_ledger(
    store: Store,
    order_id: OrderId,
    amount_inr: Annotated[float, Field(description="Refund amount in Indian rupees.")],
    reason: Annotated[str, Field(description="Short reason recorded on the ledger entry.")],
    *,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Record a refund in the payments ledger. Idempotent per key."""
    if idempotency_key is None:
        raise ValueError("refund_via_ledger needs an idempotency key")
    return public_dict(store.add_ledger_entry(order_id, amount_inr, reason, idempotency_key=idempotency_key))


def build_alternatives() -> SteadyhandRegistry:
    """The frozen registry of alternatives the switch strategy may call."""
    registry = SteadyhandRegistry(reserved_names=())
    registry.register(
        refund_via_ledger, LEDGER_POLICY, check_issue_refund, result_schema(LedgerEntry), security=STORE_WRITE_SECURITY
    )
    return registry.freeze()


def issue_refund_to_ledger(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Map ``issue_refund`` arguments onto ``refund_via_ledger``'s (the same three fields)."""
    return {
        "order_id": arguments["order_id"],
        "amount_inr": arguments["amount_inr"],
        "reason": arguments["reason"],
    }


# (failing tool, alternative) -> how to map the original arguments.
ARGUMENT_MAPS = {("issue_refund", "refund_via_ledger"): issue_refund_to_ledger}


__all__ = [
    "ARGUMENT_MAPS",
    "LEDGER_POLICY",
    "build_alternatives",
    "issue_refund_to_ledger",
    "refund_via_ledger",
]

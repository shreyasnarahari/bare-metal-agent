import json
from decimal import Decimal

import pytest

from steadyhand import NotFoundError, StoreConflictError, build_registry, load_fixture
from steadyhand.store import FIXTURE_PATH, money
from steadyhand.tools import (
    HANDLERS,
    IDEMPOTENT_WRITE_HANDLERS,
    NON_IDEMPOTENT_WRITE_HANDLERS,
    READ_HANDLERS,
    add_order_note,
    cancel_order,
    create_ticket,
    get_customer,
    get_order,
    get_ticket,
    issue_refund,
    list_payments,
    search_orders,
    set_ticket_status,
)
from toolsmith.evaluation.models import canonical_hash
from toolsmith.grammar import GBNFCompiler, resolve_tool_choice
from toolsmith.protocols import ToolProtocol


def test_fixture_load_is_deterministic_and_round_trips():
    first = load_fixture()
    second = load_fixture()

    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert first.to_dict() == second.to_dict() == raw
    assert first.fixture_sha256 == second.fixture_sha256 == canonical_hash(raw)
    assert first.snapshot_sha256() == first.fixture_sha256


def test_fixture_is_about_a_dozen_rows_with_every_entity():
    data = load_fixture().to_dict()
    counts = {name: len(data[name]) for name in ("customers", "orders", "payments", "tickets", "refunds")}

    assert all(count >= 1 for count in counts.values())
    assert 10 <= sum(counts.values()) <= 20


def test_clone_isolates_writes_from_the_original():
    original = load_fixture()
    clone = original.clone()

    issue_refund(clone, "ord_1001", 120.0, "Dented lid")

    assert original.snapshot_sha256() == original.fixture_sha256
    assert len(original.refunds) == 1
    assert len(clone.refunds) == 2
    assert clone.fixture_sha256 == original.fixture_sha256
    assert clone.snapshot_sha256() != original.snapshot_sha256()


def test_issue_refund_twice_creates_two_refunds():
    store = load_fixture().clone()

    first = issue_refund(store, "ord_1001", 120, "Dented lid")
    second = issue_refund(store, "ord_1001", 120, "Dented lid")

    assert (first["refund_id"], second["refund_id"]) == ("ref_7002", "ref_7003")
    assert first["amount_inr"] == second["amount_inr"] == "120.00"
    assert first["created_at"] == store.clock
    assert len(store.refunds_for("ord_1001")) == 2


def test_issue_refund_rejects_non_positive_amounts_without_writing():
    store = load_fixture().clone()

    with pytest.raises(ValueError):
        issue_refund(store, "ord_1001", -120, "Wrong sign")
    with pytest.raises(ValueError):
        issue_refund(store, "ord_1001", float("nan"), "Not a number")

    assert store.snapshot_sha256() == store.fixture_sha256


def test_cancel_order_twice_leaves_a_single_effect():
    store = load_fixture().clone()

    first = cancel_order(store, "ord_1003")
    after_first = store.snapshot_sha256()
    second = cancel_order(store, "ord_1003")

    assert first == second
    assert second["status"] == "cancelled"
    assert second["cancelled_at"] == store.clock
    assert store.snapshot_sha256() == after_first


def test_cancel_order_refuses_shipped_or_delivered_orders():
    store = load_fixture().clone()

    with pytest.raises(StoreConflictError, match="delivered"):
        cancel_order(store, "ord_1001")

    assert store.snapshot_sha256() == store.fixture_sha256


def test_set_ticket_status_to_the_same_value_twice_leaves_a_single_transition():
    store = load_fixture().clone()

    first = set_ticket_status(store, "tkt_9001", "pending")
    after_first = store.snapshot_sha256()
    second = set_ticket_status(store, "tkt_9001", "pending")

    assert first == second
    assert [transition["status"] for transition in second["transitions"]] == ["open", "pending"]
    assert store.snapshot_sha256() == after_first


def test_create_ticket_and_add_order_note_are_non_idempotent():
    store = load_fixture().clone()

    tickets = [create_ticket(store, "cus_101", "Late delivery", "ord_1002") for _ in range(2)]
    notes = [add_order_note(store, "ord_1003", "Call before delivery.") for _ in range(2)]

    assert [ticket["ticket_id"] for ticket in tickets] == ["tkt_9003", "tkt_9004"]
    assert all(ticket["status"] == "open" for ticket in tickets)
    assert [note["note"]["note_id"] for note in notes] == ["note_2", "note_3"]
    assert [note["text"] for note in get_order(store, "ord_1003")["notes"]] == [
        "Call before delivery.",
        "Call before delivery.",
    ]


def test_create_ticket_rejects_an_order_owned_by_another_customer():
    store = load_fixture().clone()

    with pytest.raises(StoreConflictError):
        create_ticket(store, "cus_103", "Not my order", "ord_1001")


@pytest.mark.parametrize(
    ("handler", "arguments"),
    [
        (get_order, {"order_id": "ord_9999"}),
        (get_customer, {"customer_id": "cus_999"}),
        (list_payments, {"order_id": "ord_9999"}),
        (get_ticket, {"ticket_id": "tkt_1"}),
        (search_orders, {"customer_id": "cus_999"}),
        (issue_refund, {"order_id": "ord_9999", "amount_inr": 1, "reason": "x"}),
        (cancel_order, {"order_id": "ord_9999"}),
    ],
)
def test_unknown_ids_raise_not_found(handler, arguments):
    store = load_fixture().clone()

    with pytest.raises(NotFoundError):
        handler(store, **arguments)
    assert store.snapshot_sha256() == store.fixture_sha256


def test_reads_return_json_values_and_do_not_mutate():
    store = load_fixture().clone()

    payments = list_payments(store, "ord_1005")
    orders = search_orders(store, "cus_101", "shipped")
    for value in (
        get_order(store, "ord_1001"),
        get_customer(store, "cus_101"),
        payments,
        get_ticket(store, "tkt_9002"),
        orders,
    ):
        json.dumps(value, allow_nan=False)

    assert payments["captured_total_inr"] == "5999.00"
    assert payments["refunded_total_inr"] == "500.00"
    assert [order["order_id"] for order in orders["orders"]] == ["ord_1002"]
    assert store.snapshot_sha256() == store.fixture_sha256


def test_money_formats_two_decimals():
    assert money(120) == "120.00"
    assert money(0.1 + 0.2) == "0.30"
    assert money(Decimal("5.005")) == "5.00"


def test_registry_has_the_curriculum_effect_split():
    registry = build_registry()

    assert len(registry) == 10
    assert (len(READ_HANDLERS), len(NON_IDEMPOTENT_WRITE_HANDLERS), len(IDEMPOTENT_WRITE_HANDLERS)) == (5, 3, 2)
    assert [tool.name for tool in registry] == [handler.__name__ for handler in HANDLERS]
    for tool, handler in zip(registry, HANDLERS):
        assert tool.handler is handler
        assert tool.description
        assert tool.parameters["additionalProperties"] is False
        assert "store" not in tool.parameters["properties"]


def test_set_ticket_status_schema_enumerates_statuses():
    schema = build_registry().get("set_ticket_status").parameters

    assert schema["properties"]["status"]["enum"] == ["open", "pending", "resolved"]
    assert schema["required"] == ["ticket_id", "status"]


@pytest.mark.parametrize("protocol", list(ToolProtocol))
def test_tool_schemas_compile_to_grammars(protocol):
    tools = tuple(build_registry())

    grammar = GBNFCompiler(protocol=protocol).compile_tools(
        tools,
        resolve_tool_choice(tools, "auto"),
    )

    assert set(grammar.tool_names) == {tool.name for tool in tools}

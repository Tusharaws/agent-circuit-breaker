"""Tests for the retail order-support agent's state schema + 3 tools
(Retail Demo, Story 1, subtask 1). Written before the implementation
(order_support_agent does not exist yet) -- run `pytest` to see them fail
with a collection error until order_support_agent.py exists.

Each tool is tested directly via `.invoke(...)` (the same LangChain tool
interface a real graph node calls), without building any graph at all --
proving the AC's "independently unit-testable without a full graph."
"""
import pytest

from order_support_agent import (
    FAKE_ORDERS,
    OrderSupportState,
    check_shipping_status,
    escalate_to_human,
    lookup_order,
)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


def test_state_schema_accepts_all_expected_fields():
    state: OrderSupportState = {
        "query": "ORD-10293",
        "attempts": ["ORD-10293"],
        "resolved": True,
        "order_id": "ORD-10293",
        "shipping_status": "out for delivery",
        "escalated": False,
    }
    assert state["query"] == "ORD-10293"
    assert state["attempts"] == ["ORD-10293"]
    assert state["resolved"] is True
    assert state["escalated"] is False


# ---------------------------------------------------------------------------
# The fake order database -- a real, if small, dataset, not a single record
# ---------------------------------------------------------------------------


def test_fake_orders_has_at_least_3_distinct_orders():
    assert len(FAKE_ORDERS) >= 3


# ---------------------------------------------------------------------------
# lookup_order
# ---------------------------------------------------------------------------


def test_lookup_order_finds_by_exact_order_id():
    result = lookup_order.invoke({"query": "ORD-10293"})
    assert result["found"] is True
    assert result["order_id"] == "ORD-10293"


def test_lookup_order_finds_by_email():
    result = lookup_order.invoke({"query": "jane.doe@example.com"})
    assert result["found"] is True
    assert result["order_id"] == "ORD-10293"


def test_lookup_order_finds_by_name_case_insensitive():
    result = lookup_order.invoke({"query": "jane doe"})
    assert result["found"] is True
    assert result["order_id"] == "ORD-10293"


def test_lookup_order_is_case_insensitive_on_order_id_too():
    result = lookup_order.invoke({"query": "ord-10293"})
    assert result["found"] is True
    assert result["order_id"] == "ORD-10293"


def test_lookup_order_returns_not_found_for_an_unresolvable_query():
    result = lookup_order.invoke({"query": "some garbled order reference"})
    assert result["found"] is False
    assert "message" in result


@pytest.mark.parametrize("query", ["ORD-58412", "sam.lee@example.com", "Sam Lee"])
def test_lookup_order_resolves_every_seeded_order_by_every_field(query):
    result = lookup_order.invoke({"query": query})
    assert result["found"] is True
    assert result["order_id"] == "ORD-58412"


# ---------------------------------------------------------------------------
# check_shipping_status
# ---------------------------------------------------------------------------


def test_check_shipping_status_returns_the_real_status_for_a_known_order():
    result = check_shipping_status.invoke({"order_id": "ORD-58412"})
    assert result["order_id"] == "ORD-58412"
    assert result["status"] == FAKE_ORDERS["ORD-58412"]["status"]


def test_check_shipping_status_errors_on_an_unknown_order_id():
    result = check_shipping_status.invoke({"order_id": "ORD-does-not-exist"})
    assert "error" in result


# ---------------------------------------------------------------------------
# escalate_to_human
# ---------------------------------------------------------------------------


def test_escalate_to_human_confirms_the_handoff_with_the_given_reason():
    result = escalate_to_human.invoke({"reason": "customer order cannot be located after repeated attempts"})
    assert result["escalated"] is True
    assert result["reason"] == "customer order cannot be located after repeated attempts"

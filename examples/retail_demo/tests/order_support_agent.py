"""Order-support agent fixtures for the retail demo (Retail Demo, Story 1).

A small, self-contained in-memory fake order database and the 3 tools a
LangGraph order-support agent calls against it -- no real API calls
anywhere. Exists to give the guardian system something realistic (but
fully local and deterministic) to watch for a genuine semantic loop
against: a customer whose order can't be resolved causes the agent to
retry `lookup_order` with paraphrased-but-equivalent queries rather than
exact repeats (built in a later subtask) -- the case a naive repeat
counter misses but semantic detection should catch.
"""
from typing import TypedDict

from langchain_core.tools import tool

FAKE_ORDERS = {
    "ORD-10293": {"email": "jane.doe@example.com", "name": "Jane Doe", "status": "out for delivery"},
    "ORD-58412": {"email": "sam.lee@example.com", "name": "Sam Lee", "status": "delivered"},
    "ORD-77190": {"email": "priya.singh@example.com", "name": "Priya Singh", "status": "processing"},
}


class OrderSupportState(TypedDict):
    """Shared state shape for both the looping and non-looping variants
    (built in later subtasks) -- `attempts` records every distinct search
    string tried so far, the thing that actually distinguishes a
    genuine semantic loop (many distinct attempts, never resolved) from
    normal progress (one or two attempts, then resolved)."""

    query: str
    attempts: list[str]
    resolved: bool
    order_id: str | None
    shipping_status: str | None
    escalated: bool


@tool
def lookup_order(query: str) -> dict:
    """Searches the order database by order id, customer email, or
    customer name (case-insensitive; name matches on substring)."""
    normalized = query.strip().lower()
    for order_id, record in FAKE_ORDERS.items():
        if (
            normalized == order_id.lower()
            or normalized == record["email"].lower()
            or normalized in record["name"].lower()
        ):
            return {"found": True, "order_id": order_id, "email": record["email"], "name": record["name"]}
    return {"found": False, "message": f"no order found matching {query!r}"}


@tool
def check_shipping_status(order_id: str) -> dict:
    """Returns the shipping status for a known order id."""
    record = FAKE_ORDERS.get(order_id)
    if record is None:
        return {"error": f"unknown order_id {order_id!r}"}
    return {"order_id": order_id, "status": record["status"]}


@tool
def escalate_to_human(reason: str) -> dict:
    """Hands the conversation off to a human support agent."""
    return {"escalated": True, "reason": reason}

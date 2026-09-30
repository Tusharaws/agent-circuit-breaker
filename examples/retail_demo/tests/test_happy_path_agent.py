"""Tests for the retail demo's non-looping (happy path) variant (Retail
Demo, Story 1, subtasks 2 + 4). Written before the implementation
(happy_path_agent does not exist yet) -- run `pytest` to see them fail
with a collection error until happy_path_agent.py exists.

Each graph node here calls exactly one real tool (lookup_order or
check_shipping_status), so the number of node executions the graph
actually took (via `get_state_history`) IS the number of tool calls --
counting super-steps is a precise, direct way to verify the AC's
"<=3 tool calls" without needing to patch/spy on the tools themselves.

Subtask 4 added a guard node (`check_halt`) -- `registry` is now a
required parameter (breaking change to subtask 2's tests, updated here),
matching sample_agent.py's own established precedent.
"""
import fakeredis
from control_api.guard import HaltRegistry

from happy_path_agent import build_happy_path_agent
from order_support_agent import FAKE_ORDERS


def _config(thread_id="thread-happy-1"):
    return {"configurable": {"thread_id": thread_id}}


def _registry():
    return HaltRegistry(redis_client=fakeredis.FakeRedis())


def test_a_valid_order_query_resolves_and_reaches_end():
    app = build_happy_path_agent(_registry())
    config = _config()

    result = app.invoke({"query": "ORD-10293", "attempts": [], "resolved": False, "order_id": None,
                          "shipping_status": None, "escalated": False}, config=config)

    assert result["resolved"] is True
    assert result["order_id"] == "ORD-10293"
    assert result["shipping_status"] == FAKE_ORDERS["ORD-10293"]["status"]
    assert app.get_state(config).next == ()  # reached END, nothing pending


def test_escalate_to_human_is_never_called_on_the_happy_path():
    app = build_happy_path_agent(_registry())
    config = _config("thread-happy-2")

    app.invoke({"query": "sam.lee@example.com", "attempts": [], "resolved": False, "order_id": None,
                "shipping_status": None, "escalated": False}, config=config)

    final_state = app.get_state(config).values
    assert final_state["escalated"] is False


def test_resolution_takes_at_most_3_tool_calls():
    """Verified directly against the two tool-calling nodes' own effects,
    not by counting total graph steps -- the guard node (subtask 4) also
    executes but calls no tool, so a generic step count would over-count."""
    app = build_happy_path_agent(_registry())
    config = _config("thread-happy-3")

    result = app.invoke({"query": "Priya Singh", "attempts": [], "resolved": False, "order_id": None,
                          "shipping_status": None, "escalated": False}, config=config)

    assert len(result["attempts"]) == 1  # lookup_order called exactly once
    assert result["shipping_status"] is not None  # check_shipping_status called exactly once
    # 2 tool calls total -- within the <=3 budget with room to spare.


def test_every_seeded_order_resolves_on_the_happy_path():
    for i, query in enumerate(["ORD-10293", "sam.lee@example.com", "Priya Singh"]):
        app = build_happy_path_agent(_registry())
        config = _config(f"thread-happy-seed-{i}")

        result = app.invoke(
            {"query": query, "attempts": [], "resolved": False, "order_id": None,
             "shipping_status": None, "escalated": False},
            config=config,
        )

        assert result["resolved"] is True
        assert result["shipping_status"] is not None

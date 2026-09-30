"""Proves GraphCaptureHandler captures real events from both retail demo
variants (Retail Demo, Story 1, subtask 4) -- the other half of the
subtask, alongside the guard-node/halt wiring already covered in
test_happy_path_agent.py / test_looping_agent.py.

No config-forwarding needed on the inner `tool.invoke(...)` calls inside
each node's plain Python function body -- confirmed already proven in
this exact shape by `integration-tests/tests/sample_agent.py` /
`test_full_pipeline.py` (its own `tool_node` calls `search.invoke({...})`
with no explicit config passed through, and `tool_start` still shows up in
the real captured trace) -- relying on the same, already-verified
behavior here rather than re-deriving it.
"""
import fakeredis
from control_api.guard import HaltRegistry
from happy_path_agent import build_happy_path_agent
from interceptor.graph_hooks import GraphCaptureHandler
from looping_agent import build_looping_agent

AGENT_ID = "retail-demo-agent"


class FakeDispatcher:
    def __init__(self):
        self.captured = []

    def capture(self, event):
        self.captured.append(event)


def _registry():
    return HaltRegistry(redis_client=fakeredis.FakeRedis())


def _initial_state(query):
    return {
        "query": query,
        "attempts": [],
        "resolved": False,
        "order_id": None,
        "shipping_status": None,
        "escalated": False,
    }


def test_happy_path_events_are_captured():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    app = build_happy_path_agent(_registry())
    config = {"configurable": {"thread_id": "thread-capture-happy"}, "callbacks": [handler]}

    app.invoke(_initial_state("ORD-10293"), config=config)

    assert len(dispatcher.captured) > 0
    event_types = {event.event_type for event in dispatcher.captured}
    assert "node_enter" in event_types
    assert "node_exit" in event_types
    assert "tool_start" in event_types
    assert "tool_end" in event_types
    assert all(event.trace_id == "thread-capture-happy" for event in dispatcher.captured)
    assert all(event.agent_id == AGENT_ID for event in dispatcher.captured)


def test_looping_agent_events_are_captured():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    app = build_looping_agent(_registry(), max_iterations=3, iteration_delay=0.0)
    config = {"configurable": {"thread_id": "thread-capture-loop"}, "callbacks": [handler]}

    app.invoke(_initial_state("ORD-99999"), config=config)

    event_types = {event.event_type for event in dispatcher.captured}
    assert "node_enter" in event_types
    assert "tool_start" in event_types  # lookup_order calls captured too
    assert all(event.trace_id == "thread-capture-loop" for event in dispatcher.captured)


def test_captured_tool_start_events_carry_the_real_distinct_queries():
    """Ties the two subtask-4 concerns together: the capture pipeline
    faithfully records the same distinct paraphrased attempts subtask 3's
    own tests already verify happen -- not just that *some* tool events
    fired, but that they reflect the agent's real, varying behavior."""
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    app = build_looping_agent(_registry(), max_iterations=4, iteration_delay=0.0)
    config = {"configurable": {"thread_id": "thread-capture-loop-2"}, "callbacks": [handler]}

    app.invoke(_initial_state("ORD-99999"), config=config)

    tool_start_queries = [
        event.payload.get("args", {}).get("query")
        for event in dispatcher.captured
        if event.event_type == "tool_start"
    ]
    assert len(tool_start_queries) == 4
    assert len(set(tool_start_queries)) == 4  # every captured attempt is distinct, matching subtask 3

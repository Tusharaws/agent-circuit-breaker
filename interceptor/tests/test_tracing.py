"""Phase 6: end-to-end tracing/logging. Each of the 5 services logs a
structured (JSON) line carrying the trace_id at its own choke point, so a
single trace_id is genuinely followable across all of them via logs alone.

This file covers interceptor's contribution: GraphCaptureHandler logs once
per captured event, on a dedicated "interceptor.trace" logger (DEBUG --
opt-in verbosity, since this fires on every node/llm/tool transition, not a
low-frequency decision point).
"""
import json
import logging

from interceptor.capture import CaptureDispatcher
from interceptor.graph_hooks import GraphCaptureHandler

AGENT_ID = "agent-1"


class FakeDispatcher:
    def __init__(self):
        self.captured = []

    def capture(self, event):
        self.captured.append(event)


def test_captured_event_logs_a_structured_trace_line(caplog):
    handler = GraphCaptureHandler(dispatcher=FakeDispatcher(), agent_id=AGENT_ID)

    with caplog.at_level(logging.DEBUG, logger="interceptor.trace"):
        handler.on_chain_start(
            {}, {}, run_id="run-1", metadata={"thread_id": "trace-abc", "langgraph_node": "planner"}
        )

    assert len(caplog.records) == 1
    logged = json.loads(caplog.records[0].getMessage())
    assert logged["trace_id"] == "trace-abc"
    assert logged["service"] == "interceptor"
    assert logged["stage"] == "captured"


def test_trace_log_does_not_fire_on_the_default_root_logger(caplog):
    """The trace log must be on its own dedicated logger (DEBUG-level,
    opt-in) -- it must not also spam the root/module logger at a default
    level an operator hasn't opted into."""
    handler = GraphCaptureHandler(dispatcher=FakeDispatcher(), agent_id=AGENT_ID)

    with caplog.at_level(logging.INFO):
        handler.on_chain_start(
            {}, {}, run_id="run-1", metadata={"thread_id": "trace-abc", "langgraph_node": "planner"}
        )

    assert len(caplog.records) == 0


def test_multiple_events_for_the_same_thread_all_carry_the_same_trace_id(caplog):
    handler = GraphCaptureHandler(dispatcher=FakeDispatcher(), agent_id=AGENT_ID)

    with caplog.at_level(logging.DEBUG, logger="interceptor.trace"):
        handler.on_chain_start(
            {}, {}, run_id="run-1", metadata={"thread_id": "trace-xyz", "langgraph_node": "planner"}
        )
        handler.on_llm_start(
            {}, ["prompt"], run_id="run-2", metadata={"thread_id": "trace-xyz"}
        )

    trace_ids = {json.loads(record.getMessage())["trace_id"] for record in caplog.records}
    assert trace_ids == {"trace-xyz"}

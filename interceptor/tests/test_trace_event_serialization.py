"""Proves real captured events serialize into the Phase 0 Trace Event
schema with zero JSON Schema validation errors (Phase 1, "Serialize events
into Trace Event schema"). GraphCaptureHandler already only ever produces
valid schemas.TraceEvent instances by construction (pydantic validates
eagerly) -- this file makes that guarantee explicit and portable: any
consumer can validate against TraceEvent.model_json_schema() without
needing Python or pydantic at all.

Written before running (jsonschema was already available locally, but the
dependency is declared properly in pyproject.toml regardless).
"""
import json
import threading
import time
from typing import TypedDict

import jsonschema
import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from interceptor.capture import CaptureDispatcher
from interceptor.graph_hooks import GraphCaptureHandler
from schemas.trace_event import TraceEvent

TRACE_EVENT_SCHEMA = TraceEvent.model_json_schema()


def _wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class _State(TypedDict):
    input: str
    plan: str
    tool_result: str


def _build_sample_agent():
    llm = FakeListLLM(responses=["make a plan"])

    @tool
    def fake_tool(query: str) -> str:
        """A fake tool for the serialization test."""
        return f"tool-result-for-{query}"

    def planner_node(state: _State) -> dict:
        return {"plan": llm.invoke(state["input"])}

    def tool_node(state: _State) -> dict:
        return {"tool_result": fake_tool.invoke({"query": state["plan"]})}

    graph = StateGraph(_State)
    graph.add_node("planner", planner_node)
    graph.add_node("tool", tool_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "tool")
    graph.add_edge("tool", END)
    return graph.compile()


def _run_sample_agent_and_capture() -> list[TraceEvent]:
    calls: list[TraceEvent] = []
    lock = threading.Lock()

    def recording_sink(event):
        with lock:
            calls.append(event)

    app = _build_sample_agent()
    with CaptureDispatcher(sink=recording_sink) as dispatcher:
        handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id="agent-1")
        app.invoke(
            {"input": "hello"},
            config={"callbacks": [handler], "configurable": {"thread_id": "thread-json"}},
        )
        assert _wait_until(lambda: len(calls) == 8)
    return calls


def _as_json_instance(event: TraceEvent) -> dict:
    """Round-trip through JSON text, same as what actually crosses the
    queue (queue_client / Redis carry JSON, not live Python objects)."""
    return json.loads(event.model_dump_json())


# ---------------------------------------------------------------------------
# Happy path: every real captured event validates
# ---------------------------------------------------------------------------


def test_every_real_captured_event_validates_against_trace_event_json_schema():
    events = _run_sample_agent_and_capture()
    assert len(events) == 8

    for event in events:
        jsonschema.validate(instance=_as_json_instance(event), schema=TRACE_EVENT_SCHEMA)


def test_event_with_token_usage_and_latency_present_validates():
    events = _run_sample_agent_and_capture()
    llm_end = next(e for e in events if e.event_type == "llm_end")

    assert llm_end.latency_ms is not None
    jsonschema.validate(instance=_as_json_instance(llm_end), schema=TRACE_EVENT_SCHEMA)


def test_event_with_token_usage_and_latency_absent_validates():
    events = _run_sample_agent_and_capture()
    node_enter = next(e for e in events if e.event_type == "node_enter")

    assert node_enter.token_usage is None
    assert node_enter.latency_ms is None
    jsonschema.validate(instance=_as_json_instance(node_enter), schema=TRACE_EVENT_SCHEMA)


def test_error_payload_event_validates_despite_different_payload_shape():
    """payload is free-form by design (schemas/trace_event.py), so an
    error-shaped payload from on_chain_error/on_llm_error/on_tool_error
    must validate just as well as a happy-path payload."""
    calls: list[TraceEvent] = []
    lock = threading.Lock()

    def sink(event):
        with lock:
            calls.append(event)

    with CaptureDispatcher(sink=sink) as dispatcher:
        handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id="agent-1")
        handler.on_chain_start(
            {},
            {},
            run_id="run-err",
            metadata={"thread_id": "thread-err", "langgraph_node": "planner", "langgraph_step": 0},
        )
        handler.on_chain_error(RuntimeError("boom"), run_id="run-err")
        assert _wait_until(lambda: len(calls) == 2)

    error_event = calls[-1]
    assert error_event.payload["error"] is True
    jsonschema.validate(instance=_as_json_instance(error_event), schema=TRACE_EVENT_SCHEMA)


# ---------------------------------------------------------------------------
# Negative control: the validator must actually reject broken data
# ---------------------------------------------------------------------------


def test_validator_rejects_a_deliberately_broken_sample():
    events = _run_sample_agent_and_capture()
    broken = _as_json_instance(events[0])
    del broken["trace_id"]  # required field

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=broken, schema=TRACE_EVENT_SCHEMA)


def test_validator_rejects_an_unexpected_extra_top_level_field():
    events = _run_sample_agent_and_capture()
    broken = _as_json_instance(events[0])
    broken["not_a_real_field"] = "nope"

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=broken, schema=TRACE_EVENT_SCHEMA)

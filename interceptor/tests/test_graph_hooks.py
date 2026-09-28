"""Tests for GraphCaptureHandler (Phase 1: production LangGraph node-hook
capture). Written before the implementation (interceptor.graph_hooks does
not exist yet) — run `pytest` to see them fail with a collection error
until src/interceptor/graph_hooks.py exists.

Two layers of tests:
- Unit-level: call the handler's callback methods directly with hand-built
  args (a fake dispatcher records what was captured). Fast, precise, and
  the only practical way to exercise token_usage extraction and error
  paths without a real model/tool failure.
- Integration-level: run a real, small LangGraph graph through the handler
  and a real CaptureDispatcher, proving the pieces work together for real
  (mirrors interceptor/tests/test_capture_completeness.py's approach).

Facts this file relies on, confirmed empirically first (not assumed):
- LangGraph auto-propagates `thread_id` from `config["configurable"]` into
  every callback's `metadata` dict, but does NOT propagate arbitrary custom
  keys the same way — hence agent_id is a constructor parameter here, not
  read from metadata.
- `metadata['langgraph_step']` gives LangGraph's own per-thread step
  counter — reused as `step_index` instead of hand-rolling one.
- `_end`/`_error` callback methods (on_chain_end, on_llm_end, on_tool_end,
  and their _error counterparts) do NOT receive `metadata` at all — only
  `_start` methods do. So any context needed at `_end` time (thread_id,
  step_index, node name, start time) must be cached at `_start` time,
  keyed by run_id, not re-read from metadata at `_end` time.
"""
import threading
import time
from datetime import timedelta
from typing import TypedDict

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from interceptor.capture import CaptureDispatcher
from interceptor.graph_hooks import GraphCaptureHandler
from schemas.trace_event import TraceEvent


class FakeDispatcher:
    """Records TraceEvents synchronously, no threading — keeps unit tests
    fast and deterministic. Distinct from the real CaptureDispatcher, which
    the integration tests below use instead."""

    def __init__(self):
        self.captured: list[TraceEvent] = []

    def capture(self, event):
        self.captured.append(event)


def _wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


AGENT_ID = "agent-1"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_rejects_empty_agent_id():
    with pytest.raises(ValueError):
        GraphCaptureHandler(dispatcher=FakeDispatcher(), agent_id="")


def test_construction_rejects_none_agent_id():
    with pytest.raises(ValueError):
        GraphCaptureHandler(dispatcher=FakeDispatcher(), agent_id=None)


# ---------------------------------------------------------------------------
# node_enter / node_exit (unit-level, direct callback invocation)
# ---------------------------------------------------------------------------


def test_node_enter_and_exit_emit_matching_trace_events():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-1"
    metadata = {"thread_id": "thread-1", "langgraph_node": "planner", "langgraph_step": 2}

    handler.on_chain_start({}, {}, run_id=run_id, metadata=metadata)
    handler.on_chain_end({}, run_id=run_id)

    assert len(dispatcher.captured) == 2
    enter, exit_ = dispatcher.captured
    assert enter.event_type == "node_enter"
    assert exit_.event_type == "node_exit"
    for event in (enter, exit_):
        assert event.trace_id == "thread-1"
        assert event.agent_id == AGENT_ID
        assert event.step_index == 2
        assert event.payload["node"] == "planner"
    assert exit_.latency_ms is not None
    assert exit_.latency_ms >= 0


def test_outer_graph_level_chain_is_filtered_out():
    """No `langgraph_node` in metadata => not a node, no event at all."""
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-outer"

    handler.on_chain_start({}, {}, run_id=run_id, metadata={"thread_id": "thread-1"})
    handler.on_chain_end({}, run_id=run_id)

    assert dispatcher.captured == []


def test_chain_end_for_unknown_run_id_is_a_no_op():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)

    handler.on_chain_end({}, run_id="never-started")

    assert dispatcher.captured == []


# ---------------------------------------------------------------------------
# on_chain_start: missing thread_id must never crash the caller
# ---------------------------------------------------------------------------


def test_missing_thread_id_does_not_raise_and_drops_the_event():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)

    handler.on_chain_start(
        {}, {}, run_id="run-x", metadata={"langgraph_node": "planner"}
    )  # no thread_id -- must not raise
    handler.on_chain_end({}, run_id="run-x")

    assert dispatcher.captured == []


# ---------------------------------------------------------------------------
# llm_start / llm_end
# ---------------------------------------------------------------------------


def test_llm_start_and_end_emit_matching_events_with_latency():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-llm-1"
    metadata = {"thread_id": "thread-1", "langgraph_node": "planner", "langgraph_step": 1}

    handler.on_llm_start({}, ["hello"], run_id=run_id, metadata=metadata)
    response = LLMResult(generations=[[Generation(text="hi there")]], llm_output=None)
    handler.on_llm_end(response, run_id=run_id)

    assert len(dispatcher.captured) == 2
    start, end = dispatcher.captured
    assert start.event_type == "llm_start"
    assert end.event_type == "llm_end"
    assert end.latency_ms is not None and end.latency_ms >= 0


def test_llm_end_response_payload_is_list_of_all_generations():
    """An LLM call can return multiple candidate completions (n>1) --
    capturing only the first would silently drop the rest."""
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-llm-multi"
    handler.on_llm_start({}, ["hi"], run_id=run_id, metadata={"thread_id": "t", "langgraph_step": 0})

    response = LLMResult(
        generations=[[Generation(text="candidate one"), Generation(text="candidate two")]],
        llm_output=None,
    )
    handler.on_llm_end(response, run_id=run_id)

    assert dispatcher.captured[-1].payload["response"] == ["candidate one", "candidate two"]


def test_llm_end_single_generation_is_a_one_element_list():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-llm-single"
    handler.on_llm_start({}, ["hi"], run_id=run_id, metadata={"thread_id": "t", "langgraph_step": 0})

    response = LLMResult(generations=[[Generation(text="only candidate")]], llm_output=None)
    handler.on_llm_end(response, run_id=run_id)

    assert dispatcher.captured[-1].payload["response"] == ["only candidate"]


def test_llm_end_without_token_usage_leaves_field_none():
    """Confirmed empirically: FakeListLLM's response.llm_output is None."""
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-llm-2"
    handler.on_llm_start({}, ["hi"], run_id=run_id, metadata={"thread_id": "t", "langgraph_step": 0})

    response = LLMResult(generations=[[Generation(text="ok")]], llm_output=None)
    handler.on_llm_end(response, run_id=run_id)

    assert dispatcher.captured[-1].token_usage is None


def test_llm_end_with_token_usage_extracts_it_correctly():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-llm-3"
    handler.on_llm_start({}, ["hi"], run_id=run_id, metadata={"thread_id": "t", "langgraph_step": 0})

    response = LLMResult(
        generations=[[Generation(text="ok")]],
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    )
    handler.on_llm_end(response, run_id=run_id)

    usage = dispatcher.captured[-1].token_usage
    assert usage is not None
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


def test_llm_error_emits_llm_end_with_error_payload_and_no_raise():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-llm-err"
    handler.on_llm_start({}, ["hi"], run_id=run_id, metadata={"thread_id": "t", "langgraph_step": 0})

    handler.on_llm_error(RuntimeError("model exploded"), run_id=run_id)  # must not raise

    event = dispatcher.captured[-1]
    assert event.event_type == "llm_end"
    assert event.payload["error"] is True
    assert "model exploded" in event.payload["error_message"]


# ---------------------------------------------------------------------------
# tool_start / tool_end
# ---------------------------------------------------------------------------


def test_tool_start_and_end_emit_matching_events():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-tool-1"
    metadata = {"thread_id": "thread-1", "langgraph_step": 3}

    handler.on_tool_start({"name": "search"}, "query text", run_id=run_id, metadata=metadata)
    handler.on_tool_end("search result", run_id=run_id)

    assert len(dispatcher.captured) == 2
    start, end = dispatcher.captured
    assert start.event_type == "tool_start"
    assert start.payload["tool"] == "search"
    assert end.event_type == "tool_end"


def test_tool_start_captures_structured_args_from_inputs_kwarg():
    """kwargs['inputs'] gives the real structured dict, much richer than
    the flattened input_str repr (confirmed via inline probe)."""
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-tool-args"
    metadata = {"thread_id": "thread-1", "langgraph_step": 0}

    handler.on_tool_start(
        {"name": "search"},
        "{'query': 'hello', 'limit': 3}",
        run_id=run_id,
        metadata=metadata,
        inputs={"query": "hello", "limit": 3},
    )

    payload = dispatcher.captured[-1].payload
    assert payload["args"] == {"query": "hello", "limit": 3}
    assert payload["input_str"] == "{'query': 'hello', 'limit': 3}"


def test_tool_start_args_is_none_when_inputs_kwarg_absent():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)

    handler.on_tool_start(
        {"name": "search"}, "plain string input", run_id="run-tool-no-args",
        metadata={"thread_id": "t", "langgraph_step": 0},
    )

    payload = dispatcher.captured[-1].payload
    assert payload["args"] is None
    assert payload["input_str"] == "plain string input"


def test_tool_error_emits_tool_end_with_error_payload_and_no_raise():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-tool-err"
    handler.on_tool_start({"name": "search"}, "q", run_id=run_id, metadata={"thread_id": "t", "langgraph_step": 0})

    handler.on_tool_error(RuntimeError("tool exploded"), run_id=run_id)  # must not raise

    event = dispatcher.captured[-1]
    assert event.event_type == "tool_end"
    assert event.payload["error"] is True


# ---------------------------------------------------------------------------
# chain (node) error path
# ---------------------------------------------------------------------------


def test_chain_error_emits_node_exit_with_error_payload_and_no_raise():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    run_id = "run-node-err"
    handler.on_chain_start(
        {}, {}, run_id=run_id, metadata={"thread_id": "t", "langgraph_node": "planner", "langgraph_step": 0}
    )

    handler.on_chain_error(RuntimeError("node exploded"), run_id=run_id)  # must not raise

    event = dispatcher.captured[-1]
    assert event.event_type == "node_exit"
    assert event.payload["error"] is True
    assert event.payload["node"] == "planner"


# ---------------------------------------------------------------------------
# step_index sourcing
# ---------------------------------------------------------------------------


def test_step_index_taken_from_metadata_langgraph_step():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    handler.on_chain_start(
        {}, {}, run_id="r1", metadata={"thread_id": "t", "langgraph_node": "a", "langgraph_step": 7}
    )
    assert dispatcher.captured[-1].step_index == 7


def test_step_index_defaults_to_zero_when_absent():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    handler.on_chain_start({}, {}, run_id="r1", metadata={"thread_id": "t", "langgraph_node": "a"})
    assert dispatcher.captured[-1].step_index == 0


def test_sequential_node_transitions_increase_step_index():
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)

    handler.on_chain_start({}, {}, run_id="r1", metadata={"thread_id": "t", "langgraph_node": "a", "langgraph_step": 1})
    handler.on_chain_end({}, run_id="r1")
    handler.on_chain_start({}, {}, run_id="r2", metadata={"thread_id": "t", "langgraph_node": "b", "langgraph_step": 2})
    handler.on_chain_end({}, run_id="r2")

    steps = [event.step_index for event in dispatcher.captured]
    assert steps == sorted(steps)
    assert steps[0] < steps[-1]


# ---------------------------------------------------------------------------
# Integration: a real LangGraph agent through a real CaptureDispatcher
# ---------------------------------------------------------------------------


class _State(TypedDict):
    input: str
    plan: str
    tool_result: str


def _build_sample_agent():
    llm = FakeListLLM(responses=["make a plan"])

    @tool
    def fake_tool(query: str) -> str:
        """A fake tool for the integration test."""
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


def test_real_graph_run_produces_complete_ordered_trace_via_real_dispatcher():
    calls = []
    lock = threading.Lock()

    def recording_sink(event):
        with lock:
            calls.append(event)

    app = _build_sample_agent()

    with CaptureDispatcher(sink=recording_sink) as dispatcher:
        handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
        app.invoke({"input": "hello"}, config={"callbacks": [handler], "configurable": {"thread_id": "thread-abc"}})

        assert _wait_until(lambda: len(calls) == 8)

        event_types = [event.event_type for event in calls]
        assert event_types == [
            "node_enter",
            "llm_start",
            "llm_end",
            "node_exit",
            "node_enter",
            "tool_start",
            "tool_end",
            "node_exit",
        ]
        assert all(isinstance(event, TraceEvent) for event in calls)
        assert all(event.trace_id == "thread-abc" for event in calls)
        assert all(event.agent_id == AGENT_ID for event in calls)


def test_real_graph_run_has_no_measurable_latency_added_to_agent_thread():
    """Sanity check on the AC's 'no measurable latency added' wording: the
    graph invocation itself (on the calling thread) should complete in a
    time dominated by the fake LLM/tool calls, not by capture overhead."""

    def recording_sink(event):
        pass

    app = _build_sample_agent()

    with CaptureDispatcher(sink=recording_sink) as dispatcher:
        handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
        start = time.monotonic()
        app.invoke({"input": "hello"}, config={"callbacks": [handler], "configurable": {"thread_id": "thread-perf"}})
        elapsed = time.monotonic() - start

    assert elapsed < 0.5

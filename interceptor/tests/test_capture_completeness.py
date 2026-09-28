"""Capture-completeness harness: runs a real, deterministic LangGraph agent
and asserts the exact expected sequence of captured events, in order, with
no gaps or duplicates.

Scope note: the callback-to-capture glue below (`CaptureCallbackHandler`)
is test harness code, not application code — it exists to drive the real
`CaptureDispatcher` (interceptor/src/interceptor/capture.py) with real
LangGraph/LangChain callback events. The actual LangGraph node-hook wiring
that would ship as part of `interceptor` itself (mapping callbacks into
`schemas.TraceEvent` instances) is a separate, not-yet-built feature; this
harness deliberately doesn't build or assume it.

Event order was confirmed empirically against langgraph==1.2.11 /
langchain-core==1.6.3 before being hardcoded here (see the probe used
during development) — not guessed.
"""
import threading
import time
from typing import TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from interceptor.capture import CaptureDispatcher


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
    """A minimal, deterministic 2-node LangGraph agent: planner (fake LLM
    call) -> tool (fake tool call). Deterministic so the exact captured
    sequence is reproducible run over run, in CI or locally."""
    llm = FakeListLLM(responses=["make a plan"])

    @tool
    def fake_tool(query: str) -> str:
        """A fake tool for the sample agent."""
        return f"tool-result-for-{query}"

    def planner_node(state: _State) -> dict:
        result = llm.invoke(state["input"])
        return {"plan": result}

    def tool_node(state: _State) -> dict:
        result = fake_tool.invoke({"query": state["plan"]})
        return {"tool_result": result}

    graph = StateGraph(_State)
    graph.add_node("planner", planner_node)
    graph.add_node("tool", tool_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "tool")
    graph.add_edge("tool", END)
    return graph.compile()


class CaptureCallbackHandler(BaseCallbackHandler):
    """Maps real LangGraph/LangChain callback events to
    CaptureDispatcher.capture() calls, tagged with interceptor's existing
    event_type vocabulary (schemas/config/event_types.md): node_enter,
    node_exit, llm_start, llm_end, tool_start, tool_end.

    LangGraph wraps the whole graph invocation itself as an outer chain
    (metadata has no `langgraph_node`) in addition to each node's own chain
    — only the latter should become node_enter/node_exit, so the outer one
    is filtered out.
    """

    def __init__(self, dispatcher: CaptureDispatcher) -> None:
        self._dispatcher = dispatcher
        self._node_names_by_run_id: dict[str, str] = {}

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        node_name = (metadata or {}).get("langgraph_node")
        if node_name is None:
            return
        self._node_names_by_run_id[str(run_id)] = node_name
        self._dispatcher.capture(f"node_enter:{node_name}")

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, tags=None, **kwargs):
        node_name = self._node_names_by_run_id.pop(str(run_id), None)
        if node_name is None:
            return
        self._dispatcher.capture(f"node_exit:{node_name}")

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        self._dispatcher.capture("llm_start")

    def on_llm_end(self, response, *, run_id, parent_run_id=None, tags=None, **kwargs):
        self._dispatcher.capture("llm_end")

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        self._dispatcher.capture("tool_start")

    def on_tool_end(self, output, *, run_id, parent_run_id=None, tags=None, **kwargs):
        self._dispatcher.capture("tool_end")


EXPECTED_SEQUENCE = [
    "node_enter:planner",
    "llm_start",
    "llm_end",
    "node_exit:planner",
    "node_enter:tool",
    "tool_start",
    "tool_end",
    "node_exit:tool",
]


def test_capture_contains_exact_expected_sequence_with_no_gaps_or_duplicates():
    calls = []
    lock = threading.Lock()

    def recording_sink(event):
        with lock:
            calls.append(event)

    app = _build_sample_agent()

    with CaptureDispatcher(sink=recording_sink) as dispatcher:
        handler = CaptureCallbackHandler(dispatcher)
        app.invoke({"input": "hello"}, config={"callbacks": [handler]})

        # A single exact-list equality proves order, completeness (no
        # gaps), and no duplicates all at once: any of those three failure
        # modes would make `calls` diverge from EXPECTED_SEQUENCE.
        assert _wait_until(lambda: len(calls) == len(EXPECTED_SEQUENCE))
        assert calls == EXPECTED_SEQUENCE

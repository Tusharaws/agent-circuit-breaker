"""Phase 7: LangChain's current agent API, `create_agent()`, against our
existing, UNMODIFIED GraphCaptureHandler.

The original Phase 7 backlog task ("Add LangChain AgentExecutor adapter")
targeted `langchain.agents.AgentExecutor`, which does not exist in
LangChain 1.x -- confirmed empirically (this project already pins
`langchain-core>=1,<2`/`langgraph>=1,<2` everywhere; legacy `langchain==0.x`,
the last version with `AgentExecutor`, is incompatible with that pin, not
just deprecated). Its replacement, `create_agent()`, returns a real
`langgraph.graph.state.CompiledStateGraph` -- LangGraph under the hood, not
a separate runtime. This test proves `GraphCaptureHandler` (built for
LangGraph in Phase 1, unmodified here) already captures a `create_agent()`
agent's full event trace correctly, with zero new production code.

`langchain` (the full package, not just `langchain-core`) is a test-only
dependency here -- only needed to build a real `create_agent()` fixture;
the production adapter code needed is exactly zero.
"""
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from interceptor.graph_hooks import GraphCaptureHandler


class _ToolCallingFakeModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel doesn't implement bind_tools (confirmed
    empirically -- it raises NotImplementedError), but create_agent()
    always calls it. A no-op override is enough since responses are
    pre-scripted regardless of what tools were bound."""

    def bind_tools(self, tools, **kwargs):
        return self


class FakeDispatcher:
    def __init__(self):
        self.captured = []

    def capture(self, event):
        self.captured.append(event)


@tool
def search(query: str) -> str:
    """A fake search tool."""
    return "no results"


def _build_create_agent_fixture():
    model = _ToolCallingFakeModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "call1"}]),
            AIMessage(content="done"),
        ]
    )
    return create_agent(model=model, tools=[search])


def test_graph_capture_handler_captures_a_real_create_agent_run():
    agent = _build_create_agent_fixture()
    dispatcher = FakeDispatcher()
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id="agent-1")
    config = {"configurable": {"thread_id": "thread-create-agent"}, "callbacks": [handler]}

    agent.invoke({"messages": [("user", "search for x")]}, config=config)

    assert len(dispatcher.captured) > 0
    event_types = [event.event_type for event in dispatcher.captured]
    # The model/tools node cycle create_agent() actually builds internally.
    assert "node_enter" in event_types
    assert "node_exit" in event_types
    assert "llm_start" in event_types
    assert "llm_end" in event_types
    assert "tool_start" in event_types
    assert "tool_end" in event_types
    assert all(event.trace_id == "thread-create-agent" for event in dispatcher.captured)


def test_create_agent_returns_a_real_compiled_state_graph():
    """The actual architectural fact this whole test file leans on --
    documented here as an explicit, checked assertion, not just a comment,
    so a future langchain upgrade that breaks this assumption fails loudly
    rather than silently invalidating the capture proof above."""
    from langgraph.graph.state import CompiledStateGraph

    agent = _build_create_agent_fixture()
    assert isinstance(agent, CompiledStateGraph)

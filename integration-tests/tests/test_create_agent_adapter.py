"""Phase 7: the halt half of "LangChain agent support" (re-scoped from the
original "Add LangChain AgentExecutor adapter" task -- see
interceptor/tests/test_create_agent_capture.py for why AgentExecutor is
gone in LangChain 1.x and create_agent() is LangGraph under the hood).

create_agent()'s graph shape is fixed (model/tools nodes it builds
internally) -- a customer can't insert their own "guard" node into it the
way sample_agent.py does for a hand-built LangGraph graph. Its `middleware`
extension point is the equivalent hook: `AgentMiddleware.before_model` runs
once per agent-loop iteration, the same cadence a guard node would.

Design decision (documented, not shipped as new production code): follow
the same pattern as the existing LangGraph support. check_halt() /
HaltRegistry are the shipped, generic surface (control-api, unmodified);
the customer writes a small middleware hook calling it, same as
sample_agent.py's guard_node does for a plain LangGraph node. This needs
zero new control-api code and zero new production dependencies -- langchain
is a test-only dependency here, same as interceptor's capture proof.
"""
import threading
import time

import fakeredis
from control_api.guard import HaltRegistry, check_halt
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config

ITERATION_DELAY = 0.05
MAX_ITERATIONS = 20


class _ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool
def search(query: str) -> str:
    """A fake search tool."""
    return "no results"


class HaltCheckMiddleware(AgentMiddleware):
    """The customer-side integration pattern for create_agent(), documented
    in control-api/README.md -- functionally equivalent to sample_agent.py's
    guard_node, just wired through create_agent()'s middleware hook instead
    of a hand-authored graph node, since create_agent()'s own graph shape
    isn't customer-editable."""

    def __init__(self, registry: HaltRegistry):
        super().__init__()
        self._registry = registry

    def before_model(self, state, runtime):
        thread_id = get_config()["configurable"]["thread_id"]
        check_halt(thread_id, self._registry)  # raises via interrupt() if halted
        time.sleep(ITERATION_DELAY)  # simulate real per-step work
        return None


def _build_create_agent(registry: HaltRegistry, num_tool_calls: int):
    """`num_tool_calls` rounds of tool-calling, then a final plain finish
    message (no tool_calls) so the agent loop actually terminates on its
    own when never halted."""
    responses = [
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": f"q{i}"}, "id": f"call{i}"}])
        for i in range(num_tool_calls)
    ]
    responses.append(AIMessage(content="done"))
    model = _ToolCallingFakeModel(responses=responses)
    return create_agent(
        model=model,
        tools=[search],
        middleware=[HaltCheckMiddleware(registry)],
        checkpointer=InMemorySaver(),
    )


def test_halt_check_middleware_freezes_a_real_create_agent_run():
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
    agent = _build_create_agent(registry, num_tool_calls=MAX_ITERATIONS)
    config = {"configurable": {"thread_id": "thread-create-agent-halt"}}

    result_holder = {}

    def run_agent():
        result_holder["result"] = agent.invoke({"messages": [("user", "go")]}, config=config)

    agent_thread = threading.Thread(target=run_agent)
    agent_thread.start()
    time.sleep(ITERATION_DELAY * 3)  # let a few real iterations run
    registry.request_halt("thread-create-agent-halt")
    agent_thread.join(timeout=5.0)

    assert not agent_thread.is_alive()
    # Same shape as any other LangGraph interrupt -- invoke() returns
    # normally (no exception to the caller), with __interrupt__ in the result.
    assert "__interrupt__" in result_holder["result"]

    state = agent.get_state(config)
    assert state.next == ("HaltCheckMiddleware.before_model",)


def test_thread_without_a_halt_request_completes_normally():
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
    agent = _build_create_agent(registry, num_tool_calls=3)
    config = {"configurable": {"thread_id": "thread-create-agent-normal"}}

    result = agent.invoke({"messages": [("user", "go")]}, config=config)

    assert "__interrupt__" not in result
    assert agent.get_state(config).next == ()

"""A real, cyclic LangGraph agent used across Phase 6 integration tests --
loops guard -> planner -> tool for `max_iterations` steps (or until
halted), optionally repeating the exact same tool call each time to
simulate a genuine runaway loop.

The guard node is a real, separate node (not a callback) -- confirmed in
control-api's own design task that `interrupt()` can only be called from
inside a node's own execution, never from a callback or from outside the
running thread.
"""
import time
from typing import TypedDict

from control_api.guard import HaltRegistry, check_halt
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph


class State(TypedDict):
    step: int


def build_sample_agent(
    registry: HaltRegistry,
    looping: bool,
    max_iterations: int = 20,
    iteration_delay: float = 0.05,
    tool_output: str = "no results found",
):
    """`looping=True`: the tool node repeats the exact same query every
    iteration (a genuine runaway loop). `looping=False`: the query varies
    each time (normal, progressing work). `iteration_delay` simulates
    real per-step work (an instant fake LLM/tool would otherwise complete
    all iterations before a concurrent halt request could ever land).
    `tool_output` is the tool node's return value -- overridable so a
    controlled scenario can seed real PII into a live agent run (default
    preserves the original fixed, PII-free response)."""
    llm = FakeListLLM(responses=["ok"] * max_iterations)

    @tool
    def search(query: str) -> str:
        """A fake search tool."""
        return tool_output

    def planner_node(state: State) -> dict:
        time.sleep(iteration_delay)
        llm.invoke("plan the next step")
        return {"step": state["step"] + 1}

    def tool_node(state: State) -> dict:
        query = "stuck query" if looping else f"query-{state['step']}"
        search.invoke({"query": query})
        return {}

    def guard_node(state: State, config) -> dict:
        thread_id = config["configurable"]["thread_id"]
        check_halt(thread_id, registry)  # raises via interrupt() if halted
        return {}

    def should_continue(state: State) -> str:
        return END if state["step"] >= max_iterations else "guard"

    graph = StateGraph(State)
    graph.add_node("guard", guard_node)
    graph.add_node("planner", planner_node)
    graph.add_node("tool", tool_node)
    graph.set_entry_point("guard")
    graph.add_edge("guard", "planner")
    graph.add_edge("planner", "tool")
    graph.add_conditional_edges("tool", should_continue, {"guard": "guard", END: END})

    return graph.compile(checkpointer=InMemorySaver())

"""The non-looping (happy path) variant of the retail order-support agent
(Retail Demo, Story 1, subtasks 2 + 4) -- a valid order query resolves and
the agent reaches END normally, in at most 3 tool calls, without ever
escalating to a human.

Each node calls exactly one real tool (lookup_order / check_shipping_status)
against the fake order database in order_support_agent.py -- no LLM
involved, since there's no ambiguity to reason about on this path (that's
what the looping variant is for).

The guard node + optional per-step delay (subtask 4) are the same,
already-tested primitives used everywhere else in this project --
`control_api.guard.check_halt`/`HaltRegistry`, not a reimplementation.
`iteration_delay` defaults to 0.0 here (unlike the looping variant's real
default delay) -- this path is meant to resolve fast; there's no
mid-run-halt behavior to test on a path that's over in 2 tool calls.
"""
import time

from control_api.guard import HaltRegistry, check_halt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from order_support_agent import OrderSupportState, check_shipping_status, lookup_order


def build_happy_path_agent(registry: HaltRegistry, iteration_delay: float = 0.0):
    def guard_node(state: OrderSupportState, config) -> dict:
        thread_id = config["configurable"]["thread_id"]
        check_halt(thread_id, registry)
        return {}

    def lookup_node(state: OrderSupportState) -> dict:
        time.sleep(iteration_delay)
        result = lookup_order.invoke({"query": state["query"]})
        attempts = state["attempts"] + [state["query"]]
        if result["found"]:
            return {"attempts": attempts, "resolved": True, "order_id": result["order_id"]}
        return {"attempts": attempts, "resolved": False}

    def shipping_node(state: OrderSupportState) -> dict:
        time.sleep(iteration_delay)
        result = check_shipping_status.invoke({"order_id": state["order_id"]})
        return {"shipping_status": result.get("status")}

    def after_lookup(state: OrderSupportState) -> str:
        return "shipping" if state["resolved"] else END

    graph = StateGraph(OrderSupportState)
    graph.add_node("guard", guard_node)
    graph.add_node("lookup", lookup_node)
    graph.add_node("shipping", shipping_node)
    graph.set_entry_point("guard")
    graph.add_edge("guard", "lookup")
    graph.add_conditional_edges("lookup", after_lookup, {"shipping": "shipping", END: END})
    graph.add_edge("shipping", END)

    return graph.compile(checkpointer=InMemorySaver())

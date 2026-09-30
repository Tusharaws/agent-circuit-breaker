"""The looping (non-converging, paraphrased-retry) variant of the retail
order-support agent (Retail Demo, Story 1, subtasks 3 + 4) -- given an
unresolvable order query, the agent retries `lookup_order` with textually
different but semantically equivalent paraphrases, never converges, and
never calls `escalate_to_human` even after many attempts. This is the
genuine semantic loop the guardian's SLM-based detection is meant to
catch: no two attempts are identical strings, so a naive exact-repeat
counter would miss it, but the agent still isn't making real progress.

`escalate_to_human` is deliberately never wired into this graph at all --
that's the point being demonstrated (a real, plausible failure mode: an
agent that keeps retrying instead of recognizing it should hand off),
not a placeholder to fill in later.

The guard node + real per-step delay (subtask 4) are the same, already-
tested primitives used everywhere else in this project --
`control_api.guard.check_halt`/`HaltRegistry`, not a reimplementation --
matching `integration-tests/tests/sample_agent.py`'s exact
`guard -> work -> (guard | END)` shape.
"""
import time

from control_api.guard import HaltRegistry, check_halt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from order_support_agent import OrderSupportState, lookup_order

# Different phrasings of the SAME underlying reference -- textually
# distinct, semantically equivalent. Cycling through these (not a single
# fixed rewrite) is what makes every attempt a genuinely different string.
_PARAPHRASE_TEMPLATES = [
    "order {ref}",
    "order number {ref}",
    "my order {ref}",
    "the order I placed, {ref}",
    "looking for order {ref}",
    "can you find order {ref}",
    "order id {ref}",
    "my purchase {ref}",
]


def _paraphrase(reference: str, attempt_index: int) -> str:
    template = _PARAPHRASE_TEMPLATES[attempt_index % len(_PARAPHRASE_TEMPLATES)]
    return template.format(ref=reference)


def build_looping_agent(registry: HaltRegistry, max_iterations: int = 20, iteration_delay: float = 0.05):
    """`iteration_delay` simulates real per-step work (an instant loop
    would complete before a concurrent halt request could ever land --
    see sample_agent.py's own identical reasoning)."""

    def guard_node(state: OrderSupportState, config) -> dict:
        thread_id = config["configurable"]["thread_id"]
        check_halt(thread_id, registry)  # raises via interrupt() if this thread is halted
        return {}

    def lookup_node(state: OrderSupportState) -> dict:
        time.sleep(iteration_delay)
        attempt_index = len(state["attempts"])
        query = _paraphrase(state["query"], attempt_index)
        result = lookup_order.invoke({"query": query})
        attempts = state["attempts"] + [query]
        return {"attempts": attempts, "resolved": result["found"]}

    def should_continue(state: OrderSupportState) -> str:
        if state["resolved"] or len(state["attempts"]) >= max_iterations:
            return END
        return "guard"

    graph = StateGraph(OrderSupportState)
    graph.add_node("guard", guard_node)
    graph.add_node("lookup", lookup_node)
    graph.set_entry_point("guard")
    graph.add_edge("guard", "lookup")
    graph.add_conditional_edges("lookup", should_continue, {"guard": "guard", END: END})

    return graph.compile(checkpointer=InMemorySaver())

"""Phase 6: the specific edge case ARCHITECTURE_NOTES.md flags as a known
limitation -- "latency-to-halt is bounded by node granularity, not
instant... if a single node runs a long tool call before its next
checkpoint, the halt won't land until that node completes."

Every other halt-latency test in this project (control-api's own
HALT_SLA_MS test, the Phase 6 controlled-scenarios runaway-loop test) uses
a loop spanning multiple node transitions with a guard check between each
-- the favorable case. This file tests the unfavorable one: a "loop"
living entirely inside ONE node's function body, where there is no
opportunity to check for a halt until that single node call returns.

Both fixture graphs are bespoke to this file (not added to sample_agent.py,
which models a different, already-covered shape) -- same precedent as
control-api/tests/test_guard.py's own local `_build_guarded_graph` helper.
"""
import threading
import time
from typing import TypedDict

import fakeredis
from control_api.guard import HaltRegistry, check_halt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

TOTAL_WORK_SECONDS = 1.0


class _State(TypedDict):
    step: int


def _build_single_long_node_graph(registry: HaltRegistry, node_duration: float, max_iterations: int = 5):
    """The loop lives entirely inside ONE node's function body -- e.g. one
    large batch tool call or expensive computation with no internal halt
    check. A guard node exists between iterations, but only ever runs
    BEFORE/AFTER a given long-node call, never during it -- so a halt
    requested mid-call can only ever land at the *next* guard check, after
    that in-flight call finishes."""

    def long_running_node(state: _State) -> dict:
        time.sleep(node_duration)  # the "loop," entirely inside this one call
        return {"step": state["step"] + 1}

    def guard_node(state: _State, config) -> dict:
        check_halt(config["configurable"]["thread_id"], registry)
        return {}

    def should_continue(state: _State) -> str:
        return END if state["step"] >= max_iterations else "guard"

    graph = StateGraph(_State)
    graph.add_node("guard", guard_node)
    graph.add_node("long_node", long_running_node)
    graph.set_entry_point("guard")
    graph.add_edge("guard", "long_node")
    graph.add_conditional_edges("long_node", should_continue, {"guard": "guard", END: END})
    return graph.compile(checkpointer=InMemorySaver())


def _build_fine_grained_graph(registry: HaltRegistry, node_duration: float, num_iterations: int):
    """The exact same TOTAL work as the single-long-node graph
    (node_duration * num_iterations), but split into many short node calls
    with a guard check between each -- the mitigation the granularity
    gotcha implies: keep individual nodes short."""

    def short_node(state: _State) -> dict:
        time.sleep(node_duration)
        return {"step": state["step"] + 1}

    def guard_node(state: _State, config) -> dict:
        check_halt(config["configurable"]["thread_id"], registry)
        return {}

    def should_continue(state: _State) -> str:
        return END if state["step"] >= num_iterations else "guard"

    graph = StateGraph(_State)
    graph.add_node("guard", guard_node)
    graph.add_node("short_node", short_node)
    graph.set_entry_point("guard")
    graph.add_edge("guard", "short_node")
    graph.add_conditional_edges("short_node", should_continue, {"guard": "guard", END: END})
    return graph.compile(checkpointer=InMemorySaver())


def _run_and_measure_halt_latency(app, thread_id: str, registry: HaltRegistry, halt_after: float = 0.05):
    config = {"configurable": {"thread_id": thread_id}}

    def run_agent():
        app.invoke({"step": 0}, config=config)

    thread = threading.Thread(target=run_agent)
    start = time.perf_counter()
    thread.start()

    time.sleep(halt_after)  # request the halt while the first node call is in flight
    registry.request_halt(thread_id)

    thread.join(timeout=10.0)
    elapsed_s = time.perf_counter() - start

    assert not thread.is_alive()
    return elapsed_s, app.get_state(config)


def test_halt_during_a_single_long_running_node_lands_only_after_that_node_completes():
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
    app = _build_single_long_node_graph(registry, node_duration=TOTAL_WORK_SECONDS)

    elapsed_s, state = _run_and_measure_halt_latency(app, "thread-single-long-node", registry)

    assert state.next == ("guard",)  # frozen before the guard check after the long node
    # The key property: latency-to-halt is LOWER-BOUNDED by the in-flight
    # node's own completion -- it cannot land before the node returns,
    # regardless of when mid-node the halt was requested.
    assert elapsed_s >= TOTAL_WORK_SECONDS * 0.9

    print(
        f"\nsingle long node ({TOTAL_WORK_SECONDS:.2f}s of internal work): "
        f"halt requested at ~50ms in, froze at {elapsed_s * 1000:.0f}ms -- "
        f"bounded by that one node's own completion, not sooner"
    )


def test_same_total_work_split_into_fine_grained_nodes_halts_much_sooner():
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
    num_iterations = 20
    node_duration = TOTAL_WORK_SECONDS / num_iterations  # same total work, finer granularity
    app = _build_fine_grained_graph(registry, node_duration=node_duration, num_iterations=num_iterations)

    elapsed_s, state = _run_and_measure_halt_latency(
        app, "thread-fine-grained", registry, halt_after=node_duration / 2
    )

    assert state.next == ("guard",)
    assert state.values["step"] < num_iterations  # froze well before all 20 iterations ran
    # Bounded by ONE small node's duration, not the full total work --
    # concrete, measured answer to "does splitting nodes tighten the SLA."
    assert elapsed_s < TOTAL_WORK_SECONDS / 2

    print(
        f"\nfine-grained ({num_iterations} x {node_duration * 1000:.0f}ms nodes, "
        f"same {TOTAL_WORK_SECONDS:.2f}s total work): halt requested at "
        f"~{node_duration * 500:.0f}ms in, froze at {elapsed_s * 1000:.0f}ms -- "
        f"node splitting bounds halt latency to one small node's duration"
    )

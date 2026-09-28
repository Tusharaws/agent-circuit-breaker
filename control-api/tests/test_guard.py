"""Tests for the runtime-side halt listener (Phase 5): HaltRegistry +
check_halt(). Written before the implementation (control_api.guard does
not exist yet) -- run `pytest` to see them fail with a collection error
until src/control_api/guard.py exists.

The mechanism here was verified empirically before any code was written
(see the update posted to this backlog item): calling app.update_state()
from outside a running invoke() does NOT propagate to that in-flight run
-- LangGraph doesn't re-fetch checkpointed state between node executions
within one invoke() call. interrupt() can only be called from inside a
node's own execution, triggered by a LIVE external lookup (not graph
state) a guard node performs each time it runs.
"""
import threading
import time
from typing import TypedDict

import fakeredis
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from control_api.guard import HaltRegistry, check_halt

HALT_SLA_MS = 200.0


@pytest.fixture
def registry():
    return HaltRegistry(redis_client=fakeredis.FakeRedis())


# ---------------------------------------------------------------------------
# HaltRegistry: basic behavior
# ---------------------------------------------------------------------------


def test_unknown_thread_is_not_halted(registry):
    assert registry.is_halted("no-such-thread") is False


def test_request_halt_marks_a_thread_as_halted(registry):
    registry.request_halt("thread-1")
    assert registry.is_halted("thread-1") is True


# ---------------------------------------------------------------------------
# HaltRegistry: timestamp + listing (needed by the TTL reaper)
# ---------------------------------------------------------------------------


def test_halted_at_is_none_for_an_unhalted_thread(registry):
    assert registry.halted_at("no-such-thread") is None


def test_halted_at_returns_a_tz_aware_timestamp_close_to_now(registry):
    import time as time_module
    from datetime import datetime, timezone

    before = datetime.now(timezone.utc)
    registry.request_halt("thread-1")
    after = datetime.now(timezone.utc)

    halted_at = registry.halted_at("thread-1")
    assert halted_at is not None
    assert halted_at.tzinfo is not None
    assert before <= halted_at <= after


def test_list_halted_thread_ids_returns_all_halted_threads(registry):
    registry.request_halt("thread-a")
    registry.request_halt("thread-b")

    assert set(registry.list_halted_thread_ids()) == {"thread-a", "thread-b"}


def test_list_halted_thread_ids_is_empty_when_nothing_halted(registry):
    assert registry.list_halted_thread_ids() == []


def test_clear_halt_removes_the_entry(registry):
    registry.request_halt("thread-1")
    registry.clear_halt("thread-1")

    assert registry.is_halted("thread-1") is False
    assert registry.list_halted_thread_ids() == []


def test_halt_is_isolated_per_thread(registry):
    registry.request_halt("thread-1")
    assert registry.is_halted("thread-2") is False


def test_construction_requires_either_url_or_injected_client():
    with pytest.raises(ValueError):
        HaltRegistry()


# ---------------------------------------------------------------------------
# check_halt(): the guard-node primitive
# ---------------------------------------------------------------------------


def test_check_halt_does_not_raise_when_not_halted(registry):
    check_halt("thread-1", registry)  # must not raise


def test_check_halt_attempts_to_call_interrupt_when_halted(registry):
    """interrupt() needs an active LangGraph runtime context to actually
    interrupt anything -- called outside one (as here, a narrow unit test
    with no real graph), it raises a plain RuntimeError instead (verified
    empirically). That's still proof check_halt reached the interrupt()
    call when halted; the full mechanism (a real freeze) is proven by
    test_seeded_loop_causes_the_graph_to_freeze_within_sla below, with a
    real graph and checkpointer."""
    registry.request_halt("thread-1")
    with pytest.raises(RuntimeError):
        check_halt("thread-1", registry)


# ---------------------------------------------------------------------------
# Full mechanism: a real graph, a real checkpointer, a seeded "loop"
# ---------------------------------------------------------------------------


class _State(TypedDict):
    step: int


def _build_guarded_graph(registry: HaltRegistry):
    def slow_node(state: _State) -> dict:
        time.sleep(0.05)  # simulates a node mid-execution when halt is requested
        return {"step": 1}

    def guard_node(state: _State, config) -> dict:
        thread_id = config["configurable"]["thread_id"]
        check_halt(thread_id, registry)
        return {"step": 2}

    def final_node(state: _State) -> dict:
        return {"step": 3}

    graph = StateGraph(_State)
    graph.add_node("slow_node", slow_node)
    graph.add_node("guard_node", guard_node)
    graph.add_node("final_node", final_node)
    graph.add_edge(START, "slow_node")
    graph.add_edge("slow_node", "guard_node")
    graph.add_edge("guard_node", "final_node")
    graph.add_edge("final_node", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_seeded_loop_causes_the_graph_to_freeze_within_sla(registry):
    app = _build_guarded_graph(registry)
    config = {"configurable": {"thread_id": "thread-loop"}}

    def run_graph():
        app.invoke({"step": 0}, config=config)

    thread = threading.Thread(target=run_graph)
    start = time.perf_counter()
    thread.start()

    time.sleep(0.01)  # let slow_node start
    registry.request_halt("thread-loop")  # simulated Control API detection

    thread.join(timeout=2.0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    state = app.get_state(config)
    assert state.next == ("guard_node",)  # frozen before completing guard_node
    assert state.values == {"step": 1}  # final_node never ran, step never became 3
    assert elapsed_ms < HALT_SLA_MS


def test_thread_without_a_halt_request_completes_normally(registry):
    app = _build_guarded_graph(registry)
    config = {"configurable": {"thread_id": "thread-normal"}}

    result = app.invoke({"step": 0}, config=config)

    assert result == {"step": 3}
    assert app.get_state(config).next == ()  # completed, nothing pending

"""Tests for the retail demo's looping (non-converging, paraphrased-retry)
variant (Retail Demo, Story 1, subtasks 3 + 4). Written before the
implementation (looping_agent does not exist yet) -- run `pytest` to see
them fail with a collection error until looping_agent.py exists.

This is the genuine semantic loop the guardian's SLM-based detection is
meant to catch: no two attempts are identical strings (a naive exact-
repeat counter would miss it), yet the agent never actually converges and
never calls escalate_to_human even after many attempts -- the real,
plausible failure mode a repeat-counter-only detector would sail past.

Subtask 4 added a guard node (`check_halt`) and real per-step delay --
`registry` is now a required parameter (breaking change to subtask 3's
tests, updated here), matching sample_agent.py's own established
precedent (`build_sample_agent(registry, ...)`, registry required).
"""
import threading
import time

import fakeredis
from control_api.guard import HaltRegistry
from looping_agent import build_looping_agent


def _config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _initial_state(query):
    return {
        "query": query,
        "attempts": [],
        "resolved": False,
        "order_id": None,
        "shipping_status": None,
        "escalated": False,
    }


def _registry():
    return HaltRegistry(redis_client=fakeredis.FakeRedis())


def test_unresolvable_query_never_converges_within_max_iterations():
    app = build_looping_agent(_registry(), max_iterations=10, iteration_delay=0.0)
    config = _config("thread-loop-1")

    result = app.invoke(_initial_state("ORD-99999"), config=config)

    assert result["resolved"] is False
    assert len(result["attempts"]) == 10


def test_every_attempt_is_a_textually_distinct_paraphrase_not_an_exact_repeat():
    app = build_looping_agent(_registry(), max_iterations=8, iteration_delay=0.0)
    config = _config("thread-loop-2")

    result = app.invoke(_initial_state("ORD-99999"), config=config)

    attempts = result["attempts"]
    assert len(attempts) == 8
    assert len(set(attempts)) == 8  # every attempt is a distinct string -- no exact repeats
    assert "ORD-99999" not in attempts  # not even an untouched echo of the raw query


def test_never_calls_escalate_to_human_even_after_many_attempts():
    app = build_looping_agent(_registry(), max_iterations=20, iteration_delay=0.0)
    config = _config("thread-loop-3")

    result = app.invoke(_initial_state("ORD-99999"), config=config)

    assert result["escalated"] is False


# ---------------------------------------------------------------------------
# Subtask 4: guard node + real per-step delay -- a halt requested mid-run
# actually freezes the agent before it reaches max_iterations
# ---------------------------------------------------------------------------


def test_halting_mid_run_freezes_before_reaching_max_iterations():
    registry = _registry()
    app = build_looping_agent(registry, max_iterations=30, iteration_delay=0.05)
    config = _config("thread-loop-halt")

    def run_agent():
        app.invoke(_initial_state("ORD-99999"), config=config)

    agent_thread = threading.Thread(target=run_agent)
    agent_thread.start()
    time.sleep(0.2)  # let several real iterations run
    registry.request_halt("thread-loop-halt")
    agent_thread.join(timeout=5.0)

    assert not agent_thread.is_alive()
    state = app.get_state(config)
    assert state.next == ("guard",)  # frozen before the next guard check, not completed
    assert len(state.values["attempts"]) < 30  # didn't reach the full iteration budget

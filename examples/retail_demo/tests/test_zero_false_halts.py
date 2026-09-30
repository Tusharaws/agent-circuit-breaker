"""Retail Demo, Story 2, task 2: run the non-looping (happy path) variant
against the same real pipeline (real Redis-backed stores, real SLM) and
confirm zero false halts.

Two, not one, false-halt checks: that the agent itself was never actually
halted (HaltRegistry.is_halted() stays false throughout), AND that the
real evaluator's own verdict on the resulting window is genuinely
is_loop=False -- proving the detection pipeline doesn't produce a false
positive on this conversation, not just that nobody manually triggered a
halt. Mirrors integration-tests/tests/test_controlled_scenarios.py's own
"normal run" scenario, substituting the retail happy-path agent.
"""
import pytest
from evaluator.pipeline import evaluate_thread
from evaluator.slm import SLMClient
from happy_path_agent import build_happy_path_agent
from retail_pipeline import build_wired_pipeline, initial_state

EVAL_WINDOW_SIZE = 40


@pytest.mark.integration
def test_happy_path_produces_zero_false_halts():
    queue_client, registry, dispatcher, handler = build_wired_pipeline()
    slm_client = SLMClient()

    thread_id = "thread-retail-happy"
    app = build_happy_path_agent(registry)
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]}

    result = app.invoke(initial_state("ORD-10293"), config=config)
    dispatcher.stop(timeout=2.0)

    # The agent itself completed normally, never halted.
    assert result["resolved"] is True
    assert app.get_state(config).next == ()
    assert registry.is_halted(thread_id) is False

    # And the real detection pipeline agrees -- not just "nobody
    # manually triggered a halt," but a genuine non-loop verdict.
    verdict = evaluate_thread(queue_client, slm_client, thread_id, window_size=EVAL_WINDOW_SIZE)
    print(f"\nreal verdict on retail happy path: {verdict}")
    assert verdict.is_loop is False

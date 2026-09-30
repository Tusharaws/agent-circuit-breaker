"""Retail Demo, Story 2, task 1: run the looping variant against the real
evaluator pipeline (real SLM, real /halt endpoint) and measure detect-to-
halt latency.

Mirrors integration-tests/tests/test_controlled_scenarios.py's own
runaway-loop scenario exactly (same real components, same wiring), just
substituting the retail demo's looping_agent for sample_agent's generic
"stuck query" loop -- proving the same real, closed-loop detect-and-halt
cycle holds for a realistic retail failure mode, not just an abstract one.
"""
import threading
import time

import pytest
from control_api.app import create_app
from evaluator.control_api_client import maybe_trigger_halt
from evaluator.pipeline import evaluate_thread
from evaluator.slm import SLMClient
from evaluator.window import build_window
from fastapi.testclient import TestClient
from looping_agent import build_looping_agent
from retail_pipeline import API_TOKEN, END_TO_END_HALT_SLA_MS, build_wired_pipeline, initial_state

EVAL_WINDOW_SIZE = 40


@pytest.mark.integration
def test_real_evaluator_detects_and_halts_the_retail_looping_agent():
    queue_client, registry, dispatcher, handler = build_wired_pipeline()
    slm_client = SLMClient()
    control_api_client = TestClient(
        create_app(registry, api_token=API_TOKEN), headers={"Authorization": f"Bearer {API_TOKEN}"}
    )

    thread_id = "thread-retail-loop"
    app = build_looping_agent(registry, max_iterations=30, iteration_delay=0.1)
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]}

    def run_agent():
        app.invoke(initial_state("ORD-99999"), config=config)

    agent_thread = threading.Thread(target=run_agent)
    agent_thread.start()
    time.sleep(1.0)  # let real repeated (paraphrased) content build up in the queue

    detect_start = time.perf_counter()
    window = build_window(queue_client, thread_id, window_size=EVAL_WINDOW_SIZE)
    verdict = evaluate_thread(queue_client, slm_client, thread_id, window_size=EVAL_WINDOW_SIZE)
    print(f"\nreal verdict on retail seeded loop: {verdict}")

    triggered = maybe_trigger_halt(window, verdict, control_api_client)
    assert triggered is True, f"expected a real loop verdict above threshold, got {verdict}"

    agent_thread.join(timeout=10.0)
    elapsed_ms = (time.perf_counter() - detect_start) * 1000

    assert not agent_thread.is_alive()
    state = app.get_state(config)
    assert state.next == ("guard",)  # frozen mid-loop, not completed
    assert len(state.values["attempts"]) < 30  # didn't reach the full iteration budget

    print(f"retail demo detect-to-halt: {elapsed_ms:.0f}ms (SLA {END_TO_END_HALT_SLA_MS:.0f}ms)")
    assert elapsed_ms < END_TO_END_HALT_SLA_MS

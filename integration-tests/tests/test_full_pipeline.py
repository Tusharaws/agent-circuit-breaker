"""Phase 6: wires all 5 components together against a real LangGraph
agent for the first time. Each component was individually complete and
fully tested in isolation before this task -- this is where wiring bugs
between them would surface.

A real bug WAS found and fixed while building this (not assumed away):
SanitizingSink calls its inner sink with one argument (the event), but
QueueClient.append_event(thread_id, payload) needs two -- fixed with a
new interceptor.queue_sink.make_queue_sink() adapter (see that package's
tests/README for the full story).

Mechanism verified via manual registry.request_halt() first (proving the
plumbing works end-to-end without paying real SLM inference cost), then
via the REAL evaluator pipeline (real SLM, real /halt endpoint) detecting
and halting a real seeded loop for the fully closed-loop proof.
"""
import threading
import time

import fakeredis
import pytest
from control_api.app import create_app
from control_api.guard import HaltRegistry
from evaluator.control_api_client import maybe_trigger_halt
from evaluator.pipeline import evaluate_thread
from evaluator.slm import SLMClient
from evaluator.window import build_window
from fastapi.testclient import TestClient
from interceptor.capture import CaptureDispatcher
from interceptor.graph_hooks import GraphCaptureHandler
from interceptor.queue_sink import make_queue_sink
from queue_client.client import QueueClient
from sanitizer.pipeline import SanitizingSink

from sample_agent import build_sample_agent

API_TOKEN = "test-token"
EVAL_WINDOW_SIZE = 40  # spans several loop iterations, not just one


def _build_wired_pipeline():
    """Assembles all 5 components exactly as production wiring would:
    GraphCaptureHandler -> CaptureDispatcher -> SanitizingSink ->
    make_queue_sink -> QueueClient, plus a real control-api app/registry."""
    queue_client = QueueClient(redis_client=fakeredis.FakeRedis())
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())

    sink = SanitizingSink(inner=make_queue_sink(queue_client.append_event), policy={})
    dispatcher = CaptureDispatcher(sink=sink)
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id="agent-1")

    control_api_client = TestClient(
        create_app(registry, api_token=API_TOKEN),
        headers={"Authorization": f"Bearer {API_TOKEN}"},
    )

    return queue_client, registry, dispatcher, handler, control_api_client


# ---------------------------------------------------------------------------
# Mechanism proof: manual halt request, no SLM cost -- proves the plumbing
# (capture -> sanitize -> queue -> guard node) works end-to-end
# ---------------------------------------------------------------------------


def test_all_5_components_wired_capture_events_and_honor_a_halt():
    queue_client, registry, dispatcher, handler, _control_api_client = _build_wired_pipeline()

    app = build_sample_agent(registry, looping=True, max_iterations=20, iteration_delay=0.1)
    config = {"configurable": {"thread_id": "thread-wired"}, "callbacks": [handler]}

    def run_agent():
        app.invoke({"step": 0}, config=config)

    thread = threading.Thread(target=run_agent)
    thread.start()
    time.sleep(0.3)  # let several iterations run
    registry.request_halt("thread-wired")
    thread.join(timeout=5.0)

    assert not thread.is_alive()

    state = app.get_state(config)
    assert state.next == ("guard",)  # frozen mid-loop, not completed
    assert state.values["step"] < 20  # didn't reach the end

    dispatcher.stop(timeout=1.0)  # flush in-flight captures before reading
    window = queue_client.read_window("thread-wired", limit=500)
    assert len(window) > 0  # real events actually made it through all 3 hops
    event_types = {record.payload["event_type"] for record in window}
    assert "tool_start" in event_types  # real capture content, not empty


def test_a_normal_non_looping_run_completes_and_is_never_halted():
    queue_client, registry, dispatcher, handler, _control_api_client = _build_wired_pipeline()

    app = build_sample_agent(registry, looping=False, max_iterations=5, iteration_delay=0.01)
    config = {"configurable": {"thread_id": "thread-normal"}, "callbacks": [handler]}

    result = app.invoke({"step": 0}, config=config)

    assert result["step"] == 5
    assert app.get_state(config).next == ()
    assert registry.is_halted("thread-normal") is False


# ---------------------------------------------------------------------------
# Fully closed loop: the REAL evaluator (real SLM) detects a real seeded
# loop and calls the REAL /halt endpoint -- no manual request_halt()
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_evaluator_detects_and_halts_a_real_seeded_loop():
    queue_client, registry, dispatcher, handler, control_api_client = _build_wired_pipeline()
    slm_client = SLMClient()

    app = build_sample_agent(registry, looping=True, max_iterations=30, iteration_delay=0.1)
    config = {"configurable": {"thread_id": "thread-real-loop"}, "callbacks": [handler]}

    def run_agent():
        app.invoke({"step": 0}, config=config)

    agent_thread = threading.Thread(target=run_agent)
    agent_thread.start()

    # give the agent a head start so there's real repeated content in the
    # queue before the evaluator looks -- an evaluator polling loop would
    # do this naturally; here we do one deliberate cycle for the test
    time.sleep(1.0)

    window = build_window(queue_client, "thread-real-loop", window_size=EVAL_WINDOW_SIZE)
    verdict = evaluate_thread(queue_client, slm_client, "thread-real-loop", window_size=EVAL_WINDOW_SIZE)
    print(f"\nreal verdict on seeded loop: {verdict}")

    triggered = maybe_trigger_halt(window, verdict, control_api_client)
    assert triggered is True, f"expected a real loop verdict above threshold, got {verdict}"

    agent_thread.join(timeout=10.0)
    assert not agent_thread.is_alive()

    state = app.get_state(config)
    assert state.next == ("guard",)  # halted mid-loop by the real detection
    assert state.values["step"] < 30

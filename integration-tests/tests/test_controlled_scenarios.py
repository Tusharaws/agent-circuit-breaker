"""Phase 6: controlled scenarios -- normal, PII leak, runaway loop, high
load. All 4 run through the actual full, real 5-component pipeline (real
interceptor, real sanitizer, real queue-client, real evaluator/SLM where
noted, real control-api), not isolated unit-level fakes. Every prior
Phase 4-6 test proves one component or one wiring seam; this file is the
first to put numbers on the 4 named business scenarios end-to-end. See
CONTROLLED_SCENARIOS_REPORT.md for the documented methodology and results.
"""
import json
import time

import fakeredis
import pytest
from control_api.guard import HaltRegistry
from evaluator.control_api_client import maybe_trigger_halt
from evaluator.pipeline import evaluate_thread
from evaluator.slm import SLMClient
from evaluator.window import build_window
from fastapi.testclient import TestClient
from control_api.app import create_app
from interceptor.capture import CaptureDispatcher
from interceptor.graph_hooks import GraphCaptureHandler
from interceptor.queue_sink import make_queue_sink
from queue_client.client import QueueClient
from sanitizer.ner import NerScrubber
from sanitizer.pipeline import SanitizingSink

from sample_agent import build_sample_agent

API_TOKEN = "test-token"

# Measured empirically (see CONTROLLED_SCENARIOS_REPORT.md), then fixed
# with margin -- not assumed. Detect-to-halt includes real SLM inference
# (~143ms per evaluator/EVAL_REPORT.md) plus one guard-node check cycle.
END_TO_END_HALT_SLA_MS = 2000.0

# queue.Queue.put_nowait() is microsecond-scale (see interceptor/README.md);
# a generous per-iteration budget, not a tight one, since this proves
# "no measurable penalty," not "optimal."
HIGH_LOAD_OVERHEAD_BUDGET_MS_PER_ITERATION = 5.0
HIGH_LOAD_ITERATIONS = 100


def _build_wired_pipeline(policy=None, ner_scrubber=None):
    queue_client = QueueClient(redis_client=fakeredis.FakeRedis())
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
    sink = SanitizingSink(
        inner=make_queue_sink(queue_client.append_event), policy=policy, ner_scrubber=ner_scrubber
    )
    dispatcher = CaptureDispatcher(sink=sink)
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id="agent-1")
    return queue_client, registry, dispatcher, handler


# ---------------------------------------------------------------------------
# Scenario 1: normal run, zero false halts
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_scenario_normal_run_has_zero_false_halts():
    slm_client = SLMClient()

    for i in range(3):
        thread_id = f"thread-scenario-normal-{i}"
        queue_client, registry, dispatcher, handler = _build_wired_pipeline()
        app = build_sample_agent(registry, looping=False, max_iterations=10, iteration_delay=0.0)
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]}

        result = app.invoke({"step": 0}, config=config)
        dispatcher.stop(timeout=2.0)

        assert result["step"] == 10  # completed, never halted
        assert app.get_state(config).next == ()

        verdict = evaluate_thread(queue_client, slm_client, thread_id, window_size=10)
        print(f"\nscenario normal run {i}: verdict={verdict}")
        assert verdict.is_loop is False, f"false halt on a genuinely normal run: {verdict}"


# ---------------------------------------------------------------------------
# Scenario 2: seeded PII is fully scrubbed end-to-end
# ---------------------------------------------------------------------------

SEEDED_PII = {
    "email": "john.doe@example.com",
    "phone": "415-555-2671",
    "ssn": "123-45-6789",
    "name": "John Doe",
}
SEEDED_PII_TOOL_OUTPUT = (
    f"Contact {SEEDED_PII['name']} at {SEEDED_PII['email']} or {SEEDED_PII['phone']}, "
    f"SSN {SEEDED_PII['ssn']}"
)


def test_scenario_pii_leak_is_fully_scrubbed_end_to_end():
    queue_client, registry, dispatcher, handler = _build_wired_pipeline(ner_scrubber=NerScrubber())

    app = build_sample_agent(
        registry, looping=False, max_iterations=5, iteration_delay=0.0, tool_output=SEEDED_PII_TOOL_OUTPUT
    )
    config = {"configurable": {"thread_id": "thread-scenario-pii"}, "callbacks": [handler]}
    app.invoke({"step": 0}, config=config)
    dispatcher.stop(timeout=2.0)

    window = queue_client.read_window("thread-scenario-pii", limit=500)
    assert len(window) > 0  # a real run actually happened, not an empty one

    raw_dump = json.dumps([record.payload for record in window])
    for label, value in SEEDED_PII.items():
        assert value not in raw_dump, f"seeded {label} leaked through to the queue: {value!r}"


# ---------------------------------------------------------------------------
# Scenario 3: a seeded runaway loop triggers a real halt within SLA
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_scenario_runaway_loop_triggers_halt_within_sla():
    import threading

    queue_client, registry, dispatcher, handler = _build_wired_pipeline()
    slm_client = SLMClient()
    control_api_client = TestClient(
        create_app(registry, api_token=API_TOKEN), headers={"Authorization": f"Bearer {API_TOKEN}"}
    )

    thread_id = "thread-scenario-loop"
    app = build_sample_agent(registry, looping=True, max_iterations=30, iteration_delay=0.1)
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]}

    def run_agent():
        app.invoke({"step": 0}, config=config)

    agent_thread = threading.Thread(target=run_agent)
    agent_thread.start()
    time.sleep(1.0)  # let real repeated content build up in the queue

    detect_start = time.perf_counter()
    window = build_window(queue_client, thread_id, window_size=40)
    verdict = evaluate_thread(queue_client, slm_client, thread_id, window_size=40)
    triggered = maybe_trigger_halt(window, verdict, control_api_client)
    assert triggered is True, f"expected a real loop verdict above threshold, got {verdict}"

    agent_thread.join(timeout=10.0)
    elapsed_ms = (time.perf_counter() - detect_start) * 1000

    assert not agent_thread.is_alive()
    state = app.get_state(config)
    assert state.next == ("guard",)  # halted mid-loop
    assert state.values["step"] < 30

    print(f"\nscenario runaway loop: detect-to-halt = {elapsed_ms:.0f}ms (SLA {END_TO_END_HALT_SLA_MS:.0f}ms)")
    assert elapsed_ms < END_TO_END_HALT_SLA_MS


# ---------------------------------------------------------------------------
# Scenario 4: high load shows no measurable latency penalty
# ---------------------------------------------------------------------------


def test_scenario_high_load_shows_no_measurable_latency_penalty():
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())

    baseline_app = build_sample_agent(
        registry, looping=False, max_iterations=HIGH_LOAD_ITERATIONS, iteration_delay=0.0
    )
    baseline_config = {"configurable": {"thread_id": "thread-scenario-load-baseline"}}
    start = time.perf_counter()
    baseline_app.invoke({"step": 0}, config=baseline_config)
    baseline_elapsed_s = time.perf_counter() - start

    queue_client, _registry, dispatcher, handler = _build_wired_pipeline()
    captured_app = build_sample_agent(
        registry, looping=False, max_iterations=HIGH_LOAD_ITERATIONS, iteration_delay=0.0
    )
    captured_config = {
        "configurable": {"thread_id": "thread-scenario-load-captured"},
        "callbacks": [handler],
    }
    start = time.perf_counter()
    captured_app.invoke({"step": 0}, config=captured_config)
    captured_elapsed_s = time.perf_counter() - start
    dispatcher.stop(timeout=5.0)

    window = queue_client.read_window("thread-scenario-load-captured", limit=2000)
    assert len(window) > 0  # capture actually happened under this load

    overhead_ms_per_iteration = (
        (captured_elapsed_s - baseline_elapsed_s) / HIGH_LOAD_ITERATIONS
    ) * 1000
    print(
        f"\nscenario high load: baseline={baseline_elapsed_s * 1000:.1f}ms, "
        f"captured={captured_elapsed_s * 1000:.1f}ms, "
        f"overhead/iteration={overhead_ms_per_iteration:.4f}ms "
        f"(budget {HIGH_LOAD_OVERHEAD_BUDGET_MS_PER_ITERATION}ms)"
    )
    assert overhead_ms_per_iteration < HIGH_LOAD_OVERHEAD_BUDGET_MS_PER_ITERATION

"""Phase 6: end-to-end tracing/logging -- the cross-package proof.

Each of the 5 packages was given its own dedicated trace logger
(interceptor.trace, sanitizer.trace, queue_client.trace, evaluator.trace,
control_api.trace) and its own unit tests proving it logs correctly in
isolation. None of those prove the actual AC on their own: that a SINGLE
trace_id is genuinely followable ACROSS all 5 in one real run. This test
wires all 5 together (same production wiring as test_full_pipeline.py) and
proves exactly that.

Uses a fake SLM for evaluator (not the real one from test_full_pipeline.py's
@pytest.mark.integration test) so this stays in the fast, no-model-load
tier -- the evaluator's own real-SLM behavior is already proven elsewhere;
here we only need its trace log to fire and carry the right trace_id.
"""
import json
import logging
import threading
import time

import fakeredis
from control_api.guard import HaltRegistry, check_halt
from evaluator.pipeline import evaluate_thread
from interceptor.capture import CaptureDispatcher
from interceptor.graph_hooks import GraphCaptureHandler
from interceptor.queue_sink import make_queue_sink
from queue_client.client import QueueClient
from sanitizer.pipeline import SanitizingSink

from sample_agent import build_sample_agent

TRACE_ID = "thread-tracing-e2e"
ALL_TRACE_LOGGERS = [
    "interceptor.trace",
    "sanitizer.trace",
    "queue_client.trace",
    "evaluator.trace",
    "control_api.trace",
]


class FakeSLM:
    def generate(self, prompt, max_tokens=200):
        return '{"is_loop": true, "confidence": 0.9, "reason": "runaway_loop"}'


def test_a_single_trace_id_is_followable_across_all_5_services(caplog):
    queue_client = QueueClient(redis_client=fakeredis.FakeRedis())
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
    sink = SanitizingSink(inner=make_queue_sink(queue_client.append_event), policy={})
    dispatcher = CaptureDispatcher(sink=sink)
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id="agent-1")

    app = build_sample_agent(registry, looping=True, max_iterations=20, iteration_delay=0.1)
    config = {"configurable": {"thread_id": TRACE_ID}, "callbacks": [handler]}

    with caplog.at_level(logging.DEBUG):
        for logger_name in ALL_TRACE_LOGGERS:
            caplog.set_level(logging.DEBUG, logger=logger_name)

        def run_agent():
            app.invoke({"step": 0}, config=config)

        agent_thread = threading.Thread(target=run_agent)
        agent_thread.start()
        time.sleep(0.3)  # let several iterations run: interceptor, sanitizer,
        # queue_client, and control_api's check_halt all fire during this
        registry.request_halt(TRACE_ID)
        agent_thread.join(timeout=5.0)
        assert not agent_thread.is_alive()

        dispatcher.stop(timeout=1.0)  # flush in-flight captures before reading

        # evaluator isn't in the live-agent loop -- run it directly against
        # the same trace_id's queued events, same as a real evaluator poll would
        evaluate_thread(queue_client, FakeSLM(), TRACE_ID, window_size=40)

    records_by_logger = {name: [] for name in ALL_TRACE_LOGGERS}
    for record in caplog.records:
        if record.name in records_by_logger:
            records_by_logger[record.name].append(json.loads(record.getMessage()))

    for logger_name, records in records_by_logger.items():
        assert records, f"expected at least one {logger_name} record, got none"
        trace_ids = {r["trace_id"] for r in records}
        assert trace_ids == {TRACE_ID}, f"{logger_name} carried unexpected trace_id(s): {trace_ids}"

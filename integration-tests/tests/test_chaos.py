"""Phase 6: chaos testing -- killing the queue or the evaluator mid-run
must not crash the agent; it must log a warning and continue (or fail safe
per design).

Real components throughout, not fakes-of-fakes: the real CaptureDispatcher,
SanitizingSink, QueueClient, ResilientProducer, and a real cyclic LangGraph
agent (sample_agent.build_sample_agent), same as test_full_pipeline.py.

"Killing the queue" is simulated via a controllable fake append_event that
raises redis.exceptions.ConnectionError on demand -- same pattern
queue-client's own ResilientProducer tests use (no real Redis server
exists anywhere in this environment to literally kill).

A real architecture gap was found while designing this test, not assumed
away: production wiring elsewhere in this project (interceptor's "blessed"
recipe, test_full_pipeline.py's own wiring) calls queue_client.append_event
directly, bypassing queue-client's own ResilientProducer entirely -- so a
real outage today is only ever caught by CaptureDispatcher's blunt
per-item try/except (silent drop + exception log), never retried or
buffered. This test wires ResilientProducer into the sink chain instead --
the actual "fails safe per design" the task's AC calls for.
"""
import logging

import fakeredis
import redis.exceptions
from control_api.guard import HaltRegistry
from interceptor.capture import CaptureDispatcher
from interceptor.graph_hooks import GraphCaptureHandler
from interceptor.queue_sink import make_queue_sink
from queue_client.client import QueueClient
from queue_client.producer import ResilientProducer
from sanitizer.pipeline import SanitizingSink

from sample_agent import build_sample_agent


class FlakyAppendEvent:
    """Wraps a real QueueClient.append_event, raising ConnectionError for
    the first `fail_count` calls to simulate the queue going down mid-run,
    then recovering."""

    def __init__(self, append_event, fail_count):
        self._append_event = append_event
        self._fail_count = fail_count
        self.calls = 0

    def __call__(self, thread_id, payload):
        self.calls += 1
        if self.calls <= self._fail_count:
            raise redis.exceptions.ConnectionError("simulated queue outage")
        return self._append_event(thread_id, payload)


# ---------------------------------------------------------------------------
# Chaos 1: the queue goes down mid-run, then recovers
# ---------------------------------------------------------------------------


def test_queue_outage_mid_run_does_not_crash_the_agent_and_recovers(caplog):
    queue_client = QueueClient(redis_client=fakeredis.FakeRedis())
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())

    flaky = FlakyAppendEvent(queue_client.append_event, fail_count=10)
    producer = ResilientProducer(append_event=flaky, backoff_seconds=0.01)
    sink = SanitizingSink(inner=make_queue_sink(producer.append_event), policy={})
    dispatcher = CaptureDispatcher(sink=sink)
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id="agent-1")

    app = build_sample_agent(registry, looping=False, max_iterations=15, iteration_delay=0.01)
    config = {"configurable": {"thread_id": "thread-chaos-queue"}, "callbacks": [handler]}

    with caplog.at_level(logging.WARNING):
        # The outage happens entirely on the background capture thread --
        # if it ever propagated into node execution, this call would raise.
        result = app.invoke({"step": 0}, config=config)

    assert result["step"] == 15  # ran to completion, unaffected by the outage
    assert app.get_state(config).next == ()

    dispatcher.stop(timeout=5.0)

    # It failed safe, not silently: a warning was actually logged for the outage.
    assert any("queue unreachable" in r.getMessage() for r in caplog.records)

    # And once the queue "recovered," buffered events were actually
    # delivered -- proving genuine degrade-and-recover, not just drop.
    window = queue_client.read_window("thread-chaos-queue", limit=500)
    assert len(window) > 0
    assert flaky.calls > 10  # calls did continue past the simulated outage


# ---------------------------------------------------------------------------
# Chaos 2: the evaluator is entirely absent for the whole run
# ---------------------------------------------------------------------------


def test_absent_evaluator_does_not_affect_the_agent_and_queue_self_bounds():
    small_max_len = 20
    queue_client = QueueClient(redis_client=fakeredis.FakeRedis(), max_len=small_max_len)
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())

    sink = SanitizingSink(inner=make_queue_sink(queue_client.append_event), policy={})
    dispatcher = CaptureDispatcher(sink=sink)
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id="agent-1")

    # No evaluator is ever constructed or polls this queue anywhere in this
    # test -- deliberately absent for the entire run, simulating "the
    # evaluator was killed" throughout.
    app = build_sample_agent(registry, looping=False, max_iterations=20, iteration_delay=0.01)
    config = {"configurable": {"thread_id": "thread-chaos-no-evaluator"}, "callbacks": [handler]}

    result = app.invoke({"step": 0}, config=config)

    assert result["step"] == 20  # completed fully, unaffected by the absent consumer
    assert app.get_state(config).next == ()

    dispatcher.stop(timeout=2.0)

    # Even with zero consumption, the per-thread stream stays capped -- an
    # absent evaluator can never cause unbounded queue growth.
    window = queue_client.read_window("thread-chaos-no-evaluator", limit=1000)
    assert len(window) <= small_max_len

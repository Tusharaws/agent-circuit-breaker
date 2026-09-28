"""Phase 6: end-to-end tracing/logging -- evaluator's contribution.

evaluate_thread logs once per window evaluated, on a dedicated
"evaluator.trace" logger. INFO level (not DEBUG like the high-frequency
per-event loggers elsewhere in the pipeline) -- a verdict is a meaningful,
low-frequency decision point, not a per-node pass-through.
"""
import json
import logging
from datetime import datetime, timezone

import fakeredis
import pytest

from queue_client.client import QueueClient
from schemas.trace_event import TraceEvent

from evaluator.pipeline import evaluate_thread


def _seed_event(queue_client, thread_id, step_index, event_type, payload):
    event = TraceEvent(
        trace_id=thread_id,
        agent_id="agent-1",
        step_index=step_index,
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        payload=payload,
    )
    queue_client.append_event(thread_id, event.model_dump(mode="json"))


def _seed_normal_thread(queue_client, thread_id):
    _seed_event(queue_client, thread_id, 0, "node_enter", {"node": "planner"})
    _seed_event(queue_client, thread_id, 1, "tool_start", {"tool": "search", "args": {"query": "lookup-a"}})


def _seed_looping_thread(queue_client, thread_id):
    for i in range(5):
        _seed_event(
            queue_client, thread_id, i, "tool_start", {"tool": "search", "args": {"query": "stuck query"}}
        )


class FakeSLM:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt, max_tokens=200):
        return self.response


@pytest.fixture
def queue_client():
    return QueueClient(redis_client=fakeredis.FakeRedis())


def test_filtered_window_logs_a_structured_verdict_line(queue_client, caplog):
    _seed_normal_thread(queue_client, "thread-normal")

    with caplog.at_level(logging.INFO, logger="evaluator.trace"):
        evaluate_thread(queue_client, None, "thread-normal")

    trace_logs = [json.loads(r.getMessage()) for r in caplog.records if r.name == "evaluator.trace"]
    assert len(trace_logs) == 1
    assert trace_logs[0]["trace_id"] == "thread-normal"
    assert trace_logs[0]["service"] == "evaluator"
    assert trace_logs[0]["stage"] == "evaluated"
    assert trace_logs[0]["escalated"] is False
    assert trace_logs[0]["is_loop"] is False


def test_escalated_window_logs_a_structured_verdict_line(queue_client, caplog):
    _seed_looping_thread(queue_client, "thread-loop")
    fake_slm = FakeSLM('{"is_loop": true, "confidence": 0.95, "reason": "repeated_tool_calls"}')

    with caplog.at_level(logging.INFO, logger="evaluator.trace"):
        evaluate_thread(queue_client, fake_slm, "thread-loop")

    trace_logs = [json.loads(r.getMessage()) for r in caplog.records if r.name == "evaluator.trace"]
    assert len(trace_logs) == 1
    assert trace_logs[0]["trace_id"] == "thread-loop"
    assert trace_logs[0]["escalated"] is True
    assert trace_logs[0]["is_loop"] is True
    assert trace_logs[0]["confidence"] == 0.95

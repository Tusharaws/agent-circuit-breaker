"""Phase 6: end-to-end tracing/logging -- queue-client's contribution.

append_event and read_window each log once, on a dedicated
"queue_client.trace" logger (DEBUG), carrying thread_id (the same value as
trace_id elsewhere in the pipeline, by design).
"""
import json
import logging

import fakeredis
import pytest

from queue_client.client import QueueClient


@pytest.fixture
def client():
    return QueueClient(redis_client=fakeredis.FakeRedis())


def test_append_event_logs_a_structured_trace_line(client, caplog):
    with caplog.at_level(logging.DEBUG, logger="queue_client.trace"):
        event_id = client.append_event("trace-abc", {"step": 1})

    assert len(caplog.records) == 1
    logged = json.loads(caplog.records[0].getMessage())
    assert logged["trace_id"] == "trace-abc"
    assert logged["service"] == "queue_client"
    assert logged["stage"] == "enqueued"
    assert logged["event_id"] == event_id


def test_read_window_logs_a_structured_trace_line(client, caplog):
    client.append_event("trace-abc", {"step": 1})
    client.append_event("trace-abc", {"step": 2})

    with caplog.at_level(logging.DEBUG, logger="queue_client.trace"):
        client.read_window("trace-abc")

    trace_logs = [json.loads(r.getMessage()) for r in caplog.records if r.name == "queue_client.trace"]
    read_logs = [r for r in trace_logs if r["stage"] == "window_read"]
    assert len(read_logs) == 1
    assert read_logs[0]["trace_id"] == "trace-abc"
    assert read_logs[0]["count"] == 2

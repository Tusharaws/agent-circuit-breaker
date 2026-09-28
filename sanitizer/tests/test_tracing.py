"""Phase 6: end-to-end tracing/logging -- sanitizer's contribution.

SanitizingSink logs once per event it processes, on a dedicated
"sanitizer.trace" logger (DEBUG), carrying the same trace_id the event
itself carries -- whether the event is a schemas.TraceEvent (trace_id
attribute) or a plain dict (queue_client's generic payload shape,
"trace_id" key). An event with no trace_id anywhere (neither shape) is
still sanitized and forwarded; it just isn't trace-logged, since there's
nothing to correlate it by.
"""
import json
import logging

from sanitizer.pipeline import SanitizingSink

POLICY = {"email": "mask"}


class RecordingSink:
    def __init__(self):
        self.received = []

    def __call__(self, event):
        self.received.append(event)


def test_dict_event_with_trace_id_logs_a_structured_trace_line(caplog):
    sink = SanitizingSink(inner=RecordingSink(), policy=POLICY)

    with caplog.at_level(logging.DEBUG, logger="sanitizer.trace"):
        sink({"trace_id": "trace-abc", "payload": {"note": "clean"}})

    assert len(caplog.records) == 1
    logged = json.loads(caplog.records[0].getMessage())
    assert logged["trace_id"] == "trace-abc"
    assert logged["service"] == "sanitizer"
    assert logged["stage"] == "sanitized"


def test_trace_event_with_trace_id_attribute_logs_a_structured_trace_line(caplog):
    from datetime import datetime, timezone

    from schemas.trace_event import TraceEvent

    event = TraceEvent(
        trace_id="trace-xyz",
        agent_id="agent-1",
        step_index=0,
        timestamp=datetime.now(timezone.utc),
        event_type="llm_end",
        payload={"response": ["ok"]},
    )
    sink = SanitizingSink(inner=RecordingSink(), policy=POLICY)

    with caplog.at_level(logging.DEBUG, logger="sanitizer.trace"):
        sink(event)

    assert len(caplog.records) == 1
    logged = json.loads(caplog.records[0].getMessage())
    assert logged["trace_id"] == "trace-xyz"


def test_event_with_no_trace_id_is_still_forwarded_without_logging(caplog):
    inner = RecordingSink()
    sink = SanitizingSink(inner=inner, policy=POLICY)

    with caplog.at_level(logging.DEBUG, logger="sanitizer.trace"):
        sink({"payload": {"note": "clean"}})

    assert len(caplog.records) == 0
    assert inner.received == [{"payload": {"note": "clean"}}]

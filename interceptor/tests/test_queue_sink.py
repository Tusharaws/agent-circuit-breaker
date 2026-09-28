"""Tests for make_queue_sink() (Phase 6 wiring bug fix). Written before
the implementation -- run `pytest` to see them fail with a collection
error until src/interceptor/queue_sink.py exists.

Found as a real integration bug: SanitizingSink calls its inner sink with
one argument (the event), but QueueClient.append_event(thread_id,
payload) needs two. Each side was individually correct and fully tested
in isolation; nothing exercised the seam between them until wired
together for real in the Phase 6 integration test.
"""
from datetime import datetime, timezone

from schemas.trace_event import TraceEvent

from interceptor.queue_sink import make_queue_sink, make_sanitized_queue_sink


def _event(trace_id="thread-1", event_type="node_enter", payload=None):
    return TraceEvent(
        trace_id=trace_id,
        agent_id="agent-1",
        step_index=0,
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        payload=payload or {"node": "planner"},
    )


class RecordingAppendEvent:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, thread_id: str, payload: dict) -> str:
        self.calls.append((thread_id, payload))
        return "event-1"


def test_sink_calls_append_event_with_thread_id_and_payload_separately():
    append_event = RecordingAppendEvent()
    sink = make_queue_sink(append_event)
    event = _event(trace_id="thread-abc")

    sink(event)

    assert len(append_event.calls) == 1
    thread_id, payload = append_event.calls[0]
    assert thread_id == "thread-abc"
    assert payload["trace_id"] == "thread-abc"
    assert payload["event_type"] == "node_enter"


def test_thread_id_comes_from_the_events_trace_id_field():
    append_event = RecordingAppendEvent()
    sink = make_queue_sink(append_event)

    sink(_event(trace_id="thread-x"))
    sink(_event(trace_id="thread-y"))

    thread_ids = [call[0] for call in append_event.calls]
    assert thread_ids == ["thread-x", "thread-y"]


def test_payload_is_the_full_dumped_trace_event():
    append_event = RecordingAppendEvent()
    sink = make_queue_sink(append_event)
    event = _event(payload={"prompts": ["hello"]})

    sink(event)

    _, payload = append_event.calls[0]
    assert payload == event.model_dump(mode="json")


def test_sink_works_with_a_real_queue_client():
    """Integration-shaped unit test: proves the adapter genuinely closes
    the gap against the real QueueClient.append_event, not just a fake
    matching its signature."""
    import fakeredis

    from queue_client.client import QueueClient

    queue_client = QueueClient(redis_client=fakeredis.FakeRedis())
    sink = make_queue_sink(queue_client.append_event)

    sink(_event(trace_id="thread-real"))

    window = queue_client.read_window("thread-real")
    assert len(window) == 1
    assert window[0].payload["trace_id"] == "thread-real"


# ---------------------------------------------------------------------------
# make_sanitized_queue_sink: the blessed, single-call production wiring
# (Phase 7 security review -- see sanitizer/SECURITY_REVIEW.md's
# "no structural guarantee against bypassing sanitization" finding)
# ---------------------------------------------------------------------------


def test_sanitized_sink_scrubs_pii_before_it_reaches_append_event():
    append_event = RecordingAppendEvent()
    sink = make_sanitized_queue_sink(append_event, policy={"email": "mask"})
    event = _event(payload={"note": "contact alice@example.com"})

    sink(event)

    _, payload = append_event.calls[0]
    assert "alice@example.com" not in str(payload)


def test_sanitized_sink_still_routes_by_the_events_own_trace_id():
    append_event = RecordingAppendEvent()
    sink = make_sanitized_queue_sink(append_event, policy={})

    sink(_event(trace_id="thread-sanitized"))

    thread_id, _ = append_event.calls[0]
    assert thread_id == "thread-sanitized"


def test_sanitized_sink_works_end_to_end_with_a_real_queue_client():
    import fakeredis

    from queue_client.client import QueueClient

    queue_client = QueueClient(redis_client=fakeredis.FakeRedis())
    sink = make_sanitized_queue_sink(queue_client.append_event, policy={"email": "mask"})

    sink(_event(trace_id="thread-real-sanitized", payload={"note": "email me at bob@example.com"}))

    window = queue_client.read_window("thread-real-sanitized")
    assert len(window) == 1
    assert "bob@example.com" not in str(window[0].payload)

"""Tests for detection window assembly (Phase 4). Written before the
implementation (evaluator.window does not exist yet) -- run `pytest` to
see them fail with a collection error until src/evaluator/window.py
exists.
"""
from datetime import datetime, timezone

import fakeredis
import pytest

from queue_client.client import QueueClient
from schemas.trace_event import TraceEvent

from evaluator.window import DEFAULT_WINDOW_SIZE, build_window


@pytest.fixture
def queue_client():
    return QueueClient(redis_client=fakeredis.FakeRedis())


def _seed_event(queue_client, thread_id, step_index, event_type="node_enter"):
    event = TraceEvent(
        trace_id=thread_id,
        agent_id="agent-1",
        step_index=step_index,
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        payload={"step": step_index},
    )
    queue_client.append_event(thread_id, event.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Basic assembly and ordering
# ---------------------------------------------------------------------------


def test_window_assembles_events_oldest_to_newest(queue_client):
    for i in range(5):
        _seed_event(queue_client, "thread-1", i)

    window = build_window(queue_client, "thread-1", window_size=10)

    assert [e.step_index for e in window.events] == [0, 1, 2, 3, 4]
    assert window.thread_id == "thread-1"


def test_window_events_are_real_validated_trace_events(queue_client):
    _seed_event(queue_client, "thread-1", 0)

    window = build_window(queue_client, "thread-1", window_size=10)

    assert isinstance(window.events[0], TraceEvent)
    assert window.events[0].event_type == "node_enter"


def test_window_carries_event_ids_parallel_to_events(queue_client):
    """HaltSignal.triggering_window needs the actual Redis-assigned event
    ids -- TraceEvent itself doesn't carry them (that's
    TraceEventRecord's job), so the window must capture them separately."""
    for i in range(3):
        _seed_event(queue_client, "thread-1", i)

    window = build_window(queue_client, "thread-1", window_size=10)

    assert len(window.event_ids) == len(window.events) == 3
    assert all(isinstance(event_id, str) and event_id for event_id in window.event_ids)
    assert len(set(window.event_ids)) == 3  # all distinct


# ---------------------------------------------------------------------------
# window_size limiting
# ---------------------------------------------------------------------------


def test_window_respects_window_size_limit(queue_client):
    for i in range(20):
        _seed_event(queue_client, "thread-1", i)

    window = build_window(queue_client, "thread-1", window_size=5)

    assert len(window.events) == 5
    assert [e.step_index for e in window.events] == [15, 16, 17, 18, 19]  # most recent 5


def test_thread_with_fewer_events_than_window_size_returns_all_of_them(queue_client):
    for i in range(3):
        _seed_event(queue_client, "thread-1", i)

    window = build_window(queue_client, "thread-1", window_size=10)

    assert len(window.events) == 3


def test_default_window_size_is_used_when_not_specified(queue_client):
    for i in range(DEFAULT_WINDOW_SIZE + 5):
        _seed_event(queue_client, "thread-1", i)

    window = build_window(queue_client, "thread-1")

    assert len(window.events) == DEFAULT_WINDOW_SIZE


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_unknown_thread_returns_an_empty_window(queue_client):
    window = build_window(queue_client, "no-such-thread")
    assert window.events == []
    assert window.thread_id == "no-such-thread"


@pytest.mark.parametrize("bad_size", [0, -1])
def test_invalid_window_size_is_rejected(queue_client, bad_size):
    with pytest.raises(ValueError):
        build_window(queue_client, "thread-1", window_size=bad_size)

"""Tests for the QueueClient (Phase 3).

These tests are written before the implementation (queue_client.client does
not exist yet) and define the contract for review. Run `pytest` here to see
them fail with a collection error until src/queue_client/client.py exists.

Redis is faked via `fakeredis.FakeRedis()` — no real server needed for these
unit tests (mirrors how `MemorySaver` is used for the checkpointer's
unit-test tier; see ARCHITECTURE_NOTES.md).
"""
from datetime import datetime, timezone

import fakeredis
import pytest

from queue_client.client import QueueClient, TraceEventRecord

DEFAULT_MAX_LEN = 500


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def client(fake_redis):
    return QueueClient(redis_client=fake_redis)


# ---------------------------------------------------------------------------
# append_event
# ---------------------------------------------------------------------------


def test_append_event_returns_nonempty_string_id(client):
    event_id = client.append_event("thread-1", {"node": "planner", "step": 1})
    assert isinstance(event_id, str)
    assert event_id != ""


def test_append_event_ids_are_unique_and_increasing(client):
    first_id = client.append_event("thread-1", {"step": 1})
    second_id = client.append_event("thread-1", {"step": 2})
    assert first_id != second_id
    # Redis stream ids are lexically ordered by generation time.
    assert first_id < second_id


def test_append_event_rejects_non_json_serializable_payload(client):
    with pytest.raises(TypeError):
        client.append_event("thread-1", {"bad": {1, 2, 3}})


# ---------------------------------------------------------------------------
# read_window: basic behavior
# ---------------------------------------------------------------------------


def test_read_window_on_unknown_thread_returns_empty_list(client):
    assert client.read_window("no-such-thread") == []


def test_read_window_returns_events_oldest_to_newest(client):
    client.append_event("thread-1", {"step": 1})
    client.append_event("thread-1", {"step": 2})
    client.append_event("thread-1", {"step": 3})

    window = client.read_window("thread-1")

    assert [record.payload["step"] for record in window] == [1, 2, 3]


def test_read_window_round_trips_payload_exactly(client):
    payload = {
        "node": "tool_call",
        "args": {"query": "foo", "nested": [1, 2, {"three": 3}]},
        "ok": True,
        "score": 0.5,
    }
    client.append_event("thread-1", payload)

    window = client.read_window("thread-1")

    assert window[0].payload == payload


def test_read_window_records_carry_thread_id_and_timestamp(client):
    event_id = client.append_event("thread-1", {"step": 1})

    window = client.read_window("thread-1")

    record = window[0]
    assert isinstance(record, TraceEventRecord)
    assert record.event_id == event_id
    assert record.thread_id == "thread-1"
    assert isinstance(record.enqueued_at, datetime)
    assert record.enqueued_at.tzinfo is not None


# ---------------------------------------------------------------------------
# read_window: thread isolation
# ---------------------------------------------------------------------------


def test_read_window_isolates_events_by_thread_id(client):
    client.append_event("thread-a", {"step": 1})
    client.append_event("thread-b", {"step": 1})
    client.append_event("thread-a", {"step": 2})

    window_a = client.read_window("thread-a")
    window_b = client.read_window("thread-b")

    assert [record.payload["step"] for record in window_a] == [1, 2]
    assert [record.payload["step"] for record in window_b] == [1]


# ---------------------------------------------------------------------------
# read_window: limit and MAXLEN capping
# ---------------------------------------------------------------------------


def test_read_window_limit_returns_only_the_most_recent_n(client):
    for step in range(1, 6):
        client.append_event("thread-1", {"step": step})

    window = client.read_window("thread-1", limit=2)

    assert [record.payload["step"] for record in window] == [4, 5]


def test_read_window_default_limit_falls_back_to_max_len():
    fake_redis = fakeredis.FakeRedis()
    small_client = QueueClient(redis_client=fake_redis, max_len=3)
    for step in range(1, 6):
        small_client.append_event("thread-1", {"step": step})

    window = small_client.read_window("thread-1")

    assert len(window) <= 3


def test_max_len_caps_stream_so_oldest_events_are_trimmed():
    fake_redis = fakeredis.FakeRedis()
    small_client = QueueClient(redis_client=fake_redis, max_len=3)
    for step in range(1, 11):
        small_client.append_event("thread-1", {"step": step})

    window = small_client.read_window("thread-1", limit=100)

    steps = [record.payload["step"] for record in window]
    assert steps == sorted(steps)
    assert 1 not in steps
    assert steps[-1] == 10


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_client_can_be_constructed_from_redis_url():
    built_client = QueueClient(redis_url="redis://localhost:6379/0")
    assert built_client is not None


def test_client_requires_either_url_or_injected_client():
    with pytest.raises(ValueError):
        QueueClient()


def test_client_uses_injected_redis_client_not_a_new_connection(fake_redis):
    injected = QueueClient(redis_client=fake_redis)
    injected.append_event("thread-1", {"step": 1})

    # Reading directly off the same fake instance proves no separate
    # connection/client was created under the hood.
    raw_entries = fake_redis.xrange("trace_events:thread-1")
    assert len(raw_entries) == 1


# ---------------------------------------------------------------------------
# stream naming convention: per-agent (per-thread_id) streams, with a
# configurable prefix rather than a hardcoded literal
# ---------------------------------------------------------------------------


def test_default_stream_prefix_is_trace_events(fake_redis):
    client = QueueClient(redis_client=fake_redis)
    client.append_event("thread-1", {"step": 1})
    assert len(fake_redis.xrange("trace_events:thread-1")) == 1


def test_custom_stream_prefix_produces_expected_key(fake_redis):
    client = QueueClient(redis_client=fake_redis, stream_prefix="custom_prefix")
    client.append_event("thread-1", {"step": 1})

    assert len(fake_redis.xrange("custom_prefix:thread-1")) == 1
    assert len(fake_redis.xrange("trace_events:thread-1")) == 0


def test_different_prefixes_isolate_the_same_thread_id_on_the_same_redis(fake_redis):
    client_a = QueueClient(redis_client=fake_redis, stream_prefix="tenant_a")
    client_b = QueueClient(redis_client=fake_redis, stream_prefix="tenant_b")

    client_a.append_event("thread-1", {"tenant": "a"})
    client_b.append_event("thread-1", {"tenant": "b"})

    window_a = client_a.read_window("thread-1")
    window_b = client_b.read_window("thread-1")

    assert [r.payload["tenant"] for r in window_a] == ["a"]
    assert [r.payload["tenant"] for r in window_b] == ["b"]


# ---------------------------------------------------------------------------
# get_events_by_id: resolves specific event ids back to their content
# (Phase 7 dashboard task -- a halt's triggering_window is a list of event
# ids; the dashboard needs the actual event content to show "why," not
# just the ids)
# ---------------------------------------------------------------------------


def test_get_events_by_id_returns_only_the_requested_events(client):
    id1 = client.append_event("thread-1", {"step": 1})
    id2 = client.append_event("thread-1", {"step": 2})
    client.append_event("thread-1", {"step": 3})

    records = client.get_events_by_id("thread-1", [id1, id2])

    assert [r.payload["step"] for r in records] == [1, 2]


def test_get_events_by_id_returns_oldest_to_newest_regardless_of_input_order(client):
    id1 = client.append_event("thread-1", {"step": 1})
    id2 = client.append_event("thread-1", {"step": 2})

    records = client.get_events_by_id("thread-1", [id2, id1])  # reversed input order

    assert [r.payload["step"] for r in records] == [1, 2]


def test_get_events_by_id_silently_skips_ids_that_no_longer_exist():
    """An id trimmed by MAXLEN, or from the wrong thread, or just wrong --
    a dashboard drill-down shouldn't crash on a stale/partial reference,
    it should show whatever is still available."""
    fake_redis = fakeredis.FakeRedis()
    small_client = QueueClient(redis_client=fake_redis, max_len=2)
    id1 = small_client.append_event("thread-1", {"step": 1})  # will be trimmed
    id2 = small_client.append_event("thread-1", {"step": 2})
    id3 = small_client.append_event("thread-1", {"step": 3})

    records = small_client.get_events_by_id("thread-1", [id1, id2, id3, "9999999999999-0"])

    assert [r.payload["step"] for r in records] == [2, 3]


def test_get_events_by_id_with_empty_list_returns_empty_list(client):
    assert client.get_events_by_id("thread-1", []) == []


# No "isolates by thread_id" test using another thread's event id: Redis
# Stream ids are only unique WITHIN a stream, not globally (confirmed
# empirically -- two different stream keys created in the same
# millisecond can produce the identical id), so such a test would be
# flaky by construction, not a meaningful correctness check. The real
# scoping property is already covered above: get_events_by_id only ever
# looks up ids within the specified thread_id's own stream, which is
# exactly how it's used in practice (a HaltSignal.triggering_window's ids
# always come from that same thread's stream).

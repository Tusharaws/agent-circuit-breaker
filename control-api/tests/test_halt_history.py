"""Tests for HaltHistoryStore (Phase 7 dashboard task). Written before the
implementation (control_api.halt_history does not exist yet) -- run
`pytest` to see them fail with a collection error until
src/control_api/halt_history.py exists.

Only "is this thread currently halted" was queryable before this
(HaltRegistry) -- the dashboard's "find a halt and see its triggering
context in a few clicks" AC needs a real, persisted, queryable history of
past halt decisions (reason/confidence/triggering_window), not just
current status. A Redis sorted set (score = timestamp) is used rather than
a stream, since this needs time-range queries (ZRANGEBYSCORE) and doesn't
need the per-thread stream isolation trace events do.
"""
from datetime import datetime, timedelta, timezone

import fakeredis
import pytest
from schemas.halt_signal import HaltSignal

from control_api.halt_history import HaltHistoryStore


def _signal(trace_id="thread-1", reason="runaway_loop", confidence=0.9, at=None):
    return HaltSignal(
        trace_id=trace_id,
        reason=reason,
        confidence=confidence,
        triggering_window=["evt-1", "evt-2"],
        timestamp=at if at is not None else datetime.now(timezone.utc),
    )


@pytest.fixture
def store():
    return HaltHistoryStore(redis_client=fakeredis.FakeRedis())


def test_query_on_empty_store_returns_empty_list(store):
    assert store.query() == []


def test_recorded_halt_is_returned_by_an_unfiltered_query(store):
    store.record(_signal(trace_id="thread-1"))

    results = store.query()

    assert len(results) == 1
    assert results[0].trace_id == "thread-1"
    assert results[0].reason == "runaway_loop"
    assert results[0].confidence == 0.9
    assert results[0].triggering_window == ["evt-1", "evt-2"]


def test_each_record_gets_a_unique_halt_id(store):
    store.record(_signal(trace_id="thread-1"))
    store.record(_signal(trace_id="thread-1"))

    results = store.query()

    assert len(results) == 2
    assert results[0].halt_id != results[1].halt_id


def test_query_filters_by_trace_id(store):
    store.record(_signal(trace_id="thread-a"))
    store.record(_signal(trace_id="thread-b"))

    results = store.query(trace_id="thread-a")

    assert len(results) == 1
    assert results[0].trace_id == "thread-a"


def test_query_filters_by_time_range(store):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.record(_signal(trace_id="thread-old", at=base))
    store.record(_signal(trace_id="thread-new", at=base + timedelta(days=10)))

    results = store.query(since=base + timedelta(days=5))

    assert len(results) == 1
    assert results[0].trace_id == "thread-new"


def test_query_filters_by_trace_id_and_time_range_together(store):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.record(_signal(trace_id="thread-a", at=base))
    store.record(_signal(trace_id="thread-a", at=base + timedelta(days=10)))
    store.record(_signal(trace_id="thread-b", at=base + timedelta(days=10)))

    results = store.query(trace_id="thread-a", since=base + timedelta(days=5))

    assert len(results) == 1
    assert results[0].trace_id == "thread-a"


def test_results_are_ordered_oldest_to_newest(store):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.record(_signal(trace_id="thread-1", at=base + timedelta(days=1)))
    store.record(_signal(trace_id="thread-1", at=base))

    results = store.query()

    assert [r.timestamp for r in results] == sorted(r.timestamp for r in results)


def test_query_until_excludes_records_after_the_given_time(store):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.record(_signal(trace_id="thread-early", at=base))
    store.record(_signal(trace_id="thread-late", at=base + timedelta(days=10)))

    results = store.query(until=base + timedelta(days=5))

    assert len(results) == 1
    assert results[0].trace_id == "thread-early"

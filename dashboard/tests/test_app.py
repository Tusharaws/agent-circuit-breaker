"""Tests for the dashboard's FastAPI backend (Phase 7). Written before the
implementation (dashboard.app does not exist yet) -- run `pytest` to see
them fail with a collection error until src/dashboard/app.py exists.

The frontend (a single static page) isn't unit-testable the way this API
is -- this file proves the API contract the frontend depends on: "find a
halt and see its triggering context in a few clicks" only actually works
if these endpoints return real, correct data, which is the honest,
testable boundary here.
"""
from datetime import datetime, timedelta, timezone

import fakeredis
import pytest
from control_api.halt_history import HaltHistoryStore
from fastapi.testclient import TestClient
from queue_client.client import QueueClient
from schemas.halt_signal import HaltSignal

from dashboard.app import create_app


@pytest.fixture
def queue_client():
    return QueueClient(redis_client=fakeredis.FakeRedis())


@pytest.fixture
def halt_history():
    return HaltHistoryStore(redis_client=fakeredis.FakeRedis())


@pytest.fixture
def client(queue_client, halt_history):
    app = create_app(queue_client, halt_history)
    return TestClient(app)


def _signal(trace_id="thread-1", reason="runaway_loop", confidence=0.9, triggering_window=None, at=None):
    return HaltSignal(
        trace_id=trace_id,
        reason=reason,
        confidence=confidence,
        triggering_window=triggering_window or ["placeholder"],
        timestamp=at if at is not None else datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# GET /api/traces/{trace_id}
# ---------------------------------------------------------------------------


def test_get_trace_returns_full_ordered_timeline(client, queue_client):
    queue_client.append_event("thread-1", {"step": 1})
    queue_client.append_event("thread-1", {"step": 2})
    queue_client.append_event("thread-1", {"step": 3})

    response = client.get("/api/traces/thread-1")

    assert response.status_code == 200
    events = response.json()
    assert [e["payload"]["step"] for e in events] == [1, 2, 3]


def test_get_trace_for_unknown_thread_returns_empty_list(client):
    response = client.get("/api/traces/no-such-thread")

    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# GET /api/halts
# ---------------------------------------------------------------------------


def test_list_halts_with_no_filters_returns_all(client, halt_history):
    halt_history.record(_signal(trace_id="thread-a"))
    halt_history.record(_signal(trace_id="thread-b"))

    response = client.get("/api/halts")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_list_halts_filters_by_trace_id(client, halt_history):
    halt_history.record(_signal(trace_id="thread-a"))
    halt_history.record(_signal(trace_id="thread-b"))

    response = client.get("/api/halts", params={"trace_id": "thread-a"})

    results = response.json()
    assert len(results) == 1
    assert results[0]["trace_id"] == "thread-a"


def test_list_halts_filters_by_time_range(client, halt_history):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    halt_history.record(_signal(trace_id="thread-old", at=base))
    halt_history.record(_signal(trace_id="thread-new", at=base + timedelta(days=10)))

    response = client.get("/api/halts", params={"since": (base + timedelta(days=5)).isoformat()})

    results = response.json()
    assert len(results) == 1
    assert results[0]["trace_id"] == "thread-new"


# ---------------------------------------------------------------------------
# GET /api/halts/{halt_id}/context -- the "few clicks" drill-down
# ---------------------------------------------------------------------------


def test_get_halt_context_resolves_triggering_window_to_real_events(client, queue_client, halt_history):
    id1 = queue_client.append_event("thread-1", {"tool": "search", "query": "stuck"})
    id2 = queue_client.append_event("thread-1", {"tool": "search", "query": "stuck"})
    halt_id = halt_history.record(_signal(trace_id="thread-1", triggering_window=[id1, id2]))

    response = client.get(f"/api/halts/{halt_id}/context")

    assert response.status_code == 200
    events = response.json()
    assert len(events) == 2
    assert all(e["payload"]["query"] == "stuck" for e in events)


def test_get_halt_context_for_unknown_halt_id_returns_404(client):
    response = client.get("/api/halts/no-such-halt-id/context")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------


def test_root_serves_the_dashboard_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

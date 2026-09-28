"""Phase 7: the dashboard's cross-package proof -- a real POST /halt (the
same endpoint the real evaluator calls) actually lands in HaltHistoryStore
and is genuinely queryable/drillable-down-to through the dashboard's own
API, not just through dashboard's own isolated tests where
HaltHistoryStore.record() is called directly by the test itself.
"""
from datetime import datetime, timezone

import fakeredis
from control_api.app import create_app as create_control_api_app
from control_api.guard import HaltRegistry
from control_api.halt_history import HaltHistoryStore
from dashboard.app import create_app as create_dashboard_app
from fastapi.testclient import TestClient
from queue_client.client import QueueClient

API_TOKEN = "test-token"

VALID_HALT_PAYLOAD = {
    "trace_id": "thread-dashboard-e2e",
    "reason": "runaway_loop",
    "confidence": 0.93,
    "triggering_window": ["placeholder-event-id"],
    "timestamp": datetime.now(timezone.utc).isoformat(),
}


def test_a_real_halt_request_is_queryable_and_drillable_through_the_dashboard():
    queue_client = QueueClient(redis_client=fakeredis.FakeRedis())
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
    halt_history = HaltHistoryStore(redis_client=fakeredis.FakeRedis())

    # Real events for the thread, so the drill-down has something real to resolve.
    event_id = queue_client.append_event(
        "thread-dashboard-e2e", {"event_type": "tool_start", "tool": "search", "args": {"query": "stuck"}}
    )
    payload = {**VALID_HALT_PAYLOAD, "triggering_window": [event_id]}

    control_api_client = TestClient(
        create_control_api_app(registry, api_token=API_TOKEN, halt_history=halt_history),
        headers={"Authorization": f"Bearer {API_TOKEN}"},
    )
    dashboard_client = TestClient(create_dashboard_app(queue_client, halt_history))

    # The real control-api call an evaluator would make.
    halt_response = control_api_client.post("/halt", json=payload)
    assert halt_response.status_code == 202
    assert registry.is_halted("thread-dashboard-e2e") is True  # the real halt mechanism also fired

    # Now the dashboard's own API, completely independently, can find it.
    halts = dashboard_client.get("/api/halts", params={"trace_id": "thread-dashboard-e2e"}).json()
    assert len(halts) == 1
    halt_id = halts[0]["halt_id"]
    assert halts[0]["reason"] == "runaway_loop"

    # And drill down to the exact real event that triggered it -- "a few clicks."
    context = dashboard_client.get(f"/api/halts/{halt_id}/context").json()
    assert len(context) == 1
    assert context[0]["event_id"] == event_id
    assert context[0]["payload"]["tool"] == "search"

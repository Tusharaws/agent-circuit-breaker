"""Tests for the POST /halt endpoint (Phase 5). Written before the
implementation (control_api.app does not exist yet) -- run `pytest` to
see them fail with a collection error until src/control_api/app.py
exists.

Verified empirically before writing any handler code: FastAPI validates
the request body against a declared pydantic type (HaltSignal) natively,
returning 422 with per-field detail on anything invalid -- confirmed via
a standalone check, not assumed from documentation.
"""
import fakeredis
import pytest
from fastapi.testclient import TestClient

from control_api.app import create_app
from control_api.guard import HaltRegistry
from control_api.halt_history import HaltHistoryStore

VALID_PAYLOAD = {
    "trace_id": "thread-abc",
    "reason": "repeated_tool_calls",
    "confidence": 0.92,
    "triggering_window": ["evt-101", "evt-102"],
    "timestamp": "2026-09-18T12:34:56Z",
}


API_TOKEN = "test-token"


@pytest.fixture
def registry():
    return HaltRegistry(redis_client=fakeredis.FakeRedis())


@pytest.fixture
def client(registry):
    """Pre-authenticated client -- TestClient's `headers` param applies to
    every request from this instance, so existing tests (written before
    auth existed) don't need per-call header changes; only the fixture
    changes. Auth-specific tests below use their own clients."""
    app = create_app(registry, api_token=API_TOKEN)
    return TestClient(app, headers={"Authorization": f"Bearer {API_TOKEN}"})


# ---------------------------------------------------------------------------
# Valid payload
# ---------------------------------------------------------------------------


def test_valid_halt_signal_returns_202(client):
    response = client.post("/halt", json=VALID_PAYLOAD)
    assert response.status_code == 202


def test_valid_halt_signal_actually_marks_the_thread_halted(client, registry):
    client.post("/halt", json=VALID_PAYLOAD)
    assert registry.is_halted("thread-abc") is True


def test_response_body_includes_the_trace_id(client):
    response = client.post("/halt", json=VALID_PAYLOAD)
    assert response.json()["trace_id"] == "thread-abc"


# ---------------------------------------------------------------------------
# Invalid payloads: clear 4xx, and the thread is NOT halted
# ---------------------------------------------------------------------------


def test_missing_required_fields_returns_422(client):
    response = client.post("/halt", json={"trace_id": "thread-x"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_invalid_reason_value_is_rejected(client):
    payload = {**VALID_PAYLOAD, "reason": "not_a_real_reason"}
    response = client.post("/halt", json=payload)
    assert response.status_code == 422


def test_invalid_confidence_out_of_range_is_rejected(client):
    payload = {**VALID_PAYLOAD, "confidence": 1.5}
    response = client.post("/halt", json=payload)
    assert response.status_code == 422


def test_invalid_payload_does_not_halt_any_thread(client, registry):
    client.post("/halt", json={"trace_id": "thread-x"})
    assert registry.is_halted("thread-x") is False


def test_completely_empty_body_returns_422(client):
    response = client.post("/halt", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Auth: missing/invalid tokens rejected, valid token succeeds
# ---------------------------------------------------------------------------


def test_missing_authorization_header_returns_401(registry):
    app = create_app(registry, api_token=API_TOKEN)
    unauthenticated_client = TestClient(app)  # no default headers

    response = unauthenticated_client.post("/halt", json=VALID_PAYLOAD)

    assert response.status_code == 401


def test_wrong_token_returns_401(registry):
    app = create_app(registry, api_token=API_TOKEN)
    wrong_token_client = TestClient(app, headers={"Authorization": "Bearer wrong-token"})

    response = wrong_token_client.post("/halt", json=VALID_PAYLOAD)

    assert response.status_code == 401


def test_valid_token_succeeds_and_halts_the_thread(client, registry):
    response = client.post("/halt", json=VALID_PAYLOAD)

    assert response.status_code == 202
    assert registry.is_halted("thread-abc") is True


def test_unauthenticated_request_does_not_halt_any_thread(registry):
    app = create_app(registry, api_token=API_TOKEN)
    unauthenticated_client = TestClient(app)

    unauthenticated_client.post("/halt", json=VALID_PAYLOAD)

    assert registry.is_halted("thread-abc") is False


def test_create_app_without_a_token_or_env_var_raises(registry, monkeypatch):
    monkeypatch.delenv("CONTROL_API_TOKEN", raising=False)

    with pytest.raises(ValueError):
        create_app(registry)


def test_create_app_picks_up_token_from_env_var(registry, monkeypatch):
    monkeypatch.setenv("CONTROL_API_TOKEN", "env-token")

    app = create_app(registry)
    env_client = TestClient(app, headers={"Authorization": "Bearer env-token"})

    response = env_client.post("/halt", json=VALID_PAYLOAD)
    assert response.status_code == 202


# ---------------------------------------------------------------------------
# Halt-decision logging: full context, queryable/searchable
# ---------------------------------------------------------------------------


def test_successful_halt_logs_full_context(client, caplog):
    import json
    import logging

    with caplog.at_level(logging.INFO, logger="control_api.halt_audit"):
        client.post("/halt", json=VALID_PAYLOAD)

    assert len(caplog.records) == 1
    logged = json.loads(caplog.records[0].getMessage())  # proves it's genuinely structured
    assert logged["trace_id"] == "thread-abc"
    assert logged["reason"] == "repeated_tool_calls"
    assert logged["confidence"] == 0.92
    assert logged["triggering_window"] == ["evt-101", "evt-102"]
    assert "timestamp" in logged


def test_rejected_request_produces_no_halt_audit_log(client, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="control_api.halt_audit"):
        client.post("/halt", json={"trace_id": "thread-x"})  # missing required fields -> 422

    assert len(caplog.records) == 0


# ---------------------------------------------------------------------------
# Halt history (Phase 7 dashboard task): optional, additive, backward-
# compatible -- every test above uses create_app(registry, api_token=...)
# with no halt_history and must keep passing unchanged.
# ---------------------------------------------------------------------------


def test_create_app_without_halt_history_still_works(registry):
    """Backward compatibility: halt_history is optional, defaulting to no
    persisted history rather than a required new argument."""
    app = create_app(registry, api_token=API_TOKEN)
    no_history_client = TestClient(app, headers={"Authorization": f"Bearer {API_TOKEN}"})

    response = no_history_client.post("/halt", json=VALID_PAYLOAD)

    assert response.status_code == 202


def test_successful_halt_is_recorded_in_halt_history(registry):
    halt_history = HaltHistoryStore(redis_client=fakeredis.FakeRedis())
    app = create_app(registry, api_token=API_TOKEN, halt_history=halt_history)
    history_client = TestClient(app, headers={"Authorization": f"Bearer {API_TOKEN}"})

    history_client.post("/halt", json=VALID_PAYLOAD)

    records = halt_history.query(trace_id="thread-abc")
    assert len(records) == 1
    assert records[0].reason == "repeated_tool_calls"
    assert records[0].confidence == 0.92
    assert records[0].triggering_window == ["evt-101", "evt-102"]


def test_rejected_request_is_not_recorded_in_halt_history(registry):
    halt_history = HaltHistoryStore(redis_client=fakeredis.FakeRedis())
    app = create_app(registry, api_token=API_TOKEN, halt_history=halt_history)
    history_client = TestClient(app, headers={"Authorization": f"Bearer {API_TOKEN}"})

    history_client.post("/halt", json={"trace_id": "thread-x"})  # missing required fields -> 422

    assert halt_history.query() == []

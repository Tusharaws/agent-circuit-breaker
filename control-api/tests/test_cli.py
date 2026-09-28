"""Tests for the manual override trigger (Phase 5). Written before the
implementation (control_api.cli does not exist yet) -- run `pytest` to
see them fail with a collection error until src/control_api/cli.py
exists.

The endpoint itself already doesn't distinguish who/what sends a
request -- these tests prove a manual trigger produces IDENTICAL
observable behavior (registry state, halt-audit log shape) to an
automated one, not just that the code path is theoretically the same.
"""
import json
import logging

import fakeredis
import pytest
from fastapi.testclient import TestClient

from control_api.app import create_app
from control_api.cli import DEFAULT_MANUAL_REASON, main, trigger_manual_halt
from control_api.guard import HaltRegistry

API_TOKEN = "test-token"


@pytest.fixture
def registry():
    return HaltRegistry(redis_client=fakeredis.FakeRedis())


@pytest.fixture
def http_client(registry):
    app = create_app(registry, api_token=API_TOKEN)
    return TestClient(app, headers={"Authorization": f"Bearer {API_TOKEN}"})


# ---------------------------------------------------------------------------
# trigger_manual_halt(): halts identically to an automated trigger
# ---------------------------------------------------------------------------


def test_manual_trigger_halts_the_thread(http_client, registry):
    trigger_manual_halt("thread-manual", http_client=http_client)

    assert registry.is_halted("thread-manual") is True


def test_manual_trigger_uses_manual_override_reason_by_default(http_client, caplog):
    with caplog.at_level(logging.INFO, logger="control_api.halt_audit"):
        trigger_manual_halt("thread-manual", http_client=http_client)

    logged = json.loads(caplog.records[0].getMessage())
    assert logged["reason"] == "manual_override" == DEFAULT_MANUAL_REASON


def test_manual_trigger_produces_the_same_audit_log_shape_as_automated(http_client, caplog):
    """Same fields, same structure -- a post-mortem reviewer can't tell
    from the log shape alone whether a halt was automated or manual,
    only from the reason value."""
    with caplog.at_level(logging.INFO, logger="control_api.halt_audit"):
        trigger_manual_halt("thread-manual", http_client=http_client)

    logged = json.loads(caplog.records[0].getMessage())
    assert set(logged.keys()) == {"trace_id", "reason", "confidence", "triggering_window", "timestamp"}


def test_manual_trigger_defaults_to_full_confidence(http_client, caplog):
    with caplog.at_level(logging.INFO, logger="control_api.halt_audit"):
        trigger_manual_halt("thread-manual", http_client=http_client)

    logged = json.loads(caplog.records[0].getMessage())
    assert logged["confidence"] == 1.0


def test_manual_trigger_accepts_a_custom_reason_confidence_and_window(http_client, caplog):
    with caplog.at_level(logging.INFO, logger="control_api.halt_audit"):
        trigger_manual_halt(
            "thread-manual",
            http_client=http_client,
            reason="runaway_loop",
            confidence=0.8,
            triggering_window=["evt-1", "evt-2"],
        )

    logged = json.loads(caplog.records[0].getMessage())
    assert logged["reason"] == "runaway_loop"
    assert logged["confidence"] == 0.8
    assert logged["triggering_window"] == ["evt-1", "evt-2"]


def test_manual_trigger_returns_the_response_body(http_client):
    result = trigger_manual_halt("thread-manual", http_client=http_client)
    assert result["trace_id"] == "thread-manual"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def test_cli_entry_point_triggers_a_halt(http_client, registry, monkeypatch):
    """The CLI entry point builds its own client from --control-api-url;
    here we monkeypatch the client construction to reuse the test's
    in-process ASGI TestClient instead of hitting a real network address."""
    import control_api.cli as cli_module

    monkeypatch.setattr(cli_module, "_build_http_client", lambda url, token: http_client)

    exit_code = main(["--trace-id", "thread-cli", "--control-api-url", "http://x", "--token", API_TOKEN])

    assert exit_code == 0
    assert registry.is_halted("thread-cli") is True

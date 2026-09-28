"""Tests for wiring a positive verdict to Control API's /halt endpoint
(Phase 4, now unblocked by control-api's /halt endpoint existing).
Written before the implementation (evaluator.control_api_client does not
exist yet) -- run `pytest` to see them fail with a collection error until
src/evaluator/control_api_client.py exists.

Uses fastapi.testclient.TestClient against control-api's REAL app (real
routing, real HaltSignal validation, real HaltRegistry write) -- verified
empirically before designing this that TestClient's .post(url, json=...)
signature is compatible with httpx.Client's, so the same client object
shape works in production (a real httpx.Client) and in this test (a real
ASGI test harness, not a mock).
"""
from datetime import datetime, timezone

import fakeredis
import pytest
from fastapi.testclient import TestClient
from schemas.trace_event import TraceEvent

from control_api.app import create_app
from control_api.guard import HaltRegistry
from evaluator.control_api_client import DEFAULT_CONFIDENCE_THRESHOLD, maybe_trigger_halt
from evaluator.verdict import Verdict
from evaluator.window import DetectionWindow


def _event(step, event_type, payload):
    return TraceEvent(
        trace_id="thread-1",
        agent_id="agent-1",
        step_index=step,
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        payload=payload,
    )


@pytest.fixture
def registry():
    return HaltRegistry(redis_client=fakeredis.FakeRedis())


API_TOKEN = "test-token"


@pytest.fixture
def http_client(registry):
    """A real httpx.Client in production would carry the same
    Authorization header, configured once at construction time -- the
    same way this TestClient is configured here."""
    app = create_app(registry, api_token=API_TOKEN)
    return TestClient(app, headers={"Authorization": f"Bearer {API_TOKEN}"})


@pytest.fixture
def loop_window():
    return DetectionWindow(
        thread_id="thread-loop",
        events=[_event(i, "tool_start", {"tool": "search"}) for i in range(3)],
        event_ids=["1-0", "1-1", "1-2"],
    )


# ---------------------------------------------------------------------------
# Loop verdict above threshold: real call fires, thread genuinely halted
# ---------------------------------------------------------------------------


def test_loop_verdict_above_threshold_triggers_a_real_halt_call(loop_window, http_client, registry):
    verdict = Verdict(is_loop=True, confidence=0.95, reason="repeated_tool_calls")

    triggered = maybe_trigger_halt(loop_window, verdict, http_client)

    assert triggered is True
    assert registry.is_halted("thread-loop") is True  # the real effect, not just a mock call


def test_loop_verdict_at_exactly_the_threshold_triggers(loop_window, http_client, registry):
    verdict = Verdict(is_loop=True, confidence=DEFAULT_CONFIDENCE_THRESHOLD, reason="runaway_loop")

    triggered = maybe_trigger_halt(loop_window, verdict, http_client)

    assert triggered is True
    assert registry.is_halted("thread-loop") is True


# ---------------------------------------------------------------------------
# Below threshold or no loop: no call, thread not halted
# ---------------------------------------------------------------------------


def test_loop_verdict_below_threshold_does_not_trigger(loop_window, http_client, registry):
    verdict = Verdict(is_loop=True, confidence=0.5, reason="repeated_tool_calls")

    triggered = maybe_trigger_halt(loop_window, verdict, http_client)

    assert triggered is False
    assert registry.is_halted("thread-loop") is False


def test_no_loop_verdict_never_triggers_regardless_of_confidence(loop_window, http_client, registry):
    verdict = Verdict(is_loop=False, confidence=0.99, reason=None)

    triggered = maybe_trigger_halt(loop_window, verdict, http_client)

    assert triggered is False
    assert registry.is_halted("thread-loop") is False


def test_custom_confidence_threshold_is_respected(loop_window, http_client, registry):
    verdict = Verdict(is_loop=True, confidence=0.6, reason="runaway_loop")

    triggered = maybe_trigger_halt(loop_window, verdict, http_client, confidence_threshold=0.5)

    assert triggered is True
    assert registry.is_halted("thread-loop") is True


# ---------------------------------------------------------------------------
# The HaltSignal sent matches the window + verdict that produced it
# ---------------------------------------------------------------------------


class _SpyClient:
    """Wraps a real client, recording what was sent while still
    delegating to it -- proves both the payload shape AND the real
    effect (registry write) in one real integration test, rather than
    a mock that only proves a call happened."""

    def __init__(self, real_client):
        self._real_client = real_client
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, json):
        self.calls.append((url, json))
        return self._real_client.post(url, json=json)


def test_halt_signal_sent_matches_window_and_verdict(loop_window, http_client, registry):
    spy = _SpyClient(http_client)
    verdict = Verdict(is_loop=True, confidence=0.88, reason="repeated_generation")

    maybe_trigger_halt(loop_window, verdict, spy)

    assert len(spy.calls) == 1
    url, payload = spy.calls[0]
    assert url == "/halt"
    assert payload["trace_id"] == loop_window.thread_id
    assert payload["reason"] == "repeated_generation"
    assert payload["confidence"] == 0.88
    assert payload["triggering_window"] == loop_window.event_ids

    assert registry.is_halted(loop_window.thread_id) is True  # the real effect went through too

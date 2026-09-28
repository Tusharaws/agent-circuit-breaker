"""Tests for verdict parsing and the detection prompt (Phase 4). Written
before the implementation (evaluator.verdict does not exist yet) -- run
`pytest` to see them fail with a collection error until
src/evaluator/verdict.py exists.

parse_verdict()'s lenient design is grounded in real, empirically-observed
model output (not assumed): asked for exactly 3 keys, the real Qwen2.5-0.5B
model returned 2 extra unrequested null keys; asked for `reason` to be one
of 3 enum values, it sometimes returned a full free-text explanation
instead.
"""
import json
from datetime import datetime, timezone

import pytest

from schemas.halt_signal import HaltSignal
from schemas.trace_event import TraceEvent

from evaluator.verdict import Verdict, build_halt_signal, build_prompt, parse_verdict
from evaluator.window import DetectionWindow


def _event(step_index, event_type="tool_start", payload=None):
    return TraceEvent(
        trace_id="thread-1",
        agent_id="agent-1",
        step_index=step_index,
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        payload=payload or {"tool": "search", "args": {"query": f"q{step_index}"}},
    )


# ---------------------------------------------------------------------------
# build_prompt()
# ---------------------------------------------------------------------------


def test_prompt_includes_each_events_type_and_payload():
    window = DetectionWindow(
        thread_id="thread-1",
        events=[_event(0, event_type="tool_start", payload={"tool": "search"})],
        event_ids=["1-0"],
    )

    prompt = build_prompt(window)

    assert "tool_start" in prompt
    assert "search" in prompt


def test_prompt_requests_the_three_expected_fields():
    window = DetectionWindow(thread_id="thread-1", events=[_event(0)], event_ids=["1-0"])
    prompt = build_prompt(window)

    for field in ["is_loop", "confidence", "reason"]:
        assert field in prompt


# ---------------------------------------------------------------------------
# parse_verdict(): well-formed responses
# ---------------------------------------------------------------------------


def test_well_formed_loop_verdict_parses_correctly():
    raw = '{"is_loop": true, "confidence": 0.92, "reason": "repeated_tool_calls"}'
    verdict = parse_verdict(raw)

    assert verdict == Verdict(is_loop=True, confidence=0.92, reason="repeated_tool_calls")


def test_well_formed_no_loop_verdict_parses_correctly():
    raw = '{"is_loop": false, "confidence": 0.1, "reason": null}'
    verdict = parse_verdict(raw)

    assert verdict.is_loop is False
    assert verdict.reason is None


# ---------------------------------------------------------------------------
# parse_verdict(): real observed model quirks
# ---------------------------------------------------------------------------


def test_extra_unrequested_keys_are_ignored():
    """Confirmed real behavior: the model added 'repeated_generation':
    null and 'runaway_loop': null alongside the requested 'reason' key."""
    raw = (
        '{"is_loop": true, "confidence": 1.0, "reason": "repeated_tool_calls", '
        '"repeated_generation": null, "runaway_loop": null}'
    )
    verdict = parse_verdict(raw)

    assert verdict == Verdict(is_loop=True, confidence=1.0, reason="repeated_tool_calls")


def test_invalid_free_text_reason_falls_back_to_default():
    """Confirmed real behavior: the model sometimes ignores the enum
    constraint and returns a free-text explanation instead."""
    raw = '{"is_loop": true, "confidence": 0.9, "reason": "It is repeating the same search over and over."}'
    verdict = parse_verdict(raw)

    assert verdict.is_loop is True
    assert verdict.reason == "runaway_loop"  # DEFAULT_REASON fallback


def test_no_loop_verdict_never_carries_a_reason_even_if_model_included_one():
    raw = '{"is_loop": false, "confidence": 0.2, "reason": "repeated_tool_calls"}'
    verdict = parse_verdict(raw)

    assert verdict.is_loop is False
    assert verdict.reason is None


# ---------------------------------------------------------------------------
# parse_verdict(): value coercion / clipping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw_confidence,expected", [(1.5, 1.0), (-0.3, 0.0), ("0.7", 0.7)])
def test_confidence_is_clipped_and_coerced(raw_confidence, expected):
    payload = {"is_loop": True, "confidence": raw_confidence, "reason": "runaway_loop"}
    verdict = parse_verdict(json.dumps(payload))
    assert verdict.confidence == pytest.approx(expected)


# ---------------------------------------------------------------------------
# parse_verdict(): fail closed on unparseable input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["not json at all", "", "{broken json", "I think it might be looping but not sure"])
def test_unparseable_response_fails_closed_to_no_loop(raw):
    verdict = parse_verdict(raw)

    assert verdict.is_loop is False
    assert verdict.confidence == 0.0
    assert verdict.reason is None


def test_json_embedded_in_surrounding_prose_is_still_extracted():
    raw = 'Here is my answer: {"is_loop": true, "confidence": 0.8, "reason": "runaway_loop"} Hope that helps!'
    verdict = parse_verdict(raw)

    assert verdict.is_loop is True
    assert verdict.confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# build_halt_signal()
# ---------------------------------------------------------------------------


def test_build_halt_signal_from_a_positive_verdict():
    window = DetectionWindow(
        thread_id="thread-abc",
        events=[_event(0), _event(1)],
        event_ids=["1-0", "1-1"],
    )
    verdict = Verdict(is_loop=True, confidence=0.85, reason="repeated_tool_calls")

    signal = build_halt_signal(window, verdict)

    assert isinstance(signal, HaltSignal)
    assert signal.trace_id == "thread-abc"
    assert signal.reason == "repeated_tool_calls"
    assert signal.confidence == 0.85
    assert signal.triggering_window == ["1-0", "1-1"]
    assert signal.timestamp.tzinfo is not None


def test_build_halt_signal_rejects_a_no_loop_verdict():
    window = DetectionWindow(thread_id="thread-1", events=[_event(0)], event_ids=["1-0"])
    verdict = Verdict(is_loop=False, confidence=0.1, reason=None)

    with pytest.raises(ValueError):
        build_halt_signal(window, verdict)

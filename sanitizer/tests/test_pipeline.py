"""Tests for the sanitize-before-queue pipeline step (Phase 2). Written
before the implementation (sanitizer.pipeline does not exist yet) -- run
`pytest` to see them fail with a collection error until
src/sanitizer/pipeline.py exists.
"""
import pytest

from sanitizer.pipeline import SanitizingSink, sanitize_value

POLICY = {"email": "mask", "ssn": "drop", "phone": "mask", "credit_card": "mask", "api_key": "drop"}


class RecordingSink:
    def __init__(self):
        self.received = []

    def __call__(self, event):
        self.received.append(event)


# ---------------------------------------------------------------------------
# sanitize_value(): recursive walk
# ---------------------------------------------------------------------------


def test_sanitize_value_scrubs_a_plain_string():
    assert sanitize_value("contact alice@example.com", POLICY) == "contact [REDACTED]"


def test_sanitize_value_recurses_into_nested_dicts():
    value = {"outer": {"inner": "email me at bob@example.com please"}}
    result = sanitize_value(value, POLICY)
    assert "bob@example.com" not in result["outer"]["inner"]


def test_sanitize_value_recurses_into_lists():
    value = ["clean text", "ssn is 123-45-6789 here"]
    result = sanitize_value(value, POLICY)
    assert "123-45-6789" not in result[1]
    assert result[0] == "clean text"


def test_sanitize_value_recurses_into_lists_of_dicts():
    value = [{"note": "card 4111111111111111 on file"}]
    result = sanitize_value(value, POLICY)
    assert "4111111111111111" not in result[0]["note"]


@pytest.mark.parametrize("value", [42, 3.14, True, False, None])
def test_sanitize_value_leaves_non_string_scalars_untouched(value):
    assert sanitize_value(value, POLICY) == value


# ---------------------------------------------------------------------------
# SanitizingSink: construction
# ---------------------------------------------------------------------------


def test_construction_rejects_non_callable_inner():
    with pytest.raises(TypeError):
        SanitizingSink(inner="not-callable")


# ---------------------------------------------------------------------------
# SanitizingSink: a plain dict event (queue_client's own generic payload
# shape -- SanitizingSink must not assume TraceEvent specifically)
# ---------------------------------------------------------------------------


def test_plain_dict_event_payload_is_sanitized():
    inner = RecordingSink()
    sink = SanitizingSink(inner=inner, policy=POLICY)

    sink({"thread_id": "t-1", "payload": {"prompt": "call me at 415-555-2671"}})

    assert "415-555-2671" not in inner.received[0]["payload"]["prompt"]


def test_dict_event_fields_without_pii_are_unaffected():
    inner = RecordingSink()
    sink = SanitizingSink(inner=inner, policy=POLICY)

    sink({"thread_id": "t-1", "step_index": 3, "payload": {"note": "clean text"}})

    event = inner.received[0]
    assert event["thread_id"] == "t-1"
    assert event["step_index"] == 3
    assert event["payload"]["note"] == "clean text"


# ---------------------------------------------------------------------------
# SanitizingSink: a real schemas.TraceEvent (nested payload) -- proves the
# recursive walk reaches nested structures, not just top-level fields
# ---------------------------------------------------------------------------


def _build_trace_event_with_pii():
    from datetime import datetime, timezone

    from schemas.trace_event import TraceEvent

    return TraceEvent(
        trace_id="thread-pii",
        agent_id="agent-1",
        step_index=0,
        timestamp=datetime.now(timezone.utc),
        event_type="llm_end",
        payload={
            "response": ["please email alice@example.com or call 415-555-2671"],
            "nested": {"card": "card number is 4111111111111111 for billing"},
        },
    )


def test_trace_event_nested_payload_pii_is_fully_removed():
    inner = RecordingSink()
    sink = SanitizingSink(inner=inner, policy=POLICY)
    event = _build_trace_event_with_pii()

    sink(event)

    received = inner.received[0]
    dumped = received.model_dump(mode="json")
    assert "alice@example.com" not in str(dumped)
    assert "415-555-2671" not in str(dumped)
    assert "4111111111111111" not in str(dumped)


def test_trace_event_survives_sanitization_as_a_valid_trace_event():
    from schemas.trace_event import TraceEvent

    inner = RecordingSink()
    sink = SanitizingSink(inner=inner, policy=POLICY)
    sink(_build_trace_event_with_pii())

    assert isinstance(inner.received[0], TraceEvent)
    assert inner.received[0].trace_id == "thread-pii"


# ---------------------------------------------------------------------------
# The "attempted bypass" regression guard: without SanitizingSink, the
# same seeded PII DOES leak -- proving why the wrapper is mandatory, not
# just documented advice.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Optional ner_scrubber: SanitizingSink also catches unstructured PHI when
# one is provided (default None preserves today's regex-only behavior,
# proven by all the tests above that never pass ner_scrubber).
# ---------------------------------------------------------------------------


def test_ner_scrubber_none_by_default_leaves_names_untouched():
    """Regression: without an ner_scrubber, SanitizingSink is regex-only --
    a name is not a data type any regex pattern targets."""
    inner = RecordingSink()
    sink = SanitizingSink(inner=inner, policy=POLICY)

    sink({"payload": {"note": "Patient John Smith visited today"}})

    assert "John Smith" in inner.received[0]["payload"]["note"]


def test_ner_scrubber_when_provided_also_scrubs_names_and_medical_terms():
    from sanitizer.ner import NerScrubber

    inner = RecordingSink()
    ner_policy = {**POLICY, "person": "mask", "medical_condition": "drop"}
    sink = SanitizingSink(inner=inner, policy=ner_policy, ner_scrubber=NerScrubber())

    sink({"payload": {"note": "Patient John Smith has asthma and email a@b.com"}})

    note = inner.received[0]["payload"]["note"]
    assert "John Smith" not in note
    assert "asthma" not in note
    assert "a@b.com" not in note  # regex scrubbing still applies too


def test_bypassing_sanitizing_sink_leaks_raw_pii_demonstrating_the_risk():
    inner = RecordingSink()
    event = _build_trace_event_with_pii()

    inner(event)  # directly, with no SanitizingSink -- the "bypass"

    dumped = inner.received[0].model_dump(mode="json")
    assert "alice@example.com" in str(dumped)  # leaked -- this is the risk closed above

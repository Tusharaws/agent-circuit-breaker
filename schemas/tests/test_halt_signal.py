"""Tests for the Halt Signal schema (Phase 0).

These tests are written before the implementation (schemas.halt_signal does
not exist yet) and define the contract for review. Run `pytest` here to see
them fail with a collection error until src/schemas/halt_signal.py exists.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from schemas.halt_signal import HaltReason, HaltSignal

VALID_PAYLOAD = {
    "trace_id": "trace-abc123",
    "reason": "repeated_tool_calls",
    "confidence": 0.92,
    "triggering_window": ["evt-101", "evt-102", "evt-103"],
    "timestamp": "2026-09-18T12:34:56Z",
}

EXPECTED_FIELDS = {"trace_id", "reason", "confidence", "triggering_window", "timestamp"}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_halt_signal_parses():
    signal = HaltSignal.model_validate(VALID_PAYLOAD)

    assert signal.trace_id == "trace-abc123"
    assert signal.reason == HaltReason.REPEATED_TOOL_CALLS
    assert signal.confidence == pytest.approx(0.92)
    assert signal.triggering_window == ["evt-101", "evt-102", "evt-103"]
    assert signal.timestamp == datetime(2026, 9, 18, 12, 34, 56, tzinfo=timezone.utc)


def test_sample_json_string_validates():
    """A raw JSON sample (as Control API would receive over the wire) validates."""
    raw = json.dumps(VALID_PAYLOAD)
    signal = HaltSignal.model_validate_json(raw)
    assert signal.trace_id == VALID_PAYLOAD["trace_id"]


def test_dumped_signal_matches_fields_control_api_expects():
    signal = HaltSignal.model_validate(VALID_PAYLOAD)
    dumped = signal.model_dump(mode="json")
    assert set(dumped.keys()) == EXPECTED_FIELDS


@pytest.mark.parametrize("reason", list(HaltReason))
def test_each_defined_reason_is_accepted(reason):
    payload = {**VALID_PAYLOAD, "reason": reason.value}
    signal = HaltSignal.model_validate(payload)
    assert signal.reason == reason


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_serialization():
    original = HaltSignal.model_validate(VALID_PAYLOAD)
    round_tripped = HaltSignal.model_validate_json(original.model_dump_json())
    assert original == round_tripped


# ---------------------------------------------------------------------------
# Required fields / missing fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", sorted(EXPECTED_FIELDS))
def test_missing_required_field_is_rejected(missing_field):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}
    with pytest.raises(ValidationError) as exc_info:
        HaltSignal.model_validate(payload)
    assert missing_field in str(exc_info.value)


def test_empty_payload_is_rejected():
    with pytest.raises(ValidationError):
        HaltSignal.model_validate({})


# ---------------------------------------------------------------------------
# Unknown / extra fields
# ---------------------------------------------------------------------------


def test_unexpected_extra_field_is_rejected():
    payload = {**VALID_PAYLOAD, "unexpected_field": "should not be allowed"}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


# ---------------------------------------------------------------------------
# trace_id edge cases
# ---------------------------------------------------------------------------


def test_trace_id_empty_string_is_rejected():
    payload = {**VALID_PAYLOAD, "trace_id": ""}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


@pytest.mark.parametrize("bad_trace_id", [None, 123, 1.5, ["trace-abc123"], {"id": "x"}])
def test_trace_id_wrong_type_is_rejected(bad_trace_id):
    payload = {**VALID_PAYLOAD, "trace_id": bad_trace_id}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


# ---------------------------------------------------------------------------
# reason edge cases
# ---------------------------------------------------------------------------


def test_reason_unknown_value_is_rejected():
    payload = {**VALID_PAYLOAD, "reason": "made_up_reason"}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


def test_reason_is_case_sensitive():
    payload = {**VALID_PAYLOAD, "reason": "REPEATED_TOOL_CALLS"}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


@pytest.mark.parametrize("bad_reason", [None, 1, ["repeated_tool_calls"]])
def test_reason_wrong_type_is_rejected(bad_reason):
    payload = {**VALID_PAYLOAD, "reason": bad_reason}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


# ---------------------------------------------------------------------------
# confidence edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary_value", [0.0, 0.5, 1.0])
def test_confidence_inclusive_boundaries_are_accepted(boundary_value):
    payload = {**VALID_PAYLOAD, "confidence": boundary_value}
    signal = HaltSignal.model_validate(payload)
    assert signal.confidence == boundary_value


@pytest.mark.parametrize("out_of_range_value", [-0.0001, -1, 1.0001, 2, 100])
def test_confidence_out_of_range_is_rejected(out_of_range_value):
    payload = {**VALID_PAYLOAD, "confidence": out_of_range_value}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


@pytest.mark.parametrize("bad_confidence", [None, "high", [0.9], {"value": 0.9}])
def test_confidence_wrong_type_is_rejected(bad_confidence):
    payload = {**VALID_PAYLOAD, "confidence": bad_confidence}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


def test_confidence_numeric_string_is_coerced():
    """Pydantic float fields coerce numeric strings; document that behavior explicitly."""
    payload = {**VALID_PAYLOAD, "confidence": "0.5"}
    signal = HaltSignal.model_validate(payload)
    assert signal.confidence == 0.5


# ---------------------------------------------------------------------------
# triggering_window edge cases
# ---------------------------------------------------------------------------


def test_triggering_window_empty_list_is_rejected():
    payload = {**VALID_PAYLOAD, "triggering_window": []}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


def test_triggering_window_single_item_is_accepted():
    payload = {**VALID_PAYLOAD, "triggering_window": ["evt-101"]}
    signal = HaltSignal.model_validate(payload)
    assert signal.triggering_window == ["evt-101"]


def test_triggering_window_not_a_list_is_rejected():
    payload = {**VALID_PAYLOAD, "triggering_window": "evt-101"}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


def test_triggering_window_with_non_string_item_is_rejected():
    payload = {**VALID_PAYLOAD, "triggering_window": ["evt-101", 102]}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


def test_triggering_window_with_empty_string_item_is_rejected():
    payload = {**VALID_PAYLOAD, "triggering_window": ["evt-101", ""]}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


def test_triggering_window_duplicate_entries_are_allowed():
    """Duplicates aren't inherently invalid (the same event id could recur in a window)."""
    payload = {**VALID_PAYLOAD, "triggering_window": ["evt-101", "evt-101"]}
    signal = HaltSignal.model_validate(payload)
    assert signal.triggering_window == ["evt-101", "evt-101"]


def test_triggering_window_none_is_rejected():
    payload = {**VALID_PAYLOAD, "triggering_window": None}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


# ---------------------------------------------------------------------------
# timestamp edge cases
# ---------------------------------------------------------------------------


def test_timestamp_with_offset_is_accepted():
    payload = {**VALID_PAYLOAD, "timestamp": "2026-09-18T12:34:56+05:30"}
    signal = HaltSignal.model_validate(payload)
    assert signal.timestamp.utcoffset() == timedelta(hours=5, minutes=30)


def test_timestamp_naive_datetime_is_rejected():
    """A timestamp with no timezone info is ambiguous and must be rejected."""
    payload = {**VALID_PAYLOAD, "timestamp": "2026-09-18T12:34:56"}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


def test_timestamp_naive_datetime_object_is_rejected():
    payload = {**VALID_PAYLOAD, "timestamp": datetime(2026, 9, 18, 12, 34, 56)}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


def test_timestamp_invalid_format_is_rejected():
    payload = {**VALID_PAYLOAD, "timestamp": "not-a-timestamp"}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


@pytest.mark.parametrize("bad_timestamp", [None, ["2026-09-18T12:34:56Z"], {"a": 1}])
def test_timestamp_wrong_type_is_rejected(bad_timestamp):
    payload = {**VALID_PAYLOAD, "timestamp": bad_timestamp}
    with pytest.raises(ValidationError):
        HaltSignal.model_validate(payload)


def test_timestamp_unix_epoch_number_is_coerced_to_utc():
    """Pydantic interprets a bare int/float as a Unix timestamp, which is
    already tz-aware (UTC) once parsed, so it passes our tz-aware check.
    Documented explicitly since it's easy to assume numbers are rejected."""
    payload = {**VALID_PAYLOAD, "timestamp": 1789732496}
    signal = HaltSignal.model_validate(payload)
    assert signal.timestamp.tzinfo is not None

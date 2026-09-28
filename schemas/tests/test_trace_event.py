"""Tests for the Trace Event schema (Phase 0).

These tests are written before the implementation (schemas.trace_event does
not exist yet) and define the contract for review. Run `pytest` here to see
them fail with a collection error until src/schemas/trace_event.py (and its
config/event_types.md sibling to halt_reasons.md) exist.

Design decisions this test file assumes (confirmed with the user before
writing it):
- `event_type` is validated against a config-driven allowed list, the same
  pattern `HaltSignal.reason` uses against `halt_reasons.md` — here against
  `config/event_types.md`, seeded with the LangChain-compatible callback
  names Phase 1/Interceptor already documents reusing: `node_enter`,
  `node_exit`, `llm_start`, `llm_end`, `tool_start`, `tool_end`.
- `token_usage` is a structured breakdown (`prompt_tokens`,
  `completion_tokens`, `total_tokens`), not a single int, and its total must
  equal the sum of the two parts (a cost-control product needs that sum to
  be trustworthy, not just present).
- `payload` is a fully free-form JSON-serializable dict — different
  event_types will carry different shapes, and constraining it now means
  guessing at Interceptor internals that don't exist yet.
- `token_usage` and `latency_ms` are optional (default `None`): a `_start`
  event fires before either is known, so requiring them would force
  Interceptor to send meaningless placeholder zeros.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import schemas.trace_event as trace_event_module
from schemas.trace_event import TokenUsage, TraceEvent, load_allowed_event_types

VALID_PAYLOAD = {
    "trace_id": "trace-abc123",
    "agent_id": "agent-planner-1",
    "step_index": 3,
    "timestamp": "2026-09-18T12:34:56Z",
    "event_type": "llm_end",
    "payload": {"content": "the generated answer", "model": "claude-sonnet-5"},
    "token_usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    "latency_ms": 842.5,
}

EXPECTED_FIELDS = {
    "trace_id",
    "agent_id",
    "step_index",
    "timestamp",
    "event_type",
    "payload",
    "token_usage",
    "latency_ms",
}

REQUIRED_FIELDS = EXPECTED_FIELDS - {"token_usage", "latency_ms"}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_trace_event_parses():
    event = TraceEvent.model_validate(VALID_PAYLOAD)

    assert event.trace_id == "trace-abc123"
    assert event.agent_id == "agent-planner-1"
    assert event.step_index == 3
    assert event.timestamp == datetime(2026, 9, 18, 12, 34, 56, tzinfo=timezone.utc)
    assert event.event_type == "llm_end"
    assert event.payload == {"content": "the generated answer", "model": "claude-sonnet-5"}
    assert event.token_usage == TokenUsage(prompt_tokens=120, completion_tokens=30, total_tokens=150)
    assert event.latency_ms == pytest.approx(842.5)


def test_sample_json_string_validates():
    """The acceptance criterion: a sample JSON event validates against the schema."""
    raw = json.dumps(VALID_PAYLOAD)
    event = TraceEvent.model_validate_json(raw)
    assert event.trace_id == VALID_PAYLOAD["trace_id"]


def test_dumped_event_matches_fields_consumers_expect():
    event = TraceEvent.model_validate(VALID_PAYLOAD)
    dumped = event.model_dump(mode="json")
    assert set(dumped.keys()) == EXPECTED_FIELDS


@pytest.mark.parametrize("event_type", load_allowed_event_types())
def test_each_defined_event_type_is_accepted(event_type):
    payload = {**VALID_PAYLOAD, "event_type": event_type}
    event = TraceEvent.model_validate(payload)
    assert event.event_type == event_type


def test_round_trip_serialization():
    original = TraceEvent.model_validate(VALID_PAYLOAD)
    round_tripped = TraceEvent.model_validate_json(original.model_dump_json())
    assert original == round_tripped


# ---------------------------------------------------------------------------
# Optional fields: token_usage / latency_ms
# ---------------------------------------------------------------------------


def test_start_event_without_token_usage_or_latency_parses():
    """A `_start` event fires before token usage or latency are known."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k not in ("token_usage", "latency_ms")}
    payload["event_type"] = "llm_start"

    event = TraceEvent.model_validate(payload)

    assert event.token_usage is None
    assert event.latency_ms is None


def test_explicit_none_for_token_usage_and_latency_is_accepted():
    payload = {**VALID_PAYLOAD, "token_usage": None, "latency_ms": None}
    event = TraceEvent.model_validate(payload)
    assert event.token_usage is None
    assert event.latency_ms is None


# ---------------------------------------------------------------------------
# Required fields / missing fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", sorted(REQUIRED_FIELDS))
def test_missing_required_field_is_rejected(missing_field):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}
    with pytest.raises(ValidationError) as exc_info:
        TraceEvent.model_validate(payload)
    assert missing_field in str(exc_info.value)


def test_empty_payload_dict_body_is_rejected():
    with pytest.raises(ValidationError):
        TraceEvent.model_validate({})


# ---------------------------------------------------------------------------
# Unknown / extra fields
# ---------------------------------------------------------------------------


def test_unexpected_extra_field_is_rejected():
    payload = {**VALID_PAYLOAD, "unexpected_field": "nope"}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


# ---------------------------------------------------------------------------
# trace_id / agent_id edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["trace_id", "agent_id"])
def test_id_field_empty_string_is_rejected(field):
    payload = {**VALID_PAYLOAD, field: ""}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


@pytest.mark.parametrize("field", ["trace_id", "agent_id"])
@pytest.mark.parametrize("bad_value", [None, 123, 1.5, ["x"], {"id": "x"}])
def test_id_field_wrong_type_is_rejected(field, bad_value):
    payload = {**VALID_PAYLOAD, field: bad_value}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


@pytest.mark.parametrize("field", ["trace_id", "agent_id"])
def test_id_field_whitespace_only_is_currently_accepted(field):
    """Gap worth knowing about, not a behavior change: `NonEmptyStr` only
    checks `min_length=1`, it doesn't strip whitespace (same as HaltSignal's
    `trace_id`), so a whitespace-only id currently passes. Documented so this
    is a known, not accidental, gap."""
    payload = {**VALID_PAYLOAD, field: "   "}
    event = TraceEvent.model_validate(payload)
    assert getattr(event, field) == "   "


# ---------------------------------------------------------------------------
# step_index edge cases
# ---------------------------------------------------------------------------


def test_step_index_zero_is_accepted():
    payload = {**VALID_PAYLOAD, "step_index": 0}
    event = TraceEvent.model_validate(payload)
    assert event.step_index == 0


def test_step_index_negative_is_rejected():
    payload = {**VALID_PAYLOAD, "step_index": -1}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_step_index_fractional_float_is_rejected():
    payload = {**VALID_PAYLOAD, "step_index": 3.5}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_step_index_whole_number_float_is_coerced_to_int():
    """Pydantic int fields coerce a float with no fractional part; document
    that behavior explicitly rather than assume it's rejected."""
    payload = {**VALID_PAYLOAD, "step_index": 3.0}
    event = TraceEvent.model_validate(payload)
    assert event.step_index == 3


@pytest.mark.parametrize("bad_value", [None, "not-a-number", [3], {"v": 3}])
def test_step_index_wrong_type_is_rejected(bad_value):
    payload = {**VALID_PAYLOAD, "step_index": bad_value}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_step_index_bool_is_coerced_to_int():
    """`bool` is a subclass of `int` in Python, so pydantic's default (non-strict)
    int validation accepts it. Documented explicitly since `step_index: true`
    silently becoming `step_index: 1` is a real footgun for malformed producer
    data, not something to assume is rejected."""
    payload = {**VALID_PAYLOAD, "step_index": True}
    event = TraceEvent.model_validate(payload)
    assert event.step_index == 1


# ---------------------------------------------------------------------------
# event_type edge cases
# ---------------------------------------------------------------------------


def test_event_type_unknown_value_is_rejected():
    payload = {**VALID_PAYLOAD, "event_type": "made_up_event"}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_event_type_is_case_sensitive():
    payload = {**VALID_PAYLOAD, "event_type": "LLM_END"}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


@pytest.mark.parametrize("bad_value", [None, 1, ["llm_end"]])
def test_event_type_wrong_type_is_rejected(bad_value):
    payload = {**VALID_PAYLOAD, "event_type": bad_value}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_event_type_accepts_entry_newly_added_to_config(tmp_path, monkeypatch):
    config_file = tmp_path / "event_types.md"
    config_file.write_text("- llm_end\n- retrieval_start\n")
    monkeypatch.setattr(trace_event_module, "_EVENT_TYPES_FILE", config_file)

    payload = {**VALID_PAYLOAD, "event_type": "retrieval_start"}
    event = TraceEvent.model_validate(payload)
    assert event.event_type == "retrieval_start"


def test_event_type_rejects_entry_removed_from_config(tmp_path, monkeypatch):
    config_file = tmp_path / "event_types.md"
    config_file.write_text("- tool_start\n")
    monkeypatch.setattr(trace_event_module, "_EVENT_TYPES_FILE", config_file)

    payload = {**VALID_PAYLOAD, "event_type": "llm_end"}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_event_type_rejects_everything_when_config_is_empty(tmp_path, monkeypatch):
    config_file = tmp_path / "event_types.md"
    config_file.write_text("# Event Types\n\n(none defined yet)\n")
    monkeypatch.setattr(trace_event_module, "_EVENT_TYPES_FILE", config_file)

    payload = {**VALID_PAYLOAD, "event_type": "llm_end"}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


# ---------------------------------------------------------------------------
# payload edge cases
# ---------------------------------------------------------------------------


def test_payload_empty_dict_is_accepted():
    """Free-form: no required sub-keys, so an empty dict is valid on its own terms."""
    payload = {**VALID_PAYLOAD, "payload": {}}
    event = TraceEvent.model_validate(payload)
    assert event.payload == {}


def test_payload_nested_structures_round_trip():
    nested_payload = {
        "args": {"query": "foo", "filters": [1, 2, {"three": 3}]},
        "ok": True,
        "score": 0.5,
        "note": None,
    }
    payload = {**VALID_PAYLOAD, "payload": nested_payload}
    event = TraceEvent.model_validate(payload)
    round_tripped = TraceEvent.model_validate_json(event.model_dump_json())
    assert round_tripped.payload == nested_payload


@pytest.mark.parametrize("bad_value", [None, "a string", ["not", "a", "dict"], 123])
def test_payload_non_dict_is_rejected(bad_value):
    payload = {**VALID_PAYLOAD, "payload": bad_value}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_payload_non_json_native_python_value_is_silently_accepted_and_coerced():
    """`payload: dict[str, Any]` doesn't validate JSON-serializability at
    construction time when built from Python objects (only JSON text is
    inherently JSON-native). A `set` is accepted here and silently turned
    into a list on serialization — lossy (order not guaranteed, and it's no
    longer a set) and NOT rejected. This is a real gap: a non-JSON-serializable
    Python object passed by an in-process caller (e.g. Interceptor) won't be
    caught by this schema at all; it only surfaces as changed data after a
    round trip. Documented here rather than silently assumed away."""
    event = TraceEvent.model_validate({**VALID_PAYLOAD, "payload": {"weird": {1, 2, 3}}})
    assert event.payload["weird"] == {1, 2, 3}

    round_tripped = TraceEvent.model_validate_json(event.model_dump_json())
    assert isinstance(round_tripped.payload["weird"], list)
    assert set(round_tripped.payload["weird"]) == {1, 2, 3}


# ---------------------------------------------------------------------------
# token_usage edge cases
# ---------------------------------------------------------------------------


def test_token_usage_total_mismatch_is_rejected():
    payload = {
        **VALID_PAYLOAD,
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 999},
    }
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_token_usage_negative_subfield_is_rejected(field):
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    usage[field] = -1
    payload = {**VALID_PAYLOAD, "token_usage": usage}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


@pytest.mark.parametrize("missing_field", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_token_usage_missing_subfield_is_rejected(missing_field):
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    del usage[missing_field]
    payload = {**VALID_PAYLOAD, "token_usage": usage}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_token_usage_extra_subfield_is_rejected():
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cached_tokens": 2}
    payload = {**VALID_PAYLOAD, "token_usage": usage}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_token_usage_zero_values_are_accepted():
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    payload = {**VALID_PAYLOAD, "token_usage": usage}
    event = TraceEvent.model_validate(payload)
    assert event.token_usage == TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


@pytest.mark.parametrize("bad_value", ["not-an-object", [1, 2, 3], 5])
def test_token_usage_wrong_type_is_rejected(bad_value):
    payload = {**VALID_PAYLOAD, "token_usage": bad_value}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
@pytest.mark.parametrize("bad_value", [None, "abc", [1], {"v": 1}])
def test_token_usage_subfield_wrong_type_is_rejected(field, bad_value):
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    usage[field] = bad_value
    payload = {**VALID_PAYLOAD, "token_usage": usage}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_token_usage_subfield_fractional_float_is_rejected(field):
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    usage[field] = usage[field] + 0.5
    payload = {**VALID_PAYLOAD, "token_usage": usage}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_token_usage_subfield_whole_number_float_is_coerced(field):
    """Mirrors step_index's whole-number-float coercion: documented, not assumed."""
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    usage[field] = float(usage[field])
    payload = {**VALID_PAYLOAD, "token_usage": usage}
    event = TraceEvent.model_validate(payload)
    assert getattr(event.token_usage, field) == int(usage[field])


def test_token_usage_subfield_bool_is_coerced_to_int():
    """Same footgun as step_index: bool is an int subclass, so pydantic accepts
    it by default. Kept consistent (sum still matches) and documented."""
    usage = {"prompt_tokens": True, "completion_tokens": 0, "total_tokens": 1}
    payload = {**VALID_PAYLOAD, "token_usage": usage}
    event = TraceEvent.model_validate(payload)
    assert event.token_usage.prompt_tokens == 1


def test_latency_ms_present_without_token_usage():
    """The two optional fields are independent — an event can report timing
    without a token breakdown (e.g. a tool call has latency but no tokens)."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "token_usage"}
    payload["event_type"] = "tool_end"
    event = TraceEvent.model_validate(payload)
    assert event.token_usage is None
    assert event.latency_ms == pytest.approx(842.5)


def test_token_usage_present_without_latency_ms():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "latency_ms"}
    event = TraceEvent.model_validate(payload)
    assert event.latency_ms is None
    assert event.token_usage is not None


# ---------------------------------------------------------------------------
# latency_ms edge cases
# ---------------------------------------------------------------------------


def test_latency_ms_zero_is_accepted():
    payload = {**VALID_PAYLOAD, "latency_ms": 0}
    event = TraceEvent.model_validate(payload)
    assert event.latency_ms == 0


def test_latency_ms_negative_is_rejected():
    payload = {**VALID_PAYLOAD, "latency_ms": -0.01}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_latency_ms_numeric_string_is_coerced():
    """Pydantic float fields coerce numeric strings; document that explicitly."""
    payload = {**VALID_PAYLOAD, "latency_ms": "12.5"}
    event = TraceEvent.model_validate(payload)
    assert event.latency_ms == pytest.approx(12.5)


@pytest.mark.parametrize("bad_value", ["not-a-number", [1.0], {"v": 1.0}])
def test_latency_ms_wrong_type_is_rejected(bad_value):
    payload = {**VALID_PAYLOAD, "latency_ms": bad_value}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_latency_ms_accepts_int_and_coerces_to_float():
    payload = {**VALID_PAYLOAD, "latency_ms": 500}
    event = TraceEvent.model_validate(payload)
    assert event.latency_ms == 500.0
    assert isinstance(event.latency_ms, float)


# ---------------------------------------------------------------------------
# timestamp edge cases (mirrors HaltSignal's timestamp contract)
# ---------------------------------------------------------------------------


def test_timestamp_with_offset_is_accepted():
    payload = {**VALID_PAYLOAD, "timestamp": "2026-09-18T12:34:56+05:30"}
    event = TraceEvent.model_validate(payload)
    assert event.timestamp.utcoffset() == timedelta(hours=5, minutes=30)


def test_timestamp_naive_datetime_is_rejected():
    payload = {**VALID_PAYLOAD, "timestamp": "2026-09-18T12:34:56"}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_timestamp_naive_datetime_object_is_rejected():
    payload = {**VALID_PAYLOAD, "timestamp": datetime(2026, 9, 18, 12, 34, 56)}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_timestamp_invalid_format_is_rejected():
    payload = {**VALID_PAYLOAD, "timestamp": "not-a-timestamp"}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


@pytest.mark.parametrize("bad_value", [None, ["2026-09-18T12:34:56Z"], {"a": 1}])
def test_timestamp_wrong_type_is_rejected(bad_value):
    payload = {**VALID_PAYLOAD, "timestamp": bad_value}
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_timestamp_unix_epoch_number_is_coerced_to_utc():
    payload = {**VALID_PAYLOAD, "timestamp": 1789732496}
    event = TraceEvent.model_validate(payload)
    assert event.timestamp.tzinfo is not None

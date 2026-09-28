"""Phase 7: security review finding -- a sanitization failure must never
leak the raw, unsanitized event (potentially containing real PII) into
application logs.

Before this fix, SanitizingSink.__call__ had no try/except around its own
sanitization call. If it ever raised (a scrubber bug, a Presidio/spaCy
edge case, a future code change), the exception propagated up to
CaptureDispatcher's generic catch-all, which logs the event via `%r` --
including whatever raw PII was in its payload. The content that triggers a
sanitization failure is often exactly the content most likely to contain
real PII, making this a realistic, not theoretical, leak path.

Fix: SanitizingSink now catches its own sanitization errors, logs only the
trace_id (an identifier, not sensitive content) at ERROR level, and drops
the event -- fail-closed, matching the existing behavior for the queue
side (an unhandled exception already meant the event never reached the
queue); this only makes the LOG side safe too.
"""
import logging

import pytest

from sanitizer.pipeline import SanitizingSink

POLICY = {"email": "mask"}
RAW_PII = "super-secret-value-alice@example.com-should-never-be-logged"


class RecordingSink:
    def __init__(self):
        self.received = []

    def __call__(self, event):
        self.received.append(event)


class _RaisingNerScrubber:
    """Simulates a real scrubbing failure (e.g. a Presidio/spaCy edge
    case) -- deliberately generic, not tied to one specific library bug,
    since the point is the failure-handling contract, not one exact repro."""

    def redact_with_policy(self, text, policy):
        raise RuntimeError("simulated scrubber failure")


def test_sanitization_failure_drops_the_event_instead_of_forwarding_it_unsanitized():
    inner = RecordingSink()
    sink = SanitizingSink(inner=inner, policy=POLICY, ner_scrubber=_RaisingNerScrubber())

    sink({"trace_id": "thread-1", "payload": {"note": RAW_PII}})

    assert inner.received == []  # fail-closed: never forwarded, sanitized or not


def test_sanitization_failure_log_never_contains_the_raw_payload(caplog):
    inner = RecordingSink()
    sink = SanitizingSink(inner=inner, policy=POLICY, ner_scrubber=_RaisingNerScrubber())

    with caplog.at_level(logging.ERROR):
        sink({"trace_id": "thread-1", "payload": {"note": RAW_PII}})

    full_log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert RAW_PII not in full_log_text
    assert "alice@example.com" not in full_log_text


def test_sanitization_failure_log_includes_trace_id_for_correlation(caplog):
    inner = RecordingSink()
    sink = SanitizingSink(inner=inner, policy=POLICY, ner_scrubber=_RaisingNerScrubber())

    with caplog.at_level(logging.ERROR):
        sink({"trace_id": "thread-correlate-me", "payload": {"note": RAW_PII}})

    full_log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "thread-correlate-me" in full_log_text


def test_sanitization_failure_is_logged_at_error_level(caplog):
    inner = RecordingSink()
    sink = SanitizingSink(inner=inner, policy=POLICY, ner_scrubber=_RaisingNerScrubber())

    with caplog.at_level(logging.ERROR):
        sink({"trace_id": "thread-1", "payload": {"note": RAW_PII}})

    assert len(caplog.records) >= 1
    assert all(record.levelno >= logging.ERROR for record in caplog.records)


def test_successful_sanitization_is_unaffected_by_the_new_error_handling():
    inner = RecordingSink()
    sink = SanitizingSink(inner=inner, policy=POLICY)  # no raising scrubber -- the normal path

    sink({"trace_id": "thread-1", "payload": {"note": "contact alice@example.com"}})

    assert len(inner.received) == 1
    assert "alice@example.com" not in inner.received[0]["payload"]["note"]

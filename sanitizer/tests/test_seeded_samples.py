"""End-to-end seeded PII/PHI/financial sample suite (Phase 2). Runs every
seeded sample from test_scrubbers.py (structured: email/phone/ssn/
credit_card/api_key) and test_ner.py (unstructured: names/addresses/
medical terms) through ONE combined SanitizingSink -- the same entry
point production code would use -- and asserts zero instances of each
original sensitive value survive, per this task's AC.

NerScrubber is expensive to construct (loads a spaCy model): built once
per test module via a module-scoped fixture.
"""
import threading

import pytest

from sanitizer.ner import NerScrubber
from sanitizer.pipeline import SanitizingSink

POLICY = {
    "email": "mask",
    "phone": "mask",
    "ssn": "drop",
    "credit_card": "mask",
    "api_key": "drop",
    "person": "mask",
    "address": "mask",
    "medical_condition": "drop",
}

# Structured (regex-caught): PII + financial.
STRUCTURED_SAMPLES = [
    "john.doe@example.com", "jane_smith99@sub.domain.co.uk", "test+tag@gmail.com", "a.b-c@x.io",
    "+1 415-555-2671", "(415) 555-2671", "415.555.2671", "+44 20 7946 0958",
    "123-45-6789", "987-65-4321", "001-01-0001", "555-55-5555",
    "4111111111111111", "5500000000000004", "340000000000009", "6011000000000004",
    "sk-ABCDEFGHIJKLMNOPQRSTUVWX", "sk-1234567890abcdefghijklmn",
    "AKIAABCDEFGHIJKLMNOP", "AKIAIOSFODNN7EXAMPLE",
]

# Unstructured (NER-caught): PHI.
UNSTRUCTURED_SAMPLES = [
    "John Smith", "Mary Johnson", "Robert Brown", "Linda Davis",
    "Michael Wilson", "Patricia Moore", "James Taylor",
    "221B Baker Street", "1600 Pennsylvania Avenue", "42 Wallaby Way",
    "10 Downing Street", "350 Fifth Avenue", "1 Infinite Loop Drive", "500 Main Street",
    "diabetes mellitus", "hypertension", "asthma", "epilepsy",
    "chronic kidney disease", "myocardial infarction", "pneumonia",
]

ALL_SAMPLES = STRUCTURED_SAMPLES + UNSTRUCTURED_SAMPLES


@pytest.fixture(scope="module")
def ner_scrubber():
    return NerScrubber()


@pytest.fixture
def sink(ner_scrubber):
    received = []
    lock = threading.Lock()

    def recording_inner(event):
        with lock:
            received.append(event)

    sanitizing_sink = SanitizingSink(inner=recording_inner, policy=POLICY, ner_scrubber=ner_scrubber)
    sanitizing_sink.received = received
    return sanitizing_sink


def test_at_least_twenty_combined_seeded_samples_are_defined():
    assert len(ALL_SAMPLES) >= 20


@pytest.mark.parametrize("sample", ALL_SAMPLES)
def test_seeded_sample_has_zero_instances_in_output(sink, sample):
    event = {"payload": {"content": f"Here is some context: {sample} and more text after."}}

    sink(event)

    output = sink.received[0]["payload"]["content"]
    assert sample not in output


def test_sentence_combining_pii_phi_and_financial_data_is_fully_scrubbed(sink):
    event = {
        "payload": {
            "content": (
                "Patient John Smith, living at 221B Baker Street, was diagnosed with "
                "diabetes mellitus. Contact billing at alice@example.com or "
                "415-555-2671, card 4111111111111111, SSN 123-45-6789."
            )
        }
    }

    sink(event)

    output = sink.received[0]["payload"]["content"]
    for sensitive in [
        "John Smith", "221B Baker Street", "diabetes mellitus",
        "alice@example.com", "415-555-2671", "4111111111111111", "123-45-6789",
    ]:
        assert sensitive not in output


def test_clean_text_with_no_sensitive_data_passes_through(sink):
    event = {"payload": {"content": "This is a perfectly normal message with nothing sensitive."}}

    sink(event)

    assert sink.received[0]["payload"]["content"] == event["payload"]["content"]

"""Tests for NER-based unstructured PII/PHI detection (Phase 2). Written
before the implementation (sanitizer.ner does not exist yet) -- run
`pytest` to see them fail with a collection error until
src/sanitizer/ner.py exists.

NerScrubber loads a spaCy model via Presidio, which is slow to construct
(~1s+) -- built once per test module via a module-scoped fixture, not
per test.
"""
import pytest

from sanitizer.ner import NerScrubber

NAME_SAMPLES = [
    "John Smith", "Mary Johnson", "Robert Brown", "Linda Davis",
    "Michael Wilson", "Patricia Moore", "James Taylor",
]
ADDRESS_SAMPLES = [
    "221B Baker Street", "1600 Pennsylvania Avenue", "42 Wallaby Way",
    "10 Downing Street", "350 Fifth Avenue", "1 Infinite Loop Drive",
    "500 Main Street",
]
MEDICAL_SAMPLES = [
    "diabetes mellitus", "hypertension", "asthma", "epilepsy",
    "chronic kidney disease", "myocardial infarction", "pneumonia",
]

ALL_SAMPLES = (
    [("PERSON", s) for s in NAME_SAMPLES]
    + [("ADDRESS", s) for s in ADDRESS_SAMPLES]
    + [("MEDICAL_CONDITION", s) for s in MEDICAL_SAMPLES]
)


@pytest.fixture(scope="module")
def scrubber():
    return NerScrubber()


def _detected(scrubber, entity_type, sample):
    text = f"Context before. {sample} appears here in the middle of the sentence."
    results = scrubber.detect(text)
    return any(r.entity_type == entity_type and text[r.start : r.end] == sample for r in results)


def test_at_least_twenty_seeded_samples_are_defined():
    assert len(ALL_SAMPLES) >= 20


@pytest.mark.parametrize("entity_type,sample", ALL_SAMPLES)
def test_seeded_sample_is_detected(scrubber, entity_type, sample):
    assert _detected(scrubber, entity_type, sample), f"{sample!r} ({entity_type}) was not detected"


def test_false_negative_rate_is_documented_and_under_threshold(scrubber):
    misses = [s for (t, s) in ALL_SAMPLES if not _detected(scrubber, t, s)]
    rate = len(misses) / len(ALL_SAMPLES)
    print(f"NER false-negative rate: {rate:.2%} ({len(misses)}/{len(ALL_SAMPLES)} missed: {misses})")
    assert rate < 0.05


def test_clean_text_with_no_pii_produces_no_detections(scrubber):
    text = "This is a perfectly ordinary sentence about nothing sensitive."
    assert scrubber.detect(text) == []


def test_multiple_entity_types_in_one_sentence_are_all_detected(scrubber):
    text = "Patient John Smith of 221B Baker Street was diagnosed with asthma."
    types = {r.entity_type for r in scrubber.detect(text)}
    assert "PERSON" in types
    assert "ADDRESS" in types
    assert "MEDICAL_CONDITION" in types


# ---------------------------------------------------------------------------
# redact(): shares apply_action with the regex scrubbers (mask/hash/drop)
# ---------------------------------------------------------------------------


def test_redact_mask_action_replaces_with_placeholder(scrubber):
    redacted = scrubber.redact("Patient John Smith has asthma.", action="mask")
    assert "John Smith" not in redacted
    assert "asthma" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_hash_action_is_distinguishable_from_mask(scrubber):
    redacted = scrubber.redact("Patient John Smith has asthma.", action="hash")
    assert "John Smith" not in redacted
    assert "REDACTED:" in redacted
    assert "[REDACTED]" not in redacted


def test_redact_drop_action_leaves_nothing_in_place(scrubber):
    redacted = scrubber.redact("Patient John Smith has asthma.", action="drop")
    assert "John Smith" not in redacted
    assert "asthma" not in redacted
    assert "REDACTED" not in redacted


def test_redact_default_action_is_mask(scrubber):
    redacted = scrubber.redact("Patient John Smith visited.")
    assert "[REDACTED]" in redacted


# ---------------------------------------------------------------------------
# redact_with_policy(): per-entity-type action lookup (new method, doesn't
# change redact()'s existing signature/behavior above)
# ---------------------------------------------------------------------------


def test_redact_with_policy_applies_different_action_per_entity_type(scrubber):
    policy = {"person": "mask", "medical_condition": "drop"}
    redacted = scrubber.redact_with_policy("Patient John Smith has asthma.", policy)

    assert "John Smith" not in redacted
    assert "asthma" not in redacted
    assert "[REDACTED]" in redacted  # from the masked name
    # the dropped medical condition leaves nothing in its place, unlike mask
    assert redacted.count("REDACTED") == 1


def test_redact_with_policy_falls_back_to_mask_for_unlisted_entity_type(scrubber):
    redacted = scrubber.redact_with_policy("Patient John Smith visited.", policy={})
    assert "John Smith" not in redacted
    assert "[REDACTED]" in redacted

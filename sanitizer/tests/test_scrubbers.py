"""Tests for regex scrubbers (Phase 2). Written before the implementation
(sanitizer.scrubbers does not exist yet) -- run `pytest` to see them fail
with a collection error until src/sanitizer/scrubbers.py exists.
"""
import pytest

from sanitizer.scrubbers import scrub_text

DEFAULT_POLICY = {
    "email": "mask",
    "phone": "mask",
    "ssn": "drop",
    "credit_card": "mask",
    "api_key": "drop",
}

# >=20 seeded samples across the 5 data types, 4 each, per the AC.
SEEDED_SAMPLES = [
    ("email", "john.doe@example.com"),
    ("email", "jane_smith99@sub.domain.co.uk"),
    ("email", "test+tag@gmail.com"),
    ("email", "a.b-c@x.io"),
    ("phone", "+1 415-555-2671"),
    ("phone", "(415) 555-2671"),
    ("phone", "415.555.2671"),
    ("phone", "+44 20 7946 0958"),
    ("ssn", "123-45-6789"),
    ("ssn", "987-65-4321"),
    ("ssn", "001-01-0001"),
    ("ssn", "555-55-5555"),
    ("credit_card", "4111111111111111"),  # Visa
    ("credit_card", "5500000000000004"),  # MasterCard
    ("credit_card", "340000000000009"),  # Amex
    ("credit_card", "6011000000000004"),  # Discover
    ("api_key", "sk-ABCDEFGHIJKLMNOPQRSTUVWX"),
    ("api_key", "sk-1234567890abcdefghijklmn"),
    ("api_key", "AKIAABCDEFGHIJKLMNOP"),
    ("api_key", "AKIAIOSFODNN7EXAMPLE"),
]


def test_at_least_twenty_seeded_samples_are_defined():
    assert len(SEEDED_SAMPLES) >= 20


@pytest.mark.parametrize("data_type,sample", SEEDED_SAMPLES)
def test_seeded_sample_is_fully_removed_from_output(data_type, sample):
    text = f"here is some context: {sample} and more text after"

    scrubbed = scrub_text(text, DEFAULT_POLICY)

    assert sample not in scrubbed


def test_clean_text_with_no_pii_passes_through_unchanged():
    text = "This is a perfectly normal sentence with no sensitive data at all."
    assert scrub_text(text, DEFAULT_POLICY) == text


# ---------------------------------------------------------------------------
# Action semantics
# ---------------------------------------------------------------------------


def test_mask_action_replaces_with_a_placeholder():
    policy = {"email": "mask"}
    scrubbed = scrub_text("contact: alice@example.com", policy)
    assert "alice@example.com" not in scrubbed
    assert "[REDACTED]" in scrubbed


def test_hash_action_replaces_with_a_stable_hash_not_the_placeholder():
    policy = {"email": "hash"}
    scrubbed = scrub_text("contact: alice@example.com", policy)
    assert "alice@example.com" not in scrubbed
    assert "[REDACTED]" not in scrubbed  # distinguishable from mask
    assert "REDACTED:" in scrubbed


def test_hash_action_is_stable_for_the_same_input():
    policy = {"email": "hash"}
    first = scrub_text("contact: alice@example.com", policy)
    second = scrub_text("contact: alice@example.com", policy)
    assert first == second


def test_hash_action_differs_for_different_input():
    policy = {"email": "hash"}
    a = scrub_text("contact: alice@example.com", policy)
    b = scrub_text("contact: bob@example.com", policy)
    assert a != b


def test_drop_action_removes_the_match_entirely():
    policy = {"ssn": "drop"}
    scrubbed = scrub_text("ssn on file: 123-45-6789 end", policy)
    assert "123-45-6789" not in scrubbed
    assert "[REDACTED" not in scrubbed  # nothing left in its place


# ---------------------------------------------------------------------------
# Ordering: more specific patterns must run before the permissive phone one
# ---------------------------------------------------------------------------


def test_credit_card_is_scrubbed_before_phone_pattern_could_mangle_it():
    policy = {"credit_card": "drop", "phone": "mask"}
    scrubbed = scrub_text("card number 4111111111111111 on file", policy)
    assert "4111111111111111" not in scrubbed
    # if phone ran first/independently it could partially match a digit
    # run inside the card number and leave a mangled remainder behind
    assert not any(char.isdigit() for char in scrubbed)


# ---------------------------------------------------------------------------
# Unlisted data type defaults (uses sanitizer.policy.get_action's default)
# ---------------------------------------------------------------------------


def test_data_type_not_in_policy_still_gets_masked_by_default():
    scrubbed = scrub_text("contact: alice@example.com", policy={})
    assert "alice@example.com" not in scrubbed
    assert "[REDACTED]" in scrubbed

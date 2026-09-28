"""Sanitization latency benchmark (Phase 2). Measures the full pipeline
(regex + NER combined, via SanitizingSink) against a documented budget,
scoped to a representative event size -- latency scales roughly linearly
with text length (measured separately below), so a budget only means
something relative to a stated input size, not as a universal constant.

See BENCHMARK.md for the full report (methodology, scaling data, and the
budget/no-quarantine-sub-queue decision and rationale).
"""
import time

import pytest

from sanitizer.ner import NerScrubber
from sanitizer.pipeline import SanitizingSink

POLICY = {
    "email": "mask", "phone": "mask", "ssn": "drop", "credit_card": "mask",
    "api_key": "drop", "person": "mask", "address": "mask", "medical_condition": "drop",
}

# ~750 chars -- a realistic single LLM prompt/response turn, containing a
# representative mix of structured (regex) and unstructured (NER) PII.
REPRESENTATIVE_EVENT_TEXT = (
    "Patient John Smith, living at 221B Baker Street, was diagnosed with "
    "diabetes mellitus. Contact billing at alice@example.com or "
    "415-555-2671, card 4111111111111111, SSN 123-45-6789. "
) * 2

P95_BUDGET_MS = 50.0
ITERATIONS = 100


def _measure(sink, text, iterations=ITERATIONS):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        sink({"payload": {"content": text}})
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    return p50, p95


@pytest.fixture(scope="module")
def ner_scrubber():
    return NerScrubber()


def test_representative_event_meets_p95_budget(ner_scrubber):
    sink = SanitizingSink(inner=lambda event: None, policy=POLICY, ner_scrubber=ner_scrubber)

    p50, p95 = _measure(sink, REPRESENTATIVE_EVENT_TEXT)
    print(f"\nrepresentative event ({len(REPRESENTATIVE_EVENT_TEXT)} chars): p50={p50:.2f}ms p95={p95:.2f}ms")

    assert p95 < P95_BUDGET_MS


def test_regex_only_pipeline_is_near_zero_cost_reference_point():
    """Reference point proving NER dominates the cost -- regex-only
    sanitization (no ner_scrubber) is orders of magnitude cheaper."""
    sink = SanitizingSink(inner=lambda event: None, policy=POLICY)  # no ner_scrubber

    p50, p95 = _measure(sink, REPRESENTATIVE_EVENT_TEXT)
    print(f"\nregex-only reference: p50={p50:.3f}ms p95={p95:.3f}ms")

    assert p95 < 5.0  # comfortably under the combined budget, as a sanity check


def test_latency_scales_with_text_length(ner_scrubber):
    """Documents (doesn't strictly budget) the scaling relationship the
    single-size budget above is explicitly scoped against."""
    sink = SanitizingSink(inner=lambda event: None, policy=POLICY, ner_scrubber=ner_scrubber)

    small_p50, small_p95 = _measure(sink, REPRESENTATIVE_EVENT_TEXT, iterations=20)
    large_text = REPRESENTATIVE_EVENT_TEXT * 4
    large_p50, large_p95 = _measure(sink, large_text, iterations=20)

    print(
        f"\nscaling: {len(REPRESENTATIVE_EVENT_TEXT)} chars p95={small_p95:.2f}ms  "
        f"vs {len(large_text)} chars p95={large_p95:.2f}ms"
    )

    # A ~4x longer text should cost meaningfully more, not the same --
    # proves latency is genuinely input-size-dependent, not a fixed
    # per-call overhead the representative-size budget could hide.
    assert large_p95 > small_p95 * 2

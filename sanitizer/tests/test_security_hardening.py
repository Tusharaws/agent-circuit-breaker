"""Phase 7: security review findings for the sanitization layer.

Two confirmed, fixed issues, both found via adversarial (not realistic)
input -- every other sanitizer test assumes well-formed input, which is
exactly what a security review should NOT assume.

1. ReDoS in the email regex: `[\\w.+-]+@[\\w-]+\\.[a-zA-Z]{2,}` is
   quadratic-time on any input with a long run of word characters and no
   `@` -- measured 14.6s on a 100,000-char string of "4"s. Since
   SanitizingSink runs synchronously on CaptureDispatcher's single
   background thread, one such event backs up ALL capture -- a real DoS,
   not just a slow edge case.
2. A previously-undiscovered correctness gap the ReDoS fix exposed: the
   existing regex only fully redacts single-label domains -- multi-label
   domains (subdomains, ".co.uk"-style TLDs) leaked their trailing labels.
   The existing test (`test_seeded_sample_is_fully_removed_from_output`)
   didn't catch this because it only asserts the *whole* seeded string is
   absent from the output, which is true even when a fragment (the domain
   suffix) survives.
"""
import time

from sanitizer.scrubbers import scrub_text

POLICY = {"email": "mask"}

# Large enough to make quadratic-vs-linear behavior obvious, small enough
# to run fast even if a regression reintroduces the O(n^2) behavior partway.
ADVERSARIAL_SIZE = 100_000
MAX_ALLOWED_SECONDS = 1.0  # generous margin over the ~20ms fixed-pattern measurement


def test_long_word_char_run_with_no_at_sign_does_not_cause_quadratic_slowdown():
    adversarial_text = "4" * ADVERSARIAL_SIZE  # no '@' anywhere -- the ReDoS trigger

    start = time.perf_counter()
    scrub_text(adversarial_text, POLICY)
    elapsed = time.perf_counter() - start

    assert elapsed < MAX_ALLOWED_SECONDS, (
        f"scrub_text took {elapsed:.2f}s on a {ADVERSARIAL_SIZE}-char adversarial "
        f"input with no '@' -- likely quadratic/exponential backtracking regression"
    )


def test_long_dot_separated_run_with_no_terminating_letters_does_not_cause_quadratic_slowdown():
    """The domain-matching fix itself uses a repeated group
    (`(?:[\\w-]{1,63}\\.)+`) -- this specifically stress-tests THAT group's
    own backtracking behavior, not just the original local-part group."""
    adversarial_text = "a." * (ADVERSARIAL_SIZE // 2)  # many dots, never a valid TLD

    start = time.perf_counter()
    scrub_text(adversarial_text, POLICY)
    elapsed = time.perf_counter() - start

    assert elapsed < MAX_ALLOWED_SECONDS, (
        f"scrub_text took {elapsed:.2f}s on a dot-separated adversarial input -- "
        f"likely quadratic/exponential backtracking regression"
    )


def test_multi_label_domain_is_fully_redacted_not_just_the_first_label():
    text = "contact jane_smith99@sub.domain.co.uk for details"

    scrubbed = scrub_text(text, POLICY)

    # The existing "sample not in scrubbed" check passes even when only
    # PART of the domain is redacted -- assert no fragment of the original
    # domain survives, not just that the whole string is gone.
    for leaked_fragment in ("sub.domain", "co.uk", ".co.uk", "domain.co.uk"):
        assert leaked_fragment not in scrubbed, f"domain fragment {leaked_fragment!r} leaked: {scrubbed!r}"
    assert "jane_smith99" not in scrubbed


def test_subdomain_email_is_fully_redacted():
    text = "reach me at user@mail.corp.internal.example.com anytime"

    scrubbed = scrub_text(text, POLICY)

    assert "mail.corp.internal.example.com" not in scrubbed
    assert "corp.internal.example.com" not in scrubbed
    assert "example.com" not in scrubbed

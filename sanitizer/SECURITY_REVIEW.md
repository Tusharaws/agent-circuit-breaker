# Security Review: Sanitization Layer (Phase 7)

Self-conducted structured review of `sanitizer` and its integration point
with the rest of the pipeline. Every prior sanitizer test (seeded PII
samples, NER coverage, latency benchmark) assumes well-formed, realistic
input; this review specifically looks for what happens with hostile,
malformed, or oversized input instead -- a different question, and one
that found real gaps.

Reproducible: `pytest tests/test_security_hardening.py tests/test_sanitization_failure_handling.py -v`

## Findings

### [HIGH, FIXED] ReDoS in the email regex

`[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}` is quadratic-time on any input containing a
long run of word characters with no `@` -- measured **14.6 seconds** to
scrub a 100,000-character string of `"4"`s. This is not a contrived shape:
any large tool-output blob, base64 payload, or long token/hash with no `@`
triggers it. Since `SanitizingSink` runs synchronously on
`CaptureDispatcher`'s single background thread (by design -- see
`interceptor/README.md`), one such event backs up **all** capture for the
duration, a real DoS vector, not just a slow edge case.

**Fix:** bounded the quantifiers to real RFC 5321 (local part <= 64
octets) / DNS (label <= 63 octets) limits -- not arbitrary numbers, actual
protocol limits no legitimate email exceeds:

```
before: [\w.+-]+@[\w-]+\.[a-zA-Z]{2,}
after:  [\w.+-]{1,64}@(?:[\w-]{1,63}\.)+[a-zA-Z]{2,24}
```

Measured after the fix: the same 100,000-char adversarial input drops to
**~20ms** (~700x). Verified against a second adversarial shape too (a long
run of `"a."` with no terminating letters, stress-testing the new repeated
domain-label group specifically) -- also ~20ms.

### [HIGH, FIXED] A previously-undiscovered correctness gap the ReDoS fix exposed

The *original* regex only fully redacted single-label domains. A
multi-label domain -- a subdomain, or a `.co.uk`-style two-part TLD --
only had its first label consumed, leaking the rest:

```
input:  jane_smith99@sub.domain.co.uk
output: [REDACTED].co.uk        <- domain suffix leaked
```

**Why the existing test suite didn't catch this:**
`sanitizer/tests/test_scrubbers.py::test_seeded_sample_is_fully_removed_from_output`
(which already seeds `"jane_smith99@sub.domain.co.uk"` as a sample)
asserts only that the *whole original string* is absent from the output --
true even when a fragment of it (the domain suffix) survives. A "fully
removed" test that only checks whole-string absence has a blind spot for
partial redaction.

**Fix:** the same regex change above (`(?:[\w-]{1,63}\.)+`, one-or-more
dot-separated labels before the TLD) fixes this as a side effect. New
tests in `test_security_hardening.py` assert **no fragment** of the
domain survives, not just the whole string.

### [HIGH, FIXED] Raw, unsanitized PII could leak into application logs on a sanitization failure

`SanitizingSink.__call__` had no `try`/`except` around its own
sanitization call. If sanitization ever raised (a scrubber bug, a
Presidio/spaCy edge case, or a future code change), the exception
propagated up to `CaptureDispatcher`'s generic catch-all
(`interceptor/src/interceptor/capture.py`), which logs
`"capture sink raised for event %r"` -- and `event` there is the
**original, pre-sanitization** object, logged with `%r`, i.e. with
whatever raw PII was in its payload. The content that triggers a
sanitization failure is often exactly the content most likely to actually
contain real PII (adversarial or unusual input), making this a realistic
leak path, not a theoretical one.

**Fix:** `SanitizingSink` now catches its own sanitization errors, logs
only `trace_id` (a bare identifier, not sensitive content) at `ERROR`
level, and drops the event -- fail-closed, matching the *existing*
behavior on the queue side (an unhandled exception already meant the
event never reached the queue); this only makes the **log** side safe
too, not a behavior change to what reaches the queue.

### [HIGH, FIXED] No structural guarantee against bypassing sanitization

Nothing prevented production wiring
`CaptureDispatcher(sink=make_queue_sink(queue_client.append_event))` --
skipping `SanitizingSink` entirely -- from compiling or running. This was
convention/documentation only ("Production wiring: ..." in
`interceptor/README.md`), enforced by nothing in the type system or
runtime. `sanitizer/tests/test_pipeline.py::test_bypassing_sanitizing_sink_leaks_raw_pii_demonstrating_the_risk`
actually *demonstrates* this works (PII leaks) rather than blocking it --
despite Phase 2's own original acceptance criteria wanting "an integration
test attempting to bypass it fails/is blocked."

Two remediation options were considered, each with real tradeoffs (a
runtime marker/guard on `QueueClient.append_event` would give genuine
structural enforcement, but conflicts with `queue_client`'s established
design principle of staying generic and payload-agnostic -- it doesn't
know what "sanitization" is, deliberately). Decided (user confirmed): a
"blessed," pre-composed factory function.

**Fix:** `interceptor.queue_sink.make_sanitized_queue_sink(append_event,
policy=None, ner_scrubber=None)` composes `SanitizingSink` +
`make_queue_sink` in one call -- `CaptureDispatcher(sink=
make_sanitized_queue_sink(queue_client.append_event))` is now the
recommended production entry point (see `interceptor/README.md`). This
doesn't structurally *prevent* a caller from still reaching for the
unwrapped `make_queue_sink` (kept available for non-customer-data uses,
e.g. `evaluator`'s own eval-set harness seeding synthetic traces directly)
-- it makes the safe, sanitized path the single easiest thing to reach
for, rather than a 2-step manual composition someone could forget half of.
`sanitizer` is now a production dependency of `interceptor` as a result.

### Verified safe, no fix needed

The other 4 regex patterns (`credit_card`, `ssn`, `api_key`, `phone`) were
stress-tested against the same adversarial inputs (100,000-character runs
of digits/punctuation) that broke the email pattern. All bounded
throughout (`{n}` / `{n,m}` counts, no unbounded quantifier ever precedes
a required literal that might not appear) -- all completed in under 30ms.

### Lower severity / informational, not GA-blocking

- **spaCy model supply chain:** `en_core_web_sm` is installed from a
  GitHub releases URL (`sanitizer/pyproject.toml`) rather than a package
  index, with no hash pinning. Worth hardening eventually, not urgent.
- **Policy config fails loud, correctly (positive finding, not a gap):**
  `load_redaction_policy()` raises on a missing file (`FileNotFoundError`)
  or an unrecognized action (`ValueError`) rather than silently defaulting
  to no redaction. Confirmed this is fail-closed behavior, not an
  oversight worth flagging as a risk.

## Summary

| Finding | Severity | Status |
|---|---|---|
| Email regex ReDoS | High | **Fixed** |
| Multi-label domain leak | High | **Fixed** |
| Raw PII in failure logs | High | **Fixed** |
| No structural anti-bypass guarantee | High | **Fixed** |
| Other regex patterns | — | Verified safe |
| spaCy model supply chain | Low | Documented, not urgent |
| Policy config fail-closed behavior | — | Verified correct (positive finding) |

All 4 high-severity findings fixed and regression-tested (153/153
sanitizer tests, 85/85 interceptor tests, 11/11 fast integration-tests
green). GA-blocking work for this review is complete.

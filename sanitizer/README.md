# sanitizer

Cleans and normalizes trace events captured by `interceptor` before they are
handed to `queue-client` for publishing.

## Redaction policy config

`load_redaction_policy()` (`src/sanitizer/policy.py`) reads
`config/redaction_policy.md`, a `- data_type: action` mapping where action
is one of `mask`, `hash`, or `drop` (a fixed, non-extensible vocabulary —
unlike `event_type`/halt-`reason`, these three are hardcoded in
`VALID_ACTIONS`). `get_action(policy, data_type, default="mask")` looks up
the action for a data type, falling back to `default` if unlisted.

Unlike the `schemas` package's config files (`halt_reasons.md`,
`event_types.md`), which live inside the installed package and need a
rebuild to change, this loader also checks the
`SANITIZER_REDACTION_POLICY_PATH` environment variable first — an operator
can point it at an external, mounted file and change redaction behavior
without a code change *or a redeploy*, which this task's acceptance
criteria specifically calls for.

## Regex scrubbers

`scrub_text(text, policy)` (`src/sanitizer/scrubbers.py`) detects and
redacts 5 data types — `email`, `credit_card` (Visa/MasterCard/Amex/
Discover BIN ranges), `ssn`, `api_key` (`sk-*`, `AKIA*`), `phone`
(US and international) — applying whatever action `sanitizer.policy`
configures for each (falling back to `get_action`'s default for an
unlisted type).

Patterns run in a fixed order — email, credit_card, ssn, api_key, then
**phone last** — because phone regexes are inherently the most permissive
(short digit runs with optional separators); each pattern only sees what
earlier ones left in the text, so a credit card or SSN digit run can't get
partially matched and mangled by the phone pattern first.

`mask` replaces a match with `[REDACTED]`; `hash` with a stable
`[REDACTED:<12-hex-char sha256 prefix>]` (same input always hashes the
same, different input hashes differently); `drop` removes the match
entirely with nothing left in its place.

## Sanitize-before-queue pipeline step

`SanitizingSink` (`src/sanitizer/pipeline.py`) is the mandatory
sanitization step between capture and any network-facing sink — a
composable `sink` matching `CaptureDispatcher`'s existing
`Callable[[Any], None]` contract, following the same pattern
`interceptor`'s `BatchingSink` established. Production wiring:
`CaptureDispatcher(sink=SanitizingSink(inner=queue_client_instance.append_event))`.

`sanitize_value()` recursively scrubs every string anywhere in a value —
nested dicts and lists included — so it reaches deeply-nested `payload`
content, not just top-level fields. Handles both a real
`schemas.TraceEvent` (dumped, sanitized, rebuilt as a `TraceEvent`) and a
plain dict (`queue_client`'s own generic payload shape) without assuming
`TraceEvent` specifically.

**Architecture note:** sanitization is enforced as a composable wrapper,
not baked into `queue_client.append_event` itself — keeps `queue_client`
generic/payload-agnostic per its own original design, at the cost of this
being a reviewed-wiring guarantee (always compose `SanitizingSink`) rather
than one the transport layer enforces by itself. `tests/test_pipeline.py`
includes a regression guard demonstrating the risk this closes: the exact
same seeded-PII event, sent to a sink *without* `SanitizingSink`, leaks
the raw PII — proving why the wrapper is mandatory, not just documented
advice.

## Unstructured PII/PHI detection: NerScrubber

`NerScrubber` (`src/sanitizer/ner.py`) catches what the regex scrubbers
structurally can't — names, addresses, and medical terms in free text,
which have no fixed format to pattern-match. Built on Presidio +
spaCy (`en_core_web_sm`, the smallest model — sufficient for this scale).

Presidio's own default recognizers do **not** cover street addresses or
medical terms out of the box (confirmed empirically, not assumed —
Presidio alone only caught `PERSON` and a bare city name as `LOCATION` in
a sentence containing a full street address and a diagnosis). Both gaps
are closed with custom `PatternRecognizer`s: a regex-based one for
`ADDRESS`, and a curated ~20-term deny-list for `MEDICAL_CONDITION` (not
exhaustive — real medical NER at scale needs a specialized model like
scispaCy, out of scope here).

`redact(text, action="mask")` reuses `scrubbers.apply_action` — the same
mask/hash/drop vocabulary the regex scrubbers use, so NER-based and
regex-based redaction produce visibly consistent output. Constructing
`NerScrubber` loads a spaCy model (build once, reuse the instance).

Measured false-negative rate on 21 seeded samples (7 each: names,
addresses, medical terms): **0%**, comfortably under the AC's <5%
threshold.

`redact_with_policy(text, policy)` looks up the action per detected
entity type (`person`, `address`, `medical_condition` — added to the
default `redaction_policy.md`) instead of applying one action uniformly,
so NER-based redaction is config-driven the same way regex-based
redaction is. Added as a new method rather than changing `redact()`'s
existing signature, avoiding an unnecessary breaking change to
already-tested surface.

**Now wired into the production pipeline**: `SanitizingSink`/`sanitize_value`
take an optional `ner_scrubber` (default `None` = regex-only, preserving
prior behavior for existing callers) — when provided, every string gets
both a regex pass and an NER pass. `tests/test_seeded_samples.py` proves
this end-to-end: all 20 structured (email/phone/ssn/credit_card/api_key)
and 21 unstructured (names/addresses/medical terms) seeded samples — 41
total — have zero instances surviving in output through one combined
`SanitizingSink`, including a single sentence mixing all three categories
at once.

## Latency benchmark

See [BENCHMARK.md](BENCHMARK.md) for the full report. Summary: p95 ~7.2ms
for a representative ~362-char event through the complete pipeline (regex
+ NER), against a documented 50ms budget — comfortably under, so no
quarantine sub-queue was built (the AC's own conditional doesn't trigger).
Latency scales roughly linearly with text length; the budget is
explicitly scoped to the representative size, not a universal constant.

## Install

```bash
pip install -e ".[test]"
```

## End-to-end tracing (Phase 6)

`SanitizingSink.__call__` logs one structured JSON line per event on a
dedicated `"sanitizer.trace"` logger (DEBUG): `{"trace_id": ..., "service":
"sanitizer", "stage": "sanitized"}`. `trace_id` is pulled from whichever
shape the event has -- a `trace_id` attribute (a real `TraceEvent`) or a
`"trace_id"` dict key (queue_client's generic payload shape); an event
carrying neither is still sanitized and forwarded, just not trace-logged.
Same `trace_id`/`service`/`stage` convention as interceptor/queue-client/
evaluator/control-api's own trace loggers -- see `interceptor/README.md`
for why this is a shared convention, not a shared library.

## Test

```bash
pytest
```

## Security review (Phase 7)

See [SECURITY_REVIEW.md](SECURITY_REVIEW.md) for the full report. Summary:
found and fixed a ReDoS in the email regex (14.6s -> ~20ms on a
100,000-char adversarial input) and a related multi-label-domain
redaction gap it exposed, made sanitization failures fail closed on the
logging side (no raw PII in error logs, previously possible), and closed
a structural gap where production wiring could skip `SanitizingSink`
entirely (`interceptor.queue_sink.make_sanitized_queue_sink` is now the
blessed, single-call production entry point). All 4 high-severity
findings fixed.

## Status

Phase 2 complete: redaction policy config, regex scrubbers, the
sanitize-before-queue pipeline step (regex + NER combined), NER-based
unstructured PII/PHI detection, the end-to-end seeded PII/PHI/financial
sample suite, and latency benchmarking all implemented and tested. Phase 7
security review complete (all 4 high findings fixed; see above).

# Sanitization Latency Benchmark

Measured against the complete pipeline: regex scrubbing (`scrub_text`) +
NER-based detection (`NerScrubber`) combined via `SanitizingSink`, the same
entry point production code uses. Reproducible via
`pytest tests/test_latency_benchmark.py -s`.

## Methodology

Sanitization runs on `CaptureDispatcher`'s background thread, not the
customer's agent thread — this benchmark measures background-pipeline
throughput headroom, not anything the customer's agent waits on.

Latency scales roughly linearly with input text length (NER/spaCy
inference dominates the cost; regex scrubbing alone is near-zero). A
budget only means something relative to a stated representative event
size — a single millisecond number with no size reference would be
meaningless (generous for a 200-char message, impossible for a
10,000-char one). So this benchmark states a representative size
explicitly, plus the scaling relationship, rather than hiding it.

## Results

| Text size | p50 | p95 |
|---|---|---|
| 362 chars (representative single LLM turn, mixed PII) | ~7.0ms | ~7.2ms |
| 1,448 chars (~4x representative) | — | ~26.3ms |
| Regex-only (no NER), same representative text | ~0.02ms | ~0.03ms |

Measured with `en_core_web_sm` (spaCy's smallest English model) on this
development machine — not a production benchmark environment. Numbers
are indicative of the *shape* (NER dominates; roughly linear in text
length), not a guaranteed production SLA.

## Budget

**p95 < 50ms**, scoped to the ~362-char representative event size above.

Chosen with roughly 7x headroom over the measured p95 (~7.2ms) at that
size — generous enough for a slower environment or a somewhat larger
typical event, without being an arbitrary number disconnected from real
data. Whoever revisits this if typical event sizes grow substantially
larger should re-run the benchmark rather than assume the 50ms figure
still holds — the scaling data above shows it won't hold indefinitely
(a ~1,450-char event already measures ~26ms).

## Quarantine sub-queue: not built

The AC is conditional: *"if p95 exceeds the agreed budget, a quarantine
sub-queue is implemented."* Measured p95 (~7.2ms) is comfortably under
the 50ms budget at the representative size, so the condition is false —
building the sub-queue anyway would be unrequested scope. This is stated
explicitly here so the omission reads as a decision, not an oversight,
and so it's easy to revisit if the budget is ever exceeded in a real
deployment (e.g. much larger typical event sizes, or a slower/more
loaded environment than this benchmark ran on).

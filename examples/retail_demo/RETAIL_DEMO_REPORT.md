# Retail Demo — Validation Report (Story 2)

Reproducible via `pytest examples/retail_demo/tests -s` (fast scenarios only:
`pytest examples/retail_demo/tests -m "not integration" -s`). Mirrors the
methodology and format of `integration-tests/CONTROLLED_SCENARIOS_REPORT.md`,
applied to a realistic order-support scenario instead of the abstract
"stuck query" fixture used elsewhere in this project.

All 3 scenarios run through the actual full, real pipeline — real
`GraphCaptureHandler`, real `SanitizingSink` (via the blessed
`make_sanitized_queue_sink` entry point), real `QueueClient`, real
`evaluator`/`SLMClient` where noted, real `control-api` app — not isolated
unit-level fakes.

## Scenario 1: Runaway loop — detected and halted for real

**Target:** a customer whose order can't be resolved (an unresolvable
order reference) causes the agent to retry `lookup_order` with
paraphrased-but-equivalent queries — no exact repeats, a genuine semantic
loop — and the real evaluator detects and halts it within the same
`END_TO_END_HALT_SLA_MS = 2000ms` budget the main project's own controlled
scenarios use (real SLM inference + one guard-node check cycle, with
margin).

**Method:** the looping agent runs continuously against an unresolvable
reference (`ORD-99999`); after 1s of real retries building up in the
queue, the real evaluator (real SLM) evaluates the live window and, on a
positive verdict, calls the real `/halt` endpoint.

**Result:** real verdict `Verdict(is_loop=True, confidence=1.0,
reason='runaway_loop')`. Detect-to-halt measured at **630ms** (SLA
2000ms, ~3.2x margin). The agent froze mid-loop
(`state.next == ("guard",)`, fewer than 30 attempts made), confirming this
was a genuine detection-driven halt, not a run that simply completed.
**Pass.**

## Scenario 2: Normal support flow — zero false halts

**Target:** a customer whose order resolves normally never gets
halted — checked two ways: the agent's own halt status, and the real
evaluator's own verdict on the resulting window.

**Method:** the happy-path agent resolves a valid order query
(`ORD-10293`) in 2 real tool calls (`lookup_order` + `check_shipping_status`)
and reaches `END`. `HaltRegistry.is_halted()` is checked directly, and the
real evaluator (real SLM available) separately evaluates the same window.

**Result:** agent completed normally, `is_halted() == False` throughout.
Real verdict: `Verdict(is_loop=False, confidence=0.0, reason=None)` —
filtered before ever reaching the SLM, consistent with a clearly
non-repetitive, 2-step conversation. **Pass.**

## Scenario 3: Seeded PII — fully scrubbed end-to-end

**Target:** a customer looking up their order by email (a realistic
PII-bearing query, not a contrived string) never has that email land in
the durable queue.

**Method:** the happy-path agent resolves `jane.doe@example.com` (a real
seeded order's email) via `lookup_order` — the agent's own execution uses
the raw value internally, since it needs the real value to match. The
actual stored Redis payload (via `QueueClient.read_window`) is inspected
directly, not the sanitizer's own already-covered isolated unit tests.

**Result:** the agent resolved correctly (`order_id == "ORD-10293"`), and
the raw email **never** appears anywhere in the stored payload — replaced
with `[REDACTED]`, confirming active masking (not coincidental absence).
**Pass.**

## Summary

| Scenario | Target | Measured | Result |
|---|---|---|---|
| Runaway loop (paraphrased retries) | halt within SLA (2000ms) | 630ms | **Pass** |
| Normal support flow | 0 false halts | `is_loop=False, confidence=0.0` | **Pass** |
| PII in a real customer query | fully scrubbed | 0 leaked, actively masked | **Pass** |

All 3 scenarios pass. Unlike the main project's `CONTROLLED_SCENARIOS_REPORT.md`
(an abstract "stuck query" fixture and a synthetic PII string), this
report exercises the exact same real pipeline against a plausible,
narratively coherent retail failure mode — a customer support agent that
retries instead of escalating — closer to what a real deployment would
actually need to catch.

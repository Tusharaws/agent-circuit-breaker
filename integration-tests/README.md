# integration-tests

Cross-package integration tests (Phase 6) — wires `interceptor`,
`sanitizer`, `queue-client`, `evaluator`, and `control-api` together for
real, for the first time. None of the 5 components should depend on all 4
others in production; this is a test-only concern with its own package.

## `sample_agent.py`

A real, cyclic LangGraph agent (`guard` → `planner` → `tool` → loop back
or end) shared across Phase 6 tests. `looping=True` repeats the exact
same tool call every iteration (a genuine runaway loop); `looping=False`
varies it (normal, progressing work). `iteration_delay` simulates real
per-step work — an instant fake LLM/tool would otherwise complete every
iteration before a concurrent halt request could ever land.

## `test_full_pipeline.py`

**A real bug was found and fixed while building this, not designed
upfront:** `SanitizingSink` calls its inner sink with one argument (the
event); `QueueClient.append_event(thread_id, payload)` needs two. Each
side was individually correct and fully tested in isolation — nothing
exercised the seam between them until wired together for real. Fixed with
a new `interceptor.queue_sink.make_queue_sink()` adapter (see
`interceptor`'s own README/tests).

Two layers of proof:
- **Mechanism proof** (`test_all_5_components_wired_capture_events_and_honor_a_halt`,
  `test_a_normal_non_looping_run_completes_and_is_never_halted`) — manual
  `registry.request_halt()`, no SLM cost. Proves capture → sanitize →
  queue → guard-node all genuinely work together.
- **Fully closed loop** (`test_real_evaluator_detects_and_halts_a_real_seeded_loop`,
  `@pytest.mark.integration`) — the real evaluator pipeline (real SLM)
  detects a real seeded loop and calls the real `/halt` endpoint (via
  `TestClient` against `control-api`'s actual app) — no manual
  `request_halt()` anywhere. The agent, running concurrently in a
  background thread, is observably halted mid-loop by this real
  detection.

## `test_tracing.py`

Each of the 5 packages has its own dedicated trace logger
(`interceptor.trace`, `sanitizer.trace`, `queue_client.trace`,
`evaluator.trace`, `control_api.trace`), each proven correct in isolation
by that package's own tests. None of those prove the actual AC alone: that
a single `trace_id` is genuinely followable ACROSS all 5 in one real run.
`test_a_single_trace_id_is_followable_across_all_5_services` wires all 5
together (same production wiring as `test_full_pipeline.py`), runs a real
seeded-loop agent through it, calls `evaluate_thread` (fake SLM, to stay in
the fast tier) against the same thread, and asserts every one of the 5
loggers produced at least one record and every record carries the exact
same `trace_id` -- not just that each service logs *something*.

## `test_chaos.py`

Two chaos scenarios, both with real components, not fakes-of-fakes:
- **Queue killed mid-run**: `append_event` is wrapped in a controllable
  fake that raises `redis.exceptions.ConnectionError` for the first N
  calls (simulating an outage), then recovers -- same pattern
  queue-client's own `ResilientProducer` tests use, since no real Redis
  server exists anywhere in this environment to literally kill. **A real
  architecture gap was found while designing this, not assumed away:**
  every existing production wiring recipe in this project (including this
  package's own `test_full_pipeline.py`) calls `queue_client.append_event`
  directly, bypassing `queue_client.producer.ResilientProducer` entirely --
  so a real outage was previously only ever caught by `CaptureDispatcher`'s
  blunt per-item try/except (silent drop + exception log), never retried
  or buffered. This test instead wires `ResilientProducer` into the sink
  chain (`make_queue_sink(resilient_producer.append_event)`) -- the actual
  "fails safe per design" the AC calls for -- and it passed on the first
  real run: the agent's own execution never raises, a warning is genuinely
  logged for the outage, and once the queue recovers, every buffered event
  is actually flushed through, not silently dropped. See
  `interceptor/README.md` for the updated recommended production recipe.
- **Evaluator killed mid-run**: the real looping agent runs to completion
  through the full capture pipeline with literally no evaluator ever
  polling the queue -- proving there's no synchronous coupling from
  agent-side code to the evaluator, and that `QueueClient`'s own `MAXLEN`
  still bounds the stream even when nothing ever drains it (an absent
  evaluator can't cause unbounded queue growth either).

## `test_controlled_scenarios.py`

The 4 named business scenarios, each run through the full real pipeline
and measured (not asserted from architecture reasoning alone): a normal
run has zero false halts (3 real runs, real SLM); a seeded PII leak
(email/phone/SSN/name) is fully scrubbed end-to-end, including NER for the
name; a seeded runaway loop triggers a real halt in 680ms (SLA 2000ms);
sustained high load (100 zero-delay iterations) adds only ~1.5ms/iteration
of capture overhead (budget 5ms). See `CONTROLLED_SCENARIOS_REPORT.md` for
full methodology and results.

## `test_single_long_node_halt_sla.py`

The specific edge case `ARCHITECTURE_NOTES.md` flags as a known
limitation: every other halt-latency test in this project uses a loop
spanning multiple node transitions with a guard check between each --
the favorable case. This tests the unfavorable one: a "loop" living
entirely inside ONE node's function body, with no opportunity to check
for a halt until that single call returns. Two bespoke fixture graphs,
same total work (1.0s), different granularity:
- **One long node**: halt requested ~50ms in, froze at **~1018ms** --
  confirms latency-to-halt is genuinely lower-bounded by the in-flight
  node's own completion, not sooner.
- **20 short (50ms) nodes, same total work**: halt requested ~25ms in,
  froze at **~62ms** -- a direct, measured demonstration that splitting a
  long operation into shorter nodes with guard checks between them tightens
  the worst-case halt SLA proportionally.

## `test_dashboard.py`

Phase 7's dashboard package has its own isolated backend tests
(`dashboard/tests/test_app.py`), but those call `HaltHistoryStore.record()`
directly. This test proves the real cross-package path: a real `POST
/halt` (the same call the real evaluator makes) genuinely lands in
`HaltHistoryStore`, and is then independently queryable and
drillable-down-to (trace_id filter -> the exact real triggering event)
through the dashboard's own API -- not just through the dashboard's own
tests seeding its own data.

## Install

```bash
pip install -e ".[test]"
```

## Test

```bash
pytest              # includes the real-SLM end-to-end test
pytest -m "not integration"   # fast wiring-only tests, no model load
```

## Status

Phase 6 complete: all 5 components wired end-to-end with a fully real
(no mocks/manual triggers) detect-and-halt cycle, end-to-end
tracing/logging across the full path (a single trace_id followable across
all 5 services' logs), chaos testing (a real queue outage degrades
gracefully and recovers; an absent evaluator doesn't affect the agent and
the queue self-bounds), all 4 controlled scenarios (normal, PII leak,
runaway loop, high load) passing with real measured numbers, and the
single-long-running-node halt-SLA edge case measured and documented. Phase
7's dashboard/halt-history addition proven wired end-to-end via a real
POST /halt.

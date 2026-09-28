# queue-client

Client for the evaluation queue (Phase 3): carries trace events for the
`evaluator` to consume.

Note: this is intentionally separate from the LangGraph checkpointer, which
carries resumable graph state — the two are different kinds of "state" and
should not be conflated (see `ARCHITECTURE_NOTES.md`).

## QueueClient

`QueueClient` (`src/queue_client/client.py`) — one Redis stream *per
agent thread* (`{stream_prefix}:{thread_id}`, prefix configurable, default
`trace_events`), capped with `MAXLEN ~ max_len` so a thread's stream is
naturally the window of recent steps the Evaluator needs.
`append_event(thread_id, payload)` JSON-encodes and `XADD`s, returning the
event id; `read_window(thread_id, limit=None)` returns the most recent
`limit` (default `max_len`) events oldest-to-newest as `TraceEventRecord`s.
Construct from a `redis_url` or an injected `redis_client` (the latter is
how `fakeredis.FakeRedis()` swaps in for tests without monkeypatching).

**Naming convention:** per-agent-thread streams, not a single global
stream all threads share — this is what makes windowed per-thread reads
possible at all, and is the reason Redis Streams was chosen over SQS/
RabbitMQ in the first place (see `ARCHITECTURE_NOTES.md`). `stream_prefix`
is configurable so an operator can run isolated deployments (e.g. per
environment or tenant) against the same Redis instance without a code
change — no stream name is hardcoded outside this one configurable field.

`get_events_by_id(thread_id, event_ids)` (Phase 7 dashboard task) resolves
specific event ids back to their content — e.g. a `HaltSignal.triggering_window`
is a list of event ids, and the dashboard needs the actual events to show
"why," not just the ids. One `XRANGE id id` point lookup per id (ids
requested together aren't guaranteed contiguous); a missing id (trimmed by
`MAXLEN`, or just stale) is silently skipped, not an error — a drill-down
should show whatever's still available. Returns oldest-to-newest by
comparing ids as `(ms, seq)` integer tuples, not a raw string sort (a
string sort can misorder ids of different digit width; comparing as
integers is correct regardless). **Note found while building this:** Redis
Stream ids are only unique *within* a stream, not globally — two different
stream keys created in the same millisecond can produce the identical id
(confirmed empirically) — so this method's `thread_id` scoping isn't
optional plumbing, it's what makes the lookup correct at all.

## ResilientProducer: retry + local fallback buffer

`ResilientProducer` (`src/queue_client/producer.py`) wraps
`QueueClient.append_event` — same composable-wrapper pattern as
`interceptor`'s `BatchingSink`/`sanitizer`'s `SanitizingSink`, not a
change to `QueueClient` itself. On a transient Redis failure, retries
with exponential backoff up to `max_retries`; if still failing, buffers
the event in memory (bounded by `max_buffer_size`, drop-oldest when full)
instead of losing it.

**Ordering guarantee, found via TDD, not assumed:** each call first
opportunistically flushes any buffered backlog. If the backlog doesn't
fully drain (queue still unreachable), the *new* event joins the back of
the buffer too, rather than attempting its own independent send — an
earlier version let the new event race ahead and succeed before older
buffered ones once the queue recovered, which the test suite caught.

In-memory only, not disk-backed: survives the queue being unreachable
while the process keeps running, not the process itself crashing — a
stronger-durability follow-on, not built here. "Killing Redis mid-test"
(per the AC) is simulated via a controllable fake that raises
`ConnectionError` on demand, not a literal killed server process — no
test in this project runs against a real Redis server.

## ConsumerClient: consumer-group semantics

`ConsumerClient` (`src/queue_client/consumer.py`) — multiple consumer
instances in the same group split a thread's stream with no duplicate
processing (verified against fakeredis's real consumer-group behavior
before writing tests around it, not assumed). `ensure_group(thread_id)`
idempotently creates the group (works even if no producer has written to
the stream yet); `read_new(thread_id, consumer_name, count)` returns the
next unassigned messages as `TraceEventRecord`s (same type
`QueueClient.read_window` returns); `ack(thread_id, *event_ids)` marks
them processed.

Scoped to a single thread's stream per call, matching this queue's
per-agent-thread naming convention — coordinating consumption across many
different threads' streams is the Evaluator's consumption-strategy
concern, not this client's job.

## QueueMonitor: depth, lag, dropped messages

`QueueMonitor` (`src/queue_client/monitor.py`) exposes `depth(thread_id)`
(XLEN), `lag(thread_id, group_name)` (entries not yet delivered to any
consumer in the group), and `pending(thread_id, group_name)` (delivered
but not yet acked) — a queryable API a dashboard or logging setup can
consume (a full dashboard UI is a separate Phase 7 task, not this one).
`ResilientProducer.dropped_count` exposes how many buffered events were
evicted for space, for the same monitoring picture.

**A real library bug found via empirical verification, not assumed
correct:** the installed fakeredis version's own `XINFO GROUPS` `lag` and
`entries-read` fields are computed incorrectly (reading 2 messages only
decreased its reported lag by 1, not 2). `last-delivered-id` was verified
correct instead, so `lag()` counts stream entries after it manually via
`XRANGE`, sidestepping the buggy fields entirely rather than encoding a
wrong number into the client.

## Load test

See [LOAD_TEST.md](LOAD_TEST.md) for the full report. Summary: 10,000
events across 10 concurrent agent threads, ~29,750 events/sec, p95
~0.037ms — against the 1,000/sec target and 5ms budget, comfortably met.
**Scope caveat:** this environment has no real Redis server; every test
(including this one) runs against fakeredis (in-process). This proves the
client's own code has no hidden inefficiency, not real-Redis network
behavior. Consumer draining confirmed via `QueueMonitor`: lag and pending
both reach exactly 0 after processing the full backlog.

## End-to-end tracing (Phase 6)

`append_event` and `read_window` each log one structured JSON line on a
dedicated `"queue_client.trace"` logger (DEBUG): `{"trace_id": thread_id,
"service": "queue_client", "stage": "enqueued"|"window_read", ...}`
(`thread_id` is the field name used elsewhere in this package, but it's
the same value as `trace_id` everywhere else in the pipeline, by design).
Same convention as interceptor/sanitizer/evaluator/control-api's own trace
loggers -- see `interceptor/README.md` for why this is a shared
convention, not a shared library (this package deliberately doesn't
depend on `schemas` in production).

## Install

```bash
pip install -e ".[test]"
```

## Test

```bash
pytest
```

## Status

Phase 3 complete: QueueClient, ResilientProducer, ConsumerClient,
QueueMonitor, and the high-throughput load test all implemented and
tested.

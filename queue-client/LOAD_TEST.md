# High-Throughput Producer/Consumer Load Test

Reproducible via `pytest tests/test_load.py -s`.

## Scope caveat (read this first)

This environment has **no real Redis server** anywhere — every test in
this project, including this one, runs against `fakeredis` (in-process,
no real socket, no network round trip). This test proves the *client's
own code* has no hidden inefficiency (quadratic behavior, lock
contention, unbounded backlog growth) under sustained load. It is **not**
a real-Redis network-latency benchmark, and the throughput numbers below
will not transfer directly to a real deployment (real Redis over a real
network will be meaningfully slower per call, dominated by round-trip
time rather than in-process Python/serialization overhead).

## On "the agent's own thread"

The AC's phrasing ("zero measurable added latency on the agent's own
thread") describes `interceptor`'s `CaptureDispatcher` property, already
proven separately: the agent's own thread only ever calls the
fire-and-forget `capture()`, and never calls `queue_client` directly.
`queue_client`'s calls happen on `CaptureDispatcher`'s *background*
thread. So the relevant claim here is that `queue_client`'s own calls
stay fast under sustained load — not literally that they run on the
agent's thread, since they never do.

## Target

**1,000 events/sec sustained**, with per-call p95 latency under 5ms.
Chosen as a round, defensible number representing a genuinely
high-throughput multi-agent deployment (10 concurrent agent threads each
producing ~100 events/sec) — not derived from a specific customer
requirement, since none exists yet.

## Methodology

- **Producer test**: 10 concurrent "agent threads" (`thread_id`s) each
  producing 1,000 events (10,000 total), interleaved round-robin to
  simulate concurrent agents rather than one thread finishing before the
  next starts. Every individual `append_event` call's latency is
  measured.
- **Consumer test**: the same 10,000-event backlog drained via
  `ConsumerClient`'s consumer-group reads (200 at a time, acked as
  processed), then `QueueMonitor.lag()`/`.pending()` confirm the backlog
  reaches exactly zero — proving the consumer side keeps up rather than
  falling permanently behind.

## Results

| Metric | Measured |
|---|---|
| Total events | 10,000 |
| Elapsed | ~0.34s |
| Throughput | ~29,750 events/sec |
| p50 latency | ~0.033ms |
| p95 latency | ~0.037ms |
| p99 latency | ~0.044ms |
| Post-drain lag | 0 |
| Post-drain pending | 0 |

Both the throughput target (1,000/sec) and the p95 budget (5ms) are met
with wide margin — expected, given fakeredis has no real network cost.
The meaningful finding here isn't the raw number, it's that latency
**stays flat and low across 10,000 sustained calls** (p50 ≈ p95 ≈ p99,
no tail blowing up under load) — evidence against hidden
inefficiency, which is what this test is actually scoped to prove.

## Not covered here

Real Redis behavior under network load, connection pool exhaustion,
Redis server-side memory pressure at scale, and multi-process (not just
multi-thread) producer/consumer concurrency are all out of scope for this
in-process test. A real-Redis load test would need an actual Redis
instance and is a natural follow-on if/when one is available in this
environment.

"""High-throughput producer/consumer load test (Phase 3). See
LOAD_TEST.md for the full report (methodology, target, results, and the
fakeredis-vs-real-Redis scope caveat).

This environment has no real Redis server anywhere -- only fakeredis
(in-process, no real socket). This test proves the CLIENT's own code has
no hidden inefficiency (quadratic behavior, lock contention, unbounded
growth) under sustained load; it is not a real-Redis network-latency
benchmark.
"""
import time

import fakeredis

from queue_client.client import QueueClient
from queue_client.consumer import ConsumerClient
from queue_client.monitor import QueueMonitor

THREAD_COUNT = 10
EVENTS_PER_THREAD = 1000
TOTAL_EVENTS = THREAD_COUNT * EVENTS_PER_THREAD  # 10,000
TARGET_EVENTS_PER_SEC = 1000  # the throughput target this test is scoped against
P95_LATENCY_BUDGET_MS = 5.0


def test_sustained_high_throughput_production_stays_within_latency_budget():
    fake_redis = fakeredis.FakeRedis()
    producer = QueueClient(redis_client=fake_redis, max_len=EVENTS_PER_THREAD)

    thread_ids = [f"load-thread-{i}" for i in range(THREAD_COUNT)]
    latencies_ms = []

    start = time.perf_counter()
    for step in range(EVENTS_PER_THREAD):
        for thread_id in thread_ids:
            call_start = time.perf_counter()
            producer.append_event(thread_id, {"step": step, "thread": thread_id})
            latencies_ms.append((time.perf_counter() - call_start) * 1000)
    elapsed = time.perf_counter() - start

    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99)]
    throughput = TOTAL_EVENTS / elapsed

    print(
        f"\n{TOTAL_EVENTS} events across {THREAD_COUNT} threads in {elapsed:.2f}s "
        f"({throughput:,.0f} events/sec) -- p50={p50:.3f}ms p95={p95:.3f}ms p99={p99:.3f}ms"
    )

    assert throughput >= TARGET_EVENTS_PER_SEC
    assert p95 < P95_LATENCY_BUDGET_MS


def test_consumer_drains_high_throughput_backlog_without_unbounded_growth():
    fake_redis = fakeredis.FakeRedis()
    producer = QueueClient(redis_client=fake_redis, max_len=EVENTS_PER_THREAD)
    consumer = ConsumerClient(redis_client=fake_redis, group_name="load-test-evaluator")
    monitor = QueueMonitor(redis_client=fake_redis, group_name="load-test-evaluator")

    thread_ids = [f"load-thread-{i}" for i in range(THREAD_COUNT)]
    for thread_id in thread_ids:
        for step in range(EVENTS_PER_THREAD):
            producer.append_event(thread_id, {"step": step})
        consumer.ensure_group(thread_id)

    total_depth_before = sum(monitor.depth(t) for t in thread_ids)
    assert total_depth_before == TOTAL_EVENTS

    # drain every thread's backlog via consumer-group reads, acking as we go
    for thread_id in thread_ids:
        while True:
            records = consumer.read_new(thread_id, consumer_name="worker-1", count=200)
            if not records:
                break
            consumer.ack(thread_id, *(r.event_id for r in records))

    total_lag_after = sum(monitor.lag(t, "load-test-evaluator") for t in thread_ids)
    total_pending_after = sum(monitor.pending(t, "load-test-evaluator") for t in thread_ids)

    print(f"\nafter draining {TOTAL_EVENTS} events: lag={total_lag_after} pending={total_pending_after}")

    assert total_lag_after == 0
    assert total_pending_after == 0

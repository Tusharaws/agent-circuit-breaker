"""Tests for QueueMonitor (Phase 3): queue depth, consumer lag, and
pending-message visibility. Written before the implementation
(queue_client.monitor does not exist yet) -- run `pytest` to see them
fail with a collection error until src/queue_client/monitor.py exists.

fakeredis's XINFO GROUPS "lag" field and XPENDING summary were verified
empirically before writing these (not assumed) -- see the analysis
posted to the backlog item for the raw probe output.
"""
import fakeredis
import pytest

from queue_client.client import QueueClient
from queue_client.consumer import ConsumerClient
from queue_client.monitor import QueueMonitor


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def producer(fake_redis):
    return QueueClient(redis_client=fake_redis)


@pytest.fixture
def consumer(fake_redis):
    return ConsumerClient(redis_client=fake_redis, group_name="evaluator")


@pytest.fixture
def monitor(fake_redis):
    return QueueMonitor(redis_client=fake_redis, group_name="evaluator")


# ---------------------------------------------------------------------------
# depth(): queue depth, verified by spiking load per the AC
# ---------------------------------------------------------------------------


def test_depth_is_zero_for_an_empty_or_unknown_thread(monitor):
    assert monitor.depth("no-such-thread") == 0


def test_depth_reflects_number_of_events_in_the_stream(monitor, producer):
    for step in range(3):
        producer.append_event("thread-1", {"step": step})

    assert monitor.depth("thread-1") == 3


def test_depth_changes_as_load_is_spiked(monitor, producer):
    depths = [monitor.depth("thread-1")]
    for step in range(20):
        producer.append_event("thread-1", {"step": step})
        depths.append(monitor.depth("thread-1"))

    assert depths == list(range(21))  # depth tracks each append exactly


# ---------------------------------------------------------------------------
# lag(): entries not yet delivered to any consumer in the group
# ---------------------------------------------------------------------------


def test_lag_is_zero_when_no_group_exists(monitor):
    assert monitor.lag("no-such-thread", "evaluator") == 0


def test_lag_reflects_undelivered_entries(monitor, producer, consumer):
    for step in range(5):
        producer.append_event("thread-1", {"step": step})
    consumer.ensure_group("thread-1")

    assert monitor.lag("thread-1", "evaluator") == 5  # nothing delivered yet

    consumer.read_new("thread-1", consumer_name="worker-1", count=2)

    assert monitor.lag("thread-1", "evaluator") == 3  # 2 delivered, 3 remain


# ---------------------------------------------------------------------------
# pending(): delivered but not yet acked
# ---------------------------------------------------------------------------


def test_pending_is_zero_before_anything_is_read(monitor, producer, consumer):
    producer.append_event("thread-1", {"step": 1})
    consumer.ensure_group("thread-1")

    assert monitor.pending("thread-1", "evaluator") == 0


def test_pending_reflects_delivered_unacked_messages(monitor, producer, consumer):
    producer.append_event("thread-1", {"step": 1})
    consumer.ensure_group("thread-1")
    records = consumer.read_new("thread-1", consumer_name="worker-1")

    assert monitor.pending("thread-1", "evaluator") == 1

    consumer.ack("thread-1", *(r.event_id for r in records))

    assert monitor.pending("thread-1", "evaluator") == 0


# ---------------------------------------------------------------------------
# ResilientProducer.dropped_count: exposed for monitoring
# ---------------------------------------------------------------------------


def test_producer_dropped_count_starts_at_zero():
    from queue_client.producer import ResilientProducer

    producer = ResilientProducer(append_event=lambda t, p: "id")
    assert producer.dropped_count == 0


def test_producer_dropped_count_increments_on_buffer_eviction():
    import redis.exceptions

    from queue_client.producer import ResilientProducer

    def always_fails(thread_id, payload):
        raise redis.exceptions.ConnectionError("simulated outage")

    producer = ResilientProducer(
        append_event=always_fails, max_retries=0, backoff_seconds=0.001, max_buffer_size=2
    )

    producer.append_event("thread-1", {"step": 1})
    producer.append_event("thread-1", {"step": 2})
    assert producer.dropped_count == 0  # buffer not full yet

    producer.append_event("thread-1", {"step": 3})  # evicts step 1
    assert producer.dropped_count == 1

    producer.append_event("thread-1", {"step": 4})  # evicts step 2
    assert producer.dropped_count == 2

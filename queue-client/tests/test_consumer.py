"""Tests for ConsumerClient (Phase 3): Redis consumer-group semantics.
Written before the implementation (queue_client.consumer does not exist
yet) -- run `pytest` to see them fail with a collection error until
src/queue_client/consumer.py exists.

fakeredis's consumer-group support was verified empirically before
writing these (xgroup_create/xreadgroup/xack all behave as real Redis
would for the scenarios here), not assumed.
"""
import fakeredis
import pytest

from queue_client.client import QueueClient, TraceEventRecord
from queue_client.consumer import ConsumerClient


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def producer(fake_redis):
    return QueueClient(redis_client=fake_redis)


@pytest.fixture
def consumer(fake_redis):
    return ConsumerClient(redis_client=fake_redis, group_name="evaluator")


# ---------------------------------------------------------------------------
# Group creation
# ---------------------------------------------------------------------------


def test_ensure_group_is_idempotent(consumer, producer):
    producer.append_event("thread-1", {"step": 1})
    consumer.ensure_group("thread-1")
    consumer.ensure_group("thread-1")  # must not raise on the second call


def test_ensure_group_works_even_if_stream_does_not_exist_yet(consumer):
    consumer.ensure_group("brand-new-thread")  # no producer has written yet


# ---------------------------------------------------------------------------
# Reading: basic behavior
# ---------------------------------------------------------------------------


def test_read_new_with_no_messages_returns_empty_list(consumer):
    consumer.ensure_group("thread-1")
    assert consumer.read_new("thread-1", consumer_name="worker-1") == []


def test_read_new_returns_trace_event_records(consumer, producer):
    producer.append_event("thread-1", {"step": 1})
    consumer.ensure_group("thread-1")

    records = consumer.read_new("thread-1", consumer_name="worker-1")

    assert len(records) == 1
    assert isinstance(records[0], TraceEventRecord)
    assert records[0].payload == {"step": 1}


def test_read_new_respects_count(consumer, producer):
    for step in range(5):
        producer.append_event("thread-1", {"step": step})
    consumer.ensure_group("thread-1")

    records = consumer.read_new("thread-1", consumer_name="worker-1", count=2)

    assert len(records) == 2


# ---------------------------------------------------------------------------
# The AC's literal claim: 2+ consumers split the workload, zero duplicates
# ---------------------------------------------------------------------------


def test_two_consumers_split_workload_with_no_duplicate_processing(consumer, producer):
    for step in range(10):
        producer.append_event("thread-1", {"step": step})
    consumer.ensure_group("thread-1")

    batch_a = consumer.read_new("thread-1", consumer_name="worker-a", count=6)
    batch_b = consumer.read_new("thread-1", consumer_name="worker-b", count=6)

    steps_a = {r.payload["step"] for r in batch_a}
    steps_b = {r.payload["step"] for r in batch_b}

    assert steps_a.isdisjoint(steps_b)  # zero overlap
    assert steps_a | steps_b == set(range(10))  # every event processed exactly once


def test_three_consumers_split_workload_with_no_duplicate_processing(consumer, producer):
    for step in range(9):
        producer.append_event("thread-1", {"step": step})
    consumer.ensure_group("thread-1")

    batches = [consumer.read_new("thread-1", consumer_name=f"worker-{i}", count=3) for i in range(3)]
    all_steps = [r.payload["step"] for batch in batches for r in batch]

    assert sorted(all_steps) == list(range(9))  # no duplicates, none missing


# ---------------------------------------------------------------------------
# Acknowledgement
# ---------------------------------------------------------------------------


def test_ack_removes_messages_from_pending(consumer, producer, fake_redis):
    producer.append_event("thread-1", {"step": 1})
    consumer.ensure_group("thread-1")
    records = consumer.read_new("thread-1", consumer_name="worker-1")

    consumer.ack("thread-1", *(r.event_id for r in records))

    pending = fake_redis.xpending("trace_events:thread-1", "evaluator")
    assert pending["pending"] == 0


def test_unacked_messages_remain_pending(consumer, producer, fake_redis):
    producer.append_event("thread-1", {"step": 1})
    consumer.ensure_group("thread-1")
    consumer.read_new("thread-1", consumer_name="worker-1")
    # no ack() call

    pending = fake_redis.xpending("trace_events:thread-1", "evaluator")
    assert pending["pending"] == 1

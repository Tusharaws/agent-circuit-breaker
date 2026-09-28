"""Tests for ResilientProducer (Phase 3): retry + local fallback buffer.
Written before the implementation (queue_client.producer does not exist
yet) -- run `pytest` to see them fail with a collection error until
src/queue_client/producer.py exists.

"Killing Redis mid-test" is simulated via a controllable fake
append_event callable that raises redis.exceptions.ConnectionError on
demand -- this environment has never run a real Redis server (every test
across this project uses fakeredis, in-process, no real socket), which
can't simulate a genuine connection failure the way killing a real server
would. A controllable fake gives deterministic, portable control instead.
"""
import pytest
import redis.exceptions

from queue_client.producer import ResilientProducer


class FlakyAppendEvent:
    """Fake append_event: raises ConnectionError for the first `fail_count`
    calls, then succeeds. Records every successful call."""

    def __init__(self, fail_count: int = 0):
        self.fail_count = fail_count
        self.calls = 0
        self.succeeded: list[tuple[str, dict]] = []

    def __call__(self, thread_id: str, payload: dict) -> str:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise redis.exceptions.ConnectionError("simulated Redis outage")
        self.succeeded.append((thread_id, payload))
        return f"event-{self.calls}"


class AlwaysFailingAppendEvent:
    def __init__(self):
        self.calls = 0

    def __call__(self, thread_id: str, payload: dict) -> str:
        self.calls += 1
        raise redis.exceptions.ConnectionError("simulated Redis outage")


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_construction_rejects_non_callable_append_event():
    with pytest.raises(TypeError):
        ResilientProducer(append_event="not-callable")


@pytest.mark.parametrize("kwarg,value", [("max_retries", -1), ("backoff_seconds", -0.1), ("max_buffer_size", 0)])
def test_construction_rejects_invalid_config(kwarg, value):
    with pytest.raises(ValueError):
        ResilientProducer(append_event=lambda t, p: "id", **{kwarg: value})


# ---------------------------------------------------------------------------
# Retry: transient failure that resolves within max_retries
# ---------------------------------------------------------------------------


def test_transient_failure_within_max_retries_succeeds_without_buffering():
    fake = FlakyAppendEvent(fail_count=2)
    producer = ResilientProducer(append_event=fake, max_retries=3, backoff_seconds=0.001)

    event_id = producer.append_event("thread-1", {"step": 1})

    assert event_id == "event-3"
    assert fake.succeeded == [("thread-1", {"step": 1})]
    assert producer.buffered_count == 0


def test_no_failure_succeeds_on_first_attempt():
    fake = FlakyAppendEvent(fail_count=0)
    producer = ResilientProducer(append_event=fake)

    producer.append_event("thread-1", {"step": 1})

    assert fake.calls == 1


# ---------------------------------------------------------------------------
# Fallback buffer: failure exceeding max_retries
# ---------------------------------------------------------------------------


def test_failure_exceeding_max_retries_buffers_instead_of_raising():
    fake = AlwaysFailingAppendEvent()
    producer = ResilientProducer(append_event=fake, max_retries=2, backoff_seconds=0.001)

    result = producer.append_event("thread-1", {"step": 1})  # must not raise

    assert result is None
    assert producer.buffered_count == 1
    assert fake.calls == 3  # 1 initial + 2 retries


def test_multiple_buffered_events_accumulate_in_order():
    fake = AlwaysFailingAppendEvent()
    producer = ResilientProducer(append_event=fake, max_retries=0, backoff_seconds=0.001)

    producer.append_event("thread-1", {"step": 1})
    producer.append_event("thread-1", {"step": 2})

    assert producer.buffered_count == 2


def test_buffer_respects_max_buffer_size_and_drops_oldest():
    fake = AlwaysFailingAppendEvent()
    producer = ResilientProducer(append_event=fake, max_retries=0, backoff_seconds=0.001, max_buffer_size=2)

    producer.append_event("thread-1", {"step": 1})
    producer.append_event("thread-1", {"step": 2})
    producer.append_event("thread-1", {"step": 3})  # buffer full, drops step 1

    assert producer.buffered_count == 2


# ---------------------------------------------------------------------------
# Zero data loss: buffer flushes once the queue becomes reachable again
# ---------------------------------------------------------------------------


def test_buffer_flushes_in_order_once_queue_recovers():
    fake = FlakyAppendEvent(fail_count=2)  # first 2 calls fail, rest succeed
    producer = ResilientProducer(append_event=fake, max_retries=0, backoff_seconds=0.001)

    producer.append_event("thread-1", {"step": 1})  # fails immediately, buffered
    producer.append_event("thread-1", {"step": 2})  # fails immediately, buffered

    assert producer.buffered_count == 2
    assert fake.succeeded == []

    # queue is "back up" now (FlakyAppendEvent succeeds from call 3 onward);
    # the next append_event call should flush the backlog first, in order
    event_id = producer.append_event("thread-1", {"step": 3})

    assert producer.buffered_count == 0
    assert fake.succeeded == [
        ("thread-1", {"step": 1}),
        ("thread-1", {"step": 2}),
        ("thread-1", {"step": 3}),
    ]
    assert event_id == "event-5"  # the 3rd (new) event is the 5th call overall


def test_zero_data_loss_across_a_simulated_outage_and_recovery():
    """The literal "killing Redis mid-test" scenario: several appends
    during an outage, all survive (buffered), then all reach the queue
    once it recovers -- none lost."""
    fake = FlakyAppendEvent(fail_count=3)
    producer = ResilientProducer(append_event=fake, max_retries=0, backoff_seconds=0.001)

    sent_during_outage = [{"step": i} for i in range(3)]
    for payload in sent_during_outage:
        producer.append_event("thread-1", payload)

    assert producer.buffered_count == 3  # nothing lost, just buffered

    producer.append_event("thread-1", {"step": 99})  # triggers flush + itself succeeds

    assert producer.buffered_count == 0
    assert fake.succeeded == [("thread-1", p) for p in sent_during_outage] + [("thread-1", {"step": 99})]

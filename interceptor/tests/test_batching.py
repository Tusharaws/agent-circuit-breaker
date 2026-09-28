"""Tests for BatchingSink (Phase 1: local buffering/batching layer).

Written before the implementation (interceptor.batching does not exist
yet) -- run `pytest` to see them fail with a collection error until
src/interceptor/batching.py exists.

BatchingSink is a composable `sink` (matching CaptureDispatcher's existing
Callable[[Any], None] contract) that accumulates events locally and only
calls the real flush callable once per batch_size items or once per
flush_interval seconds, whichever comes first -- reducing network calls
from one-per-event to one-per-batch without changing CaptureDispatcher's
already-tested per-item sink contract at all.
"""
import threading
import time

import pytest

from interceptor.batching import BatchingSink


def _wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class RecordingFlush:
    """Fake flush target: records each batch it was called with."""

    def __init__(self):
        self.batches: list[list] = []
        self._lock = threading.Lock()

    def __call__(self, batch):
        with self._lock:
            self.batches.append(list(batch))

    @property
    def call_count(self):
        with self._lock:
            return len(self.batches)

    @property
    def total_events(self):
        with self._lock:
            return sum(len(b) for b in self.batches)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_batch_size_zero_is_rejected():
    with pytest.raises(ValueError):
        BatchingSink(flush=RecordingFlush(), batch_size=0)


def test_batch_size_negative_is_rejected():
    with pytest.raises(ValueError):
        BatchingSink(flush=RecordingFlush(), batch_size=-1)


def test_flush_interval_zero_is_rejected():
    with pytest.raises(ValueError):
        BatchingSink(flush=RecordingFlush(), flush_interval=0)


def test_flush_interval_negative_is_rejected():
    with pytest.raises(ValueError):
        BatchingSink(flush=RecordingFlush(), flush_interval=-1)


def test_non_callable_flush_is_rejected():
    with pytest.raises(TypeError):
        BatchingSink(flush="not-callable")


# ---------------------------------------------------------------------------
# Batch-size-triggered flush
# ---------------------------------------------------------------------------


def test_flush_triggered_exactly_at_batch_size():
    flush = RecordingFlush()
    with BatchingSink(flush=flush, batch_size=3, flush_interval=100.0) as sink:
        sink("e1")
        sink("e2")
        assert flush.call_count == 0  # not yet at batch_size
        sink("e3")
        assert _wait_until(lambda: flush.call_count == 1)
        assert flush.batches[0] == ["e1", "e2", "e3"]


def test_multiple_sequential_flushes_each_respect_batch_size():
    flush = RecordingFlush()
    with BatchingSink(flush=flush, batch_size=2, flush_interval=100.0) as sink:
        for i in range(6):
            sink(f"e{i}")

        assert _wait_until(lambda: flush.call_count == 3)
        assert all(len(batch) == 2 for batch in flush.batches)
        assert flush.total_events == 6


# ---------------------------------------------------------------------------
# Interval-triggered flush (partial batch)
# ---------------------------------------------------------------------------


def test_flush_triggered_by_interval_with_partial_batch():
    flush = RecordingFlush()
    with BatchingSink(flush=flush, batch_size=100, flush_interval=0.05) as sink:
        sink("only-one")
        assert flush.call_count == 0
        assert _wait_until(lambda: flush.call_count == 1, timeout=1.0)
        assert flush.batches[0] == ["only-one"]


def test_interval_flush_is_a_no_op_when_buffer_is_empty():
    flush = RecordingFlush()
    with BatchingSink(flush=flush, batch_size=100, flush_interval=0.05):
        time.sleep(0.2)
    assert flush.call_count == 0


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------


def test_flush_exception_does_not_propagate_to_caller():
    def raising_flush(batch):
        raise RuntimeError("boom")

    with BatchingSink(flush=raising_flush, batch_size=1, flush_interval=100.0) as sink:
        sink("event")  # must not raise


def test_flush_exception_is_logged(caplog):
    import logging

    def raising_flush(batch):
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        with BatchingSink(flush=raising_flush, batch_size=1, flush_interval=100.0) as sink:
            sink("event")
            assert _wait_until(lambda: "boom" in caplog.text)


def test_worker_survives_a_flush_exception_and_processes_later_batches():
    calls = []
    lock = threading.Lock()

    def sometimes_raising_flush(batch):
        if batch == ["bad"]:
            raise RuntimeError("boom")
        with lock:
            calls.append(batch)

    with BatchingSink(flush=sometimes_raising_flush, batch_size=1, flush_interval=100.0) as sink:
        sink("bad")
        sink("good")

        assert _wait_until(lambda: calls == [["good"]])


# ---------------------------------------------------------------------------
# stop(): final best-effort flush
# ---------------------------------------------------------------------------


def test_stop_flushes_remaining_buffered_events():
    flush = RecordingFlush()
    sink = BatchingSink(flush=flush, batch_size=100, flush_interval=100.0)
    sink("leftover-1")
    sink("leftover-2")
    assert flush.call_count == 0

    sink.stop(timeout=2.0)

    assert flush.call_count == 1
    assert flush.batches[0] == ["leftover-1", "leftover-2"]


def test_context_manager_stops_timer_thread_on_exit():
    with BatchingSink(flush=RecordingFlush(), flush_interval=100.0) as sink:
        thread = sink._timer_thread
        assert thread.is_alive()
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# The AC's specific measured claim: >90% reduction in call count
# ---------------------------------------------------------------------------


def test_batching_reduces_call_count_by_more_than_90_percent():
    flush = RecordingFlush()
    event_count = 1000
    batch_size = 100

    with BatchingSink(flush=flush, batch_size=batch_size, flush_interval=100.0) as sink:
        for i in range(event_count):
            sink(f"event-{i}")

        assert _wait_until(lambda: flush.total_events == event_count)

    naive_call_count = event_count  # one network call per event, no batching
    reduction = 1 - (flush.call_count / naive_call_count)

    assert reduction > 0.90
    assert flush.call_count == event_count // batch_size

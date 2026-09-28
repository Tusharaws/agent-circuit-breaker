"""Tests for CaptureDispatcher (Phase 1 execution model).

These tests are written before the implementation (interceptor.capture does
not exist yet) and define the contract for review. Run `pytest` here to see
them fail with a collection error until src/interceptor/capture.py exists.

Acceptance criteria under test: capture runs on a background thread and is
non-blocking for both sync and async callers; a deliberately-raised
exception inside capture (in the sink, or in the enqueue path itself) never
propagates to the caller and never crashes the parent run; errors are
logged instead.
"""
import asyncio
import gc
import logging
import threading
import time
import weakref
from datetime import timedelta

import pytest

from interceptor.capture import CaptureDispatcher


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Non-blocking behavior
# ---------------------------------------------------------------------------


def test_capture_returns_immediately_while_sink_is_still_running():
    release = threading.Event()
    started = threading.Event()

    def slow_sink(event):
        started.set()
        release.wait(timeout=2.0)

    with CaptureDispatcher(sink=slow_sink) as dispatcher:
        start = time.monotonic()
        dispatcher.capture("event-1")
        elapsed = time.monotonic() - start

        assert elapsed < 0.05
        assert _wait_until(started.is_set)
        release.set()


def test_capture_is_non_blocking_when_called_from_async_context():
    """A `_start` node event fires from inside an `async def` LangGraph node.
    capture() must not need `await` and must not meaningfully block the
    event loop — proves one background thread serves both sync and async
    callers, no separate asyncio-native dispatch path needed."""
    release = threading.Event()
    started = threading.Event()

    def slow_sink(event):
        started.set()
        release.wait(timeout=2.0)

    with CaptureDispatcher(sink=slow_sink) as dispatcher:
        async def call_capture():
            start = time.monotonic()
            dispatcher.capture("event-1")
            return time.monotonic() - start

        elapsed = asyncio.run(call_capture())

        assert elapsed < 0.05
        assert _wait_until(started.is_set)
        release.set()


# ---------------------------------------------------------------------------
# Exception isolation (the core acceptance criterion)
# ---------------------------------------------------------------------------


def test_sink_exception_does_not_propagate_to_caller():
    def raising_sink(event):
        raise RuntimeError("boom")

    with CaptureDispatcher(sink=raising_sink) as dispatcher:
        dispatcher.capture("event-1")  # must not raise


def test_sink_exception_is_logged(caplog):
    def raising_sink(event):
        raise RuntimeError("boom")

    with CaptureDispatcher(sink=raising_sink) as dispatcher:
        with caplog.at_level(logging.ERROR):
            dispatcher.capture("event-1")
            assert _wait_until(lambda: "boom" in caplog.text)


def test_parent_run_completes_normally_after_a_capture_exception():
    """Simulates the acceptance criterion directly: an agent run that calls
    capture() (which raises internally) still completes and produces its
    normal result, with no exception surfacing to the caller."""

    def raising_sink(event):
        raise RuntimeError("boom")

    with CaptureDispatcher(sink=raising_sink) as dispatcher:
        def simulated_agent_run():
            dispatcher.capture("node-enter")
            result = 1 + 1
            dispatcher.capture("node-exit")
            return result

        assert simulated_agent_run() == 2


def test_worker_survives_a_raising_item_and_processes_subsequent_ones():
    """One bad event must not kill the background worker thread and
    silently stop all future capture."""
    calls = []
    lock = threading.Lock()

    def sink(event):
        if event == "bad":
            raise RuntimeError("boom")
        with lock:
            calls.append(event)

    with CaptureDispatcher(sink=sink) as dispatcher:
        dispatcher.capture("good-1")
        dispatcher.capture("bad")
        dispatcher.capture("good-2")

        assert _wait_until(lambda: calls == ["good-1", "good-2"])


# ---------------------------------------------------------------------------
# Buffer-full policy
# ---------------------------------------------------------------------------


def test_buffer_full_drops_new_event_without_blocking_or_raising(caplog):
    release = threading.Event()
    started = threading.Event()
    calls = []
    lock = threading.Lock()

    def sink(event):
        if event == "block-me":
            started.set()
            release.wait(timeout=2.0)
        with lock:
            calls.append(event)

    with CaptureDispatcher(sink=sink, max_queue_size=1) as dispatcher:
        dispatcher.capture("block-me")
        assert _wait_until(started.is_set)  # worker now busy; queue itself is empty

        dispatcher.capture("fills-queue")  # queue now at capacity (size 1)

        with caplog.at_level(logging.WARNING):
            start = time.monotonic()
            dispatcher.capture("dropped")  # must not block or raise
            elapsed = time.monotonic() - start
            assert elapsed < 0.05
            assert _wait_until(lambda: "buffer full" in caplog.text.lower())

        release.set()
        assert _wait_until(lambda: calls == ["block-me", "fills-queue"])


# ---------------------------------------------------------------------------
# Age-based eviction
# ---------------------------------------------------------------------------


def test_item_older_than_max_age_is_dropped_when_dequeued(caplog):
    release = threading.Event()
    started = threading.Event()
    calls = []
    lock = threading.Lock()

    def sink(event):
        if event == "block-me":
            started.set()
            release.wait(timeout=2.0)
        with lock:
            calls.append(event)

    with CaptureDispatcher(sink=sink, max_age=timedelta(milliseconds=20)) as dispatcher:
        dispatcher.capture("block-me")
        assert _wait_until(started.is_set)  # worker busy; next item will sit queued

        dispatcher.capture("stale-event")
        time.sleep(0.05)  # let it age past max_age while still queued

        with caplog.at_level(logging.WARNING):
            release.set()  # worker finishes "block-me", moves on to "stale-event"
            assert _wait_until(lambda: "older than max_age" in caplog.text.lower())

        time.sleep(0.05)
        assert calls == ["block-me"]  # stale-event was dropped, never reached sink


# ---------------------------------------------------------------------------
# Thread properties
# ---------------------------------------------------------------------------


def test_background_thread_is_a_daemon_thread():
    with CaptureDispatcher(sink=lambda event: None) as dispatcher:
        assert dispatcher._thread.daemon is True


# ---------------------------------------------------------------------------
# Resource lifecycle: stop() / context manager (fixes the GC-leak gap)
#
# Without a way to stop the background thread, `threading.Thread(target=
# self._run, ...)` holds a strong reference to `self` for the thread's
# entire (currently infinite) lifetime, so a CaptureDispatcher can never be
# garbage collected. These tests require a `stop()` method and context
# manager support that don't exist in the current implementation.
# ---------------------------------------------------------------------------


def test_stop_joins_the_background_thread():
    dispatcher = CaptureDispatcher(sink=lambda event: None)
    thread = dispatcher._thread
    assert thread.is_alive()

    dispatcher.stop(timeout=1.0)

    assert not thread.is_alive()


def test_dispatcher_used_as_context_manager_stops_thread_on_exit():
    with CaptureDispatcher(sink=lambda event: None) as dispatcher:
        thread = dispatcher._thread
        assert thread.is_alive()

    assert not thread.is_alive()


def test_dispatcher_is_garbage_collectable_after_stop():
    dispatcher = CaptureDispatcher(sink=lambda event: None)
    dispatcher.stop(timeout=1.0)

    ref = weakref.ref(dispatcher)
    del dispatcher
    gc.collect()

    assert ref() is None


# ---------------------------------------------------------------------------
# Construction validation
#
# Currently unvalidated: max_queue_size=0 means *unbounded* in stdlib
# queue.Queue semantics (the opposite of what it looks like it should mean),
# a non-callable sink only "works" by accident via the per-item
# try/except, and max_age<=0 has never been decided one way or the other.
# ---------------------------------------------------------------------------


def test_max_queue_size_zero_is_rejected():
    with pytest.raises(ValueError):
        CaptureDispatcher(sink=lambda event: None, max_queue_size=0)


def test_max_queue_size_negative_is_rejected():
    with pytest.raises(ValueError):
        CaptureDispatcher(sink=lambda event: None, max_queue_size=-1)


def test_sink_not_callable_is_rejected():
    with pytest.raises(TypeError):
        CaptureDispatcher(sink="not-callable")


def test_sink_none_is_rejected():
    with pytest.raises(TypeError):
        CaptureDispatcher(sink=None)


def test_max_age_zero_is_rejected():
    with pytest.raises(ValueError):
        CaptureDispatcher(sink=lambda event: None, max_age=timedelta(0))


def test_max_age_negative_is_rejected():
    with pytest.raises(ValueError):
        CaptureDispatcher(sink=lambda event: None, max_age=timedelta(seconds=-1))


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_captures_from_multiple_threads_are_all_processed():
    calls = []
    lock = threading.Lock()

    def sink(event):
        with lock:
            calls.append(event)

    with CaptureDispatcher(sink=sink) as dispatcher:
        def producer(producer_id):
            for i in range(50):
                dispatcher.capture(f"{producer_id}-{i}")

        threads = [threading.Thread(target=producer, args=(p,)) for p in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = {f"{p}-{i}" for p in range(5) for i in range(50)}
        assert _wait_until(lambda: len(calls) == len(expected), timeout=5.0)
        assert set(calls) == expected


def test_events_are_processed_in_fifo_order():
    calls = []
    lock = threading.Lock()

    def sink(event):
        with lock:
            calls.append(event)

    with CaptureDispatcher(sink=sink) as dispatcher:
        for i in range(200):
            dispatcher.capture(i)

        assert _wait_until(lambda: len(calls) == 200, timeout=5.0)
        assert calls == list(range(200))


def test_sink_can_call_capture_reentrantly_without_deadlock():
    calls = []
    lock = threading.Lock()
    holder = {}

    def sink(event):
        with lock:
            calls.append(event)
        if event == "trigger":
            holder["dispatcher"].capture("triggered-by-sink")

    with CaptureDispatcher(sink=sink) as dispatcher:
        holder["dispatcher"] = dispatcher
        dispatcher.capture("trigger")

        assert _wait_until(lambda: "triggered-by-sink" in calls, timeout=1.0)


# ---------------------------------------------------------------------------
# Logging precision
# ---------------------------------------------------------------------------


def test_sink_exception_is_logged_at_error_level(caplog):
    def raising_sink(event):
        raise RuntimeError("boom")

    with caplog.at_level(logging.WARNING):
        with CaptureDispatcher(sink=raising_sink) as dispatcher:
            dispatcher.capture("event-1")
            assert _wait_until(
                lambda: any(r.levelno == logging.ERROR for r in caplog.records)
            )


def test_buffer_full_is_logged_at_warning_level(caplog):
    release = threading.Event()
    started = threading.Event()

    def sink(event):
        if event == "block-me":
            started.set()
            release.wait(timeout=2.0)

    with CaptureDispatcher(sink=sink, max_queue_size=1) as dispatcher:
        dispatcher.capture("block-me")
        assert _wait_until(started.is_set)
        dispatcher.capture("fills-queue")

        with caplog.at_level(logging.WARNING):
            dispatcher.capture("dropped")
            assert _wait_until(
                lambda: any(
                    r.levelno == logging.WARNING and "buffer full" in r.getMessage().lower()
                    for r in caplog.records
                )
            )

        release.set()


def test_max_age_drop_is_logged_at_warning_level(caplog):
    release = threading.Event()
    started = threading.Event()

    def sink(event):
        if event == "block-me":
            started.set()
            release.wait(timeout=2.0)

    with CaptureDispatcher(sink=sink, max_age=timedelta(milliseconds=20)) as dispatcher:
        dispatcher.capture("block-me")
        assert _wait_until(started.is_set)
        dispatcher.capture("stale-event")
        time.sleep(0.05)

        with caplog.at_level(logging.WARNING):
            release.set()
            assert _wait_until(
                lambda: any(
                    r.levelno == logging.WARNING
                    and "older than max_age" in r.getMessage().lower()
                    for r in caplog.records
                )
            )


def test_successful_capture_produces_no_log_output(caplog):
    calls = []

    def sink(event):
        calls.append(event)

    with caplog.at_level(logging.WARNING):
        with CaptureDispatcher(sink=sink) as dispatcher:
            dispatcher.capture("event-1")
            assert _wait_until(lambda: calls == ["event-1"])
            assert caplog.records == []


def test_custom_logger_is_used_instead_of_default():
    custom_logger = logging.getLogger("interceptor.capture.custom-test")
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    custom_logger.addHandler(ListHandler())
    custom_logger.setLevel(logging.WARNING)
    custom_logger.propagate = False

    def raising_sink(event):
        raise RuntimeError("boom")

    with CaptureDispatcher(sink=raising_sink, logger=custom_logger) as dispatcher:
        dispatcher.capture("event-1")
        assert _wait_until(
            lambda: any(r.exc_info and str(r.exc_info[1]) == "boom" for r in records)
        )


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


def test_capture_accepts_dict_payload():
    calls = []

    def sink(event):
        calls.append(event)

    payload = {"trace_id": "t-1", "event_type": "llm_end", "payload": {"a": 1}}
    with CaptureDispatcher(sink=sink) as dispatcher:
        dispatcher.capture(payload)
        assert _wait_until(lambda: calls == [payload])


def test_capture_accepts_none_as_event():
    calls = []

    def sink(event):
        calls.append(event)

    with CaptureDispatcher(sink=sink) as dispatcher:
        dispatcher.capture(None)
        assert _wait_until(lambda: calls == [None])


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_worker_recovers_after_draining_a_fully_stale_backlog():
    release = threading.Event()
    started = threading.Event()
    calls = []
    lock = threading.Lock()

    def sink(event):
        if event == "block-me":
            started.set()
            release.wait(timeout=2.0)
        with lock:
            calls.append(event)

    with CaptureDispatcher(
        sink=sink, max_queue_size=5, max_age=timedelta(milliseconds=20)
    ) as dispatcher:
        dispatcher.capture("block-me")
        assert _wait_until(started.is_set)

        for i in range(4):
            dispatcher.capture(f"stale-{i}")
        time.sleep(0.05)  # whole backlog now older than max_age

        release.set()
        assert _wait_until(lambda: "block-me" in calls)

        dispatcher.capture("fresh")
        assert _wait_until(lambda: "fresh" in calls)

        assert not any(f"stale-{i}" in calls for i in range(4))

import logging
import queue
import threading
import time
from datetime import timedelta
from typing import Any, Callable

_DEFAULT_MAX_AGE = timedelta(days=2)


class CaptureDispatcher:
    """Fire-and-forget capture dispatch (Phase 1 execution model).

    Hands events off to a single background daemon thread so a node's
    execution never blocks on `sink` (e.g. queue_client.append_event), and
    never crashes/pauses the parent agent run even if `sink` raises. One
    thread-safe queue serves both sync and async callers: enqueueing is a
    microsecond-scale, non-blocking operation with no real I/O in it, so
    it's safe to call from inside an `async def` node without `await`.
    """

    def __init__(
        self,
        sink: Callable[[Any], None],
        max_queue_size: int = 1000,
        max_age: timedelta = _DEFAULT_MAX_AGE,
        logger: logging.Logger | None = None,
    ) -> None:
        if not callable(sink):
            raise TypeError(f"sink must be callable, got {type(sink).__name__}")
        if max_queue_size <= 0:
            # queue.Queue(maxsize=0) means *unbounded* in stdlib semantics,
            # the opposite of what 0 (or a negative size) looks like it means.
            raise ValueError(f"max_queue_size must be positive, got {max_queue_size}")
        if max_age <= timedelta(0):
            raise ValueError(f"max_age must be positive, got {max_age}")

        self._sink = sink
        self._max_queue_size = max_queue_size
        self._max_age = max_age
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._queue: queue.Queue[tuple[Any, float]] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="interceptor-capture", daemon=True
        )
        self._thread.start()

    def __enter__(self) -> "CaptureDispatcher":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def stop(self, timeout: float | None = None) -> None:
        """Signal the background thread to exit and wait for it to finish.

        Not a guaranteed flush: items still queued when `stop()` is called
        may not be processed before the thread exits (bounded by the
        internal poll interval). This releases the thread's reference to
        `self`, which is what allows the dispatcher to be garbage collected.
        """
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    def capture(self, event: Any) -> None:
        """Non-blocking. Never raises, regardless of what goes wrong."""
        try:
            self._enqueue(event)
        except Exception:
            self._logger.exception("interceptor: capture failed for event %r", event)

    def _enqueue(self, event: Any) -> None:
        try:
            self._queue.put_nowait((event, time.monotonic()))
        except queue.Full:
            self._logger.warning(
                "interceptor: capture buffer full (max_queue_size=%d), dropping event",
                self._max_queue_size,
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event, enqueued_at = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            age = time.monotonic() - enqueued_at
            if age > self._max_age.total_seconds():
                self._logger.warning(
                    "interceptor: dropping event older than max_age (%s): %r",
                    self._max_age,
                    event,
                )
                continue

            try:
                self._sink(event)
            except Exception:
                self._logger.exception(
                    "interceptor: capture sink raised for event %r", event
                )

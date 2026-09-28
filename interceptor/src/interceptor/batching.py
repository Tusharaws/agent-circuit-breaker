import logging
import threading
from typing import Any, Callable


class BatchingSink:
    """Local buffering/batching layer (Phase 1).

    A composable `sink` matching CaptureDispatcher's existing
    `Callable[[Any], None]` contract, so it drops in as
    `CaptureDispatcher(sink=BatchingSink(flush=...))` without any change to
    CaptureDispatcher itself. Accumulates events locally and only calls the
    real `flush` callable once per `batch_size` items or once per
    `flush_interval` seconds (whichever comes first) — trading one network
    call per event for one network call per batch.

    `stop()` performs a final best-effort flush of whatever remains
    buffered, then stops the background flush-interval timer thread.
    """

    def __init__(
        self,
        flush: Callable[[list[Any]], None],
        batch_size: int = 50,
        flush_interval: float = 1.0,
        logger: logging.Logger | None = None,
    ) -> None:
        if not callable(flush):
            raise TypeError(f"flush must be callable, got {type(flush).__name__}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if flush_interval <= 0:
            raise ValueError(f"flush_interval must be positive, got {flush_interval}")

        self._flush_callable = flush
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._buffer: list[Any] = []
        self._stop_event = threading.Event()
        self._timer_thread = threading.Thread(
            target=self._timer_loop, name="batching-sink-timer", daemon=True
        )
        self._timer_thread.start()

    def __call__(self, event: Any) -> None:
        should_flush = False
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._batch_size:
                should_flush = True
        if should_flush:
            self._flush()

    def __enter__(self) -> "BatchingSink":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        self._timer_thread.join(timeout=timeout)
        self._flush()

    def _timer_loop(self) -> None:
        while not self._stop_event.wait(self._flush_interval):
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer
            self._buffer = []
        try:
            self._flush_callable(batch)
        except Exception:
            self._logger.exception(
                "batching_sink: flush failed, %d event(s) dropped", len(batch)
            )

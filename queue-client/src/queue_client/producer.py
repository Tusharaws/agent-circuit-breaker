import logging
import time
from collections import deque
from typing import Callable

import redis.exceptions

_TRANSIENT_ERRORS = (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError)


class ResilientProducer:
    """Retry + local fallback buffer wrapper around
    `QueueClient.append_event` (Phase 3). Composable wrapper, not a change
    to `QueueClient` itself -- same pattern as `BatchingSink`/
    `SanitizingSink` in the other packages.

    On a transient Redis failure, retries with exponential backoff up to
    `max_retries`. If still failing, buffers the event in memory instead
    of losing it (bounded by `max_buffer_size`, drop-oldest when full).
    Each call first opportunistically flushes any previously-buffered
    backlog, in order, before sending the new event -- so once the queue
    recovers, nothing buffered during the outage is lost.

    In-memory only, not disk-backed: survives the queue being unreachable
    while this process keeps running, not this process itself crashing --
    a stronger-durability follow-on, not built here.
    """

    def __init__(
        self,
        append_event: Callable[[str, dict], str],
        max_retries: int = 3,
        backoff_seconds: float = 0.1,
        max_buffer_size: int = 1000,
        logger: logging.Logger | None = None,
    ) -> None:
        if not callable(append_event):
            raise TypeError(f"append_event must be callable, got {type(append_event).__name__}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if backoff_seconds < 0:
            raise ValueError(f"backoff_seconds must be >= 0, got {backoff_seconds}")
        if max_buffer_size <= 0:
            raise ValueError(f"max_buffer_size must be positive, got {max_buffer_size}")

        self._append_event = append_event
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._max_buffer_size = max_buffer_size
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._buffer: deque[tuple[str, dict]] = deque()
        self._dropped_count = 0

    @property
    def buffered_count(self) -> int:
        return len(self._buffer)

    @property
    def dropped_count(self) -> int:
        """Total events evicted (never buffer-full acceptance, always
        drop-oldest) because the buffer was full -- for monitoring, not
        just the warning log line each eviction also produces."""
        return self._dropped_count

    def append_event(self, thread_id: str, payload: dict) -> str | None:
        """Same contract as `QueueClient.append_event`, except: on
        exhausted retries this returns `None` and buffers the event
        instead of raising."""
        self._flush_buffer()
        if self._buffer:
            # backlog didn't fully drain (queue still unreachable) -- join
            # the back of it instead of racing ahead of older buffered
            # events, which would deliver them out of order once the
            # queue recovers.
            self._buffer_event(thread_id, payload)
            return None
        try:
            return self._append_with_retry(thread_id, payload)
        except _TRANSIENT_ERRORS:
            self._buffer_event(thread_id, payload)
            return None

    def _append_with_retry(self, thread_id: str, payload: dict) -> str:
        attempt = 0
        while True:
            try:
                return self._append_event(thread_id, payload)
            except _TRANSIENT_ERRORS:
                attempt += 1
                if attempt > self._max_retries:
                    raise
                time.sleep(self._backoff_seconds * (2 ** (attempt - 1)))

    def _buffer_event(self, thread_id: str, payload: dict) -> None:
        if len(self._buffer) >= self._max_buffer_size:
            self._buffer.popleft()
            self._dropped_count += 1
            self._logger.warning(
                "producer buffer full (max_buffer_size=%d), dropped oldest buffered event",
                self._max_buffer_size,
            )
        self._buffer.append((thread_id, payload))
        self._logger.warning(
            "queue unreachable, buffered event locally (%d buffered)", len(self._buffer)
        )

    def _flush_buffer(self) -> None:
        while self._buffer:
            thread_id, payload = self._buffer[0]
            try:
                self._append_event(thread_id, payload)
            except _TRANSIENT_ERRORS:
                break  # still down -- stop, leave the rest buffered
            self._buffer.popleft()

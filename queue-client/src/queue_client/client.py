import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import redis

# Dedicated logger (Phase 6 end-to-end tracing) so a trace_id (== thread_id
# here) is followable across all 5 services' logs -- DEBUG, since this
# fires on every append/read, not a low-frequency decision.
_trace_logger = logging.getLogger("queue_client.trace")


@dataclass
class TraceEventRecord:
    """A decoded stream entry -- callers work with this, not raw Redis
    byte-string field/value pairs."""

    event_id: str
    thread_id: str
    payload: dict
    enqueued_at: datetime


class QueueClient:
    """Redis Streams client (Phase 3). Naming convention: one stream
    *per agent thread* (`{stream_prefix}:{thread_id}`), not a single
    global stream all threads share -- this is what makes the windowed
    per-thread reads (`XRANGE`/`MAXLEN`) the Evaluator needs possible at
    all, and was the whole reason Redis Streams was chosen over SQS/
    RabbitMQ in the first place (see ARCHITECTURE_NOTES.md). `stream_prefix`
    is configurable (default `trace_events`) so an operator can run
    isolated deployments against the same Redis instance without a code
    change.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        redis_client: "redis.Redis | None" = None,
        max_len: int = 500,
        stream_prefix: str = "trace_events",
    ) -> None:
        if redis_client is not None:
            self._redis = redis_client
        elif redis_url is not None:
            self._redis = redis.Redis.from_url(redis_url)
        else:
            raise ValueError("either redis_url or redis_client must be provided")
        self._max_len = max_len
        self._stream_prefix = stream_prefix

    def _stream_key(self, thread_id: str) -> str:
        return f"{self._stream_prefix}:{thread_id}"

    def append_event(self, thread_id: str, payload: dict) -> str:
        """JSON-encodes `payload`, XADDs it (MAXLEN ~ max_len), returns
        the Redis-assigned event id. Raises TypeError for a
        non-JSON-serializable payload."""
        data = json.dumps(payload)
        event_id = self._redis.xadd(
            self._stream_key(thread_id),
            {"data": data},
            maxlen=self._max_len,
            approximate=True,
        )
        if isinstance(event_id, bytes):
            event_id = event_id.decode("utf-8")
        _trace_logger.debug(
            json.dumps({"trace_id": thread_id, "service": "queue_client", "stage": "enqueued", "event_id": event_id})
        )
        return event_id

    def read_window(self, thread_id: str, limit: int | None = None) -> list[TraceEventRecord]:
        """The most recent `limit` (default: max_len) events for
        `thread_id`, oldest to newest. Empty stream/unknown thread
        returns []."""
        count = limit if limit is not None else self._max_len
        entries = self._redis.xrevrange(self._stream_key(thread_id), count=count)

        records = [self._decode_entry(thread_id, entry_id, fields) for entry_id, fields in entries]

        records.reverse()  # xrevrange is newest-first; callers want oldest-to-newest
        _trace_logger.debug(
            json.dumps({"trace_id": thread_id, "service": "queue_client", "stage": "window_read", "count": len(records)})
        )
        return records

    def get_events_by_id(self, thread_id: str, event_ids: list[str]) -> list[TraceEventRecord]:
        """Resolves specific event ids back to their content (Phase 7
        dashboard task) -- e.g. a halt's `triggering_window` is a list of
        event ids, and showing "why" needs the actual events, not just
        their ids. An id that no longer exists (trimmed by MAXLEN, from a
        different thread, or simply wrong) is silently skipped rather than
        raising -- a dashboard drill-down should show whatever's still
        available, not crash on a stale/partial reference. Returns
        oldest-to-newest regardless of the input list's order (one point
        lookup per id via `XRANGE id id`, since ids requested together
        aren't guaranteed contiguous)."""
        records = []
        stream_key = self._stream_key(thread_id)
        for event_id in event_ids:
            entries = self._redis.xrange(stream_key, min=event_id, max=event_id)
            for entry_id, fields in entries:
                records.append(self._decode_entry(thread_id, entry_id, fields))
        records.sort(key=lambda record: self._id_sort_key(record.event_id))
        return records

    @staticmethod
    def _id_sort_key(event_id: str) -> tuple[int, int]:
        """(ms, seq) as integers, not the raw string -- a plain string sort
        would order same-millisecond entries correctly (sequence is always
        the same digit-width tiebreak) but could misorder different
        millisecond values with different digit widths; comparing as
        integers is correct either way."""
        ms_part, _, seq_part = event_id.partition("-")
        return (int(ms_part), int(seq_part))

    @staticmethod
    def _decode_entry(thread_id: str, entry_id, fields) -> TraceEventRecord:
        if isinstance(entry_id, bytes):
            entry_id = entry_id.decode("utf-8")
        data_field = fields.get(b"data", fields.get("data"))
        if isinstance(data_field, bytes):
            data_field = data_field.decode("utf-8")
        payload = json.loads(data_field)
        timestamp_ms = int(entry_id.split("-")[0])
        enqueued_at = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return TraceEventRecord(
            event_id=entry_id,
            thread_id=thread_id,
            payload=payload,
            enqueued_at=enqueued_at,
        )

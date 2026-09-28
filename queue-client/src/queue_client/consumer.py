import json
from datetime import datetime, timezone

import redis
import redis.exceptions

from queue_client.client import TraceEventRecord


class ConsumerClient:
    """Redis consumer-group semantics (Phase 3): multiple consumer
    instances reading from the same group split the workload -- each
    message goes to exactly one consumer in the group, with no duplicate
    processing under normal conditions.

    Scoped to a single thread's stream per call, matching this queue's
    per-agent-thread stream design (see `QueueClient`'s naming
    convention). Coordinating consumption across many different threads'
    streams is a consumption-strategy concern for whatever consumes this
    (the Evaluator), not this client's job.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        redis_client: "redis.Redis | None" = None,
        stream_prefix: str = "trace_events",
        group_name: str = "evaluator",
    ) -> None:
        if redis_client is not None:
            self._redis = redis_client
        elif redis_url is not None:
            self._redis = redis.Redis.from_url(redis_url)
        else:
            raise ValueError("either redis_url or redis_client must be provided")
        self._stream_prefix = stream_prefix
        self._group_name = group_name

    def _stream_key(self, thread_id: str) -> str:
        return f"{self._stream_prefix}:{thread_id}"

    def ensure_group(self, thread_id: str) -> None:
        """Idempotently creates the consumer group on `thread_id`'s
        stream, starting from the beginning of its history. `mkstream`
        creates the stream itself if no producer has written to it yet."""
        try:
            self._redis.xgroup_create(
                self._stream_key(thread_id), self._group_name, id="0", mkstream=True
            )
        except redis.exceptions.ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    def read_new(
        self, thread_id: str, consumer_name: str, count: int = 10
    ) -> list[TraceEventRecord]:
        """The next up-to-`count` messages nobody in this group has been
        given yet, assigned to `consumer_name`. Empty if none are
        pending."""
        response = self._redis.xreadgroup(
            self._group_name,
            consumer_name,
            {self._stream_key(thread_id): ">"},
            count=count,
        )
        if not response:
            return []

        records = []
        for _stream_key, entries in response:
            for entry_id, fields in entries:
                if isinstance(entry_id, bytes):
                    entry_id = entry_id.decode("utf-8")
                data_field = fields.get(b"data", fields.get("data"))
                if isinstance(data_field, bytes):
                    data_field = data_field.decode("utf-8")
                payload = json.loads(data_field)
                timestamp_ms = int(entry_id.split("-")[0])
                enqueued_at = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                records.append(
                    TraceEventRecord(
                        event_id=entry_id,
                        thread_id=thread_id,
                        payload=payload,
                        enqueued_at=enqueued_at,
                    )
                )
        return records

    def ack(self, thread_id: str, *event_ids: str) -> None:
        """Marks `event_ids` as processed, removing them from the
        group's pending-entries list."""
        if event_ids:
            self._redis.xack(self._stream_key(thread_id), self._group_name, *event_ids)

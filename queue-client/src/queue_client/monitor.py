import redis
import redis.exceptions


class QueueMonitor:
    """Queue depth, consumer lag, and pending-message visibility (Phase
    3) -- exposed via a queryable API a dashboard or logging setup can
    consume (a full dashboard UI is a separate, not-yet-built Phase 7
    task; this isn't it).

    `lag` (entries not yet delivered to any consumer in the group) and
    `pending` (delivered but not yet acked) are kept as two distinct
    metrics rather than conflated into one "lag" number -- they answer
    different operational questions: is the group falling behind the
    stream, vs. are consumers failing to ack what they've already
    received.
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

    def depth(self, thread_id: str) -> int:
        """Total entries currently in `thread_id`'s stream (subject to
        its MAXLEN cap). 0 for an unknown/empty thread."""
        return self._redis.xlen(self._stream_key(thread_id))

    def lag(self, thread_id: str, group_name: str | None = None) -> int:
        """Entries in the stream not yet delivered to any consumer in
        the group. 0 if the group doesn't exist.

        Deliberately NOT using XINFO GROUPS's own `lag`/`entries-read`
        fields: verified empirically that the installed fakeredis version
        computes them incorrectly (reading 2 messages only decreased its
        reported `lag` by 1, not 2). `last-delivered-id` was confirmed
        correct instead, so lag is computed manually by counting stream
        entries after it -- correct regardless of that library bug.
        """
        target = group_name or self._group_name
        target_bytes = target.encode("utf-8")
        for group in self._groups(thread_id):
            name = group.get(b"name", group.get("name"))
            if name in (target, target_bytes):
                last_delivered = group.get(b"last-delivered-id", group.get("last-delivered-id"))
                if isinstance(last_delivered, bytes):
                    last_delivered = last_delivered.decode("utf-8")
                undelivered = self._redis.xrange(
                    self._stream_key(thread_id), min=f"({last_delivered}", max="+"
                )
                return len(undelivered)
        return 0

    def pending(self, thread_id: str, group_name: str | None = None) -> int:
        """Messages delivered to a consumer but not yet acked. 0 if the
        group doesn't exist."""
        try:
            summary = self._redis.xpending(self._stream_key(thread_id), group_name or self._group_name)
        except redis.exceptions.ResponseError:
            return 0
        return summary.get("pending", 0) if summary else 0

    def _groups(self, thread_id: str) -> list:
        try:
            return self._redis.xinfo_groups(self._stream_key(thread_id))
        except redis.exceptions.ResponseError:
            return []

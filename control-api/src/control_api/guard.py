import json
import logging
from datetime import datetime, timezone

import redis
from langgraph.types import interrupt

DEFAULT_KEY_PREFIX = "halt"
DEFAULT_HALT_REASON = "halted by control-api"

# Dedicated logger (Phase 6 end-to-end tracing) -- DEBUG, since check_halt
# runs on every guard-node execution, not a low-frequency decision (the
# actual halt decision itself is separately logged, with full context, by
# control_api.app's "control_api.halt_audit" logger).
_trace_logger = logging.getLogger("control_api.trace")


class HaltRegistry:
    """Shared halt-signal store, checked live by guard nodes inside a
    customer's graph (Phase 5).

    Deliberately NOT part of LangGraph's own checkpointed state -- verified
    empirically before designing this that state updates from outside
    don't propagate to an already-in-flight `invoke()` call (LangGraph
    doesn't re-fetch checkpointed state between node executions within one
    invoke() call). A plain external store, checked fresh on every
    guard-node call, is the only mechanism that actually works for a live,
    externally-triggered halt.

    A third, distinct kind of state from both the checkpointer (LangGraph's
    own state) and `queue_client`'s Streams (trace events) -- control
    signals, not trace data or resumable graph state -- so it gets its own
    small dedicated Redis keyspace rather than reusing either.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        redis_client: "redis.Redis | None" = None,
        key_prefix: str = DEFAULT_KEY_PREFIX,
    ) -> None:
        if redis_client is not None:
            self._redis = redis_client
        elif redis_url is not None:
            self._redis = redis.Redis.from_url(redis_url)
        else:
            raise ValueError("either redis_url or redis_client must be provided")
        self._key_prefix = key_prefix

    def _key(self, thread_id: str) -> str:
        return f"{self._key_prefix}:{thread_id}"

    def is_halted(self, thread_id: str) -> bool:
        return bool(self._redis.get(self._key(thread_id)))

    def request_halt(self, thread_id: str, at: datetime | None = None) -> None:
        """Stores WHEN the halt was requested (not just a flag) -- the
        TTL reaper needs this to decide what's expired. `at` is
        injectable for deterministic tests; defaults to now."""
        timestamp = at if at is not None else datetime.now(timezone.utc)
        self._redis.set(self._key(thread_id), str(timestamp.timestamp()))

    def halted_at(self, thread_id: str) -> datetime | None:
        raw = self._redis.get(self._key(thread_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)

    def list_halted_thread_ids(self) -> list[str]:
        prefix_len = len(self._key_prefix) + 1  # +1 for the ":" separator
        keys = self._redis.keys(f"{self._key_prefix}:*")
        return [
            (key.decode("utf-8") if isinstance(key, bytes) else key)[prefix_len:]
            for key in keys
        ]

    def clear_halt(self, thread_id: str) -> None:
        self._redis.delete(self._key(thread_id))


def check_halt(
    thread_id: str, registry: HaltRegistry, reason: str = DEFAULT_HALT_REASON
) -> None:
    """Call at the start of a node (a "guard node," or wrap every node
    function with this) to honor a pending halt.

    Calling `interrupt()` here is the ONLY viable mechanism for an
    external process to stop a running graph -- confirmed empirically
    that `interrupt()` must be called from inside a node's own execution;
    it cannot be triggered on a running thread from outside it.
    """
    halted = registry.is_halted(thread_id)
    _trace_logger.debug(
        json.dumps({"trace_id": thread_id, "service": "control_api", "stage": "halt_checked", "halted": halted})
    )
    if halted:
        interrupt(reason)

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import redis
from schemas.halt_signal import HaltSignal

DEFAULT_KEY = "halt_history"


@dataclass
class HaltRecord:
    """A single past halt decision, as returned by a query -- richer than
    HaltRegistry's live is-halted/halted-at, carrying the full context
    (reason, confidence, triggering_window) the dashboard needs to show
    "why was this halted" without grepping logs."""

    halt_id: str
    trace_id: str
    reason: str
    confidence: float
    triggering_window: list[str]
    timestamp: datetime


class HaltHistoryStore:
    """Persisted, queryable history of past halt decisions (Phase 7
    dashboard task) -- a third thing, distinct from both HaltRegistry
    (live is-halted status, one value per thread, overwritten/cleared) and
    the halt-audit log (unstructured log lines, not queryable). Backed by
    one Redis sorted set (score = timestamp), not a per-thread stream like
    queue_client's trace events -- this needs time-range queries
    (ZRANGEBYSCORE) across ALL threads, which a per-thread structure
    doesn't give for free, and halt decisions are rare events (not a
    per-node-transition stream), so one global set is proportionate.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        redis_client: "redis.Redis | None" = None,
        key: str = DEFAULT_KEY,
    ) -> None:
        if redis_client is not None:
            self._redis = redis_client
        elif redis_url is not None:
            self._redis = redis.Redis.from_url(redis_url)
        else:
            raise ValueError("either redis_url or redis_client must be provided")
        self._key = key

    def record(self, signal: HaltSignal) -> str:
        """Appends a permanent record for this halt decision. Returns the
        new record's halt_id."""
        halt_id = str(uuid.uuid4())
        member = json.dumps(
            {
                "halt_id": halt_id,
                "trace_id": signal.trace_id,
                "reason": signal.reason,
                "confidence": signal.confidence,
                "triggering_window": signal.triggering_window,
                "timestamp": signal.timestamp.isoformat(),
            }
        )
        self._redis.zadd(self._key, {member: signal.timestamp.timestamp()})
        return halt_id

    def query(
        self,
        trace_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[HaltRecord]:
        """Records in [since, until] (both optional, both inclusive),
        optionally filtered further by trace_id, oldest to newest.
        trace_id filtering happens in Python after the time-range fetch --
        halt decisions are rare, so this stays cheap without a second,
        per-trace_id index to keep in sync."""
        min_score = since.timestamp() if since is not None else "-inf"
        max_score = until.timestamp() if until is not None else "+inf"
        raw_members = self._redis.zrangebyscore(self._key, min_score, max_score)

        records = []
        for raw in raw_members:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
            if trace_id is not None and data["trace_id"] != trace_id:
                continue
            records.append(
                HaltRecord(
                    halt_id=data["halt_id"],
                    trace_id=data["trace_id"],
                    reason=data["reason"],
                    confidence=data["confidence"],
                    triggering_window=data["triggering_window"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                )
            )
        return records

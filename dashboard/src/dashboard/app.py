from datetime import datetime
from pathlib import Path

from control_api.halt_history import HaltHistoryStore
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from queue_client.client import QueueClient

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(queue_client: QueueClient, halt_history: HaltHistoryStore) -> FastAPI:
    """Builds the dashboard's FastAPI app (Phase 7). A factory function,
    not a global app singleton, so tests inject fakeredis-backed
    QueueClient/HaltHistoryStore directly -- same pattern as
    control_api.app.create_app.

    Aggregates two existing, already-tested stores rather than owning any
    storage itself: `queue_client` for trace timelines,
    `halt_history` (control-api's Phase 7 addition) for halt records. This
    package's whole job is presenting that data, not producing it.
    """
    app = FastAPI()

    @app.get("/api/traces/{trace_id}")
    def get_trace(trace_id: str, limit: int = 500) -> list[dict]:
        """The full timeline for one thread, oldest to newest. Empty for
        an unknown/quiet thread rather than a 404 -- "no events yet" is a
        normal state for the dashboard to render, not an error."""
        records = queue_client.read_window(trace_id, limit=limit)
        return [
            {
                "event_id": record.event_id,
                "thread_id": record.thread_id,
                "payload": record.payload,
                "enqueued_at": record.enqueued_at.isoformat(),
            }
            for record in records
        ]

    @app.get("/api/halts")
    def list_halts(
        trace_id: str | None = None, since: str | None = None, until: str | None = None
    ) -> list[dict]:
        since_dt = datetime.fromisoformat(since) if since else None
        until_dt = datetime.fromisoformat(until) if until else None
        records = halt_history.query(trace_id=trace_id, since=since_dt, until=until_dt)
        return [
            {
                "halt_id": record.halt_id,
                "trace_id": record.trace_id,
                "reason": record.reason,
                "confidence": record.confidence,
                "triggering_window": record.triggering_window,
                "timestamp": record.timestamp.isoformat(),
            }
            for record in records
        ]

    @app.get("/api/halts/{halt_id}/context")
    def get_halt_context(halt_id: str) -> list[dict]:
        """The "few clicks" drill-down the AC asks for: resolves a halt's
        triggering_window (a list of event ids) back to the real events
        that triggered it. HaltHistoryStore has no direct by-id lookup
        (its Redis sorted set is keyed by timestamp, not halt_id) -- an
        unfiltered query() + Python-side filter is proportionate here
        since halt decisions are rare events, not a high-frequency
        stream."""
        matches = [record for record in halt_history.query() if record.halt_id == halt_id]
        if not matches:
            raise HTTPException(status_code=404, detail="halt not found")
        record = matches[0]
        events = queue_client.get_events_by_id(record.trace_id, record.triggering_window)
        return [
            {
                "event_id": event.event_id,
                "payload": event.payload,
                "enqueued_at": event.enqueued_at.isoformat(),
            }
            for event in events
        ]

    # Registered after the /api/* routes above -- Starlette matches routes
    # in registration order, so the specific API routes take precedence
    # over this catch-all mount for their own paths.
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app

# dashboard

Trace visualization and halt history dashboard for the Agent Execution
Guardian (Phase 7) — the first user-facing surface in the project;
everything else is a backend service/library.

## What it aggregates

This package produces nothing itself — it presents data two already-built
components own:
- **Trace timelines**: `queue_client.QueueClient` (Phase 3) — one Redis
  stream per agent thread.
- **Halt history**: `control_api.halt_history.HaltHistoryStore` (Phase 7,
  built alongside this task) — a new, persisted, queryable log of past
  halt decisions (`reason`/`confidence`/`triggering_window`/`timestamp`).
  This didn't previously exist: `control_api.guard.HaltRegistry` only ever
  tracked *current* is-halted status per thread (overwritten/cleared, not
  a history), and the halt-audit log (`control_api.app`) is unstructured
  log lines, not a queryable store. Without `HaltHistoryStore`, "find a
  halt and see its triggering context" would mean grepping logs.

## Backend: FastAPI

`src/dashboard/app.py` — `create_app(queue_client, halt_history) -> FastAPI`,
same factory-function pattern as `control_api.app.create_app` (tests
inject fakeredis-backed stores directly).

- `GET /api/traces/{trace_id}?limit=500` — the full timeline for one
  thread, oldest to newest. Empty list (not 404) for an unknown/quiet
  thread — "no events yet" is a normal state, not an error.
- `GET /api/halts?trace_id=...&since=...&until=...` — halt history,
  filterable by trace_id and/or a time range (all optional).
- `GET /api/halts/{halt_id}/context` — the "few clicks" drill-down the AC
  asks for: resolves a halt's `triggering_window` (a list of event ids)
  back to the actual events that triggered it, via `queue_client`'s new
  `get_events_by_id` (added alongside this task specifically for this).
  404 for an unknown `halt_id`.

`HaltHistoryStore` has no direct by-id lookup (its Redis sorted set is
keyed by timestamp, not `halt_id`) — `get_halt_context` does an unfiltered
`query()` + Python-side filter, proportionate since halt decisions are
rare events, not a high-frequency stream.

## Frontend: a single static page, no build step

`src/dashboard/static/index.html`, served by FastAPI's `StaticFiles` mount
at `/` (registered *after* the `/api/*` routes, so Starlette's
route-matching-in-registration-order gives the specific API routes
precedence over the catch-all static mount). React is loaded via CDN +
Babel Standalone (in-browser JSX) — no `package.json`/npm/Node build
toolchain, "lightweight" both in dependencies and in what a contributor
needs installed to touch it, consistent with this being an otherwise
pure-Python repo.

**Verification note:** this environment has no GUI browser available to
actually render and click through the page. The backend API contract the
page depends on is verified for real (`tests/test_app.py`, `TestClient`,
fakeredis) — genuine HTTP requests/responses, not mocked. The frontend
itself was checked for valid HTML/JS and correct `fetch()` calls against
that same API contract, but not visually confirmed rendering in a real
browser. Flagging this explicitly rather than claiming a check that
wasn't actually performed.

## Install

```bash
pip install -e ".[test]"
```

## Run

```bash
python -m dashboard  # starts uvicorn on :8000 against a real Redis
```

Requires `REDIS_URL` (or defaults to `redis://localhost:6379/0`) for both
`QueueClient` and `HaltHistoryStore` — see `src/dashboard/__main__.py`.

## Test

```bash
pytest
```

## Status

Backend API (trace timelines, halt history with trace_id/time-range
filtering, triggering-context drill-down) implemented and tested against
real fakeredis-backed stores. Frontend is a working single-page UI calling
that API, not a mockup — see the verification note above for what could
and couldn't be checked in this environment.

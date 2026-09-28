import json
import logging
import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from schemas.halt_signal import HaltSignal

from control_api.guard import HaltRegistry
from control_api.halt_history import HaltHistoryStore

_ENV_VAR = "CONTROL_API_TOKEN"
_security = HTTPBearer()

# A dedicated logger (not the module default) so operators can configure
# halt-audit logs distinctly -- a different destination/retention policy
# than general app logs, since this is a post-mortem-review audit trail,
# not routine operational logging.
_HALT_AUDIT_LOGGER = logging.getLogger("control_api.halt_audit")


def _log_halt_decision(signal: HaltSignal) -> None:
    """One JSON-serialized line per halt decision -- "queryable/
    searchable" only requires the content to be structured/parseable, not
    a specific log-aggregation backend; grep/jq on a JSON-lines file (or
    any real aggregator) can query this."""
    _HALT_AUDIT_LOGGER.info(
        json.dumps(
            {
                "trace_id": signal.trace_id,
                "reason": signal.reason,
                "confidence": signal.confidence,
                "triggering_window": signal.triggering_window,
                "timestamp": signal.timestamp.isoformat(),
            }
        )
    )


def create_app(
    registry: HaltRegistry,
    api_token: str | None = None,
    halt_history: HaltHistoryStore | None = None,
) -> FastAPI:
    """Builds the Control API FastAPI app (Phase 5). A factory function,
    not a global app singleton, so tests inject a fakeredis-backed
    HaltRegistry directly rather than needing FastAPI's
    dependency-override machinery.

    `api_token` (or the `CONTROL_API_TOKEN` env var) is required -- an
    auth mechanism that can be silently skipped by omission isn't a real
    security control, so this fails fast rather than starting an
    unauthenticated `/halt` endpoint.

    `halt_history` (Phase 7 dashboard task) is optional -- `None` (the
    default) preserves prior behavior exactly (no persisted history, every
    existing caller unaffected); pass a `HaltHistoryStore` to also record
    every accepted halt for later querying by the dashboard.
    """
    token = api_token or os.environ.get(_ENV_VAR)
    if not token:
        raise ValueError(
            f"api_token must be provided or {_ENV_VAR} must be set -- "
            "the /halt endpoint must not run without authentication configured"
        )

    app = FastAPI()

    def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(_security)) -> None:
        if credentials.credentials != token:
            raise HTTPException(status_code=401, detail="invalid or missing token")

    @app.post("/halt", status_code=202, dependencies=[Depends(_verify_token)])
    def halt(signal: HaltSignal) -> dict:
        """Declaring `signal: HaltSignal` is what gives us "invalid
        payloads return a clear 4xx" -- FastAPI validates the request
        body against the pydantic model automatically (verified
        empirically before writing this), returning 422 with per-field
        detail on anything invalid, before this function body even runs.

        202, not 200: the halt isn't immediately confirmed complete, only
        accepted for processing -- it takes effect at the guard node's
        next check, consistent with the already-documented
        node-granularity-bound halt latency.
        """
        registry.request_halt(signal.trace_id)
        _log_halt_decision(signal)
        if halt_history is not None:
            halt_history.record(signal)
        return {"status": "accepted", "trace_id": signal.trace_id}

    return app

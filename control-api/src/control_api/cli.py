import argparse
import sys
from datetime import datetime, timezone
from typing import Any

import httpx
from schemas.halt_signal import HaltSignal

DEFAULT_MANUAL_REASON = "manual_override"
DEFAULT_MANUAL_TRIGGERING_WINDOW = ["manual-override"]
DEFAULT_MANUAL_CONFIDENCE = 1.0


def _build_http_client(control_api_url: str, api_token: str) -> httpx.Client:
    return httpx.Client(base_url=control_api_url, headers={"Authorization": f"Bearer {api_token}"})


def trigger_manual_halt(
    trace_id: str,
    control_api_url: str | None = None,
    api_token: str | None = None,
    reason: str = DEFAULT_MANUAL_REASON,
    confidence: float = DEFAULT_MANUAL_CONFIDENCE,
    triggering_window: list[str] | None = None,
    http_client: Any = None,
) -> dict:
    """A human can call this (or the `main()` CLI entry point below) to
    trigger a halt manually, via the SAME `/halt` endpoint an automated
    detection uses (Phase 5) -- the agent halts identically either way,
    since the endpoint doesn't distinguish who/what sent the request.

    `confidence` defaults to full (1.0): a human's manual decision isn't a
    probabilistic judgment. `triggering_window` defaults to a sentinel
    (`HaltSignal` requires at least one entry) when a human isn't
    pointing at specific event ids.

    `http_client` is injectable (any object exposing `.post(url,
    json=...)`) -- a real `httpx.Client` by default (built from
    `control_api_url`/`api_token`), or a test double.
    """
    signal = HaltSignal(
        trace_id=trace_id,
        reason=reason,
        confidence=confidence,
        triggering_window=triggering_window or DEFAULT_MANUAL_TRIGGERING_WINDOW,
        timestamp=datetime.now(timezone.utc),
    )
    client = http_client if http_client is not None else _build_http_client(control_api_url, api_token)
    response = client.post("/halt", json=signal.model_dump(mode="json"))
    response.raise_for_status()
    return response.json()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: `python -m control_api.cli --trace-id ... --control-api-url ... --token ...`"""
    parser = argparse.ArgumentParser(description="Manually trigger a halt via Control API's /halt endpoint.")
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--control-api-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--reason", default=DEFAULT_MANUAL_REASON)
    parser.add_argument("--confidence", type=float, default=DEFAULT_MANUAL_CONFIDENCE)
    args = parser.parse_args(argv)

    client = _build_http_client(args.control_api_url, args.token)
    result = trigger_manual_halt(
        args.trace_id, reason=args.reason, confidence=args.confidence, http_client=client
    )
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

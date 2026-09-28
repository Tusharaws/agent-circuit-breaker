from typing import Any, Protocol

from evaluator.verdict import Verdict, build_halt_signal
from evaluator.window import DetectionWindow

DEFAULT_CONFIDENCE_THRESHOLD = 0.7


class _HttpClient(Protocol):
    def post(self, url: str, json: dict) -> Any: ...


def maybe_trigger_halt(
    window: DetectionWindow,
    verdict: Verdict,
    http_client: _HttpClient,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """If `verdict.is_loop` and confidence >= `confidence_threshold`,
    POST a `HaltSignal` to Control API's `/halt` endpoint (Phase 4/5
    wiring). Returns True if a halt was actually triggered.

    `confidence_threshold` is a separate, operator-tunable safety gate on
    top of `verdict.is_loop` itself -- the eval set found real SLM
    misjudgments even at high confidence, so this doesn't fully solve
    precision on its own; it establishes the mechanism for tuning it
    further without re-plumbing.

    `http_client` is injectable -- any object exposing
    `.post(url, json=...)`. A real `httpx.Client` in production;
    `fastapi.testclient.TestClient` in tests (a real, synchronous ASGI
    test harness with a compatible `.post()` signature, confirmed
    empirically before designing this -- not a mock).
    """
    if not verdict.is_loop or verdict.confidence < confidence_threshold:
        return False

    signal = build_halt_signal(window, verdict)
    response = http_client.post("/halt", json=signal.model_dump(mode="json"))
    response.raise_for_status()
    return True

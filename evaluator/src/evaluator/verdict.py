import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from schemas.halt_signal import HaltSignal

from evaluator.window import DetectionWindow

VALID_REASONS = {"repeated_tool_calls", "repeated_generation", "runaway_loop"}
DEFAULT_REASON = "runaway_loop"

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Verdict:
    is_loop: bool
    confidence: float
    reason: str | None  # always None when is_loop is False


def build_prompt(window: DetectionWindow) -> str:
    """The detection prompt: instructs the SLM to judge a window of
    recent execution steps for semantic loop/repetition, per
    ARCHITECTURE_NOTES.md ("same underlying intent/approach recurring",
    not just exact-match repetition -- that's the pre-filter's job)."""
    steps = "\n".join(
        f"{i + 1}. {event.event_type}: {json.dumps(event.payload)}"
        for i, event in enumerate(window.events)
    )
    return f"""You are judging whether an AI agent is stuck in an unproductive loop.

Below is a window of recent execution steps. Judge whether the agent is repeating the same underlying approach without making real progress -- even if wording or arguments vary slightly each time -- versus genuinely progressing toward a result.

Steps:
{steps}

Respond with ONLY a JSON object, no markdown code fences, no other text, containing exactly these three keys: "is_loop" (true or false), "confidence" (a number from 0.0 to 1.0), "reason" (one of "repeated_tool_calls", "repeated_generation", "runaway_loop" if is_loop is true, otherwise null).
"""


def parse_verdict(raw_response: str) -> Verdict:
    """Lenient parsing: the SLM (a small model) doesn't reliably follow
    the requested output shape -- confirmed empirically, not assumed. It
    has produced extra unrequested keys, and free-text instead of the
    constrained `reason` enum. This extracts just what's needed and
    validates/coerces each field independently, failing closed
    (is_loop=False) on anything unparseable -- a false positive here
    means Control API halts a healthy agent run, far more disruptive than
    missing one window's detection (the next window gets another chance).
    """
    match = _JSON_OBJECT_RE.search(raw_response)
    if not match:
        return Verdict(is_loop=False, confidence=0.0, reason=None)

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(is_loop=False, confidence=0.0, reason=None)

    if not isinstance(data, dict):
        return Verdict(is_loop=False, confidence=0.0, reason=None)

    is_loop = bool(data.get("is_loop", False))

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = None
    if is_loop:
        candidate = data.get("reason")
        reason = candidate if candidate in VALID_REASONS else DEFAULT_REASON

    return Verdict(is_loop=is_loop, confidence=confidence, reason=reason)


def build_halt_signal(window: DetectionWindow, verdict: Verdict) -> HaltSignal:
    """Only meaningful for a positive verdict -- constructs the signal
    Control API's `POST /halt` expects directly from the window + verdict
    that produced it."""
    if not verdict.is_loop:
        raise ValueError("cannot build a HaltSignal from a no-loop verdict")

    return HaltSignal(
        trace_id=window.thread_id,
        reason=verdict.reason,
        confidence=verdict.confidence,
        triggering_window=window.event_ids,
        timestamp=datetime.now(timezone.utc),
    )

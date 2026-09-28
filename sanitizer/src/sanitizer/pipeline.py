import json
import logging
from typing import Any, Callable

from sanitizer.policy import load_redaction_policy
from sanitizer.scrubbers import scrub_text

# Dedicated logger (Phase 6 end-to-end tracing) so a trace_id is followable
# across all 5 services' logs -- DEBUG, since this fires on every event.
_trace_logger = logging.getLogger("sanitizer.trace")

logger = logging.getLogger(__name__)


def _extract_trace_id(event: Any) -> str | None:
    """Handles both event shapes SanitizingSink already supports: a real
    TraceEvent (trace_id attribute) or a plain dict (queue_client's own
    generic payload shape, "trace_id" key). Returns None for either shape
    lacking one -- there's nothing to correlate by, but the event is still
    sanitized and forwarded regardless."""
    if hasattr(event, "trace_id"):
        return event.trace_id
    if isinstance(event, dict):
        return event.get("trace_id")
    return None


def sanitize_value(value: Any, policy: dict[str, str], ner_scrubber: Any = None) -> Any:
    """Recursively scrub every string found anywhere in `value` -- nested
    dicts and lists included. Non-string scalars (int/float/bool/None)
    pass through untouched. When `ner_scrubber` (a `sanitizer.ner.NerScrubber`)
    is provided, each string also gets an NER pass (names/addresses/medical
    terms) after the regex pass -- omitted by default, which preserves
    today's regex-only behavior for callers that don't pass one."""
    if isinstance(value, str):
        text = scrub_text(value, policy)
        if ner_scrubber is not None:
            text = ner_scrubber.redact_with_policy(text, policy)
        return text
    if isinstance(value, dict):
        return {key: sanitize_value(val, policy, ner_scrubber) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item, policy, ner_scrubber) for item in value]
    return value


class SanitizingSink:
    """The mandatory sanitization step between capture and any
    network-facing sink (Phase 2). Composable `sink` matching
    CaptureDispatcher's `Callable[[Any], None]` contract -- the one
    blessed way production code wires capture to a real queue sink:
    `CaptureDispatcher(sink=SanitizingSink(inner=queue_client_instance.append_event))`.

    Handles both a `schemas.TraceEvent` (dumped, sanitized, rebuilt as a
    TraceEvent) and a plain dict/JSON-serializable event (queue_client's
    own generic payload shape) -- doesn't assume TraceEvent specifically.

    `ner_scrubber` is optional (default `None` = regex-only, today's
    behavior) -- pass a `sanitizer.ner.NerScrubber` instance to also catch
    unstructured PHI (names/addresses/medical terms). Construct it once
    and reuse it (it loads a spaCy model), don't build one per sink.
    """

    def __init__(
        self,
        inner: Callable[[Any], None],
        policy: dict[str, str] | None = None,
        ner_scrubber: Any = None,
    ) -> None:
        if not callable(inner):
            raise TypeError(f"inner must be callable, got {type(inner).__name__}")
        self._inner = inner
        self._policy = policy if policy is not None else load_redaction_policy()
        self._ner_scrubber = ner_scrubber

    def __call__(self, event: Any) -> None:
        trace_id = _extract_trace_id(event)
        try:
            sanitized = self._sanitize_event(event)
        except Exception:
            # Fail-closed, and safe to log: never forward an event whose
            # sanitization failed (matches the existing behavior on the
            # queue side -- an unhandled exception already meant the event
            # never reached the queue), and never log the raw event itself
            # (%r on it would include whatever raw PII triggered the
            # failure in the first place -- the exact leak this exists to
            # prevent). Only trace_id, a bare identifier, is logged.
            logger.error("sanitizer: sanitization failed for trace_id=%s, event dropped", trace_id)
            return
        if trace_id is not None:
            _trace_logger.debug(
                json.dumps({"trace_id": trace_id, "service": "sanitizer", "stage": "sanitized"})
            )
        self._inner(sanitized)

    def _sanitize_event(self, event: Any) -> Any:
        if hasattr(event, "model_dump") and hasattr(type(event), "model_validate"):
            dumped = event.model_dump(mode="json")
            sanitized = sanitize_value(dumped, self._policy, self._ner_scrubber)
            return type(event).model_validate(sanitized)
        return sanitize_value(event, self._policy, self._ner_scrubber)

from typing import Any, Callable

from sanitizer.pipeline import SanitizingSink
from schemas.trace_event import TraceEvent


def make_queue_sink(append_event: Callable[[str, dict], str]) -> Callable[[TraceEvent], None]:
    """Adapts a two-argument sink -- `QueueClient.append_event(thread_id,
    payload)` -- into the single-argument sink shape `CaptureDispatcher`/
    `SanitizingSink` expect (`Callable[[Any], None]`).

    Found as a real integration bug during Phase 6 wiring, not designed
    upfront: each side was individually correct and fully tested in
    isolation (`SanitizingSink` calling `inner(event)` with one argument;
    `QueueClient.append_event` requiring `thread_id` and `payload`
    separately for stream routing), but nothing exercised the seam
    between them until they were actually wired together for the first
    time. `thread_id` is extracted from the event's own `trace_id` field
    -- the two are the same value by design (see the resolved trace_id vs
    thread_id identity model in ARCHITECTURE_NOTES.md).
    """

    def sink(event: TraceEvent) -> None:
        append_event(event.trace_id, event.model_dump(mode="json"))

    return sink


def make_sanitized_queue_sink(
    append_event: Callable[[str, dict], str],
    policy: dict[str, str] | None = None,
    ner_scrubber: Any = None,
) -> Callable[[TraceEvent], None]:
    """The blessed, single-call production wiring -- always sanitizes an
    event before it reaches the queue.

    A Phase 7 security review found that production wiring could
    previously skip `SanitizingSink` entirely (`CaptureDispatcher(sink=
    make_queue_sink(...))` with no wrapper) and nothing in the type system
    or runtime would catch it -- convention/documentation only, enforced
    by nothing (see `sanitizer/SECURITY_REVIEW.md`). This doesn't
    structurally *prevent* a caller from still reaching for the unwrapped
    `make_queue_sink` -- that remains available for whatever doesn't need
    sanitization (e.g. `evaluator`'s own eval-set harness, which seeds
    synthetic, non-customer data directly) -- but it makes the safe,
    sanitized path the single easiest thing to reach for in the customer
    integration case, rather than a 2-step manual composition someone
    could forget half of.

    `policy`/`ner_scrubber` are passed straight through to `SanitizingSink`
    (same defaults: `policy=None` loads the bundled redaction policy,
    `ner_scrubber=None` is regex-only).
    """
    return SanitizingSink(inner=make_queue_sink(append_event), policy=policy, ner_scrubber=ner_scrubber)

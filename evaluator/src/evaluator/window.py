from dataclasses import dataclass, field

from queue_client.client import QueueClient
from schemas.trace_event import TraceEvent

DEFAULT_WINDOW_SIZE = 10


@dataclass
class DetectionWindow:
    """The "window of recent steps" ARCHITECTURE_NOTES.md describes as
    the input to semantic loop judgment -- real, validated TraceEvent
    instances, oldest to newest, not raw queue payloads.

    `event_ids` (parallel to `events`, same order) carries the actual
    Redis-assigned event ids -- needed because `HaltSignal.triggering_window`
    requires them and `TraceEvent` itself doesn't carry one (that's
    `TraceEventRecord`'s job). Defaults to `[]` so existing direct
    constructions (e.g. in prefilter tests, which don't need ids) don't
    need to supply it.
    """

    thread_id: str
    events: list[TraceEvent]
    event_ids: list[str] = field(default_factory=list)


def build_window(
    queue_client: QueueClient, thread_id: str, window_size: int = DEFAULT_WINDOW_SIZE
) -> DetectionWindow:
    """N-steps window: the most recent `window_size` trace events for
    `thread_id`, oldest to newest. Reuses `QueueClient.read_window`
    directly -- its MAXLEN-capped per-thread stream design already
    implements exactly this; no new mechanism needed.

    A time-based (T seconds) window is a documented future extension:
    `TraceEventRecord.enqueued_at` (from `read_window`) already carries
    what's needed to filter by age on top of this, if ever required.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    records = queue_client.read_window(thread_id, limit=window_size)
    events = [TraceEvent.model_validate(record.payload) for record in records]
    event_ids = [record.event_id for record in records]
    return DetectionWindow(thread_id=thread_id, events=events, event_ids=event_ids)

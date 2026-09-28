from datetime import datetime, timedelta, timezone
from typing import Any

from control_api.guard import HaltRegistry

# Deliberately different from interceptor's 2-day local-buffer eviction
# convention -- that protects in-process memory during a Redis outage (a
# different concern entirely). This is retention time for operators to
# review a halt (see the halt-logging task) before its checkpointer state
# is cleaned up.
DEFAULT_TTL = timedelta(days=7)


def reap_expired_halts(
    registry: HaltRegistry, checkpointer: Any, ttl: timedelta = DEFAULT_TTL
) -> list[str]:
    """Deletes checkpointer state and clears the halt-registry entry for
    every halted thread older than `ttl` (Phase 5) -- LangGraph does not
    auto-expire frozen threads, so without this, halted threads' state
    accumulates in the checkpointer store forever.

    Returns the thread_ids actually reaped.
    """
    now = datetime.now(timezone.utc)
    reaped = []

    for thread_id in registry.list_halted_thread_ids():
        halted_at = registry.halted_at(thread_id)
        if halted_at is None:
            continue
        if now - halted_at >= ttl:
            checkpointer.delete_thread(thread_id)
            registry.clear_halt(thread_id)
            reaped.append(thread_id)

    return reaped

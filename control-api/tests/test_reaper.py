"""Tests for the TTL-based reaper (Phase 5). Written before the
implementation (control_api.reaper does not exist yet) -- run `pytest` to
see them fail with a collection error until src/control_api/reaper.py
exists.

Verified empirically before designing (see the update posted to this
backlog item): InMemorySaver/SqliteSaver both expose delete_thread(), and
it genuinely clears checkpointed state (confirmed with a real graph, not
assumed from documentation).
"""
from datetime import datetime, timedelta, timezone
from typing import TypedDict

import fakeredis
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from control_api.guard import HaltRegistry
from control_api.reaper import DEFAULT_TTL, reap_expired_halts


class _State(TypedDict):
    count: int


def _build_and_run_graph(checkpointer, thread_id: str):
    def inc(state: _State) -> dict:
        return {"count": state["count"] + 1}

    graph = StateGraph(_State)
    graph.add_node("inc", inc)
    graph.add_edge(START, "inc")
    graph.add_edge("inc", END)
    app = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke({"count": 0}, config=config)
    return app, config


@pytest.fixture
def registry():
    return HaltRegistry(redis_client=fakeredis.FakeRedis())


@pytest.fixture
def checkpointer():
    return InMemorySaver()


# ---------------------------------------------------------------------------
# Basic reaping behavior
# ---------------------------------------------------------------------------


def test_thread_younger_than_ttl_is_not_reaped(registry, checkpointer):
    app, config = _build_and_run_graph(checkpointer, "thread-recent")
    registry.request_halt("thread-recent")  # halted "now"

    reaped = reap_expired_halts(registry, checkpointer, ttl=timedelta(days=7))

    assert reaped == []
    assert registry.is_halted("thread-recent") is True
    assert app.get_state(config).values == {"count": 1}  # checkpointer state untouched


def test_thread_older_than_ttl_is_reaped(registry, checkpointer):
    app, config = _build_and_run_graph(checkpointer, "thread-old")
    old_timestamp = datetime.now(timezone.utc) - timedelta(days=8)
    registry.request_halt("thread-old", at=old_timestamp)

    reaped = reap_expired_halts(registry, checkpointer, ttl=timedelta(days=7))

    assert reaped == ["thread-old"]
    assert registry.is_halted("thread-old") is False  # registry entry cleared
    assert app.get_state(config).values == {}  # checkpointer state cleared


def test_never_halted_thread_is_untouched(registry, checkpointer):
    app, config = _build_and_run_graph(checkpointer, "thread-normal")

    reaped = reap_expired_halts(registry, checkpointer, ttl=timedelta(days=7))

    assert reaped == []
    assert app.get_state(config).values == {"count": 1}


def test_only_expired_threads_are_reaped_among_several(registry, checkpointer):
    _build_and_run_graph(checkpointer, "thread-old")
    _build_and_run_graph(checkpointer, "thread-recent")

    registry.request_halt("thread-old", at=datetime.now(timezone.utc) - timedelta(days=10))
    registry.request_halt("thread-recent")

    reaped = reap_expired_halts(registry, checkpointer, ttl=timedelta(days=7))

    assert reaped == ["thread-old"]
    assert registry.is_halted("thread-recent") is True


def test_default_ttl_is_seven_days():
    assert DEFAULT_TTL == timedelta(days=7)


def test_reap_uses_default_ttl_when_not_specified(registry, checkpointer):
    _build_and_run_graph(checkpointer, "thread-old")
    registry.request_halt("thread-old", at=datetime.now(timezone.utc) - timedelta(days=8))

    reaped = reap_expired_halts(registry, checkpointer)

    assert reaped == ["thread-old"]

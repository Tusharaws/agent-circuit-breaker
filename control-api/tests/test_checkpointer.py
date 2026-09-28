"""Tests for environment-driven checkpointer selection (Phase 5). Written
before the implementation (control_api.checkpointer does not exist yet)
-- run `pytest` to see them fail with a collection error until
src/control_api/checkpointer.py exists.

SqliteSaver is tested against a real in-memory (":memory:") SQLite
connection, not mocked -- confirmed empirically before writing these that
this actually works end-to-end (state persists, is retrievable). This
environment has no real Postgres server (same honest caveat as
queue-client's load test having no real Redis), so the "prod" tier's
tests verify OUR routing logic (it calls PostgresSaver.from_conn_string
with the right argument) via mocking that one call, not genuine Postgres
connectivity.
"""
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from control_api.checkpointer import get_checkpointer


# ---------------------------------------------------------------------------
# "test" environment: InMemorySaver
# ---------------------------------------------------------------------------


def test_test_environment_yields_an_in_memory_saver():
    with get_checkpointer(environment="test") as checkpointer:
        assert isinstance(checkpointer, InMemorySaver)


def test_default_environment_is_test_when_nothing_specified(monkeypatch):
    monkeypatch.delenv("CONTROL_API_ENVIRONMENT", raising=False)

    with get_checkpointer() as checkpointer:
        assert isinstance(checkpointer, InMemorySaver)


def test_environment_variable_is_used_when_no_explicit_argument(monkeypatch):
    monkeypatch.setenv("CONTROL_API_ENVIRONMENT", "test")

    with get_checkpointer() as checkpointer:
        assert isinstance(checkpointer, InMemorySaver)


# ---------------------------------------------------------------------------
# "dev" environment: SqliteSaver, tested against a real :memory: connection
# ---------------------------------------------------------------------------


def test_dev_environment_yields_a_working_sqlite_saver():
    with get_checkpointer(environment="dev", sqlite_path=":memory:") as checkpointer:
        assert isinstance(checkpointer, SqliteSaver)

        # prove it's genuinely functional, not just the right type
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class State(TypedDict):
            count: int

        def increment(state: State) -> dict:
            return {"count": state["count"] + 1}

        graph = StateGraph(State)
        graph.add_node("increment", increment)
        graph.add_edge(START, "increment")
        graph.add_edge("increment", END)
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "t-1"}}
        app.invoke({"count": 0}, config=config)

        assert app.get_state(config).values == {"count": 1}


# ---------------------------------------------------------------------------
# "prod" environment: PostgresSaver -- routing verified via mock, no real
# Postgres server exists in this environment
# ---------------------------------------------------------------------------


def test_prod_environment_requires_a_connection_string():
    with pytest.raises(ValueError):
        with get_checkpointer(environment="prod"):
            pass


def test_prod_environment_delegates_to_postgres_saver_with_the_given_conn_string():
    fake_saver = MagicMock()
    fake_context_manager = MagicMock()
    fake_context_manager.__enter__.return_value = fake_saver
    fake_context_manager.__exit__.return_value = False

    with patch(
        "control_api.checkpointer.PostgresSaver.from_conn_string", return_value=fake_context_manager
    ) as mock_from_conn_string:
        with get_checkpointer(environment="prod", postgres_conn_string="postgresql://x") as checkpointer:
            assert checkpointer is fake_saver

    mock_from_conn_string.assert_called_once_with("postgresql://x")
    fake_saver.setup.assert_called_once()


# ---------------------------------------------------------------------------
# Invalid environment
# ---------------------------------------------------------------------------


def test_invalid_environment_is_rejected():
    with pytest.raises(ValueError):
        with get_checkpointer(environment="staging"):
            pass

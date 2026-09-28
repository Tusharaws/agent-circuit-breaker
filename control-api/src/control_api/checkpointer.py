import os
from contextlib import contextmanager

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver

_ENV_VAR = "CONTROL_API_ENVIRONMENT"
VALID_ENVIRONMENTS = {"test", "dev", "prod"}
DEFAULT_ENVIRONMENT = "test"
DEFAULT_SQLITE_PATH = "checkpoints.db"


@contextmanager
def get_checkpointer(
    environment: str | None = None,
    sqlite_path: str = DEFAULT_SQLITE_PATH,
    postgres_conn_string: str | None = None,
):
    """Environment-driven checkpointer selection (Phase 5): `InMemorySaver`
    for unit tests, `SqliteSaver` for local dev, `PostgresSaver` for
    prod -- required because `interrupt()` raises without a checkpointer
    attached to the compiled graph. Picked via `environment` or the
    `CONTROL_API_ENVIRONMENT` env var; no code change needed to switch.

    A context manager: `SqliteSaver`/`PostgresSaver` both need their
    underlying connection cleaned up (confirmed via a real end-to-end
    check before writing this), so every tier is used the same way --
    `with get_checkpointer(...) as checkpointer:` -- even though
    `InMemorySaver` itself needs no cleanup.
    """
    target = environment or os.environ.get(_ENV_VAR, DEFAULT_ENVIRONMENT)
    if target not in VALID_ENVIRONMENTS:
        raise ValueError(f"environment must be one of {sorted(VALID_ENVIRONMENTS)}, got {target!r}")

    if target == "test":
        yield InMemorySaver()
    elif target == "dev":
        with SqliteSaver.from_conn_string(sqlite_path) as saver:
            yield saver
    else:  # prod
        if not postgres_conn_string:
            raise ValueError("postgres_conn_string is required for the prod environment")
        with PostgresSaver.from_conn_string(postgres_conn_string) as saver:
            saver.setup()
            yield saver

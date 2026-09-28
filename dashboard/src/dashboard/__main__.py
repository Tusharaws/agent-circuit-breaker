"""Runs the dashboard against a real Redis (`python -m dashboard`). All
other entry points into this package (the tests, `create_app` itself) use
injected fakeredis-backed stores instead -- this is the one place a real
`redis.Redis.from_url` connection is actually made."""
import os

import uvicorn
from control_api.halt_history import HaltHistoryStore
from queue_client.client import QueueClient

from dashboard.app import create_app

DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def main() -> None:
    redis_url = os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
    queue_client = QueueClient(redis_url=redis_url)
    halt_history = HaltHistoryStore(redis_url=redis_url)
    app = create_app(queue_client, halt_history)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()

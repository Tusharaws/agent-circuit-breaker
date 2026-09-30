"""Runnable entry point for the retail demo (Retail Demo, Story 3, task 2)
-- runs either variant against a REAL local Redis instance (not
fakeredis), mirroring dashboard/src/dashboard/__main__.py's pattern.

The fixture modules (looping_agent.py, happy_path_agent.py,
retail_pipeline.py) live in tests/ (matching
integration-tests/tests/sample_agent.py's own convention of co-locating
fixtures with their tests) -- added to sys.path explicitly below since
this script lives one level up, in examples/retail_demo/ itself.

Usage:
    python run_retail_demo.py --variant happy
    python run_retail_demo.py --variant looping --halt-after 1.5
    REDIS_URL=redis://localhost:6379/1 python run_retail_demo.py --variant looping
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))

import redis  # noqa: E402
from control_api.guard import HaltRegistry  # noqa: E402
from interceptor.capture import CaptureDispatcher  # noqa: E402
from interceptor.graph_hooks import GraphCaptureHandler  # noqa: E402
from interceptor.queue_sink import make_sanitized_queue_sink  # noqa: E402
from queue_client.client import QueueClient  # noqa: E402

from happy_path_agent import build_happy_path_agent  # noqa: E402
from looping_agent import build_looping_agent  # noqa: E402

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
AGENT_ID = "retail-demo-agent"


def _build_pipeline(redis_client):
    queue_client = QueueClient(redis_client=redis_client)
    registry = HaltRegistry(redis_client=redis_client)
    dispatcher = CaptureDispatcher(sink=make_sanitized_queue_sink(queue_client.append_event))
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    return queue_client, registry, dispatcher, handler


def _initial_state(query):
    return {
        "query": query,
        "attempts": [],
        "resolved": False,
        "order_id": None,
        "shipping_status": None,
        "escalated": False,
    }


def run_happy_path(redis_client, thread_id):
    queue_client, registry, dispatcher, handler = _build_pipeline(redis_client)
    app = build_happy_path_agent(registry)
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]}

    print(f"Running happy-path agent (thread_id={thread_id!r})...")
    result = app.invoke(_initial_state("ORD-10293"), config=config)
    dispatcher.stop(timeout=2.0)

    print(f"  resolved={result['resolved']} order_id={result['order_id']} "
          f"shipping_status={result['shipping_status']!r}")
    print(f"  halted={registry.is_halted(thread_id)}")
    window = queue_client.read_window(thread_id)
    print(f"  {len(window)} real events captured in Redis stream 'trace_events:{thread_id}'")


def run_looping(redis_client, thread_id, halt_after_seconds):
    queue_client, registry, dispatcher, handler = _build_pipeline(redis_client)
    app = build_looping_agent(registry, max_iterations=30, iteration_delay=0.2)
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]}

    def run_agent():
        app.invoke(_initial_state("ORD-99999"), config=config)

    print(f"Starting looping agent (thread_id={thread_id!r})...")
    agent_thread = threading.Thread(target=run_agent)
    agent_thread.start()

    print(f"Letting it run for {halt_after_seconds}s (simulating a detection window)...")
    time.sleep(halt_after_seconds)
    print("Requesting halt (this is what the real evaluator would trigger via POST /halt)...")
    registry.request_halt(thread_id)
    agent_thread.join(timeout=10.0)
    dispatcher.stop(timeout=2.0)

    state = app.get_state(config)
    print(f"  agent thread still alive: {agent_thread.is_alive()}")
    print(f"  frozen at: {state.next}")
    print(f"  attempts made before halt: {len(state.values['attempts'])} (of a possible 30)")
    print(f"  sample attempts: {state.values['attempts'][:3]}")
    print(f"  escalated: {state.values['escalated']}")


def main():
    parser = argparse.ArgumentParser(description="Run the retail demo against a real Redis instance.")
    parser.add_argument("--variant", choices=["happy", "looping"], required=True)
    parser.add_argument("--redis-url", default=os.environ.get("REDIS_URL", DEFAULT_REDIS_URL))
    parser.add_argument("--thread-id", default=None)
    parser.add_argument(
        "--halt-after", type=float, default=1.0,
        help="Seconds to let the looping agent run before requesting a halt (looping variant only).",
    )
    args = parser.parse_args()

    redis_client = redis.Redis.from_url(args.redis_url)
    thread_id = args.thread_id or f"retail-demo-{args.variant}"

    if args.variant == "happy":
        run_happy_path(redis_client, thread_id)
    else:
        run_looping(redis_client, thread_id, args.halt_after)


if __name__ == "__main__":
    main()

"""Shared real-pipeline wiring for the retail demo's Story 2 tests --
extracted out of any single test_*.py file (matching
integration-tests/tests/sample_agent.py's own precedent: shared fixture
code lives in its own module, not inside a test file).
"""
from control_api.guard import HaltRegistry
from interceptor.capture import CaptureDispatcher
from interceptor.graph_hooks import GraphCaptureHandler
from interceptor.queue_sink import make_sanitized_queue_sink
from queue_client.client import QueueClient

import fakeredis

AGENT_ID = "retail-demo-agent"
API_TOKEN = "test-token"

# Same reasoning/budget as integration-tests/tests/test_controlled_scenarios.py's
# END_TO_END_HALT_SLA_MS: real SLM inference (~143ms measured in
# evaluator/EVAL_REPORT.md) plus one guard-node check cycle, with margin.
END_TO_END_HALT_SLA_MS = 2000.0


def build_wired_pipeline():
    queue_client = QueueClient(redis_client=fakeredis.FakeRedis())
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
    # make_sanitized_queue_sink, not the raw unwrapped composition -- the
    # blessed production entry point (see root README / security review).
    sink = make_sanitized_queue_sink(queue_client.append_event)
    dispatcher = CaptureDispatcher(sink=sink)
    handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id=AGENT_ID)
    return queue_client, registry, dispatcher, handler


def initial_state(query):
    return {
        "query": query,
        "attempts": [],
        "resolved": False,
        "order_id": None,
        "shipping_status": None,
        "escalated": False,
    }

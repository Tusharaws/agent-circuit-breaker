"""Tests for the full consume -> pre-filter -> SLM -> verdict pipeline
(Phase 4). Written before the implementation (evaluator.pipeline does not
exist yet) -- run `pytest` to see them fail with a collection error until
src/evaluator/pipeline.py exists.

"Runs on a live queue" = a real, running QueueClient against fakeredis
(consistent with every other test in this project -- no real Redis
server exists in this environment). The SLM side uses a fake for fast,
deterministic unit tests, and the real SLMClient for one dedicated
integration test that measures actual latency.
"""
import time
from datetime import datetime, timezone

import fakeredis
import pytest

from queue_client.client import QueueClient
from schemas.trace_event import TraceEvent

from evaluator.pipeline import FILTERED_PATH_BUDGET_MS, evaluate_thread
from evaluator.slm import SLMClient
from evaluator.verdict import Verdict


def _seed_event(queue_client, thread_id, step_index, event_type, payload):
    event = TraceEvent(
        trace_id=thread_id,
        agent_id="agent-1",
        step_index=step_index,
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        payload=payload,
    )
    queue_client.append_event(thread_id, event.model_dump(mode="json"))


def _seed_normal_thread(queue_client, thread_id):
    _seed_event(queue_client, thread_id, 0, "node_enter", {"node": "planner"})
    _seed_event(queue_client, thread_id, 1, "llm_start", {"prompts": ["analyze the case"]})
    _seed_event(queue_client, thread_id, 2, "llm_end", {"response": ["plan ready"]})
    _seed_event(queue_client, thread_id, 3, "tool_start", {"tool": "search", "args": {"query": "lookup-a"}})
    _seed_event(queue_client, thread_id, 4, "tool_end", {"output": "result-a"})


def _seed_looping_thread(queue_client, thread_id):
    for i in range(5):
        _seed_event(
            queue_client, thread_id, i, "tool_start", {"tool": "search", "args": {"query": "stuck query"}}
        )


class FailIfCalledSLM:
    def generate(self, prompt, max_tokens=200):
        raise AssertionError("SLM must not be called for a filtered-out window")


class FakeSLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, prompt, max_tokens=200):
        self.calls += 1
        return self.response


@pytest.fixture
def queue_client():
    return QueueClient(redis_client=fakeredis.FakeRedis())


# ---------------------------------------------------------------------------
# Filtered path: normal window never reaches the SLM
# ---------------------------------------------------------------------------


def test_normal_window_is_filtered_and_never_calls_the_slm(queue_client):
    _seed_normal_thread(queue_client, "thread-normal")

    verdict = evaluate_thread(queue_client, FailIfCalledSLM(), "thread-normal")

    assert verdict == Verdict(is_loop=False, confidence=0.0, reason=None)


def test_filtered_path_meets_its_latency_budget(queue_client):
    _seed_normal_thread(queue_client, "thread-normal")

    start = time.perf_counter()
    evaluate_thread(queue_client, FailIfCalledSLM(), "thread-normal")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < FILTERED_PATH_BUDGET_MS


# ---------------------------------------------------------------------------
# Escalated path: looping window calls the SLM and parses its verdict
# ---------------------------------------------------------------------------


def test_looping_window_escalates_and_returns_parsed_verdict(queue_client):
    _seed_looping_thread(queue_client, "thread-loop")
    fake_slm = FakeSLM('{"is_loop": true, "confidence": 0.95, "reason": "repeated_tool_calls"}')

    verdict = evaluate_thread(queue_client, fake_slm, "thread-loop")

    assert fake_slm.calls == 1
    assert verdict == Verdict(is_loop=True, confidence=0.95, reason="repeated_tool_calls")


# ---------------------------------------------------------------------------
# Every window gets a verdict, per the AC
# ---------------------------------------------------------------------------


def test_every_thread_produces_some_verdict_including_an_empty_one(queue_client):
    _seed_normal_thread(queue_client, "thread-a")
    _seed_looping_thread(queue_client, "thread-b")
    fake_slm = FakeSLM('{"is_loop": true, "confidence": 0.9, "reason": "runaway_loop"}')

    for thread_id in ["thread-a", "thread-b", "thread-with-no-events-at-all"]:
        verdict = evaluate_thread(queue_client, fake_slm, thread_id)
        assert isinstance(verdict, Verdict)


def test_thread_with_no_events_is_filtered_without_calling_slm(queue_client):
    verdict = evaluate_thread(queue_client, FailIfCalledSLM(), "empty-thread")
    assert verdict == Verdict(is_loop=False, confidence=0.0, reason=None)


# ---------------------------------------------------------------------------
# Integration: real SLM, real latency measurement
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_end_to_end_pipeline_meets_escalated_path_budget(queue_client):
    from evaluator.pipeline import ESCALATED_PATH_BUDGET_MS

    _seed_looping_thread(queue_client, "thread-loop-real")
    real_slm = SLMClient()

    start = time.perf_counter()
    verdict = evaluate_thread(queue_client, real_slm, "thread-loop-real")
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\nreal end-to-end pipeline (escalated path): {elapsed_ms:.0f}ms, verdict={verdict}")

    assert elapsed_ms < ESCALATED_PATH_BUDGET_MS
    assert verdict.is_loop is True  # an obvious, exact-repeat loop

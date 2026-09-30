"""Retail Demo, Story 2, task 3: confirm sanitization behaves correctly
for real customer PII in the scenario -- a customer looking up their
order by email is about as realistic a PII-bearing query as this
scenario gets.

The agent's own internal resolution still uses the RAW email (it needs
the real value to actually match FAKE_ORDERS) -- only what's captured and
stored in the queue is sanitized, the same behavior already established
and tested throughout this project (sanitization happens on the capture
side, never on the agent's own execution). This test inspects the raw
stored Redis payload directly, not the sanitizer's own isolated unit
tests, so it proves the whole real pipeline (GraphCaptureHandler ->
CaptureDispatcher -> SanitizingSink -> QueueClient), not just the
scrubbing function in isolation.
"""
from happy_path_agent import build_happy_path_agent
from retail_pipeline import build_wired_pipeline, initial_state

SEEDED_EMAIL = "jane.doe@example.com"  # a real FAKE_ORDERS email -- the agent needs the raw value to resolve it


def test_a_customer_looking_up_by_email_never_leaks_it_into_the_queue():
    queue_client, registry, dispatcher, handler = build_wired_pipeline()

    thread_id = "thread-retail-pii"
    app = build_happy_path_agent(registry)
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]}

    result = app.invoke(initial_state(SEEDED_EMAIL), config=config)
    dispatcher.stop(timeout=2.0)

    # The agent itself still resolved correctly -- sanitization happens
    # on the capture side, not the agent's own execution.
    assert result["resolved"] is True
    assert result["order_id"] == "ORD-10293"

    window = queue_client.read_window(thread_id, limit=500)
    assert len(window) > 0  # a real run actually happened, not an empty one

    raw_dump = str([record.payload for record in window])
    assert SEEDED_EMAIL not in raw_dump
    assert "[REDACTED]" in raw_dump  # actively masked, not coincidentally absent

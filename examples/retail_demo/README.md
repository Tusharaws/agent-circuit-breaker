# Retail Demo: Order-Support Agent Walkthrough

A realistic retail scenario for exercising the Agent Execution Guardian
end-to-end — a customer-support agent that looks up orders and checks
shipping status, in two variants: one that resolves normally, and one that
gets stuck in a genuine semantic loop (not just repeated tool calls) the
way a real production agent plausibly could.

This doc assumes you've already followed the root `README.md`'s install
steps (steps 1–2: Redis + `pip install -e` each package). Everything here
runs against `fakeredis` by default — no real Redis needed just to run
these tests, same as every other test in this project.

## What's in this folder

- **`order_support_agent.py`** — the state schema (`OrderSupportState`), a
  small in-memory fake order database (3 seeded orders), and the 3 tools
  an order-support agent would call: `lookup_order`, `check_shipping_status`,
  `escalate_to_human`.
- **`happy_path_agent.py`** — the non-looping variant: `guard → lookup →
  shipping → END`. Resolves a valid order in 2 real tool calls.
- **`looping_agent.py`** — the looping variant: `guard → lookup → (guard |
  END)`. Given an unresolvable order reference, retries `lookup_order`
  with 8 different phrasings of the same reference (never an exact
  repeat) and never calls `escalate_to_human` — the realistic failure mode
  this whole system exists to catch.
- **`retail_pipeline.py`** — shared wiring: a real `QueueClient` +
  `HaltRegistry` + `CaptureDispatcher` + `GraphCaptureHandler`, composed
  through `make_sanitized_queue_sink` (the blessed production entry point —
  see the root README's security note on why not the unwrapped version).
- **`RETAIL_DEMO_REPORT.md`** — the actual measured results from running
  all of this for real (detect-to-halt latency, zero-false-halts check,
  PII scrubbing) — read this if you want the numbers, not just the code.

## Run the fast tests (no real model, no real Redis)

```bash
cd examples/retail_demo
pytest -m "not integration"
```

This covers: the state schema and all 3 tools in isolation
(`test_order_support_agent.py`), both agent variants' own logic
(`test_happy_path_agent.py`, `test_looping_agent.py` — including a real
mid-run halt using a real `HaltRegistry`), `GraphCaptureHandler` capturing
real events from both variants (`test_capture_wiring.py`), and PII
sanitization on a realistic email-based lookup (`test_pii_sanitization.py`).

## Run the real-SLM tests

```bash
pytest -m integration -s
```

This loads the actual local SLM (`mlx-lm`, Apple Silicon only — see the
root README's prerequisites) and runs the two scenarios that need real
detection:

- `test_detect_and_halt_latency.py` — the looping agent actually gets
  detected and halted, with the measured latency printed.
- `test_zero_false_halts.py` — the happy-path agent gets a real
  `is_loop=False` verdict, not just "nobody happened to call halt."

Both are slower (real model load + real inference) — that's why they're
separated from the fast suite via the `integration` marker, same
convention as every other package in this project.

## Try it yourself, interactively

```python
import fakeredis
from control_api.guard import HaltRegistry
from interceptor.capture import CaptureDispatcher
from interceptor.graph_hooks import GraphCaptureHandler
from interceptor.queue_sink import make_sanitized_queue_sink
from queue_client.client import QueueClient

from looping_agent import build_looping_agent  # run from examples/retail_demo/tests/

queue_client = QueueClient(redis_client=fakeredis.FakeRedis())
registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
dispatcher = CaptureDispatcher(sink=make_sanitized_queue_sink(queue_client.append_event))
handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id="retail-demo-agent")

app = build_looping_agent(registry, max_iterations=10, iteration_delay=0.2)
config = {"configurable": {"thread_id": "try-it-1"}, "callbacks": [handler]}

state = {"query": "ORD-99999", "attempts": [], "resolved": False,
         "order_id": None, "shipping_status": None, "escalated": False}
result = app.invoke(state, config=config)

print(result["attempts"])   # 10 distinct paraphrases, never converging
print(result["escalated"])  # False -- it never learned to give up
```

Run this from inside `examples/retail_demo/tests/` (or add that directory
to your `PYTHONPATH`) so the bare `from looping_agent import ...` import
resolves — the fixture modules aren't a pip-installed package, matching
`integration-tests/tests/sample_agent.py`'s own convention in this project.

## Next steps

- Want to see a real halt happen against the dashboard? Wire this agent
  against a real (not fakeredis) Redis and a running `control-api` +
  `dashboard`, following the root README's steps 3, 5, and 6 — swap in
  `looping_agent`/`happy_path_agent` for the guard-node example there.
- `RETAIL_DEMO_REPORT.md` has the full measured numbers from the last real
  run of every scenario above.

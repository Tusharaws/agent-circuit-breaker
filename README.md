# Agent Execution Guardian — Setup Guide for LangGraph Agents

A sidecar circuit breaker for LangGraph agents: it captures every node/LLM/tool
event from your running agent, sanitizes PII/PHI before anything leaves your
process, watches for semantically unproductive loops (not just repeat counts),
and halts the agent mid-execution when it finds one — before it burns further
tokens on a run that isn't converging.

This guide is the fastest path from a fresh checkout to a real LangGraph agent
wired up end-to-end. Every command and code snippet below uses APIs that exist
and are tested in this repo today — nothing here is aspirational.

## What you'll have running at the end

- **Redis** — the event queue and control-signal store.
- **control-api** — a small FastAPI service exposing `POST /halt`, running
  against a real Redis-backed checkpointer.
- **A detection loop** — polls recent activity and calls `/halt` when it finds
  a real loop. There's no packaged daemon for this yet (see
  [Detection loop](#4-run-the-detection-loop) below) — you'll copy a small,
  real reference script.
- **Your LangGraph agent** — with one node added (a guard node) and one
  callback attached, both calling into this system's own tested code.
- **Optionally, the dashboard** — a web UI for trace timelines and halt
  history.

## Prerequisites

- Python 3.11+
- A real Redis instance (not `fakeredis` — every test in this repo uses
  `fakeredis`, but real usage needs a real one)
- macOS on Apple Silicon, **if** you want the built-in local SLM
  (`evaluator`'s `SLMClient` uses `mlx-lm`, which is Apple Silicon-only). On
  other platforms, you'll need to swap in your own model client — `SLMClient`
  is constructor-injectable (`model`, `tokenizer`, `generate_fn`) specifically
  so this is possible without touching `evaluator`'s own code.

## 1. Start Redis

```bash
brew install redis      # if you don't have it
brew services start redis
redis-cli ping           # should print PONG
```

## 2. Install the packages

These aren't published to PyPI — install each one editable, in dependency
order, into one virtualenv:

```bash
python -m venv .venv && source .venv/bin/activate

pip install -e ./schemas
pip install -e ./queue-client
pip install -e ./sanitizer
pip install -e "./control-api[test]"   # test extras include fakeredis, used below for a quick local check
pip install -e ./interceptor
pip install -e ./evaluator
pip install -e ./dashboard              # optional
```

(Order matters: each package lists the ones before it as a plain dependency
name, e.g. `interceptor` depends on `schemas` and `sanitizer` — pip needs
those already installed in this environment to resolve them, since they don't
exist on the real package index.)

## 3. Start control-api

`control_api.app.create_app(registry, api_token=...)` is a FastAPI app
factory — there's no bundled script to run it yet, so here's a minimal real
one to copy as `run_control_api.py`. **Note:** the `/halt` service itself
never touches a checkpointer at all (`check_halt()` only reads a Redis key
via `HaltRegistry` and calls `interrupt()`) — only the reaper job below
needs one:

```python
import os

import redis
import uvicorn
from control_api.app import create_app
from control_api.guard import HaltRegistry
from control_api.halt_history import HaltHistoryStore

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
API_TOKEN = os.environ["CONTROL_API_TOKEN"]  # required -- create_app refuses to start without one

redis_client = redis.Redis.from_url(REDIS_URL)
registry = HaltRegistry(redis_client=redis_client)
halt_history = HaltHistoryStore(redis_client=redis_client)  # optional, powers the dashboard's halt history

app = create_app(registry, api_token=API_TOKEN, halt_history=halt_history)
uvicorn.run(app, host="0.0.0.0", port=8001)
```

```bash
export CONTROL_API_TOKEN="pick-a-real-secret"
python run_control_api.py
```

**The TTL reaper** (`control_api.reaper.reap_expired_halts`) *does* need a
checkpointer — the SAME one your agent process compiles with (step 5) — and
has no packaged scheduler either. It needs to run periodically (a cron job,
a scheduled task, or a simple `while True: ...; time.sleep(3600)` loop)
against that checkpointer and `registry`, or halted threads' state
accumulates forever. Default TTL is 7 days (`DEFAULT_TTL`):

```python
import time

from control_api.checkpointer import get_checkpointer
from control_api.guard import HaltRegistry
from control_api.reaper import reap_expired_halts

registry = HaltRegistry(redis_url="redis://localhost:6379/0")

with get_checkpointer(environment="dev") as checkpointer:  # same tier/connection your agent uses
    while True:
        reap_expired_halts(registry, checkpointer)
        time.sleep(3600)
```

## 4. Run the detection loop

There's no packaged polling daemon either — the pieces (`evaluate_thread`,
`maybe_trigger_halt`) are real and tested, but wiring them into a
continuously-running loop is your own small script today. **You also need to
tell it which `thread_id`s are currently active** — nothing in this system
maintains a registry of running threads; that's inherent to your own
application (you already know which agent runs you started).

```python
import time

import httpx
from evaluator.control_api_client import maybe_trigger_halt
from evaluator.pipeline import evaluate_thread
from evaluator.slm import SLMClient
from evaluator.window import build_window
from queue_client.client import QueueClient

REDIS_URL = "redis://localhost:6379/0"
CONTROL_API_URL = "http://localhost:8001"
API_TOKEN = "pick-a-real-secret"  # same value as CONTROL_API_TOKEN above

queue_client = QueueClient(redis_url=REDIS_URL)
slm_client = SLMClient()  # loads the local model once -- construct it outside the loop
control_api_client = httpx.Client(base_url=CONTROL_API_URL, headers={"Authorization": f"Bearer {API_TOKEN}"})


def poll_once(active_thread_ids: list[str]) -> None:
    for thread_id in active_thread_ids:
        window = build_window(queue_client, thread_id)
        verdict = evaluate_thread(queue_client, slm_client, thread_id)
        if verdict.is_loop:
            maybe_trigger_halt(window, verdict, control_api_client)


while True:
    poll_once(get_your_own_list_of_active_thread_ids())  # you supply this
    time.sleep(2)
```

## 5. Wire your LangGraph agent

Two things to add: a guard node that checks for a halt, and a callback handler
that captures events. Both call into this system's own tested code — this is
not a reimplementation, it's the same `check_halt`/`GraphCaptureHandler` used
throughout this repo's own test suite.

```python
from control_api.guard import HaltRegistry, check_halt
from control_api.checkpointer import get_checkpointer
from interceptor.capture import CaptureDispatcher
from interceptor.graph_hooks import GraphCaptureHandler
from interceptor.queue_sink import make_sanitized_queue_sink
from queue_client.client import QueueClient
from langgraph.graph import StateGraph, END

queue_client = QueueClient(redis_url="redis://localhost:6379/0")
registry = HaltRegistry(redis_url="redis://localhost:6379/0")

# make_sanitized_queue_sink, not make_queue_sink -- always sanitizes before
# anything reaches the queue (see interceptor/README.md's security-review note).
dispatcher = CaptureDispatcher(sink=make_sanitized_queue_sink(queue_client.append_event))
handler = GraphCaptureHandler(dispatcher=dispatcher, agent_id="my-agent")


def guard_node(state, config):
    thread_id = config["configurable"]["thread_id"]
    check_halt(thread_id, registry)  # raises via interrupt() if this thread is halted -- freezes here
    return {}


# Add `guard_node` as a real node customers' own graphs already have a
# precedent for in this repo (see integration-tests/tests/sample_agent.py) --
# wire it before whichever node(s) you want checked. A tight loop should call
# it once per iteration; see NODE_SPLITTING_GUIDE.md if any single node can
# run longer than ~500ms uninterrupted.
graph = StateGraph(...)
graph.add_node("guard", guard_node)
# ... your own nodes ...

# control-api's process needs the SAME checkpointer instance/connection your
# agent process compiles with -- interrupt() only works with a persisted
# checkpointer attached (InMemorySaver won't survive across processes).
with get_checkpointer(environment="dev") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "run-1"}, "callbacks": [handler]}
    app.invoke({...}, config=config)
```

## 6. Optional: the dashboard

```bash
cd dashboard
python -m dashboard   # serves http://localhost:8000/, needs REDIS_URL if not localhost:6379
```

Trace timelines (`/api/traces/{thread_id}`) and halt history
(`/api/halts?trace_id=...&since=...&until=...`), with drill-down from a halt
to the exact events that triggered it.

## Production notes

- **Checkpointer tier**: `InMemorySaver` (test) only works within one process
  — your agent process and `control-api`'s process must share the same
  persisted checkpointer (`SqliteSaver` for a single-box dev setup,
  `PostgresSaver` for real production) for halting across processes to work
  at all. See `control-api/README.md`.
- **Halt latency is bounded by node granularity**: `check_halt()` only runs
  where you put it. A node wrapping a long, uninterruptible call delays a
  halt by that call's own duration. See `control-api/NODE_SPLITTING_GUIDE.md`
  for a concrete threshold (500ms) and a worked before/after example.
- **Multi-tenant**: `schemas.TenantConfig` ties together isolated queue
  prefixes and per-tenant sanitization policies if you're running this for
  more than one customer — see `schemas/README.md`. `control-api`'s auth and
  the dashboard are still single-tenant; that's a documented, not-yet-built
  gap, not an oversight.
- **Security**: `sanitizer/SECURITY_REVIEW.md` documents what's been checked
  (and fixed) in the sanitization layer specifically.
- **create_agent() / AutoGen**: if your agents use LangChain's `create_agent()`
  instead of a hand-built LangGraph graph, or AutoGen instead of LangGraph
  entirely, see `control-api/README.md`'s "Integration pattern: create_agent()"
  section and `interceptor/README.md`'s "AutoGen adapter" section respectively
  — both are real, tested, different integration points from the guard-node
  pattern above.

## Verifying your setup

`integration-tests/` wires all of this together against real (not mocked)
LangGraph agents, including a full seeded-loop-to-halt cycle — run
`pytest -m "not integration"` there for the fast checks, or the full suite
(needs the real local SLM) to see an actual detect-and-halt cycle end to end.
That's the same proof this whole system's own test suite relies on; it's a
good sanity check that your local install is wired correctly before pointing
it at a real agent.

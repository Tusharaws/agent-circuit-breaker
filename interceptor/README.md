# interceptor

Per-node LangGraph instrumentation for the Agent Execution Guardian (Phase 1).

Captures a full state snapshot at each node transition (not just prompt/response
pairs) and reuses LangChain-compatible callback events (`on_llm_start/end`,
`on_tool_start/end`) that LangGraph nodes still emit under the hood.

## Capture dispatch

`CaptureDispatcher` (`src/interceptor/capture.py`) is the fire-and-forget
execution model capture runs on: a single background daemon thread draining
a bounded `queue.Queue`. `capture(event)` is non-blocking for both sync and
async callers — a plain in-memory enqueue has no real I/O in it, so one
thread-safe queue serves async LangGraph nodes too, without a second
asyncio-native dispatch path.

Guarantees:
- `capture()` never raises, and a `sink` that raises never propagates or
  crashes the parent run — errors are logged instead (`logging.exception`).
- One bad event can't kill the worker thread; subsequent events still reach
  `sink` normally.
- Buffer-full and item-too-old (`max_age`, default 2 days) both drop the
  event and log a warning rather than blocking or raising.
- Construction fails fast on nonsensical config: a non-callable `sink`, or a
  `max_queue_size`/`max_age` that's zero or negative (`max_queue_size=0`
  would silently mean *unbounded* in stdlib `queue.Queue` semantics, so
  it's rejected rather than left as a footgun).
- `CaptureDispatcher` supports `stop()` and the context manager protocol —
  without it, the background thread holds a permanent reference to the
  dispatcher and it can never be garbage collected. `stop()` isn't a
  guaranteed flush (items still queued may not be processed before the
  thread exits); it exists to release resources cleanly, not to promise
  delivery.

`sink` is an injected callable (e.g. `queue_client.QueueClient.append_event`),
not a direct dependency — keeps this package testable in isolation.

Deferred (not yet built here): state-snapshot capture beyond what's in
`payload` today, and the exact `max_queue_size` default (currently `1000`,
easy to change).

## Production node-hook wiring: GraphCaptureHandler

`GraphCaptureHandler` (`src/interceptor/graph_hooks.py`) is the real
production capture — maps LangGraph/LangChain callback events into
`schemas.TraceEvent` instances dispatched through a `CaptureDispatcher`.
`langgraph`/`langchain-core`/`schemas` are runtime dependencies of
`interceptor` because of this module (`schemas` must be installed
editable from `../schemas` in local dev — it isn't published anywhere yet).

Usage: `GraphCaptureHandler(dispatcher, agent_id="...")`, passed via
`config={"callbacks": [handler], "configurable": {"thread_id": "..."}}` on
`app.invoke(...)`.

Design notes, confirmed empirically before writing (not assumed):
- `agent_id` is a **required constructor parameter**, not read from
  LangGraph's runtime config — LangGraph only auto-propagates its own
  reserved configurable keys (like `thread_id`) into callback `metadata`,
  not arbitrary customer-supplied ones.
- `thread_id` (LangGraph's own reserved key, found in `metadata`) becomes
  `TraceEvent.trace_id` directly — reusing LangGraph's identity rather than
  inventing a second one (the `trace_id` vs `thread_id` reconciliation is
  tracked as its own backlog task; this is the provisional answer used here).
- `step_index` is sourced from `metadata['langgraph_step']` — LangGraph's
  own per-thread step counter — rather than a hand-rolled counter, which
  would need its own thread-safety handling and could drift under
  concurrent runs.
- `_end`/`_error` callbacks (`on_chain_end`, `on_llm_end`, `on_tool_end`,
  and their `_error` counterparts) do not receive `metadata` at all — only
  `_start` callbacks do. Needed context (thread_id, step_index, node name,
  start time) is cached at the matching `_start` callback, keyed by
  `run_id`, and looked up (not re-derived) at `_end`/`_error` time.
- Every callback body is wrapped so nothing here ever raises into the
  customer's graph execution — same guarantee `CaptureDispatcher` provides,
  extended to event construction itself (e.g. a missing `thread_id` logs
  and drops the event rather than crashing the run).
- Errors (`on_chain_error`/`on_llm_error`/`on_tool_error`) are folded into
  the matching exit-style event (`node_exit`/`llm_end`/`tool_end`) via an
  `{"error": true, "error_message": ...}` payload, not new `event_type`
  values — keeps `schemas/config/event_types.md` unchanged.
- `llm_end`'s `payload["response"]` is a **list** of every generation's
  text, not just the first — an LLM call can return multiple candidate
  completions (n>1); capturing only the first would silently drop the
  rest. Scoped to single-prompt calls (what a LangGraph node does).
- `tool_start`'s `payload["args"]` captures the structured `inputs` dict
  from the callback's kwargs when present (confirmed empirically — richer
  than the flattened `input_str` LangChain also provides, which is kept
  alongside it for backward compatibility/debugging).

`tests/test_trace_event_serialization.py` proves this explicitly and
portably: every real event captured from a sample agent run validates with
zero errors against `TraceEvent.model_json_schema()` (derived from the
pydantic model, not a separately hand-maintained `.schema.json` — avoids a
second source of truth that could drift). Covers events with
token_usage/latency_ms present and absent, and error-payload events.

## Local buffering/batching layer: BatchingSink

`BatchingSink` (`src/interceptor/batching.py`) is a composable `sink` —
same `Callable[[Any], None]` contract `CaptureDispatcher` already expects —
that accumulates events locally and only calls the real `flush` callable
once per `batch_size` items or once per `flush_interval` seconds
(whichever comes first), trading one network call per event for one per
batch. Usage: `CaptureDispatcher(sink=BatchingSink(flush=queue_client_instance.append_events_batch))`
(once such a batch-write method exists on the sink side — see note below).

Design notes:
- Implemented as a **separate class used as a `sink`**, not a change to
  `CaptureDispatcher` itself — `CaptureDispatcher`'s per-item sink contract
  already had passing tests; changing it to a batch-callback contract would
  have been a breaking change with real blast radius for no benefit over an
  additive wrapper.
- Same guarantees as `CaptureDispatcher`: a raising `flush` never
  propagates or crashes the caller (logged instead); construction fails
  fast on nonsensical config (`batch_size`/`flush_interval` ≤ 0, non-callable
  `flush`); supports `stop()`/context manager, and `stop()` performs a
  final best-effort flush of whatever remains buffered.
- **Scope note:** this only proves the buffering/batching behavior itself
  (tested against a fake `flush` that counts invocations). It does not by
  itself reduce real network round-trips unless the `flush` callable it
  wraps is *itself* a genuinely batched/pipelined write (e.g. a Redis
  pipeline in `queue_client`) — looping over the batch and calling a
  per-event write N times would defeat the point. That batch-write
  capability on the `queue_client` side doesn't exist yet; it's a natural
  next step, tracked separately rather than folded into this task.

## Capture-completeness test harness

`tests/test_capture_completeness.py` runs a real, deterministic 2-node
LangGraph agent (fake LLM + fake tool, no real model calls) through
`CaptureDispatcher` and asserts the captured trace is the exact expected
event sequence — in order, no gaps, no duplicates. Its `CaptureCallbackHandler`
was test-harness-only glue (bare string tags, no real `TraceEvent`); the
real production wiring is `GraphCaptureHandler` above. Also see
`tests/test_graph_hooks.py`, which tests `GraphCaptureHandler` itself at
both the unit level (direct callback invocation, e.g. for token_usage
extraction and error paths) and the integration level (a real graph run
through a real `CaptureDispatcher`).

## Install

```bash
pip install -e ".[test]"
```

## queue_sink: adapting to QueueClient (Phase 6 wiring fix)

`make_queue_sink(append_event) -> Callable[[TraceEvent], None]`
(`src/interceptor/queue_sink.py`) adapts `QueueClient.append_event(thread_id,
payload)` (two arguments) into the single-argument sink shape
`CaptureDispatcher`/`SanitizingSink` expect.

**A real integration bug, found while wiring all 5 components together
for the first time (Phase 6), not designed upfront:** `SanitizingSink`
calls its inner sink with one argument (the event); `QueueClient.append_event`
needs `thread_id` and `payload` separately for stream routing. Each side
was individually correct and fully tested in isolation — nothing
exercised the seam between them until a real end-to-end run hit a
`TypeError` immediately. `thread_id` is extracted from the event's own
`trace_id` field (the two are the same value by design).

Production wiring: `CaptureDispatcher(sink=SanitizingSink(inner=make_queue_sink(queue_client.append_event)))`.

**Blessed single-call variant (Phase 7 security review):** a review of the
sanitization layer found nothing structurally prevented wiring
`CaptureDispatcher(sink=make_queue_sink(queue_client.append_event))`
directly, skipping `SanitizingSink` entirely -- convention/documentation
only (see `sanitizer/SECURITY_REVIEW.md`). `make_sanitized_queue_sink(append_event,
policy=None, ner_scrubber=None)` composes `SanitizingSink` + `make_queue_sink`
in one call, so there's no 2-step assembly where the wrapper could be
forgotten: `CaptureDispatcher(sink=make_sanitized_queue_sink(queue_client.append_event))`.
This is now the recommended production entry point; `sanitizer` is a
production dependency of `interceptor` as a result (previously test-only
context via other packages, not a direct dependency here).

**Outage-resilient variant (Phase 6 chaos testing):** the recipe above
leaves a real Redis outage to be caught only by `CaptureDispatcher`'s own
blunt per-item try/except (silent drop + exception log) -- it never
retries or buffers, even though `queue_client.producer.ResilientProducer`
exists in this codebase for exactly that. Wrap `append_event` with it for
real deployments: `CaptureDispatcher(sink=make_sanitized_queue_sink(resilient_producer.append_event))`
where `resilient_producer = ResilientProducer(append_event=queue_client.append_event)`.
Proven end-to-end (real outage simulation, real recovery/flush, no crash)
in `integration-tests/tests/test_chaos.py` (predates `make_sanitized_queue_sink`;
composes the same way).

## End-to-end tracing (Phase 6)

`GraphCaptureHandler._emit()` logs one structured JSON line per captured
event on a dedicated `"interceptor.trace"` logger (DEBUG -- fires on every
node/llm/tool transition, so it's opt-in verbosity, not default noise):
`{"trace_id": ..., "service": "interceptor", "stage": "captured", "event_type": ...}`.
Same convention (`trace_id`/`service`/`stage` at minimum) is used
independently by sanitizer, queue-client, evaluator, and control-api's own
trace loggers -- not a shared library (sanitizer and queue-client
deliberately don't depend on `schemas` in production), just a consistent
format each package implements itself, so a single `trace_id` is
grep/jq-able across all 5 services' logs. See
`integration-tests/tests/test_tracing.py` for the real cross-package proof.

## Test

```bash
pytest
```

## Status

Capture dispatch (fire-and-forget execution model), production LangGraph
node-hook wiring (`GraphCaptureHandler`), and the `QueueClient` adapter
implemented and tested. Remaining Phase 1 work: richer state-snapshot
capture in `payload`, and the local buffering/batching layer (separate
from `CaptureDispatcher`'s in-memory queue, per the batching backlog task).

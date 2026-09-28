# control-api

Circuit breaker / control API (Phase 5).

`POST /halt` looks up a thread's checkpointer state and calls `interrupt()`
(or flips a flag a guard node checks), freezing the graph at the next
checkpointed boundary. Halts are permanent — `Command(resume=...)` is
deliberately never called on a halted thread.

Also owns the TTL-based reaper job that cleans up halted threads, since
LangGraph does not auto-expire frozen threads.

## Checkpointer selection

`get_checkpointer(environment=None, sqlite_path="checkpoints.db", postgres_conn_string=None)`
(`src/control_api/checkpointer.py`) — a context manager picking
`InMemorySaver` (test), `SqliteSaver` (dev), or `PostgresSaver` (prod) via
`environment` or the `CONTROL_API_ENVIRONMENT` env var, no code change to
switch. Required because `interrupt()` raises without a checkpointer
attached to the compiled graph.

Verified end-to-end against a real `SqliteSaver` before writing the
factory (a tiny graph, real state persistence, real `get_state()`
readback) — not assumed from documentation. This environment has no real
Postgres server; the `prod` tier's tests verify this module's own routing
logic (it calls `PostgresSaver.from_conn_string` with the right argument),
not genuine Postgres connectivity — same honest caveat as queue-client's
load test having no real Redis.

**Resolved:** the "own DB/schema vs. shared with queue's metadata store"
open question from `ARCHITECTURE_NOTES.md` turned out to be moot —
`queue_client`'s metadata lives in Redis, this prod tier lives in
Postgres. Different database technologies; there was never a scenario
where a shared schema was possible.

## Runtime-side halt listener

`HaltRegistry` + `check_halt(thread_id, registry)` (`src/control_api/guard.py`)
— the mechanism a guard node inside a customer's graph calls to honor a
pending halt.

**A real design fork resolved by testing, not assumed:** the obvious first
idea — Control API calls `interrupt()` directly on a running thread, or a
guard node checks LangGraph's own graph state for a halt flag — **both
fail**. Verified empirically: calling `app.update_state(config, {...})`
from outside while a graph is mid-`invoke()` on that thread never
propagates to the running invocation; LangGraph doesn't re-fetch
checkpointed state between node executions within one `invoke()` call.
`interrupt()` itself can only be called from *inside* a node's own
execution — there is no way to trigger it on a thread from outside it.

The only mechanism that actually works: a guard node performs a **live,
independent lookup** (a plain Redis key, `HaltRegistry` — deliberately
*not* part of LangGraph's checkpointed state) each time it runs, and calls
`interrupt()` itself if halted. This is a real integration requirement —
customers must insert a guard node (or wrap their node functions) that
calls `check_halt()`; this can't be done purely via callbacks (`interceptor`'s
`GraphCaptureHandler` is a callback/observer and structurally cannot call
`interrupt()`).

`HaltRegistry` is a third, distinct kind of state — not the checkpointer
(LangGraph's own state) and not `queue_client`'s Streams (trace events) —
control signals get their own small dedicated Redis keyspace.

**Keeping halt latency tight in your own graph:** since a guard node can
only check for a halt when it runs, a node that wraps a long-running call
with no checkpoint until it returns can delay a halt by that call's own
duration. See [NODE_SPLITTING_GUIDE.md](NODE_SPLITTING_GUIDE.md) for a
concrete threshold (500ms) and a worked before/after example.

### Integration pattern: `create_agent()` (Phase 7)

The Phase 7 backlog task "Add LangChain AgentExecutor adapter" targeted
`langchain.agents.AgentExecutor`, which **does not exist in LangChain
1.x** — confirmed empirically, not assumed (this project already pins
`langchain-core>=1,<2`/`langgraph>=1,<2` everywhere; legacy
`langchain==0.x`, the last version with `AgentExecutor`, is incompatible
with that pin). Its replacement, `langchain.agents.create_agent()`,
returns a real `langgraph.graph.state.CompiledStateGraph` — LangGraph
under the hood, not a separate runtime.

Two consequences, both verified:
- **Capture already works, unmodified.** `interceptor.GraphCaptureHandler`
  (built for LangGraph in Phase 1) captures a `create_agent()` agent's
  full event trace correctly with zero new code — proven in
  `interceptor/tests/test_create_agent_capture.py`.
- **Halting needs a different integration point, but the same primitive.**
  `create_agent()`'s graph shape (`model`/`tools` nodes) is fixed —
  customers can't insert their own guard node into it the way a
  hand-built LangGraph graph allows. Its `middleware` extension point is
  the equivalent hook instead: `AgentMiddleware.before_model` runs once
  per agent-loop iteration, the same cadence a guard node would. Calling
  `check_halt()` there works identically to a guard node — no new
  control-api code needed:

  ```python
  from langchain.agents.middleware import AgentMiddleware
  from langgraph.config import get_config
  from control_api.guard import check_halt

  class HaltCheckMiddleware(AgentMiddleware):
      def __init__(self, registry):
          super().__init__()
          self._registry = registry

      def before_model(self, state, runtime):
          thread_id = get_config()["configurable"]["thread_id"]
          check_halt(thread_id, self._registry)  # raises via interrupt() if halted
          return None

  agent = create_agent(model=..., tools=[...], middleware=[HaltCheckMiddleware(registry)],
                        checkpointer=InMemorySaver())
  ```

  This is documented as a customer integration pattern (like the
  guard-node pattern `sample_agent.py` demonstrates for plain LangGraph),
  not shipped as a new `control-api` class — that would require adding
  the full `langchain` package as a new **production** dependency here
  just for one base class, for no behavioral gain over 10 lines of
  customer glue code. Proven end-to-end (real halt mid-run, real
  `__interrupt__` in the result, and a normal run that's never halted) in
  `integration-tests/tests/test_create_agent_adapter.py`.

## TTL reaper

`reap_expired_halts(registry, checkpointer, ttl=timedelta(days=7))`
(`src/control_api/reaper.py`) deletes checkpointer state and clears the
registry entry for every halted thread older than `ttl` — LangGraph does
not auto-expire frozen threads, so without this they'd accumulate
forever. Verified empirically before designing: both `InMemorySaver` and
`SqliteSaver` expose `delete_thread()`, and it genuinely clears
checkpointed state (confirmed with a real graph).

Default TTL (7 days) is deliberately different from `interceptor`'s
2-day local-buffer eviction — that protects in-process memory during a
Redis outage (a different concern); this is retention time for operators
to review a halt before cleanup.

`HaltRegistry.request_halt()` now stores *when* the halt happened, not
just a flag (`halted_at()`, `list_halted_thread_ids()`, `clear_halt()`
added) — backward compatible, existing tests only checked truthiness.

**Scope note on "never resume":** enforced by simple omission — nothing
in this package calls `Command(resume=...)` anywhere. A genuine runtime
block would require intercepting the *customer's own* `invoke()` calls,
which control-api doesn't own and can't reach into. Documented honestly
rather than implying stronger enforcement than exists.

## POST /halt endpoint

`create_app(registry: HaltRegistry) -> FastAPI` (`src/control_api/app.py`)
— a factory function, not a global app singleton, so tests inject a
fakeredis-backed `HaltRegistry` directly. `POST /halt` accepts a
`schemas.HaltSignal` payload and returns **202** (accepted for
processing — the halt takes effect at the guard node's next check, not
immediately) and marks the thread halted via `registry.request_halt()`.

**FastAPI over Flask:** `HaltSignal` is already a pydantic model —
declaring `signal: HaltSignal` as the handler's parameter type is enough
for FastAPI to validate the request body automatically, returning **422**
with per-field detail on anything invalid, before the handler body even
runs. Verified empirically before writing any handler code. Flask would
need manual validation wiring for the same guarantee.

## Auth on the /halt endpoint

`create_app(registry, api_token=None)` now requires `api_token` (or the
`CONTROL_API_TOKEN` env var) — fails fast with a clear `ValueError` at
creation time rather than starting an unauthenticated endpoint. Bearer
token, checked via FastAPI's `HTTPBearer`. Verified empirically before
designing: both a *missing* Authorization header and a *wrong* token
return **401** (not 403 for the missing case, contrary to my initial
expectation) — one consistent status code either way.

This is a real, necessary breaking change to `create_app()`'s signature —
every existing caller now needs an `Authorization` header, including
`evaluator`'s test suite (which drives this app directly via
`TestClient`) — not a bug, just the natural consequence of retrofitting
required auth onto an existing endpoint.

## Halt-decision logging

Every successful halt logs one JSON-serialized line (`trace_id`,
`reason`, `confidence`, `triggering_window`, `timestamp`) via a dedicated
`control_api.halt_audit` logger — separate from the module's default
logger, so operators can route/retain halt-audit logs distinctly from
general app logs. "Queryable/searchable" only requires the content to be
structured (parseable JSON) — no specific log-aggregation backend or
library needed; grep/jq on a JSON-lines file or any real aggregator can
query it. A rejected request (422/401) produces no audit log entry, since
no halt actually happened.

## Manual override trigger

`trigger_manual_halt(trace_id, control_api_url, api_token, ...)` +
a `python -m control_api.cli` entry point (`src/control_api/cli.py`) — a
human triggers a halt via the exact same `/halt` endpoint an automated
detection uses. No new server-side logic: the endpoint never
distinguished who/what sent a request, so "halts identically" holds by
construction — proven with tests confirming a manual trigger produces the
same registry state and the same halt-audit log shape as an automated one.

New `manual_override` reason (`schemas/config/halt_reasons.md`, config-only
change) — a human's manual decision doesn't have to masquerade as one of
the three automated-detection reasons in the audit log. Defaults:
`confidence=1.0` (not a probabilistic judgment), `triggering_window=["manual-override"]`
sentinel (`HaltSignal` requires at least one entry; a human isn't always
pointing at specific event ids) — both overridable when real context exists.

A real dashboard button is separate, not-yet-built Phase 7 work (`Build
dashboard/UI`) — it would call this exact same endpoint too.

## End-to-end tracing (Phase 6)

`check_halt` logs one structured JSON line per call on a dedicated
`"control_api.trace"` logger (DEBUG -- it runs on every guard-node
execution, a high-frequency check, not a low-frequency decision):
`{"trace_id": ..., "service": "control_api", "stage": "halt_checked",
"halted": bool}`, logged before `interrupt()` is called so it fires even
on the halting call. This is separate from the existing
`"control_api.halt_audit"` logger (`_log_halt_decision` in `app.py`),
which already carries full context for the actual halt decision itself and
keeps its own exact log shape (locked by `test_cli.py`) -- not widened
with `service`/`stage` fields to avoid disturbing that contract. Same
`trace_id`/`service`/`stage` convention as interceptor/sanitizer/
queue-client/evaluator's own trace loggers otherwise -- see
`interceptor/README.md` for why this is a shared convention, not a shared
library.

## Install

```bash
pip install -e ".[test]"
```

## Test

```bash
pytest
```

## Status

Phase 5 complete: checkpointer selection, the runtime-side halt listener,
the TTL reaper, the `POST /halt` endpoint, auth, halt-decision logging,
and the manual override trigger all implemented and tested. Phase 7
additions: the `create_agent()` integration pattern, `HaltHistoryStore`
(a persisted, queryable halt history for the dashboard), and
[NODE_SPLITTING_GUIDE.md](NODE_SPLITTING_GUIDE.md).

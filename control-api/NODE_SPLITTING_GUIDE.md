# Node-Splitting Guidance: Keeping Your Halt SLA Tight

Customer-facing guidance for the limitation documented in
`ARCHITECTURE_NOTES.md`: **latency-to-halt is bounded by node granularity,
not instant.** `check_halt()` (see the "Runtime-side halt listener"
section above) only ever runs where *your* graph puts a guard node or
guard check — it cannot interrupt a node that's already mid-execution. If
one of your nodes wraps a long-running tool call with no checkpoint until
it returns, a halt request arriving during that call cannot take effect
until the call finishes, no matter how fast detection was.

This is not a bug to report — it's a physical property of how
LangGraph's `interrupt()` works (see "Runtime-side halt listener" above
for why the alternatives, like calling `interrupt()` from outside a node
or polling graph state, don't work at all). This guide tells you how to
keep it from becoming a problem in your own graph.

## The threshold: 500ms per node

**If a single node can run longer than 500ms before its next checkpoint,
split it.**

This number isn't arbitrary — it's the same figure as `evaluator`'s own
`ESCALATED_PATH_BUDGET_MS` (the system's own accepted worst-case detection
latency once a window escalates to the SLM). The reasoning: a single node
in *your* graph shouldn't be allowed to contribute more worst-case added
halt latency than our own detection pipeline already budgets for. If it
does, your node — not our detection speed — becomes the dominant
bottleneck in how long a runaway loop keeps running after it's been
caught, which defeats the point of having a fast detection pipeline at
all.

This is a different number from two others you'll see elsewhere in this
project, on purpose:
- `HALT_SLA_MS = 200ms` (`control_api/tests/test_guard.py`) measures only
  the guard-node mechanism itself — state-freeze time *after* a halt is
  already requested and the preceding node is already instant. It says
  nothing about how long your own node runs.
- `END_TO_END_HALT_SLA_MS = 2000ms` (`integration-tests/tests/test_controlled_scenarios.py`)
  is the full real-SLM detection-to-halt budget for the whole system,
  measured at 680ms. The 500ms-per-node threshold here is what keeps a
  single node from single-handedly blowing past that budget.

**Measured, not assumed** (`integration-tests/tests/test_single_long_node_halt_sla.py`):
a node running 1.0s of continuous internal work, halted mid-call, froze at
**~1018ms** — the node's own duration, almost exactly. The identical total
work (1.0s) split into 20 nodes of 50ms each, with a guard check between
each, froze at **~62ms** instead — bounded by one small node's duration,
not the full 1.0s.

## Worked example: before / after

Say one of your nodes processes a batch of items, each requiring a slow
call (an API request, a DB write, an LLM call) — a very common shape for
"a long-running node."

### Before: one monolithic node, no checkpoint until it's all done

```python
def process_batch_node(state):
    # 100 items x ~50ms each = ~5s of uninterruptible work. A halt
    # requested at item #1 doesn't take effect until item #100 finishes.
    results = []
    for item in state["items"]:
        results.append(slow_api_call(item))
    return {"results": results}

graph.add_node("guard", guard_node)
graph.add_node("process_batch", process_batch_node)
graph.set_entry_point("guard")
graph.add_edge("guard", "process_batch")
graph.add_edge("process_batch", END)
```

### After: chunk it, with a guard check between chunks

```python
CHUNK_SIZE = 10  # ~10 x 50ms = ~500ms per node call -- at the threshold

def process_chunk_node(state):
    start = state["processed"]
    chunk = state["items"][start : start + CHUNK_SIZE]
    new_results = [slow_api_call(item) for item in chunk]
    return {
        "results": state["results"] + new_results,
        "processed": start + len(chunk),
    }

def should_continue(state):
    return END if state["processed"] >= len(state["items"]) else "guard"

graph.add_node("guard", guard_node)  # calls check_halt(thread_id, registry)
graph.add_node("process_chunk", process_chunk_node)
graph.set_entry_point("guard")
graph.add_edge("guard", "process_chunk")
graph.add_conditional_edges("process_chunk", should_continue, {"guard": "guard", END: END})
```

Same total work, same result — but now a halt requested during item #1's
chunk takes effect after at most ~500ms (one chunk), not ~5s (the whole
batch). This is exactly the pattern proven in
`integration-tests/tests/test_single_long_node_halt_sla.py`'s
`_build_fine_grained_graph` fixture, just shaped around a realistic
per-item tool call instead of a bare `time.sleep()`.

Pick `CHUNK_SIZE` so `CHUNK_SIZE * (time per item)` stays under 500ms.
Don't have a per-item shape at all (e.g. one single call that processes
the whole batch server-side)? See the limitation below.

## When splitting doesn't apply

Node splitting only works when the long operation can be **decomposed
into checkable increments** — a loop over items, a paginated API, a
chunked/streamed response you consume incrementally. A genuinely atomic
call — one blocking request with no partial results, no pagination, no
streaming interface — has a hard floor equal to its own duration. Wrapping
it in more nodes doesn't help, because there's no natural point *inside*
the call to check anything.

If you're stuck with a truly atomic long call: this is an accepted,
documented trade-off, not something this project currently solves for
you. Options outside this project's current scope: add your own
timeout/cancellation at the call site (most HTTP clients and SDKs support
this independently of anything here), or, if the call has a bulk/batch
variant with a smaller unit of work, use that instead so it becomes
decomposable.

## Cross-references

- **Phase 5 runtime-listener design** (why a guard node + `check_halt()`
  is the only mechanism that actually works, and why `interrupt()` can't
  be called from outside a node): see "Runtime-side halt listener" above.
- **Phase 6 measured SLA gap** (the empirical basis for everything in this
  guide): `ARCHITECTURE_NOTES.md`'s "Phase 6 — Integration Testing"
  section; the actual test is
  `integration-tests/tests/test_single_long_node_halt_sla.py`.

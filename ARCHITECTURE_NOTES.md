# Agent Execution Guardian — Architecture Context Notes
> Keep this file in the repo root (or `/docs`). It's meant to be read by Claude Code / your IDE assistant
> for context while implementing each phase — not a task tracker (that lives on monday.com).

---

## Product Overview

**What it is:** A sidecar that watches an agent's execution loop in real time, pulls metrics from each step, and uses a semantic understanding of the execution pattern — not just raw counts — to detect when the agent is stuck in an unproductive loop. When it detects one, it stops the loop **mid-execution**, before the agent burns further tokens/compute on a run that isn't converging.

**The problem it solves:** Agent frameworks let an agent keep calling tools or generating output as long as its own reasoning tells it to keep going. That reasoning can get stuck — the same tool called repeatedly with near-identical arguments, or the same code/text regenerated with only cosmetic differences, without the task actually progressing. Every one of those extra steps is a real LLM call, so an unnoticed loop is a direct, uncapped cost — this is a **cost-control** product as much as a safety one.

**What makes the detection "semantic" rather than just a counter:**
- A naive guard could just say "halt after N identical tool calls." That's brittle — it misses loops where the *arguments* or *generated text* drift slightly each time without any real progress (e.g. regenerating a function with a different variable name each pass, or rephrasing the same failed approach).
- This product instead looks at a **window of recent steps** and asks a small language model (the Evaluator, Phase 4) to judge whether the *meaning* of what's happening is repeating — same underlying intent/approach recurring — rather than requiring byte-for-byte repetition.
- Concretely this means: comparing semantic similarity of tool-call arguments and generated content across a window, not just exact-match counting, so it catches "the agent is retrying the same broken approach in slightly different words" as well as "the agent is calling the same tool with the same args."

**Failure modes this specifically targets:**
1. **Repeated tool calls** — same tool invoked over and over (same or trivially varied arguments) without the task state advancing.
2. **Repeated generation of code or text** — the agent regenerates a function, document, or answer multiple times with only superficial differences and no real convergence toward a finished result.
3. (General case both of the above fall under) **Runaway loops** — any pattern where the agent's own reasoning fails to recognize it isn't making progress and keeps spending compute anyway.

**Why "stop in between," not just log and alert:** Pure observability (log it, alert a human) still lets the loop burn cost until someone notices. The value proposition here is specifically **automatic, in-flight intervention** — the circuit breaker (Phase 5) has to be able to reach into a *running* agent and stop it before more calls happen, which is exactly why the halt mechanism (see below) has to hook into the framework's actual execution primitive rather than being an out-of-band monitor.

**Non-goals for v1** (worth being explicit about, since it's easy to scope-creep a "watch the agent" product):
- Not a general observability/tracing platform (though it produces trace data as a byproduct) — the differentiator is the automatic halt, not the dashboard.
- Not correctness checking — it doesn't judge whether the agent's output is *right*, only whether the agent is *stuck repeating itself*.
- Not a cost-optimization/token-budgeting product in general — it targets the specific waste pattern of loops, not general prompt/token efficiency.

---

## Core Decision: LangGraph for v1

**Framework:** LangGraph (not vanilla LangChain `AgentExecutor`).

**Why:**
- `interrupt()` + a persistent checkpointer is a native, built-in pause/resume primitive — this becomes our circuit breaker's halt mechanism instead of a custom webhook/polling listener.
- Node-level granularity gives richer state capture per step than chain-level callbacks alone.
- LangGraph nodes still emit LangChain-compatible callback events under the hood, so `BaseCallbackHandler`-style capture still works — we're not throwing away LangChain knowledge, just building on the graph layer above it.
- AutoGen and AWS Strands adapters are deferred to Phase 7 (post-MVP framework expansion).

**Explicitly rejected for v1 (revisit later):**
- Plain LangChain `AgentExecutor` — considered too naive/first-generation for how agents are actually built today.
- AWS Strands — has a genuinely excellent native `HookProvider` lifecycle-event system (arguably closer to a "guardrail hook" out of the box), but smaller ecosystem and AWS/Bedrock-centric. Worth a second look for Phase 7 or as an alternative adapter, since its hook model maps almost 1:1 to what our Evaluator + Circuit Breaker do.

---

## Halt Mechanism: `interrupt()` + Checkpointer

This is the crux of the whole product and the one piece of "how LangGraph works" every phase needs to agree on.

### How it actually works
- `interrupt()` is called **inside a node**. It pauses graph execution at that point and serializes state via the checkpointer.
- A checkpointer is a **hard prerequisite** — `interrupt()` raises an error if the graph wasn't compiled with one attached.
- Every run needs a `thread_id` in its config so the runtime can locate the frozen state later.
- Resumption (normal LangGraph usage) happens via `Command(resume=<value>)`, which injects a value and continues execution from the interrupted node.

### How we're using it (circuit breaker, not HITL)
LangGraph's `interrupt()` is designed for human-in-the-loop pause-then-continue. We're repurposing it as an **abort**:

- On loop detection, the Control API triggers `interrupt()` on the running thread (or a wrapper/guard node checks a shared halt flag each step and calls `interrupt()` itself).
- We **deliberately never call `Command(resume=...)`** on a halted thread — the freeze is permanent, not a pause.
- **Important gotcha:** LangGraph does **not** auto-expire frozen threads. We need our own cleanup/expiry job (TTL-based reaper) or halted threads pile up in the checkpointer store forever.

### Checkpointer choice by environment
| Environment | Checkpointer | Notes |
|---|---|---|
| Unit tests | `MemorySaver` | In-process dict, lost on exit — never use beyond tests |
| Local dev | `SqliteSaver` (`langgraph-checkpoint-sqlite`) | Persists to a local file, negligible latency |
| Production | `PostgresSaver` | Default choice for durability + auditability; Redis checkpointer is an option if you need lower latency at high throughput, but Postgres is the safer default for an audit trail |

### API note
Current API (LangGraph 0.2+) calls `interrupt()` directly inside a node. The older `NodeInterrupt` exception pattern is retired — don't reach for it in new code.

---

## Phase-by-Phase Implementation Notes

### Phase 1 — Interceptor
- Instrument **per node**, not via a single bare `BaseCallbackHandler` bolted onto an executor.
- Capture a **full state snapshot per node transition**, not just prompt/response pairs — LangGraph's state object gives you this for free; use it.
- Node transitions still fire LangChain-compatible callback events internally, so existing `on_llm_start/end`, `on_tool_start/end` capture code is largely reusable inside each node.

### Phase 3 — Queue
- No LangGraph-specific change, but note: the checkpointer itself is a second source of state persistence separate from our event queue. Don't duplicate — the queue carries *trace events for evaluation*, the checkpointer carries *resumable graph state*. Keep these conceptually separate even though both involve "state."

### Phase 4 — Evaluator
- Detection windows should be built from **node-level trace events**, which gives finer-grained loop signals than chain-level events would (e.g., the same node re-entered N times with near-identical state, not just "same tool called repeatedly").

### Phase 5 — Control / Circuit Breaker API
- `POST /halt` → looks up the thread's checkpointer state → calls `interrupt()` (or flips a flag a guard node checks) → the graph freezes at the next checkpointed boundary.
- **Latency-to-halt is bounded by node granularity**, not instant — if a single node runs a long tool call before its next checkpoint, the halt won't land until that node completes. Worth documenting as a known limit, and worth checking during Phase 6 load/chaos testing whether any node needs to be split up to keep the halt SLA tight.
- Cleanup job (see gotcha above) belongs here, not as an afterthought in Phase 6.

### Phase 6 — Integration Testing
- Add a specific test case: **seeded loop inside a single long-running node** — confirms whether our halt SLA holds when the loop is *inside* a node rather than *between* nodes (this is the edge case the granularity gotcha above creates).

---

## Open Questions to Resolve Before/During Build
- [ ] Where does the thread_id come from — generated by us at agent-run start, or does the caller (e.g. the customer's app) provide it? This affects how the Control API looks up a thread to halt.
- [ ] Postgres checkpointer schema — does it live in its own DB/schema, or share infra with the event queue's metadata store?
- [ ] Do we need node splitting guidance/docs for customers whose nodes wrap long-running tool calls, so their halt SLA doesn't blow out?

# control-api

Circuit breaker / control API (Phase 5).

`POST /halt` looks up a thread's checkpointer state and calls `interrupt()`
(or flips a flag a guard node checks), freezing the graph at the next
checkpointed boundary. Halts are permanent — `Command(resume=...)` is
deliberately never called on a halted thread.

Also owns the TTL-based reaper job that cleans up halted threads, since
LangGraph does not auto-expire frozen threads.

## Install

```bash
pip install -e .
```

## Status

Skeleton only — no implementation yet.

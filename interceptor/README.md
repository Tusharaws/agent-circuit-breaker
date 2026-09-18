# interceptor

Per-node LangGraph instrumentation for the Agent Execution Guardian (Phase 1).

Captures a full state snapshot at each node transition (not just prompt/response
pairs) and reuses LangChain-compatible callback events (`on_llm_start/end`,
`on_tool_start/end`) that LangGraph nodes still emit under the hood.

## Install

```bash
pip install -e .
```

## Status

Skeleton only — no implementation yet.

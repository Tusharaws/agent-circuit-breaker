# Trace Event Types

Allowed `event_type` values for `TraceEvent` (see `../trace_event.py`). This
list mirrors the LangChain-compatible callback events the Interceptor
(Phase 1) reuses from LangGraph node transitions (see `ARCHITECTURE_NOTES.md`).

- node_enter
- node_exit
- llm_start
- llm_end
- tool_start
- tool_end

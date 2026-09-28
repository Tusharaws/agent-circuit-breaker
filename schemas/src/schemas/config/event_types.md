# Trace Event Types

Allowed `event_type` values for `TraceEvent` (see `../trace_event.py`). The
first 6 mirror the LangChain-compatible callback events the Interceptor
(Phase 1) reuses from LangGraph node transitions (see `ARCHITECTURE_NOTES.md`).
`agent_message` (Phase 7) is AutoGen-specific -- an agent's conversational
turn has no LangGraph-node equivalent, so it gets its own type rather than
being forced into `node_enter`/`node_exit`; `tool_start`/`tool_end` are
reused as-is for AutoGen tool calls since that concept genuinely is the
same one.

- node_enter
- node_exit
- llm_start
- llm_end
- tool_start
- tool_end
- agent_message

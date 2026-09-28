import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from interceptor.capture import CaptureDispatcher
from schemas.trace_event import TokenUsage, TraceEvent

logger = logging.getLogger(__name__)

# Dedicated logger (Phase 6 end-to-end tracing) so a trace_id is followable
# across all 5 services' logs -- DEBUG, since this fires on every node/llm/
# tool transition, not a low-frequency decision an operator wants by default.
_trace_logger = logging.getLogger("interceptor.trace")


@dataclass
class _RunContext:
    node_name: str | None
    thread_id: str
    step_index: int
    started_at: float


class GraphCaptureHandler(BaseCallbackHandler):
    """Production LangGraph node-hook capture (Phase 1).

    Maps real LangGraph/LangChain callback events into `schemas.TraceEvent`
    instances dispatched through a `CaptureDispatcher`. Every callback body
    is wrapped so nothing here ever raises into the customer's graph
    execution — the same non-blocking, never-crash guarantee
    `CaptureDispatcher` itself provides, extended to event construction.

    `agent_id` is a required constructor parameter rather than something
    read from LangGraph's runtime config: LangGraph only auto-propagates
    its own reserved configurable keys (like `thread_id`) into callback
    metadata, not arbitrary customer-supplied ones — relying on that for
    `agent_id` would be fragile. `thread_id` (LangGraph's own reserved key)
    becomes `TraceEvent.trace_id`, reusing LangGraph's own thread identity
    rather than inventing a second one.

    `_end`/`_error` callbacks don't receive `metadata` at all (confirmed
    empirically) — needed context (thread_id, step_index, node name, start
    time) is cached at the matching `_start` callback, keyed by run_id.
    """

    def __init__(self, dispatcher: CaptureDispatcher, agent_id: str) -> None:
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        self._dispatcher = dispatcher
        self._agent_id = agent_id
        self._contexts: dict[str, _RunContext] = {}

    # ---- chain (node) --------------------------------------------------

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        try:
            node_name = (metadata or {}).get("langgraph_node")
            if node_name is None:
                return
            ctx = self._start_context(run_id, metadata, node_name)
            self._emit(ctx, "node_enter", {"node": node_name})
        except Exception:
            logger.exception("graph_hooks: on_chain_start failed")

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, tags=None, **kwargs):
        try:
            ctx = self._contexts.pop(str(run_id), None)
            if ctx is None or ctx.node_name is None:
                return
            self._emit(ctx, "node_exit", {"node": ctx.node_name}, latency_from=ctx)
        except Exception:
            logger.exception("graph_hooks: on_chain_end failed")

    def on_chain_error(self, error, *, run_id, parent_run_id=None, tags=None, **kwargs):
        try:
            ctx = self._contexts.pop(str(run_id), None)
            if ctx is None or ctx.node_name is None:
                return
            self._emit(
                ctx,
                "node_exit",
                {"node": ctx.node_name, "error": True, "error_message": str(error)},
                latency_from=ctx,
            )
        except Exception:
            logger.exception("graph_hooks: on_chain_error failed")

    # ---- llm -------------------------------------------------------------

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        try:
            ctx = self._start_context(run_id, metadata, node_name=None)
            self._emit(ctx, "llm_start", {"prompts": list(prompts)})
        except Exception:
            logger.exception("graph_hooks: on_llm_start failed")

    def on_llm_end(self, response, *, run_id, parent_run_id=None, tags=None, **kwargs):
        try:
            ctx = self._contexts.pop(str(run_id), None)
            if ctx is None:
                return
            token_usage = self._extract_token_usage(response)
            payload = {"response": self._extract_all_generations(response)}
            self._emit(ctx, "llm_end", payload, latency_from=ctx, token_usage=token_usage)
        except Exception:
            logger.exception("graph_hooks: on_llm_end failed")

    def on_llm_error(self, error, *, run_id, parent_run_id=None, tags=None, **kwargs):
        try:
            ctx = self._contexts.pop(str(run_id), None)
            if ctx is None:
                return
            self._emit(
                ctx, "llm_end", {"error": True, "error_message": str(error)}, latency_from=ctx
            )
        except Exception:
            logger.exception("graph_hooks: on_llm_error failed")

    # ---- tool --------------------------------------------------------

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        try:
            ctx = self._start_context(run_id, metadata, node_name=None)
            name = (serialized or {}).get("name")
            args = kwargs.get("inputs")
            self._emit(ctx, "tool_start", {"tool": name, "args": args, "input_str": input_str})
        except Exception:
            logger.exception("graph_hooks: on_tool_start failed")

    def on_tool_end(self, output, *, run_id, parent_run_id=None, tags=None, **kwargs):
        try:
            ctx = self._contexts.pop(str(run_id), None)
            if ctx is None:
                return
            self._emit(ctx, "tool_end", {"output": str(output)}, latency_from=ctx)
        except Exception:
            logger.exception("graph_hooks: on_tool_end failed")

    def on_tool_error(self, error, *, run_id, parent_run_id=None, tags=None, **kwargs):
        try:
            ctx = self._contexts.pop(str(run_id), None)
            if ctx is None:
                return
            self._emit(
                ctx, "tool_end", {"error": True, "error_message": str(error)}, latency_from=ctx
            )
        except Exception:
            logger.exception("graph_hooks: on_tool_error failed")

    # ---- helpers -------------------------------------------------------

    def _start_context(self, run_id: Any, metadata: dict | None, node_name: str | None) -> _RunContext:
        metadata = metadata or {}
        thread_id = metadata.get("thread_id")
        if not thread_id:
            raise ValueError(
                "thread_id missing from callback metadata; set "
                "config={'configurable': {'thread_id': ...}} on invoke"
            )
        step_index = metadata.get("langgraph_step")
        if step_index is None:
            step_index = 0
        ctx = _RunContext(
            node_name=node_name,
            thread_id=thread_id,
            step_index=step_index,
            started_at=time.monotonic(),
        )
        self._contexts[str(run_id)] = ctx
        return ctx

    def _emit(
        self,
        ctx: _RunContext,
        event_type: str,
        payload: dict,
        latency_from: _RunContext | None = None,
        token_usage: TokenUsage | None = None,
    ) -> None:
        latency_ms = None
        if latency_from is not None:
            latency_ms = (time.monotonic() - latency_from.started_at) * 1000
        event = TraceEvent(
            trace_id=ctx.thread_id,
            agent_id=self._agent_id,
            step_index=ctx.step_index,
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            payload=payload,
            token_usage=token_usage,
            latency_ms=latency_ms,
        )
        _trace_logger.debug(
            json.dumps({"trace_id": ctx.thread_id, "service": "interceptor", "stage": "captured", "event_type": event_type})
        )
        self._dispatcher.capture(event)

    @staticmethod
    def _extract_token_usage(response: Any) -> TokenUsage | None:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage")
        if not usage:
            return None
        try:
            return TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
        except Exception:
            return None

    @staticmethod
    def _extract_all_generations(response: Any) -> list[str]:
        """All candidate completions for the (single) prompt, not just the
        first -- an LLM call can return n>1 candidates, and dropping the
        rest would silently discard data. Scoped to single-prompt calls,
        which is what a LangGraph node does; a batch of prompts isn't
        handled here."""
        try:
            return [generation.text for generation in response.generations[0]]
        except Exception:
            return []

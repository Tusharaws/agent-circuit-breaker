import logging
from datetime import datetime, timezone
from typing import Any

from autogen_core import CancellationToken, DefaultInterventionHandler, MessageContext
from control_api.guard import HaltRegistry
from schemas.trace_event import TokenUsage, TraceEvent

from interceptor.capture import CaptureDispatcher

logger = logging.getLogger(__name__)

DEFAULT_HALT_REASON = "halted by control-api"


class AutoGenCaptureHandler(DefaultInterventionHandler):
    """Production AutoGen capture + halt (Phase 7).

    Registers once on a `SingleThreadedAgentRuntime` via
    `intervention_handlers=[...]` -- AutoGen's native message-interception
    hook, confirmed empirically to see every agent turn *and* every tool
    call (nested in a `GroupChatAgentResponse`'s `inner_messages`), with no
    exception-raise hack needed.

    Unlike LangGraph, AutoGen has no customer-editable graph to insert a
    guard node into, and unlike `create_agent()` (still LangGraph
    underneath), there's no per-message auto-propagated thread identifier
    either -- so `thread_id` is a constructor parameter, one handler
    instance per running conversation, same as `agent_id` already is for
    `GraphCaptureHandler`.

    Halting uses AutoGen's own `CancellationToken` -- confirmed
    empirically (not assumed) that calling `.cancel()` on the same token
    instance passed to `team.run(cancellation_token=token)` cleanly raises
    `asyncio.CancelledError` back to the caller, whether cancelled
    externally or from inside this handler itself. This is a real native
    primitive, not the "no interrupt()/checkpointer, use exception-raise"
    the original AutoGen adapter task assumed (that assumption came from
    the LangChain AgentExecutor task and was never actually checked
    against AutoGen specifically).
    """

    def __init__(
        self,
        dispatcher: CaptureDispatcher,
        agent_id: str,
        thread_id: str,
        cancellation_token: CancellationToken,
        registry: HaltRegistry | None = None,
        halt_reason: str = DEFAULT_HALT_REASON,
    ) -> None:
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not thread_id:
            raise ValueError("thread_id must be a non-empty string")
        self._dispatcher = dispatcher
        self._agent_id = agent_id
        self._thread_id = thread_id
        self._cancellation_token = cancellation_token
        self._registry = registry
        self._halt_reason = halt_reason
        self._step_index = 0

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any:
        try:
            self._check_halt()
            self._capture(message)
        except Exception:
            logger.exception("autogen_hooks: on_publish failed")
        return message

    def _check_halt(self) -> None:
        if self._registry is None or self._cancellation_token.is_cancelled():
            return
        if self._registry.is_halted(self._thread_id):
            self._cancellation_token.cancel()

    def _capture(self, message: Any) -> None:
        # Only a GroupChatAgentResponse-shaped message (a `.response`
        # attribute) carries both the agent's turn AND any tool calls it
        # made -- the separately-published GroupChatMessage is a redundant
        # echo of the same chat_message, so it's intentionally not
        # captured again here (would double-count every turn).
        response = getattr(message, "response", None)
        if response is None:
            return
        for inner in response.inner_messages or []:
            self._emit_from_chat_message(inner)
        self._emit_from_chat_message(response.chat_message)

    def _emit_from_chat_message(self, chat_message: Any) -> None:
        type_name = type(chat_message).__name__
        source = getattr(chat_message, "source", None)
        if type_name == "ToolCallRequestEvent":
            for call in chat_message.content:
                self._emit("tool_start", {"tool": call.name, "args": call.arguments}, source=source)
        elif type_name == "ToolCallExecutionEvent":
            for result in chat_message.content:
                self._emit(
                    "tool_end",
                    {"tool": result.name, "output": result.content, "error": result.is_error},
                    source=source,
                )
        else:
            self._emit(
                "agent_message",
                {"agent": source, "content": getattr(chat_message, "content", None), "message_type": type_name},
                source=source,
                token_usage=self._extract_token_usage(chat_message),
            )

    def _emit(
        self, event_type: str, payload: dict, source: str | None, token_usage: TokenUsage | None = None
    ) -> None:
        self._step_index += 1
        event = TraceEvent(
            trace_id=self._thread_id,
            agent_id=self._agent_id,
            step_index=self._step_index,
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            payload=payload,
            token_usage=token_usage,
        )
        self._dispatcher.capture(event)

    @staticmethod
    def _extract_token_usage(chat_message: Any) -> TokenUsage | None:
        usage = getattr(chat_message, "models_usage", None)
        if usage is None:
            return None
        return TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.prompt_tokens + usage.completion_tokens,
        )

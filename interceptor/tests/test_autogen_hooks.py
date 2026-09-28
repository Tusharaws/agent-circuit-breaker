"""Tests for AutoGenCaptureHandler (Phase 7). Written before the
implementation (interceptor.autogen_hooks does not exist yet) -- run
`pytest` to see them fail with a collection error until
src/interceptor/autogen_hooks.py exists.

Two layers, same pattern as test_graph_hooks.py:
- Unit-level: call on_publish directly with hand-built AutoGen message
  objects (a fake dispatcher records what was captured).
- Integration-level: a real 2-agent autogen-agentchat conversation through
  a real CaptureDispatcher and a real HaltRegistry.

Facts this file relies on, confirmed empirically first (not assumed):
- "AutoGen" today means autogen-agentchat + autogen-core -- pyautogen is
  now just a proxy package pointing at autogen-agentchat; ag2 is a
  separate, unrelated fork.
- InterventionHandler.on_publish sees every GroupChatAgentResponse, whose
  `.response` (a Response) carries both `.chat_message` (the agent's
  turn output) and `.inner_messages` (real ToolCallRequestEvent/
  ToolCallExecutionEvent objects when the agent used a tool).
- A custom `runtime` passed into a Team is NOT auto-started/stopped by
  the Team -- that's the caller's job (confirmed from BaseGroupChat's own
  source after an initial hang during reconnaissance).
- CancellationToken.cancel() -- called either externally or from inside
  the intervention handler itself -- cleanly raises asyncio.CancelledError
  back to team.run()'s caller. This is AutoGen's real halt primitive, not
  an exception-raise hack.
"""
import asyncio

import fakeredis
import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent, ToolCallRequestEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken, FunctionCall, SingleThreadedAgentRuntime
from autogen_core.models import CreateResult, FunctionExecutionResult, ModelFamily, ModelInfo, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient
from control_api.guard import HaltRegistry

from interceptor.autogen_hooks import AutoGenCaptureHandler

AGENT_ID = "agent-1"
THREAD_ID = "thread-1"

_MODEL_INFO = ModelInfo(
    vision=False, function_calling=True, json_output=False, family=ModelFamily.UNKNOWN, structured_output=False
)


class FakeDispatcher:
    def __init__(self):
        self.captured = []

    def capture(self, event):
        self.captured.append(event)


def _response_message(chat_message, inner_messages=None):
    """A minimal stand-in for GroupChatAgentResponse -- only `.response` is
    read by the handler, so a tiny namespace-like object is enough for the
    unit-level tests (matches test_graph_hooks.py's own preference for
    hand-built args over a full real object where a real one isn't needed)."""

    class _Response:
        def __init__(self, chat_message, inner_messages):
            self.chat_message = chat_message
            self.inner_messages = inner_messages

    class _Wrapper:
        def __init__(self, response):
            self.response = response

    return _Wrapper(_Response(chat_message, inner_messages))


def _text_message(source="agent_a", content="hello"):
    return TextMessage(source=source, content=content)


def _tool_call_request(source="agent_a"):
    return ToolCallRequestEvent(
        source=source, content=[FunctionCall(id="call1", name="search", arguments='{"query": "x"}')]
    )


def _tool_call_execution(source="agent_a"):
    return ToolCallExecutionEvent(
        source=source, content=[FunctionExecutionResult(content="results", name="search", call_id="call1", is_error=False)]
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_rejects_empty_agent_id():
    with pytest.raises(ValueError):
        AutoGenCaptureHandler(
            dispatcher=FakeDispatcher(), agent_id="", thread_id=THREAD_ID, cancellation_token=CancellationToken()
        )


def test_construction_rejects_empty_thread_id():
    with pytest.raises(ValueError):
        AutoGenCaptureHandler(
            dispatcher=FakeDispatcher(), agent_id=AGENT_ID, thread_id="", cancellation_token=CancellationToken()
        )


# ---------------------------------------------------------------------------
# Capture: agent_message (unit-level, direct on_publish invocation)
# ---------------------------------------------------------------------------


def test_agent_text_message_emits_an_agent_message_event():
    dispatcher = FakeDispatcher()
    handler = AutoGenCaptureHandler(
        dispatcher=dispatcher, agent_id=AGENT_ID, thread_id=THREAD_ID, cancellation_token=CancellationToken()
    )

    asyncio.run(handler.on_publish(_response_message(_text_message(content="hi there")), message_context=None))

    assert len(dispatcher.captured) == 1
    event = dispatcher.captured[0]
    assert event.event_type == "agent_message"
    assert event.trace_id == THREAD_ID
    assert event.agent_id == AGENT_ID
    assert event.payload["content"] == "hi there"
    assert event.payload["agent"] == "agent_a"


def test_non_groupchat_response_message_is_ignored():
    """Only GroupChatAgentResponse-shaped messages (a `.response` attribute)
    carry capturable content -- everything else (GroupChatStart,
    GroupChatRequestPublish, the redundant GroupChatMessage echo) is
    plumbing, not "agent message events" in the AC's sense."""
    dispatcher = FakeDispatcher()
    handler = AutoGenCaptureHandler(
        dispatcher=dispatcher, agent_id=AGENT_ID, thread_id=THREAD_ID, cancellation_token=CancellationToken()
    )

    asyncio.run(handler.on_publish(object(), message_context=None))

    assert dispatcher.captured == []


# ---------------------------------------------------------------------------
# Capture: tool calls nested in inner_messages
# ---------------------------------------------------------------------------


def test_tool_call_request_and_execution_emit_tool_start_and_tool_end():
    dispatcher = FakeDispatcher()
    handler = AutoGenCaptureHandler(
        dispatcher=dispatcher, agent_id=AGENT_ID, thread_id=THREAD_ID, cancellation_token=CancellationToken()
    )

    message = _response_message(
        chat_message=_text_message(content="results"),
        inner_messages=[_tool_call_request(), _tool_call_execution()],
    )
    asyncio.run(handler.on_publish(message, message_context=None))

    event_types = [e.event_type for e in dispatcher.captured]
    assert event_types == ["tool_start", "tool_end", "agent_message"]

    tool_start = dispatcher.captured[0]
    assert tool_start.payload["tool"] == "search"
    assert tool_start.payload["args"] == '{"query": "x"}'

    tool_end = dispatcher.captured[1]
    assert tool_end.payload["tool"] == "search"
    assert tool_end.payload["output"] == "results"
    assert tool_end.payload["error"] is False


# ---------------------------------------------------------------------------
# Token usage extraction
# ---------------------------------------------------------------------------


def test_token_usage_is_extracted_when_present():
    dispatcher = FakeDispatcher()
    handler = AutoGenCaptureHandler(
        dispatcher=dispatcher, agent_id=AGENT_ID, thread_id=THREAD_ID, cancellation_token=CancellationToken()
    )
    message = _text_message()
    message.models_usage = RequestUsage(prompt_tokens=12, completion_tokens=8)

    asyncio.run(handler.on_publish(_response_message(message), message_context=None))

    event = dispatcher.captured[0]
    assert event.token_usage.prompt_tokens == 12
    assert event.token_usage.completion_tokens == 8
    assert event.token_usage.total_tokens == 20


def test_token_usage_absent_when_not_reported():
    dispatcher = FakeDispatcher()
    handler = AutoGenCaptureHandler(
        dispatcher=dispatcher, agent_id=AGENT_ID, thread_id=THREAD_ID, cancellation_token=CancellationToken()
    )

    asyncio.run(handler.on_publish(_response_message(_text_message()), message_context=None))

    assert dispatcher.captured[0].token_usage is None


# ---------------------------------------------------------------------------
# Halt: checks HaltRegistry, cancels the token
# ---------------------------------------------------------------------------


def test_halted_thread_cancels_the_token():
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
    registry.request_halt(THREAD_ID)
    token = CancellationToken()
    handler = AutoGenCaptureHandler(
        dispatcher=FakeDispatcher(),
        agent_id=AGENT_ID,
        thread_id=THREAD_ID,
        cancellation_token=token,
        registry=registry,
    )

    asyncio.run(handler.on_publish(_response_message(_text_message()), message_context=None))

    assert token.is_cancelled() is True


def test_non_halted_thread_does_not_cancel_the_token():
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
    token = CancellationToken()
    handler = AutoGenCaptureHandler(
        dispatcher=FakeDispatcher(),
        agent_id=AGENT_ID,
        thread_id=THREAD_ID,
        cancellation_token=token,
        registry=registry,
    )

    asyncio.run(handler.on_publish(_response_message(_text_message()), message_context=None))

    assert token.is_cancelled() is False


def test_no_registry_never_cancels():
    """registry=None (default) preserves capture-only behavior -- no halt
    check attempted at all."""
    token = CancellationToken()
    handler = AutoGenCaptureHandler(
        dispatcher=FakeDispatcher(), agent_id=AGENT_ID, thread_id=THREAD_ID, cancellation_token=token
    )

    asyncio.run(handler.on_publish(_response_message(_text_message()), message_context=None))

    assert token.is_cancelled() is False


# ---------------------------------------------------------------------------
# Fault isolation: on_publish must never raise into the runtime
# ---------------------------------------------------------------------------


def test_a_malformed_message_does_not_raise():
    dispatcher = FakeDispatcher()
    handler = AutoGenCaptureHandler(
        dispatcher=dispatcher, agent_id=AGENT_ID, thread_id=THREAD_ID, cancellation_token=CancellationToken()
    )

    class _BrokenResponse:
        @property
        def response(self):
            raise RuntimeError("boom")

    result = asyncio.run(handler.on_publish(_BrokenResponse(), message_context=None))

    assert result is not None  # still returns the message, doesn't drop it
    assert dispatcher.captured == []


# ---------------------------------------------------------------------------
# Integration: a real 2-agent conversation, real CaptureDispatcher, real halt
# ---------------------------------------------------------------------------


class _SlowReplayClient(ReplayChatCompletionClient):
    """Adds real per-turn delay -- an instant fake model would complete
    the whole conversation before a halt request could ever land, same
    reasoning as sample_agent.py's iteration_delay for the LangGraph case."""

    def __init__(self, *args, delay=0.03, **kwargs):
        super().__init__(*args, **kwargs)
        self._delay = delay

    async def create(self, *args, **kwargs):
        await asyncio.sleep(self._delay)
        return await super().create(*args, **kwargs)


def _build_real_conversation(handler, num_turns=30):
    """Construction only -- `runtime.start()` internally calls
    `asyncio.create_task(...)`, which needs an already-running event loop,
    so it must happen inside the async caller, not here."""
    model_a = _SlowReplayClient([f"turn {i} from A" for i in range(num_turns)], model_info=_MODEL_INFO)
    model_b = _SlowReplayClient([f"turn {i} from B" for i in range(num_turns)], model_info=_MODEL_INFO)
    agent_a = AssistantAgent("agent_a", model_client=model_a)
    agent_b = AssistantAgent("agent_b", model_client=model_b)

    runtime = SingleThreadedAgentRuntime(intervention_handlers=[handler])
    team = RoundRobinGroupChat(
        [agent_a, agent_b], runtime=runtime, termination_condition=MaxMessageTermination(2 * num_turns)
    )
    return runtime, team


def test_real_conversation_produces_a_complete_ordered_trace():
    from interceptor.capture import CaptureDispatcher

    dispatcher = CaptureDispatcher(sink=lambda event: captured.append(event))
    captured = []
    token = CancellationToken()
    handler = AutoGenCaptureHandler(
        dispatcher=dispatcher, agent_id=AGENT_ID, thread_id="thread-real", cancellation_token=token
    )
    runtime, team = _build_real_conversation(handler, num_turns=2)

    async def run():
        runtime.start()
        await team.run(task="Say hello.", cancellation_token=token)
        await runtime.stop_when_idle()

    asyncio.run(run())
    dispatcher.stop(timeout=2.0)

    # MaxMessageTermination(2 * num_turns) counts the initial user task
    # message too, so real agent turns = 2*num_turns - 1.
    assert len(captured) >= 2 * 2 - 1
    assert all(e.event_type == "agent_message" for e in captured)
    assert all(e.trace_id == "thread-real" for e in captured)
    sources = [e.payload["agent"] for e in captured]
    assert "agent_a" in sources and "agent_b" in sources


def test_real_halt_stops_a_running_conversation_within_a_few_turns():
    registry = HaltRegistry(redis_client=fakeredis.FakeRedis())
    registry.request_halt("thread-real-halt")  # already halted before the run starts
    token = CancellationToken()
    handler = AutoGenCaptureHandler(
        dispatcher=FakeDispatcher(),
        agent_id=AGENT_ID,
        thread_id="thread-real-halt",
        cancellation_token=token,
        registry=registry,
    )
    runtime, team = _build_real_conversation(handler, num_turns=30)

    async def run():
        runtime.start()
        with pytest.raises(asyncio.CancelledError):
            await team.run(task="Go.", cancellation_token=token)
        await runtime.stop_when_idle()

    asyncio.run(run())

    assert token.is_cancelled() is True
    # Stopped almost immediately, nowhere near the full 60-message conversation.
    assert len(handler._dispatcher.captured) < 10

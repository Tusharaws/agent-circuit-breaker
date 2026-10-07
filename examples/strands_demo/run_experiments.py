"""Measures the three open Strands questions against a real Strands Agent.

Run from this folder:  python run_experiments.py
Uses fakeredis-backed HaltRegistry and a scripted model, so results are
about Strands' cancellation semantics, not model behaviour.yes
"""
import shutil
import tempfile
import threading
import time

import fakeredis
from control_api.guard import HaltRegistry
from strands import Agent, tool
from strands.session.file_session_manager import FileSessionManager

from scripted_model import ScriptedModel
from strands_halt import StrandsHaltHook

TOOL_LOG: list[str] = []


@tool
def slow_tool(seconds: float) -> str:
    """Simulates a long, uninterruptible tool call."""
    time.sleep(seconds)
    TOOL_LOG.append("slow_tool finished")
    return f"slept {seconds}s"


@tool
def lookup_order(query: str) -> str:
    """Looks up an order; never finds it (simulates the retail loop)."""
    TOOL_LOG.append(f"lookup {query!r}")
    return "no order found"


def _registry():
    return HaltRegistry(redis_client=fakeredis.FakeRedis())


def experiment_granularity():
    """Halt lands while a 2s tool is running. Does the tool finish?"""
    print("\n== A. Granularity: halt requested while a 2.0s tool is in flight ==")
    TOOL_LOG.clear()
    registry = _registry()
    thread_id = "granularity"
    hook = StrandsHaltHook(registry, thread_id)
    model = ScriptedModel(
        [("tool", "slow_tool", {"seconds": 2.0}), ("tool", "lookup_order", {"query": "A"}), ("text", "done")]
    )
    agent = Agent(model=model, tools=[slow_tool, lookup_order], hooks=[hook], callback_handler=None)

    threading.Timer(0.5, lambda: registry.request_halt(thread_id)).start()
    start = time.perf_counter()
    result = agent("start")
    elapsed = time.perf_counter() - start

    print(f"invoke returned after {elapsed:.2f}s")
    print(f"tool ran to completion: {'slow_tool finished' in TOOL_LOG}")
    print(f"model calls made: {model.calls}; halt-check cancels on tools: {hook.tool_cancels}")
    print(f"result text: {str(result)[:120]!r}")


def experiment_loop_after_cancel():
    """Halt lands mid-loop of repeated lookups. Does the loop stop?"""
    print("\n== B. Loop after cancel: halt requested mid-loop of 20 lookups ==")
    TOOL_LOG.clear()
    registry = _registry()
    thread_id = "loop"
    hook = StrandsHaltHook(registry, thread_id)
    script = [("tool", "lookup_order", {"query": f"attempt {i}"}) for i in range(20)] + [("text", "gave up")]
    model = ScriptedModel(script, latency_s=0.2)
    agent = Agent(model=model, tools=[lookup_order], hooks=[hook], callback_handler=None)

    threading.Timer(0.5, lambda: registry.request_halt(thread_id)).start()
    start = time.perf_counter()
    result = agent("start the lookup")
    elapsed = time.perf_counter() - start

    print(f"invoke returned after {elapsed:.2f}s (of a possible ~4.2s for 20 calls)")
    print(f"lookups actually executed: {len(TOOL_LOG)} of 20")
    print(f"model calls made: {model.calls}; model-call cancels: {hook.model_cancels}")
    print(f"final result text: {str(result)[:160]!r}")
    print(f"stop_reason: {getattr(result, 'stop_reason', None)!r}")


def experiment_persistence():
    """After a halt, is conversation state persisted, and can it resume?"""
    print("\n== C. Persistence: halt, then inspect and resume the session ==")
    TOOL_LOG.clear()
    storage = tempfile.mkdtemp(prefix="strands-session-")
    try:
        registry = _registry()
        thread_id = "persist"
        hook = StrandsHaltHook(registry, thread_id)
        script = [("tool", "lookup_order", {"query": f"attempt {i}"}) for i in range(10)] + [("text", "done")]
        session = FileSessionManager(session_id=thread_id, storage_dir=storage)
        agent = Agent(
            model=ScriptedModel(script, latency_s=0.2), tools=[lookup_order], hooks=[hook],
            session_manager=session, callback_handler=None,
        )
        threading.Timer(0.5, lambda: registry.request_halt(thread_id)).start()
        agent("start the lookup")
        messages_at_halt = len(agent.messages)
        print(f"messages in memory after halt: {messages_at_halt}")

        # Is the halted state persisted by Strands' session manager?
        import os

        files = []
        for root, _, names in os.walk(storage):
            files += [os.path.join(root, n) for n in names]
        print(f"session files written: {len(files)}")

        # Resume attempt while STILL halted: the registry should stop it immediately.
        resumed = Agent(
            model=ScriptedModel([("text", "resumed")]), tools=[lookup_order],
            hooks=[StrandsHaltHook(registry, thread_id)], callback_handler=None,
            session_manager=FileSessionManager(session_id=thread_id, storage_dir=storage),
        )
        print(f"restored messages on reload: {len(resumed.messages)}")
        out = resumed("continue please")
        print(f"resume while halted -> result: {str(out)[:100]!r}")

        # Clear the halt: does Strands resume from the persisted history?
        registry.clear_halt(thread_id)
        resumed2 = Agent(
            model=ScriptedModel([("text", "resumed after clear")]), tools=[lookup_order],
            hooks=[StrandsHaltHook(registry, thread_id)], callback_handler=None,
            session_manager=FileSessionManager(session_id=thread_id, storage_dir=storage),
        )
        out2 = resumed2("continue please")
        print(f"resume after clear -> result: {str(out2)[:100]!r}; messages now {len(resumed2.messages)}")
    finally:
        shutil.rmtree(storage, ignore_errors=True)


if __name__ == "__main__":
    experiment_granularity()
    experiment_loop_after_cancel()
    experiment_persistence()

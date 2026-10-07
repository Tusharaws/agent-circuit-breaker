"""Halt enforcement for a Strands agent, using Strands' native cancel fields.

BeforeModelCallEvent.cancel and BeforeToolCallEvent.cancel_tool are set
from the shared HaltRegistry. Counters record every hook firing so the
experiments can see exactly which calls were stopped.
"""
from strands.hooks import BeforeModelCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry

HALT_MESSAGE = "halted by control-api"


class StrandsHaltHook(HookProvider):
    def __init__(self, registry, thread_id: str):
        self._registry = registry
        self._thread_id = thread_id
        self.model_checks = 0
        self.model_cancels = 0
        self.tool_checks = 0
        self.tool_cancels = 0

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(BeforeToolCallEvent, self._before_tool)

    def _halted(self) -> bool:
        return self._registry.is_halted(self._thread_id)

    def _before_model(self, event: BeforeModelCallEvent) -> None:
        self.model_checks += 1
        if self._halted():
            self.model_cancels += 1
            event.cancel = HALT_MESSAGE

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        self.tool_checks += 1
        if self._halted():
            self.tool_cancels += 1
            event.cancel_tool = HALT_MESSAGE

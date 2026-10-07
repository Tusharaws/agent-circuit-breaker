"""A deterministic Strands model that replays a script of tool calls or text.

Each script entry is either ("tool", name, input_dict) or ("text", string).
`latency_s` simulates in-flight model time so a halt can land mid-call.
Used so the experiments measure Strands' cancellation semantics, not a
real model's behaviour.
"""
import asyncio
import uuid
from typing import Any

from strands.models import Model


class ScriptedModel(Model):
    def __init__(self, script: list[tuple], latency_s: float = 0.0):
        self._script = list(script)
        self._latency_s = latency_s
        self.calls = 0
        self._config: dict[str, Any] = {}

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    def get_config(self) -> Any:
        return dict(self._config)

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError("ScriptedModel does not support structured output")
        yield  # pragma: no cover

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        await asyncio.sleep(self._latency_s)
        step = self._script.pop(0) if self._script else ("text", "done")

        yield {"messageStart": {"role": "assistant"}}
        if step[0] == "tool":
            _, name, tool_input = step
            tool_use_id = f"tu-{uuid.uuid4().hex[:8]}"
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tool_use_id, "name": name}}}}
            import json

            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_input)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": step[1]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}, "metrics": {"latencyMs": 1}}}

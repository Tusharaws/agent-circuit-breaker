import json
import logging
from typing import Protocol

from queue_client.client import QueueClient

from evaluator.prefilter import should_escalate_to_slm
from evaluator.verdict import Verdict, build_prompt, parse_verdict
from evaluator.window import DEFAULT_WINDOW_SIZE, build_window

# Dedicated logger (Phase 6 end-to-end tracing). INFO, not DEBUG like the
# high-frequency per-event loggers elsewhere in the pipeline -- a verdict
# is a meaningful, low-frequency decision point per window, not a
# per-node pass-through.
_trace_logger = logging.getLogger("evaluator.trace")

# Two separate budgets, not one blended figure: a filtered window costs
# one queue read + one hash comparison (sub-millisecond, per queue-client's
# own load test); an escalated window additionally costs a real SLM call
# (measured ~143ms) -- blending them would be meaninglessly loose for the
# filtered path or unachievably tight for the escalated one.
FILTERED_PATH_BUDGET_MS = 10.0
ESCALATED_PATH_BUDGET_MS = 500.0  # ~3.5x margin over the measured ~143ms


class _SLM(Protocol):
    def generate(self, prompt: str, max_tokens: int = ...) -> str: ...


def evaluate_thread(
    queue_client: QueueClient,
    slm_client: _SLM,
    thread_id: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> Verdict:
    """Consume -> pre-filter -> SLM -> verdict (Phase 4), for a single
    thread's current window. Always returns a Verdict, per the AC ("a
    verdict for every window") -- a filtered-out window gets a fast,
    cheap no-loop verdict without ever calling the SLM.
    """
    window = build_window(queue_client, thread_id, window_size=window_size)

    if not should_escalate_to_slm(window):
        verdict = Verdict(is_loop=False, confidence=0.0, reason=None)
        _log_verdict(thread_id, escalated=False, verdict=verdict)
        return verdict

    prompt = build_prompt(window)
    raw_response = slm_client.generate(prompt)
    verdict = parse_verdict(raw_response)
    _log_verdict(thread_id, escalated=True, verdict=verdict)
    return verdict


def _log_verdict(thread_id: str, escalated: bool, verdict: Verdict) -> None:
    _trace_logger.info(
        json.dumps(
            {
                "trace_id": thread_id,
                "service": "evaluator",
                "stage": "evaluated",
                "escalated": escalated,
                "is_loop": verdict.is_loop,
                "confidence": verdict.confidence,
            }
        )
    )

"""Eval set: known-good vs. known-loop traces, precision/recall
measurement against the REAL pipeline (Phase 4). Written to measure real
numbers first, then assert an honest target with real margin -- same
approach used for the latency budgets elsewhere in this package.

See evaluator/EVAL_REPORT.md for the full report (methodology, results,
threshold-tuning experiment, and the documented alternating-pattern
limitation this surfaced).
"""
import fakeredis
import pytest

from queue_client.client import QueueClient

from evaluator.eval_set import ALL_TRACES, KNOWN_GOOD_TRACES, KNOWN_LOOP_TRACES, run_eval
from evaluator.slm import SLMClient

# Measured: precision=0.90, recall=0.75 on this eval set. Targets set with
# real margin below the observed numbers (not exact-match), since SLM
# generation isn't guaranteed bit-for-bit deterministic run to run.
PRECISION_TARGET = 0.80
RECALL_TARGET = 0.65


def test_at_least_ten_known_good_and_ten_known_loop_traces_exist():
    assert len(KNOWN_GOOD_TRACES) >= 10
    assert len(KNOWN_LOOP_TRACES) >= 10


@pytest.mark.integration
def test_real_pipeline_meets_precision_recall_targets():
    queue_client = QueueClient(redis_client=fakeredis.FakeRedis())
    slm_client = SLMClient()

    result = run_eval(queue_client, slm_client)

    print(f"\nEval set ({len(ALL_TRACES)} traces): precision={result.precision:.2f} recall={result.recall:.2f}")
    if result.misclassified:
        print("Misclassified:")
        for label, expected, actual in result.misclassified:
            print(f"  {label}: expected={expected} actual={actual}")

    assert result.precision >= PRECISION_TARGET
    assert result.recall >= RECALL_TARGET

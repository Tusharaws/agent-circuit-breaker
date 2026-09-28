"""Tests for the cheap pre-filter heuristic (Phase 4). Written before the
implementation (evaluator.prefilter does not exist yet) -- run `pytest`
to see them fail with a collection error until
src/evaluator/prefilter.py exists.
"""
from datetime import datetime, timezone

import pytest

from schemas.trace_event import TraceEvent

from evaluator.prefilter import should_escalate_to_slm
from evaluator.window import DetectionWindow


def _event(step_index, event_type="tool_start", payload=None):
    return TraceEvent(
        trace_id="thread-1",
        agent_id="agent-1",
        step_index=step_index,
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        payload=payload or {"tool": "search", "args": {"query": f"query-{step_index}"}},
    )


def _window(events):
    return DetectionWindow(thread_id="thread-1", events=events)


# ---------------------------------------------------------------------------
# Basic escalation logic
# ---------------------------------------------------------------------------


def test_window_with_no_repeats_is_filtered_out():
    # genuinely distinct actions -- different tools each step, not just
    # different args to the same tool (that's a different, intentional
    # case -- see test_repeated_tool_with_varied_args_is_still_escalated)
    window = _window(
        [
            _event(0, event_type="tool_start", payload={"tool": "search"}),
            _event(1, event_type="tool_start", payload={"tool": "write_file"}),
            _event(2, event_type="tool_start", payload={"tool": "run_tests"}),
            _event(3, event_type="tool_start", payload={"tool": "deploy"}),
            _event(4, event_type="tool_start", payload={"tool": "notify"}),
        ]
    )
    assert should_escalate_to_slm(window) is False


def test_window_with_three_identical_tool_calls_is_escalated():
    repeated_payload = {"tool": "search", "args": {"query": "same query"}}
    window = _window(
        [_event(0, payload=repeated_payload), _event(1, payload=repeated_payload), _event(2, payload=repeated_payload)]
    )
    assert should_escalate_to_slm(window) is True


def test_two_identical_calls_below_min_repeats_is_not_escalated():
    repeated_payload = {"tool": "search", "args": {"query": "same query"}}
    window = _window(
        [_event(0, payload=repeated_payload), _event(1, payload=repeated_payload), _event(2, payload={"tool": "other"})]
    )
    assert should_escalate_to_slm(window) is False


def test_semantic_only_loop_with_zero_exact_repeats_is_now_escalated():
    """The bug this task's fix closes: same tool called repeatedly with
    paraphrased args and differently-worded results each time -- NOT one
    byte-exact repeat anywhere. Previously (full-payload fingerprint)
    this was silently filtered out and never reached the SLM at all,
    defeating the reason the SLM exists (ARCHITECTURE_NOTES.md: "misses
    loops where the arguments or generated text drift slightly")."""
    window = _window(
        [
            _event(0, payload={"tool": "search", "args": {"query": "fix json parse error"}}),
            _event(1, event_type="tool_end", payload={"output": "found 0 relevant results for json parse error"}),
            _event(2, payload={"tool": "search", "args": {"query": "resolve json parsing issue"}}),
            _event(3, event_type="tool_end", payload={"output": "found 0 relevant results for json parsing issue"}),
            _event(4, payload={"tool": "search", "args": {"query": "how to handle json error"}}),
            _event(5, event_type="tool_end", payload={"output": "found 0 relevant results for json error handling"}),
        ]
    )
    assert should_escalate_to_slm(window) is True


def test_repeated_tool_with_varied_args_is_still_escalated_even_if_legitimate():
    """Intentional trade-off, not a bug: pagination-style legitimate work
    (same tool, clearly different sub-tasks/args each time) also
    escalates. False positives here are cheap -- the SLM gets a chance to
    correctly clear them by seeing the actual varying content; a false
    negative (never reaching the SLM) is unrecoverable."""
    window = _window(
        [
            _event(0, payload={"tool": "list_items", "args": {"page": 1}}),
            _event(1, payload={"tool": "list_items", "args": {"page": 2}}),
            _event(2, payload={"tool": "list_items", "args": {"page": 3}}),
        ]
    )
    assert should_escalate_to_slm(window) is True


def test_custom_min_repeats_threshold():
    repeated_payload = {"tool": "search", "args": {"query": "same"}}
    window = _window([_event(0, payload=repeated_payload), _event(1, payload=repeated_payload)])

    assert should_escalate_to_slm(window, min_repeats=2, min_window_size=2) is True
    assert should_escalate_to_slm(window, min_repeats=3, min_window_size=2) is False


# ---------------------------------------------------------------------------
# Window-size edge cases
# ---------------------------------------------------------------------------


def test_empty_window_is_filtered():
    assert should_escalate_to_slm(_window([])) is False


def test_window_below_min_window_size_is_filtered_regardless_of_content():
    repeated_payload = {"tool": "search", "args": {"query": "same"}}
    window = _window([_event(0, payload=repeated_payload), _event(1, payload=repeated_payload)])

    assert should_escalate_to_slm(window, min_window_size=3) is False


# ---------------------------------------------------------------------------
# The AC's literal target: >=80% filter rate on a normal-run test set
# ---------------------------------------------------------------------------


def _normal_window(seed):
    """A realistic non-looping window: distinct tool/step content."""
    return _window(
        [
            _event(0, event_type="node_enter", payload={"node": "planner"}),
            _event(1, event_type="llm_start", payload={"prompts": [f"analyze case {seed}"]}),
            _event(2, event_type="llm_end", payload={"response": [f"plan for case {seed}"]}),
            _event(3, event_type="tool_start", payload={"tool": "search", "args": {"query": f"lookup-{seed}"}}),
            _event(4, event_type="tool_end", payload={"output": f"result-{seed}"}),
        ]
    )


NORMAL_WINDOWS = [_normal_window(i) for i in range(20)]

LOOPING_WINDOWS = [
    _window(
        [
            _event(i, event_type="tool_start", payload={"tool": "search", "args": {"query": "stuck query"}})
            for i in range(5)
        ]
    )
    for _ in range(5)
]


def test_at_least_80_percent_of_normal_windows_are_filtered_out():
    filtered = [w for w in NORMAL_WINDOWS if not should_escalate_to_slm(w)]
    rate = len(filtered) / len(NORMAL_WINDOWS)
    assert rate >= 0.80


def test_all_looping_windows_are_correctly_escalated_not_filtered():
    """The pre-filter must never filter out an actual loop -- it only
    exists to skip the SLM call for the OBVIOUSLY-not-a-loop case."""
    escalated = [w for w in LOOPING_WINDOWS if should_escalate_to_slm(w)]
    assert len(escalated) == len(LOOPING_WINDOWS)

from dataclasses import dataclass
from datetime import datetime, timezone

from queue_client.client import QueueClient
from schemas.trace_event import TraceEvent

from evaluator.pipeline import evaluate_thread


@dataclass
class LabeledTrace:
    label: str
    events: list[TraceEvent]
    expected_is_loop: bool


def _e(step: int, event_type: str, payload: dict) -> TraceEvent:
    return TraceEvent(
        trace_id="eval",
        agent_id="eval-agent",
        step_index=step,
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Known-good traces (expected_is_loop=False) -- includes cases deliberately
# designed to look suspicious to a naive heuristic (same tool called more
# than once) but are genuinely progressing work, to stress-test that the
# pipeline doesn't just flag "same tool repeated" as a loop.
# ---------------------------------------------------------------------------

KNOWN_GOOD_TRACES = [
    LabeledTrace(
        "simple_qa",
        [
            _e(0, "llm_start", {"prompts": ["what is 2+2"]}),
            _e(1, "llm_end", {"response": ["4"]}),
        ],
        False,
    ),
    LabeledTrace(
        "search_then_answer",
        [
            _e(0, "node_enter", {"node": "planner"}),
            _e(1, "tool_start", {"tool": "search", "args": {"query": "capital of france"}}),
            _e(2, "tool_end", {"output": "Paris"}),
            _e(3, "llm_end", {"response": ["The capital of France is Paris"]}),
        ],
        False,
    ),
    LabeledTrace(
        "multi_tool_pipeline",
        [
            _e(0, "tool_start", {"tool": "fetch_data", "args": {}}),
            _e(1, "tool_end", {"output": "raw data"}),
            _e(2, "tool_start", {"tool": "transform_data", "args": {}}),
            _e(3, "tool_end", {"output": "transformed data"}),
            _e(4, "tool_start", {"tool": "save_data", "args": {}}),
            _e(5, "tool_end", {"output": "saved"}),
        ],
        False,
    ),
    LabeledTrace(
        "pagination",
        [
            _e(0, "tool_start", {"tool": "list_items", "args": {"page": 1}}),
            _e(1, "tool_end", {"output": "items 1-10"}),
            _e(2, "tool_start", {"tool": "list_items", "args": {"page": 2}}),
            _e(3, "tool_end", {"output": "items 11-20"}),
            _e(4, "tool_start", {"tool": "list_items", "args": {"page": 3}}),
            _e(5, "tool_end", {"output": "items 21-30"}),
        ],
        False,
    ),
    LabeledTrace(
        "error_then_switch_strategy",
        [
            _e(0, "tool_start", {"tool": "primary_api", "args": {}}),
            _e(1, "tool_end", {"output": "error: rate limited"}),
            _e(2, "tool_start", {"tool": "backup_api", "args": {}}),
            _e(3, "tool_end", {"output": "success"}),
        ],
        False,
    ),
    LabeledTrace(
        "build_pipeline",
        [
            _e(0, "tool_start", {"tool": "compile", "args": {}}),
            _e(1, "tool_end", {"output": "compiled successfully"}),
            _e(2, "tool_start", {"tool": "run_tests", "args": {}}),
            _e(3, "tool_end", {"output": "all tests passed"}),
            _e(4, "tool_start", {"tool": "deploy", "args": {}}),
            _e(5, "tool_end", {"output": "deployed to production"}),
        ],
        False,
    ),
    LabeledTrace(
        "conversation",
        [
            _e(0, "llm_end", {"response": ["Hello, how can I help?"]}),
            _e(1, "llm_start", {"prompts": ["What's the weather like today"]}),
            _e(2, "llm_end", {"response": ["It's sunny today"]}),
        ],
        False,
    ),
    LabeledTrace(
        "refine_search",
        [
            _e(0, "tool_start", {"tool": "search", "args": {"query": "python error"}}),
            _e(1, "tool_end", {"output": "too many results, please narrow your search"}),
            _e(2, "tool_start", {"tool": "search", "args": {"query": "python TypeError line 42 NoneType"}}),
            _e(3, "tool_end", {"output": "found: NoneType has no attribute 'foo', check initialization"}),
        ],
        False,
    ),
    LabeledTrace(
        "distinct_subtasks",
        [
            _e(0, "tool_start", {"tool": "create_file", "args": {"path": "a.txt"}}),
            _e(1, "tool_end", {"output": "created a.txt"}),
            _e(2, "tool_start", {"tool": "create_file", "args": {"path": "b.txt"}}),
            _e(3, "tool_end", {"output": "created b.txt"}),
        ],
        False,
    ),
    LabeledTrace(
        "debug_session",
        [
            _e(0, "tool_start", {"tool": "run_debugger", "args": {}}),
            _e(1, "tool_end", {"output": "breakpoint hit at line 10"}),
            _e(2, "tool_start", {"tool": "inspect_variable", "args": {"name": "x"}}),
            _e(3, "tool_end", {"output": "x = 5"}),
            _e(4, "tool_start", {"tool": "step_over", "args": {}}),
            _e(5, "tool_end", {"output": "moved to line 11"}),
        ],
        False,
    ),
    LabeledTrace(
        "translation_task",
        [
            _e(0, "llm_start", {"prompts": ["translate to french: hello"]}),
            _e(1, "llm_end", {"response": ["bonjour"]}),
            _e(2, "llm_start", {"prompts": ["translate to spanish: hello"]}),
            _e(3, "llm_end", {"response": ["hola"]}),
        ],
        False,
    ),
    LabeledTrace(
        "crawl_different_pages",
        [
            _e(0, "tool_start", {"tool": "fetch_page", "args": {"url": "site.com/page1"}}),
            _e(1, "tool_end", {"output": "links: [a, b, c]"}),
            _e(2, "tool_start", {"tool": "fetch_page", "args": {"url": "site.com/page2"}}),
            _e(3, "tool_end", {"output": "links: [d, e, f]"}),
        ],
        False,
    ),
]

# ---------------------------------------------------------------------------
# Known-loop traces (expected_is_loop=True) -- a mix of exact-repeat (the
# pre-filter alone would flag these) and pure semantic-paraphrase loops
# (zero exact repeats anywhere -- only the SLM can catch these).
# ---------------------------------------------------------------------------

KNOWN_LOOP_TRACES = [
    LabeledTrace(
        "exact_repeat_tool",
        [
            _e(i, "tool_start", {"tool": "search", "args": {"query": "same query"}}) for i in range(5)
        ],
        True,
    ),
    LabeledTrace(
        "exact_repeat_llm",
        [_e(i, "llm_end", {"response": ["I'm not sure, let me think again"]}) for i in range(4)],
        True,
    ),
    LabeledTrace(
        "paraphrased_search_loop",
        [
            _e(0, "tool_start", {"tool": "search", "args": {"query": "fix json parse error"}}),
            _e(1, "tool_end", {"output": "found 0 relevant results for json parse error"}),
            _e(2, "tool_start", {"tool": "search", "args": {"query": "resolve json parsing issue"}}),
            _e(3, "tool_end", {"output": "found 0 relevant results for json parsing issue"}),
            _e(4, "tool_start", {"tool": "search", "args": {"query": "how to handle json error"}}),
            _e(5, "tool_end", {"output": "found 0 relevant results for json error handling"}),
        ],
        True,
    ),
    LabeledTrace(
        "code_regeneration_cosmetic",
        [
            _e(0, "llm_end", {"response": ["def foo(x): return x + 1"]}),
            _e(1, "llm_end", {"response": ["def bar(y): return y + 1"]}),
            _e(2, "llm_end", {"response": ["def baz(z): return z + 1"]}),
        ],
        True,
    ),
    LabeledTrace(
        "alternating_ab_no_progress",
        [
            _e(0, "tool_start", {"tool": "tool_a", "args": {}}),
            _e(1, "tool_end", {"output": "fail"}),
            _e(2, "tool_start", {"tool": "tool_b", "args": {}}),
            _e(3, "tool_end", {"output": "fail"}),
            _e(4, "tool_start", {"tool": "tool_a", "args": {}}),
            _e(5, "tool_end", {"output": "fail"}),
        ],
        True,
    ),
    LabeledTrace(
        "retry_same_failure",
        [
            _e(i, "tool_start", {"tool": "connect_db", "args": {}}) if i % 2 == 0
            else _e(i, "tool_end", {"output": "error: connection refused"})
            for i in range(6)
        ],
        True,
    ),
    LabeledTrace(
        "repeated_replanning",
        [
            _e(0, "llm_end", {"response": ["Plan: search then summarize"]}),
            _e(1, "llm_end", {"response": ["Plan: search then summarize again"]}),
            _e(2, "llm_end", {"response": ["New plan: search and then summarize"]}),
        ],
        True,
    ),
    LabeledTrace(
        "node_reentry_same_content",
        [
            _e(0, "node_enter", {"node": "reviewer"}),
            _e(1, "llm_end", {"response": ["needs revision"]}),
            _e(2, "node_enter", {"node": "reviewer"}),
            _e(3, "llm_end", {"response": ["needs revision"]}),
            _e(4, "node_enter", {"node": "reviewer"}),
            _e(5, "llm_end", {"response": ["needs revision"]}),
        ],
        True,
    ),
    LabeledTrace(
        "near_identical_args_typo",
        [
            _e(0, "tool_start", {"tool": "search", "args": {"query": "weathr in paris"}}),
            _e(1, "tool_end", {"output": "no results"}),
            _e(2, "tool_start", {"tool": "search", "args": {"query": "weather in pari"}}),
            _e(3, "tool_end", {"output": "no results"}),
            _e(4, "tool_start", {"tool": "search", "args": {"query": "waether in paris"}}),
            _e(5, "tool_end", {"output": "no results"}),
        ],
        True,
    ),
    LabeledTrace(
        "circular_ab_no_progress",
        [
            _e(0, "node_enter", {"node": "A"}),
            _e(1, "node_exit", {"node": "A"}),
            _e(2, "node_enter", {"node": "B"}),
            _e(3, "node_exit", {"node": "B"}),
            _e(4, "node_enter", {"node": "A"}),
            _e(5, "node_exit", {"node": "A"}),
        ],
        True,
    ),
    LabeledTrace(
        "rephrasing_without_new_info",
        [
            _e(0, "llm_end", {"response": ["The error is due to a type mismatch"]}),
            _e(1, "llm_end", {"response": ["This happens because the types don't match"]}),
            _e(2, "llm_end", {"response": ["The issue is a mismatch in types"]}),
        ],
        True,
    ),
    LabeledTrace(
        "stuck_asking_clarification",
        [
            _e(0, "llm_end", {"response": ["Could you clarify what you mean?"]}),
            _e(1, "llm_end", {"response": ["I need more details to proceed"]}),
            _e(2, "llm_end", {"response": ["Can you please provide more information?"]}),
        ],
        True,
    ),
]

ALL_TRACES = KNOWN_GOOD_TRACES + KNOWN_LOOP_TRACES


@dataclass
class EvalResult:
    precision: float
    recall: float
    predictions: list[tuple[str, bool, bool]]  # (label, expected, actual)

    @property
    def misclassified(self) -> list[tuple[str, bool, bool]]:
        return [p for p in self.predictions if p[1] != p[2]]


def run_eval(queue_client: QueueClient, slm_client, window_size: int = 10) -> EvalResult:
    """Runs the REAL pipeline (evaluate_thread) against every labeled
    trace -- measuring what the product actually delivers (pre-filter +
    SLM together), not the SLM in isolation."""
    predictions = []
    for trace in ALL_TRACES:
        thread_id = f"eval-{trace.label}"
        for event in trace.events:
            queue_client.append_event(thread_id, event.model_dump(mode="json"))
        verdict = evaluate_thread(queue_client, slm_client, thread_id, window_size=window_size)
        predictions.append((trace.label, trace.expected_is_loop, verdict.is_loop))

    true_positives = sum(1 for _, expected, actual in predictions if expected and actual)
    false_positives = sum(1 for _, expected, actual in predictions if not expected and actual)
    false_negatives = sum(1 for _, expected, actual in predictions if expected and not actual)

    precision = (
        true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 1.0
    )
    recall = (
        true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 1.0
    )

    return EvalResult(precision=precision, recall=recall, predictions=predictions)

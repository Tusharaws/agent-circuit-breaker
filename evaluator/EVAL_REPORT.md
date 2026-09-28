# Evaluator Eval Set Report

Reproducible via `pytest tests/test_eval_set.py -m integration -s`.

## Methodology

24 hand-constructed traces (12 known-good, 12 known-loop — above the
AC's ≥10/≥10 minimum), run through the **real, complete pipeline**
(`evaluate_thread`: pre-filter + real SLM together), not the SLM in
isolation. This measures what the product actually delivers, since a
pre-filter false-negative and an SLM misjudgment are both real failure
modes worth catching in the same measurement.

The set deliberately includes adversarial cases on both sides:
- **Known-good** traces that look suspicious to a naive heuristic (same
  tool called more than once) but are genuinely progressing work —
  pagination, a multi-step build pipeline, creating several distinct
  files — to stress-test that the pipeline doesn't just flag "same tool
  twice" as a loop.
- **Known-loop** traces split between exact-repeat (the pre-filter alone
  would flag these) and pure semantic-paraphrase loops (zero exact
  repeats anywhere — only the SLM's judgment can catch these).

## Results

| Metric | Measured | Target |
|---|---|---|
| Precision | 0.90 | ≥0.80 |
| Recall | 0.75 | ≥0.65 |

Targets set with real margin below the observed numbers (not an exact
match), since SLM generation isn't guaranteed bit-for-bit deterministic
run to run.

## Two real bugs found and fixed while building this eval set

Neither was caught by the pre-filter task's own original tests — both
only surfaced once realistic, varied traces were constructed here.

1. **Semantic-only loops were silently filtered out entirely.** The
   original pre-filter fingerprinted the *full payload*. A real semantic
   loop — same tool repeated with paraphrased arguments and
   differently-worded results each time, zero byte-exact repeats
   anywhere — never escalated to the SLM at all, defeating the exact
   reason the SLM exists. Fixed by fingerprinting a coarser *action
   identity* (event_type + tool/node name, ignoring variable content).

2. **`tool_end` events carry no tool identity.** `interceptor`'s
   `GraphCaptureHandler.on_tool_end` only emits `{"output": ...}` — no
   tool name. Every `tool_end`, regardless of which tool actually ran,
   collapsed to the same fingerprint, causing windows with 3+ *different*
   tools' completions (e.g. compile → test → deploy) to look like one
   repeated action. Fixed by excluding `tool_end` from identity counting
   — its `tool_start` counterpart already carries the real identity
   reliably, so no signal is lost.

Precision went from 0.71 → 0.90 after fix #2 alone (recall dropped
0.83 → 0.75 as a side effect — see the documented limitation below for
why, and why that trade-off was still correct).

## Threshold-tuning experiment (this task's own mandate)

Tried lowering `min_repeats` from 3 to 2, hoping to catch two remaining
recall misses (alternating/cyclical patterns — see below). Result: it did
fix those two cases, but also caused 6 *additional* known-good traces to
escalate unnecessarily (e.g. a 3-line conversation, a translation task) —
any window with just 2 occurrences of the same event type is extremely
common in ordinary execution. Each additional escalation costs a real SLM
call, and each is a fresh chance for the small model to misjudge —
confirmed this would very likely have made precision *worse*, not
recall better. **Kept `min_repeats=3`** (the default) as the better
trade-off; this is a genuine tuning conclusion reached by testing the
alternative, not left untried.

## Documented limitation: alternating/cyclical patterns

Two known-loop traces are still missed:
- `alternating_ab_no_progress` (tool A → tool B → tool A → tool B, no
  progress)
- `circular_ab_no_progress` (node A → node B → node A → node B, no
  progress)

Neither has any *single* action-identity repeating 3+ times within the
window (each of A and B individually appears only ~2 times) — the
repetition is in the *cycle*, not in any one fingerprint. The current
pre-filter counts single-fingerprint repeats only; catching a cycle needs
subsequence/n-gram-style detection, a structurally different (and no
longer "cheap") technique. Confirmed via the threshold experiment above
that this can't be fixed by tuning `min_repeats` without unacceptable
precision cost. Flagged as a natural follow-on for a future pre-filter
iteration, not solved here.

## Not covered here

Statistical significance at this sample size (24 traces) is limited —
this is a directional eval set for catching real, gross failure modes and
guiding threshold decisions, not a rigorous accuracy benchmark. A larger,
more systematically sampled eval set would be needed for that.

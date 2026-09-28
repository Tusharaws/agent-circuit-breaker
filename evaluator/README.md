# evaluator

Semantic loop detection (Phase 4). Builds detection windows from node-level
trace events and judges whether the *meaning* of recent steps is repeating
(same underlying intent/approach recurring), not just byte-for-byte or
exact-match repetition.

## Detection window

`build_window(queue_client, thread_id, window_size=10)`
(`src/evaluator/window.py`) assembles the "window of recent steps"
(`ARCHITECTURE_NOTES.md`) the Evaluator judges — real, validated
`schemas.TraceEvent` instances, oldest to newest, not raw queue payloads.
Defined as N-steps (reusing `QueueClient.read_window`'s existing
MAXLEN-capped per-thread stream directly), not T-seconds — a time-based
window is a documented future extension, not needed to satisfy this
task's AC.

## Pre-filter heuristic

`should_escalate_to_slm(window, min_repeats=3, min_window_size=3)`
(`src/evaluator/prefilter.py`) decides whether a window is suspicious
enough to warrant the (expensive) SLM call. Deliberately **not**
embedding-similarity thresholding, since that needs its own
model/inference cost and would undercut the entire point of a *cheap*
pre-filter.

Measured on a 20-window seeded normal-run test set: **100%** filtered out
(target: ≥80%); all 5 seeded looping windows correctly escalated, not
filtered — the pre-filter must never suppress an actual loop.

**Bug found and fixed while building the eval set task:** the original
version fingerprinted the *full payload* (event_type + all args/response
text). That meant a real semantic loop — same tool called repeatedly with
paraphrased arguments and differently-worded results each time, zero
byte-exact repeats anywhere — was silently filtered out and never reached
the SLM at all, defeating the exact reason the SLM exists
(`ARCHITECTURE_NOTES.md`: "misses loops where the arguments or generated
text drift slightly"). Fixed by fingerprinting a coarser *action identity*
(event_type + tool/node name only, ignoring variable content). This
deliberately over-escalates on legitimate repeated-tool-different-work
(e.g. paginating through several pages with the same tool) — accepted
trade-off: a false positive here is cheap (the SLM gets a chance to
correctly clear it by seeing the real content); a false negative is
unrecoverable (the window never reaches the SLM at all).

## SLM

`SLMClient` (`src/evaluator/slm.py`) — **Qwen2.5-0.5B-Instruct** via
**mlx-lm** (Apple's MLX framework, native Apple Silicon acceleration).
Chosen over Phi-3-mini/Qwen2.5-1.5B for lowest resource footprint of the
considered options, and over transformers+torch/llama-cpp-python for this
hardware specifically (native acceleration, lightweight pure-Python
install, no C++ compilation step). Verified end-to-end before writing any
wrapper code: downloaded the model (~40s one-time) and ran a real
generation, coherent output confirmed. Measured real per-call latency:
**~143ms** on this machine (M4).

`model`/`tokenizer`/`generate_fn` are all constructor-injectable
(defaulting to the real `mlx_lm.load`/`generate`) — unit tests use fakes
for fast, network-independent runs; a separate test marked
`@pytest.mark.integration` exercises the real model.

## Detection prompt and verdict parsing

`build_prompt(window)` (`src/evaluator/verdict.py`) is the detection
prompt sent to the SLM: judge whether recent steps repeat the same
underlying approach without progress, even if wording/arguments vary
(semantic repetition — exact-match is the pre-filter's job). Requests a
JSON object with `is_loop`/`confidence`/`reason`.

`parse_verdict(raw_response)` is deliberately **lenient**, not a strict
schema match — grounded in real, empirically-observed model behavior, not
assumed: asked for exactly 3 keys, the real model added 2 extra
unrequested null keys; asked for `reason` to be one of 3 enum values, it
repeatedly returned free-text explanations instead (confirmed again in an
end-to-end real-model check while building this). Extracts and
independently validates/coerces each field; a free-text or invalid
`reason` falls back to `DEFAULT_REASON = "runaway_loop"`; totally
unparseable output **fails closed** to `is_loop=False` — a false positive
here means Control API halts a healthy agent run, far worse than missing
one window's detection.

`build_halt_signal(window, verdict)` constructs a real `HaltSignal`
directly from a positive verdict — `trace_id` from `window.thread_id`,
`triggering_window` from `window.event_ids` (added to `DetectionWindow` in
this task — a real gap found here: `TraceEvent` itself doesn't carry a
Redis-assigned event id, only `TraceEventRecord` does, and `build_window`
was discarding them).

## Full pipeline: consume -> pre-filter -> SLM -> verdict

`evaluate_thread(queue_client, slm_client, thread_id, window_size=10)`
(`src/evaluator/pipeline.py`) composes everything above: assembles the
window, pre-filters, and only calls the SLM if escalated — always
returns a `Verdict`, per the AC ("a verdict for every window"), including
a fast no-loop verdict for an empty/unknown thread.

Two separate latency budgets, not one blended number: **10ms** for the
filtered path (queue read + hash comparison only, no SLM) and **500ms**
for the escalated path (includes a real SLM call). Real measured
escalated-path latency: **379ms** — notably higher than the ~143ms
measured for a small standalone prompt earlier, because a realistic
6-event window's prompt is larger and uses the default `max_tokens=200`
(vs. 10 in that earlier check). Still under budget, but with real,
honestly-reported margin (~24%), not the wider margin a smaller reference
number would have implied.

"Runs on a live queue" = a real, running `QueueClient` against fakeredis
— consistent with every other test in this project (no real Redis server
exists in this environment).

## Eval set: known-good vs. known-loop traces

See [EVAL_REPORT.md](EVAL_REPORT.md) for the full report. Summary: 24
hand-constructed traces (12 known-good, 12 known-loop) run through the
real, complete pipeline — **precision 0.90, recall 0.75** (targets: ≥0.80
/ ≥0.65). Building this set found and fixed 2 real bugs in already-Done
work: semantic-only loops (zero exact repeats) were silently filtered out
entirely, and `tool_end` events collapsed to one indistinguishable
fingerprint regardless of which tool ran, causing false escalations.
Documented limitation: alternating/cyclical patterns (A→B→A→B) aren't
caught by single-fingerprint counting — a threshold-tuning experiment
confirmed this needs a structurally different technique, not a parameter
tweak.

## Wiring to Control API

`maybe_trigger_halt(window, verdict, http_client, confidence_threshold=0.7)`
(`src/evaluator/control_api_client.py`) — a loop verdict at/above the
threshold POSTs a real `HaltSignal` to Control API's `/halt` endpoint.
`confidence_threshold` is a separate, tunable safety gate on top of
`verdict.is_loop` itself — the eval set found real SLM misjudgments even
at high confidence, so this alone doesn't solve precision; it establishes
the mechanism for tuning it further.

`http_client` is injectable — a real `httpx.Client` in production,
`fastapi.testclient.TestClient` in tests. Confirmed empirically before
designing this: `TestClient` is a real, synchronous ASGI test harness
with a `.post(url, json=...)` signature compatible with `httpx.Client`'s
— tests drive control-api's *actual* app (real routing, validation, and
`HaltRegistry` write), not a mock. `control-api` is a test-only
dependency of this package; production code never imports it.

## End-to-end tracing (Phase 6)

`evaluate_thread` logs one structured JSON line per window evaluated on a
dedicated `"evaluator.trace"` logger -- INFO, not DEBUG like the
high-frequency per-event loggers elsewhere in the pipeline, since a
verdict is a meaningful, low-frequency decision point, not a per-node
pass-through: `{"trace_id": ..., "service": "evaluator", "stage":
"evaluated", "escalated": bool, "is_loop": bool, "confidence": float}`.
Same convention as interceptor/sanitizer/queue-client/control-api's own
trace loggers -- see `interceptor/README.md` for why this is a shared
convention, not a shared library.

## Install

```bash
pip install -e ".[test]"
```

## Test

```bash
pytest              # includes the real-model integration tests
pytest -m "not integration"   # unit tests only, fast, no model load
```

## Status

Phase 4 complete: detection window, pre-filter heuristic, SLM client,
detection prompt/verdict schema, the full consume->filter->SLM->verdict
pipeline, the eval set, and wiring positive verdicts to Control API's
`/halt` endpoint all implemented and tested.

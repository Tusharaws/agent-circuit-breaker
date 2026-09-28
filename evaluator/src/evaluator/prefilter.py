from collections import Counter

from evaluator.window import DetectionWindow

DEFAULT_MIN_REPEATS = 3
DEFAULT_MIN_WINDOW_SIZE = 3

# tool_end carries no tool identity in its payload today (interceptor's
# GraphCaptureHandler.on_tool_end only emits {"output": ...}) -- found via
# the eval-set task: every tool_end collapsed to the same fingerprint
# regardless of which tool actually ran, causing 3+ *different* tools'
# completions to look like one repeated action. tool_start already
# reliably carries the real identity for the same call, so excluding
# tool_end from counting loses no real signal.
_NO_RELIABLE_IDENTITY = {"tool_end"}


def _action_fingerprint(event) -> str | None:
    """A coarse signature of WHAT KIND of action this step took --
    event_type plus the acted-upon thing's identity (tool name / node
    name) -- deliberately ignoring variable content (args, query text,
    response text). Returns None for event types with no reliable
    identity signal (excluded from repetition counting entirely).

    Bug found and fixed via the eval-set task, not caught by this task's
    own original tests: fingerprinting the FULL payload (as this function
    originally did) means a real semantic loop -- same tool called
    repeatedly with paraphrased arguments and different-worded results
    each time, zero byte-exact repeats anywhere -- never escalates to the
    SLM at all. That's exactly the failure mode the SLM step exists to
    catch, per ARCHITECTURE_NOTES.md ("misses loops where the arguments
    or generated text drift slightly"). A pre-filter's job is to triage
    for the SLM, not to make the final call -- it should be biased toward
    over-escalating (false positives are cheap: the SLM gets a chance to
    correctly clear them) rather than under-escalating (false negatives
    here are unrecoverable: the window never reaches the SLM at all).
    """
    if event.event_type in _NO_RELIABLE_IDENTITY:
        return None
    identity = event.payload.get("tool") or event.payload.get("node") or ""
    return f"{event.event_type}:{identity}"


def should_escalate_to_slm(
    window: DetectionWindow,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    min_window_size: int = DEFAULT_MIN_WINDOW_SIZE,
) -> bool:
    """Cheap pre-filter heuristic (Phase 4): True only when a window looks
    suspicious enough to warrant the expensive SLM call.

    Repetition detection on a coarse action-identity fingerprint (event
    type + tool/node name) -- no embeddings/ML, kept genuinely cheap. This
    deliberately over-escalates on legitimate repeated-tool-different-work
    cases (e.g. paginating through several pages with the same tool) --
    the SLM is the backstop that tells those apart from a real loop; the
    pre-filter's only job is to not silently drop suspicious windows
    before the SLM ever sees them.
    """
    if len(window.events) < min_window_size:
        return False  # not enough history to judge

    fingerprints = [
        fp for event in window.events if (fp := _action_fingerprint(event)) is not None
    ]
    counts = Counter(fingerprints)
    most_common_count = counts.most_common(1)[0][1] if counts else 0

    return most_common_count >= min_repeats

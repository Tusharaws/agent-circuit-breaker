# queue-client

Client for the evaluation queue (Phase 3): carries trace events for the
`evaluator` to consume.

Note: this is intentionally separate from the LangGraph checkpointer, which
carries resumable graph state — the two are different kinds of "state" and
should not be conflated (see `ARCHITECTURE_NOTES.md`).

## Install

```bash
pip install -e .
```

## Status

Skeleton only — no implementation yet.

# schemas

Shared pydantic contracts used across the Agent Execution Guardian packages
(Phase 0), so producers (`evaluator`) and consumers (`control-api`) validate
against the same definition instead of each owning their own copy.

## Contents

- `HaltSignal` — the message the Evaluator emits when it detects a loop and
  the Control API's `POST /halt` expects to receive.
- `TraceEvent` — a single node-level trace event carried on the evaluation
  queue from the Interceptor to the Evaluator. `event_type` is validated
  against `config/event_types.md`, the same config-driven pattern
  `HaltSignal.reason` uses. `token_usage` and `latency_ms` are optional,
  since a `_start` event fires before either is known.

## Install

```bash
pip install -e ".[test]"
```

## Test

```bash
pytest
```

## Status

Schema under test-first development — see `tests/` for the current contract.

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
- `TenantConfig` (Phase 7) — ties together the per-tenant configuration
  knobs that already exist as independent constructor parameters across
  `queue_client` (`stream_prefix`), `sanitizer` (`policy`), and
  `control_api` (`key_prefix`). Owns no behavior itself, just a shared data
  shape -- lives here rather than a new package since every other package
  already depends on `schemas`. Does *not* cover control-api's `/halt`
  auth (still single-tenant, one token per `create_app()` call) or
  dashboard tenant-scoping -- both documented as separate, unbuilt gaps,
  not silently assumed solved.

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

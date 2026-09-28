"""Phase 6: end-to-end tracing/logging -- control-api's contribution.

check_halt logs once per call, on a dedicated "control_api.trace" logger
(DEBUG -- it's called on every guard-node execution, a high-frequency
check, not a low-frequency decision like an actual halt). The existing
"control_api.halt_audit" log (test_app.py) already covers the actual halt
decision itself with full context; this is the lower-level "was this
thread checked" trace line that makes the guard-node's own activity
followable by trace_id too.

check_halt() calling interrupt() outside a real LangGraph runtime raises a
plain RuntimeError (confirmed empirically in test_guard.py) -- the trace
log must fire before that, not be skipped by it.
"""
import json
import logging

import fakeredis
import pytest

from control_api.guard import HaltRegistry, check_halt


@pytest.fixture
def registry():
    return HaltRegistry(redis_client=fakeredis.FakeRedis())


def test_non_halted_check_logs_a_structured_trace_line(registry, caplog):
    with caplog.at_level(logging.DEBUG, logger="control_api.trace"):
        check_halt("thread-normal", registry)

    assert len(caplog.records) == 1
    logged = json.loads(caplog.records[0].getMessage())
    assert logged["trace_id"] == "thread-normal"
    assert logged["service"] == "control_api"
    assert logged["stage"] == "halt_checked"
    assert logged["halted"] is False


def test_halted_check_logs_before_raising(registry, caplog):
    registry.request_halt("thread-halted")

    with caplog.at_level(logging.DEBUG, logger="control_api.trace"):
        with pytest.raises(RuntimeError):
            check_halt("thread-halted", registry)

    assert len(caplog.records) == 1
    logged = json.loads(caplog.records[0].getMessage())
    assert logged["trace_id"] == "thread-halted"
    assert logged["halted"] is True

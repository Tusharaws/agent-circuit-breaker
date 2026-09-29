"""Phase 7: multi-tenant support -- two simulated customers, sharing the
SAME underlying Redis instance (the harder, more honest thing to prove:
isolation comes from the prefixing scheme itself, not from separate
physical infrastructure), each with its own TenantConfig tying together
QueueClient.stream_prefix, SanitizingSink.policy, and
HaltRegistry.key_prefix.

Both tenants use the SAME thread_id ("thread-1") deliberately -- proving
isolation holds even under an identical customer-supplied identifier is a
much stronger test than using different thread_ids per tenant, which
wouldn't collide even without any isolation mechanism at all.
"""
import fakeredis
from control_api.guard import HaltRegistry
from interceptor.queue_sink import make_queue_sink
from queue_client.client import QueueClient
from sanitizer.pipeline import SanitizingSink
from schemas.tenant import TenantConfig

SHARED_THREAD_ID = "thread-1"

TENANT_A = TenantConfig(
    tenant_id="acme-corp",
    stream_prefix="acme_corp_events",
    redaction_policy={"email": "mask"},
    halt_key_prefix="acme_corp_halt",
)
TENANT_B = TenantConfig(
    tenant_id="globex",
    stream_prefix="globex_events",
    redaction_policy={"email": "drop"},
    halt_key_prefix="globex_halt",
)


def _build_tenant_pipeline(tenant: TenantConfig, redis_client):
    queue_client = QueueClient(redis_client=redis_client, stream_prefix=tenant.stream_prefix)
    sink = SanitizingSink(inner=make_queue_sink(queue_client.append_event), policy=tenant.redaction_policy)
    registry = HaltRegistry(redis_client=redis_client, key_prefix=tenant.halt_key_prefix)
    return queue_client, sink, registry


def _seed_event(sink, thread_id, payload):
    from datetime import datetime, timezone

    from schemas.trace_event import TraceEvent

    sink(
        TraceEvent(
            trace_id=thread_id,
            agent_id="agent-1",
            step_index=0,
            timestamp=datetime.now(timezone.utc),
            event_type="tool_start",
            payload=payload,
        )
    )


def test_two_tenants_events_never_cross_into_each_others_stream():
    shared_redis = fakeredis.FakeRedis()
    queue_a, sink_a, _registry_a = _build_tenant_pipeline(TENANT_A, shared_redis)
    queue_b, sink_b, _registry_b = _build_tenant_pipeline(TENANT_B, shared_redis)

    _seed_event(sink_a, SHARED_THREAD_ID, {"note": "tenant A's own data"})
    _seed_event(sink_b, SHARED_THREAD_ID, {"note": "tenant B's own data"})

    window_a = queue_a.read_window(SHARED_THREAD_ID)
    window_b = queue_b.read_window(SHARED_THREAD_ID)

    assert len(window_a) == 1
    assert len(window_b) == 1
    assert window_a[0].payload["payload"]["note"] == "tenant A's own data"
    assert window_b[0].payload["payload"]["note"] == "tenant B's own data"
    # Neither tenant's stream ever contains a fragment of the other's data.
    assert "tenant B" not in str(window_a)
    assert "tenant A" not in str(window_b)


def test_each_tenants_own_redaction_policy_applies_only_to_its_own_data():
    shared_redis = fakeredis.FakeRedis()
    queue_a, sink_a, _registry_a = _build_tenant_pipeline(TENANT_A, shared_redis)
    queue_b, sink_b, _registry_b = _build_tenant_pipeline(TENANT_B, shared_redis)

    pii_note = "contact us at alice@example.com"
    _seed_event(sink_a, SHARED_THREAD_ID, {"note": pii_note})
    _seed_event(sink_b, SHARED_THREAD_ID, {"note": pii_note})

    stored_a = queue_a.read_window(SHARED_THREAD_ID)[0].payload["payload"]["note"]
    stored_b = queue_b.read_window(SHARED_THREAD_ID)[0].payload["payload"]["note"]

    # Tenant A's policy masks emails -- a placeholder survives.
    assert "alice@example.com" not in stored_a
    assert "[REDACTED]" in stored_a
    # Tenant B's policy drops emails entirely -- nothing survives, not even a placeholder.
    assert "alice@example.com" not in stored_b
    assert "[REDACTED]" not in stored_b


def test_halting_one_tenants_thread_does_not_affect_the_others():
    shared_redis = fakeredis.FakeRedis()
    _queue_a, _sink_a, registry_a = _build_tenant_pipeline(TENANT_A, shared_redis)
    _queue_b, _sink_b, registry_b = _build_tenant_pipeline(TENANT_B, shared_redis)

    registry_a.request_halt(SHARED_THREAD_ID)

    assert registry_a.is_halted(SHARED_THREAD_ID) is True
    assert registry_b.is_halted(SHARED_THREAD_ID) is False

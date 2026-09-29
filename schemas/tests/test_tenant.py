"""Tests for TenantConfig (Phase 7 multi-tenant task). Written before the
implementation (schemas.tenant does not exist yet) -- run `pytest` to see
them fail with a collection error until src/schemas/tenant.py exists.

TenantConfig ties together per-tenant configuration knobs that already
exist as independent constructor parameters across queue_client
(stream_prefix), sanitizer (policy), and control_api (key_prefix) -- it
owns no behavior itself, just a shared data shape, which is why it lives
in schemas (every other package already depends on schemas) rather than a
new package.
"""
import pytest
from pydantic import ValidationError

from schemas.tenant import TenantConfig

VALID_KWARGS = {
    "tenant_id": "acme-corp",
    "stream_prefix": "acme_corp_events",
    "redaction_policy": {"email": "mask", "ssn": "drop"},
    "halt_key_prefix": "acme_corp_halt",
}


def test_valid_tenant_config_round_trips_all_fields():
    config = TenantConfig(**VALID_KWARGS)

    assert config.tenant_id == "acme-corp"
    assert config.stream_prefix == "acme_corp_events"
    assert config.redaction_policy == {"email": "mask", "ssn": "drop"}
    assert config.halt_key_prefix == "acme_corp_halt"


@pytest.mark.parametrize("field", ["tenant_id", "stream_prefix", "halt_key_prefix"])
def test_empty_string_fields_are_rejected(field):
    with pytest.raises(ValidationError):
        TenantConfig(**{**VALID_KWARGS, field: ""})


def test_redaction_policy_can_be_empty_dict():
    """An empty policy is valid -- it just means every data type falls
    back to sanitizer's own default action, not a config error."""
    config = TenantConfig(**{**VALID_KWARGS, "redaction_policy": {}})
    assert config.redaction_policy == {}


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        TenantConfig(**{**VALID_KWARGS, "unexpected_field": "value"})


def test_two_tenants_are_independent_objects_with_different_values():
    tenant_a = TenantConfig(**VALID_KWARGS)
    tenant_b = TenantConfig(
        tenant_id="globex",
        stream_prefix="globex_events",
        redaction_policy={"email": "drop"},
        halt_key_prefix="globex_halt",
    )

    assert tenant_a.stream_prefix != tenant_b.stream_prefix
    assert tenant_a.halt_key_prefix != tenant_b.halt_key_prefix
    assert tenant_a.redaction_policy != tenant_b.redaction_policy

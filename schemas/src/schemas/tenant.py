from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonEmptyStr = Annotated[str, Field(min_length=1)]


class TenantConfig(BaseModel):
    """Ties together the per-tenant configuration knobs that already exist
    as independent constructor parameters across `queue_client`
    (`QueueClient.stream_prefix`), `sanitizer` (`SanitizingSink.policy`),
    and `control_api` (`HaltRegistry.key_prefix`) -- Phase 7's multi-tenant
    task. Wiring one tenant's pipeline becomes one lookup instead of
    remembering to pass 3 separate values consistently everywhere.

    Lives in `schemas`, not a new package: every other package already
    depends on `schemas`, and this is purely a shared data shape, not
    behavior -- a new package/dependency for it would be disproportionate.

    Deliberately does NOT include an API token/auth scope for control-api's
    `/halt` endpoint -- that endpoint is still single-tenant (one token per
    `create_app()` call), and making it tenant-aware is a distinct, separate
    gap not built here (see this task's own board record).
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: NonEmptyStr
    stream_prefix: NonEmptyStr
    redaction_policy: dict[str, str]
    halt_key_prefix: NonEmptyStr

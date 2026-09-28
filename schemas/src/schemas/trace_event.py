from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_EVENT_TYPES_FILE = Path(__file__).parent / "config" / "event_types.md"


def load_allowed_event_types(path: Path | None = None) -> list[str]:
    """Parse bulleted event_type entries (`- event_type_name`) out of a
    markdown config file, so valid `event_type` values can be added/removed
    by editing that file instead of this module."""
    target = path if path is not None else _EVENT_TYPES_FILE
    event_types = []
    for line in target.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            event_type = stripped[1:].strip()
            if event_type:
                event_types.append(event_type)
    return event_types


NonEmptyStr = Annotated[str, Field(min_length=1)]


class TokenUsage(BaseModel):
    """Structured token breakdown for an event that reports LLM usage."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _total_matches_sum(self) -> "TokenUsage":
        expected = self.prompt_tokens + self.completion_tokens
        if self.total_tokens != expected:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) must equal "
                f"prompt_tokens + completion_tokens ({expected})"
            )
        return self


class TraceEvent(BaseModel):
    """A single node-level trace event (Phase 0). Carried on the evaluation
    queue from the Interceptor to the Evaluator (see ARCHITECTURE_NOTES.md)."""

    model_config = ConfigDict(extra="forbid")

    trace_id: NonEmptyStr
    agent_id: NonEmptyStr
    step_index: int = Field(ge=0)
    timestamp: datetime
    event_type: NonEmptyStr
    payload: dict[str, Any]
    token_usage: TokenUsage | None = None
    latency_ms: float | None = Field(default=None, ge=0)

    @field_validator("event_type")
    @classmethod
    def _event_type_must_be_allowed(cls, value: str) -> str:
        allowed = load_allowed_event_types()
        if value not in allowed:
            raise ValueError(f"event_type must be one of {allowed}, got {value!r}")
        return value

    @field_validator("timestamp")
    @classmethod
    def _require_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

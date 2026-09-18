from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

_HALT_REASONS_FILE = Path(__file__).parent / "config" / "halt_reasons.md"


def load_allowed_reasons(path: Path | None = None) -> list[str]:
    """Parse bulleted reason entries (`- reason_name`) out of a markdown
    config file, so valid `reason` values can be added/removed by editing
    that file instead of this module."""
    target = path if path is not None else _HALT_REASONS_FILE
    reasons = []
    for line in target.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            reason = stripped[1:].strip()
            if reason:
                reasons.append(reason)
    return reasons


NonEmptyStr = Annotated[str, Field(min_length=1)]


class HaltSignal(BaseModel):
    """Message the Evaluator emits when it detects a loop; the contract
    Control API's `POST /halt` validates incoming requests against."""

    model_config = ConfigDict(extra="forbid")

    trace_id: NonEmptyStr
    reason: NonEmptyStr
    confidence: float = Field(ge=0.0, le=1.0)
    triggering_window: list[NonEmptyStr] = Field(min_length=1)
    timestamp: datetime

    @field_validator("reason")
    @classmethod
    def _reason_must_be_allowed(cls, value: str) -> str:
        allowed = load_allowed_reasons()
        if value not in allowed:
            raise ValueError(f"reason must be one of {allowed}, got {value!r}")
        return value

    @field_validator("timestamp")
    @classmethod
    def _require_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

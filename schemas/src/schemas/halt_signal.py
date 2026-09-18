from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HaltReason(str, Enum):
    REPEATED_TOOL_CALLS = "repeated_tool_calls"
    REPEATED_GENERATION = "repeated_generation"
    RUNAWAY_LOOP = "runaway_loop"


NonEmptyStr = Annotated[str, Field(min_length=1)]


class HaltSignal(BaseModel):
    """Message the Evaluator emits when it detects a loop; the contract
    Control API's `POST /halt` validates incoming requests against."""

    model_config = ConfigDict(extra="forbid")

    trace_id: NonEmptyStr
    reason: HaltReason
    confidence: float = Field(ge=0.0, le=1.0)
    triggering_window: list[NonEmptyStr] = Field(min_length=1)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def _require_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

"""Memory schemas for active recall."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    DECISION_REFLECTION = "decision_reflection"
    INSTRUMENT_LESSON = "instrument_lesson"
    RISK_RULE = "risk_rule"
    DATA_CAVEAT = "data_caveat"
    USER_PREFERENCE = "user_preference"
    EXTERNAL_REFERENCE = "external_reference"


class MemoryItem(BaseModel):
    id: str
    type: str
    scope: str = "global"
    ticker: str | None = None
    market: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    last_verified_at: str | None = None
    summary: str = ""
    stale_after_days: int | None = 30
    path: str | None = None
    content: str = ""
    is_stale: bool = False
    stale_warning: str = ""


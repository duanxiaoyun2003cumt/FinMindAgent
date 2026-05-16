"""Event and audit-friendly observation models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from finmindagent.runtime.actions import AgentAction


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventType(str, Enum):
    LLM_CALL = "llm_call"
    ACTION = "action"
    TOOL_CALL = "tool_call"
    AGENT_CALL = "agent_call"
    OBSERVATION = "observation"
    PERMISSION = "permission"
    COMPACT = "compact"
    MEMORY = "memory"
    ERROR = "error"
    FINAL = "final"


class RuntimeEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    step: int
    type: EventType
    actor: str = "runtime"
    action: AgentAction | None = None
    observation: Any = None
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utc_now)


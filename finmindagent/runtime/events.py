"""Event and audit-friendly observation models."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from finmindagent.runtime.actions import AgentAction
from finmindagent.time_utils import system_timestamp


def system_now() -> str:
    """ISO-8601 system/runtime timestamp in Asia/Shanghai (+08:00).

    The system clock for RuntimeEvent / permission audit / tool results —
    always timezone-aware with an explicit offset.
    """
    return system_timestamp()


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
    # Resolved at call time so the module-level system clock can be
    # substituted in tests; behaviorally identical to system_now directly.
    timestamp: str = Field(default_factory=lambda: system_now())


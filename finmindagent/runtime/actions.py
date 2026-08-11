"""Structured action schema for the FinMindAgent runtime loop."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    CALL_AGENT = "call_agent"
    CALL_TOOL = "call_tool"
    WRITE_REPORT = "write_report"
    DEBATE_TURN = "debate_turn"
    COMPACT_CONTEXT = "compact_context"
    FINALIZE_DECISION = "finalize_decision"
    ASK_CONFIRMATION = "ask_confirmation"
    NOOP = "noop"


class AgentAction(BaseModel):
    type: ActionType
    rationale: str | None = None
    target_agent: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    requires_evidence: bool = True

    @classmethod
    def parse_action(cls, raw: Any) -> "AgentAction":
        """Parse an action from a model object, JSON string, or dict."""
        if isinstance(raw, AgentAction):
            return raw
        if isinstance(raw, BaseModel):
            raw = raw.model_dump()
        if isinstance(raw, dict):
            return cls(**raw)
        content = getattr(raw, "content", raw)
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str):
            raise ValueError(f"Cannot parse action from {type(raw)!r}")
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()
        return cls(**json.loads(content))


def repair_action(raw: Any) -> AgentAction:
    """Best-effort action repair for malformed model output."""
    text = getattr(raw, "content", raw)
    if not isinstance(text, str):
        raise ValueError("Cannot repair non-text action output")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in action output")
    return AgentAction.parse_action(text[start : end + 1])


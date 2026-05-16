"""Structured tool result wrappers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from finmindagent.runtime.events import utc_now


class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    data: str | dict | list | None = None
    error: str | None = None
    source: str | None = None
    timestamp: str = Field(default_factory=utc_now)
    truncated: bool = False
    artifact_path: str | None = None
    token_estimate: int | None = None

    @classmethod
    def success(
        cls,
        tool_name: str,
        data: Any,
        source: str | None = None,
    ) -> "ToolResult":
        return cls(ok=True, tool_name=tool_name, data=data, source=source)

    @classmethod
    def failure(cls, tool_name: str, error: str, source: str | None = None) -> "ToolResult":
        return cls(ok=False, tool_name=tool_name, error=error, source=source)


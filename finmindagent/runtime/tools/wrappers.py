"""Helpers for executing arbitrary tools with structured results."""

from __future__ import annotations

from typing import Any, Callable

from finmindagent.runtime.tools.adapters import adapt_tool_data
from finmindagent.runtime.tools.result import ToolResult


def tool_display_name(tool: Callable[..., Any]) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", tool.__class__.__name__)


def invoke_tool(tool: Callable[..., Any], tool_name: str, tool_args: dict[str, Any]) -> ToolResult:
    try:
        if hasattr(tool, "invoke"):
            data = tool.invoke(tool_args)
        else:
            data = tool(**tool_args)
        result = ToolResult.success(tool_name=tool_name, data=data, source=tool_name)
        # Parse known vendor formats into a deterministic structured payload.
        # ``data`` keeps the raw text; truncation budgets below only affect it.
        result.structured_data = adapt_tool_data(tool_name, data)
        return result
    except Exception as exc:
        return ToolResult.failure(tool_name=tool_name, error=f"{type(exc).__name__}: {exc}", source=tool_name)


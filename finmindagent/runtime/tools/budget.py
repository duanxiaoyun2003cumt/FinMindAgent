"""Output budgets for runtime tool results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finmindagent.runtime.tools.result import ToolResult


DEFAULT_CHAR_BUDGETS = {
    "news": 20_000,
    "social": 20_000,
    "search": 20_000,
    "fundamental": 30_000,
    "financial": 30_000,
    "market": 30_000,
    "stock": 30_000,
}


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return max(1, len(text) // 4)


class ToolBudgetManager:
    def __init__(self, artifact_dir: str | Path, budgets: dict[str, int] | None = None):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.budgets = budgets or DEFAULT_CHAR_BUDGETS.copy()

    def budget_for(self, tool_name: str) -> int:
        lowered = tool_name.lower()
        for key, value in self.budgets.items():
            if key in lowered:
                return value
        return 20_000

    def apply(self, result: ToolResult, run_id: str, step: int) -> ToolResult:
        if not result.ok or result.data is None:
            result.token_estimate = estimate_tokens(result.error or "")
            return result

        data_text = (
            result.data
            if isinstance(result.data, str)
            else json.dumps(result.data, ensure_ascii=False, indent=2, default=str)
        )
        limit = self.budget_for(result.tool_name)
        if isinstance(result.data, list) and "search" in result.tool_name.lower() and len(result.data) > 250:
            full = data_text
            result.data = result.data[:250]
            result.truncated = True
        elif len(data_text) > limit:
            full = data_text
            preview = data_text[:limit]
            result.data = preview + "\n\n[TRUNCATED: full output saved as artifact]"
            result.truncated = True
        else:
            result.token_estimate = estimate_tokens(data_text)
            return result

        path = self.artifact_dir / f"{run_id}_step_{step}_{result.tool_name}.txt"
        path.write_text(full, encoding="utf-8")
        result.artifact_path = str(path)
        result.token_estimate = estimate_tokens(result.data)
        return result


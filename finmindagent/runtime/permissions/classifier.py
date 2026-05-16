"""Optional LLM permission classifier.

The rule engine is authoritative. This classifier is only used for ambiguous
actions and fails closed when parsing or invocation fails.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from finmindagent.runtime.actions import AgentAction
from finmindagent.runtime.permissions.rules import PermissionCheckResult, PermissionDecision


class ClassifierDecision(BaseModel):
    decision: PermissionDecision
    reason: str


class LLMPermissionClassifier:
    def __init__(self, llm: Any | None = None):
        self.llm = llm

    def check(
        self,
        user_request: str,
        action: AgentAction,
        tool_schema: dict[str, Any] | None,
        state_summary: str,
        permission_mode: str,
    ) -> PermissionCheckResult:
        if self.llm is None:
            return PermissionCheckResult(PermissionDecision.CONFIRM, "No classifier configured.")
        prompt = {
            "user_request": user_request,
            "action": action.model_dump(mode="json"),
            "tool_schema": tool_schema or {},
            "state_summary": state_summary,
            "permission_mode": permission_mode,
            "instruction": "Return JSON with decision allow/deny/confirm and reason.",
        }
        try:
            raw = self.llm.invoke(str(prompt))
            content = getattr(raw, "content", raw)
            decision = ClassifierDecision.model_validate_json(content)
            return PermissionCheckResult(decision.decision, decision.reason)
        except Exception:
            return PermissionCheckResult(PermissionDecision.CONFIRM, "Classifier failed closed.")


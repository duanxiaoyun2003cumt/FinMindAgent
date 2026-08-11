"""Code-level runtime permission manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.permissions.audit import PermissionAuditLog
from finmindagent.runtime.permissions.classifier import LLMPermissionClassifier
from finmindagent.runtime.permissions.rules import (
    PermissionCheckResult,
    PermissionDecision,
    PermissionMode,
    action_risk_level,
    actor_for_action,
    check_agent_tool_scope,
    hard_deny_reason,
)


class PermissionManager:
    def __init__(
        self,
        mode: PermissionMode | str = PermissionMode.SAFE,
        audit_log: PermissionAuditLog | str | Path | None = None,
        classifier: LLMPermissionClassifier | None = None,
    ):
        self.mode = PermissionMode(mode)
        self.audit_log = (
            audit_log
            if isinstance(audit_log, PermissionAuditLog)
            else PermissionAuditLog(audit_log)
        )
        self.classifier = classifier

    def check(
        self,
        action: AgentAction,
        state: Any,
        actor: str | None = None,
        tool_risk: str | None = None,
        tool_schema: dict[str, Any] | None = None,
    ) -> PermissionCheckResult:
        actor = actor or actor_for_action(action)
        reason = hard_deny_reason(action)
        if reason:
            return self._record(state, action, actor, PermissionDecision.DENY, reason)

        scope_error = check_agent_tool_scope(action, actor)
        if scope_error:
            return self._record(state, action, actor, PermissionDecision.DENY, scope_error)

        if action.type == ActionType.CALL_TOOL:
            risk = action_risk_level(action, tool_risk)
            if self.mode == PermissionMode.STRICT and risk != "low":
                return self._record(
                    state, action, actor, PermissionDecision.DENY, "Strict mode allows only low-risk read-only actions."
                )
            if risk == "high" and self.mode != PermissionMode.AUTO:
                return self._record(
                    state, action, actor, PermissionDecision.CONFIRM, "High-risk tool action requires confirmation."
                )

        if self.classifier and action.type == ActionType.CALL_TOOL:
            classified = self.classifier.check(
                user_request=getattr(state, "user_request", ""),
                action=action,
                tool_schema=tool_schema,
                state_summary=f"ticker={getattr(state, 'ticker', '')}; step={getattr(state, 'step_count', '')}",
                permission_mode=self.mode.value,
            )
            if classified.decision != PermissionDecision.ALLOW:
                return self._record(state, action, actor, classified.decision, classified.reason)

        # Internal analytical finalization: writes the already-generated
        # PortfolioDecision into runtime state — no tool, no broker, no
        # exchange, no external side effect. Capability-based reason, never
        # a payload self-declaration.
        if action.type == ActionType.FINALIZE_DECISION:
            reason = "Internal analytical finalization; no external trading side effect."
        else:
            reason = "Allowed by runtime permission rules."
        return self._record(state, action, actor, PermissionDecision.ALLOW, reason)

    def _record(
        self,
        state: Any,
        action: AgentAction,
        actor: str,
        decision: PermissionDecision,
        reason: str,
    ) -> PermissionCheckResult:
        self.audit_log.write(
            {
                "run_id": getattr(state, "run_id", ""),
                "step": getattr(state, "step_count", 0),
                "agent": actor,
                "action_type": action.type.value,
                "tool_name": action.tool_name,
                "decision": decision.value,
                "reason": reason,
            }
        )
        return PermissionCheckResult(decision, reason)


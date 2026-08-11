"""Hard permission rules for runtime actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from finmindagent.runtime.actions import ActionType, AgentAction


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


class PermissionMode(str, Enum):
    SAFE = "safe"
    AUTO = "auto"
    STRICT = "strict"


@dataclass(frozen=True)
class PermissionCheckResult:
    decision: PermissionDecision
    reason: str

    @property
    def denied(self) -> bool:
        return self.decision == PermissionDecision.DENY

    @property
    def requires_confirmation(self) -> bool:
        return self.decision == PermissionDecision.CONFIRM


REAL_TRADING_TERMS = (
    "broker",
    "order",
    "exchange",
    "execute_trade",
    "place_order",
    "submit_order",
    "filled",
    "fill_order",
    "live_trade",
)

FORBIDDEN_TEXT = (
    "已下单",
    "已成交",
    "订单号",
    "executed trade",
    "order filled",
    "placed order",
)


def hard_deny_reason(action: AgentAction) -> str | None:
    """Hard denial based on structured capability fields only.

    Content/capability decoupling: ``payload`` and ``rationale`` are free
    text — research reports legitimately contain "buy / sell / limit order /
    stop loss / profit taking" vocabulary — and free text carries no
    executable side-effect capability. Hard denies therefore inspect only
    the structured capability fields (tool identity, tool arguments, target
    agent), which is where real external side effects live.

    A ``finalize_decision`` whose payload describes a trading strategy is an
    internal analytical finalization (writes the decision into runtime
    state; no broker, no order, no exchange), never a real trade execution.
    """
    text = " ".join(
        [
            action.tool_name or "",
            action.target_agent or "",
            str(action.tool_args),
        ]
    ).lower()
    if any(term in text for term in REAL_TRADING_TERMS):
        return "Real trading, broker, order, or exchange actions are forbidden."
    if ".env" in text or "api_key" in text or "credential" in text:
        return "Credential, .env, and system configuration writes are forbidden."
    if "shell" in text or "subprocess" in text or "powershell" in text:
        return "Arbitrary shell execution is forbidden."
    if "delete" in text and ("file" in text or "source" in text or "git" in text):
        return "Deleting user files or source control state is forbidden."
    if any(term in text for term in FORBIDDEN_TEXT):
        return "Misleading real-trade execution language is forbidden."
    return None


AGENT_TOOL_ALLOWLIST: dict[str, set[str]] = {
    "market_analyst": {"get_stock_data", "get_indicators"},
    "social_sentiment_analyst": {"get_news", "get_holder_context"},
    "social_analyst": {"get_news"},
    "news_analyst": {"get_news", "get_global_news", "get_insider_transactions", "get_company_calendar"},
    "fundamentals_analyst": {
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_analyst_expectations",
        "get_earnings_history",
    },
    "bull_researcher": set(),
    "bear_researcher": set(),
    "research_manager": set(),
    "trader": set(),
    "aggressive_risk_analyst": set(),
    "conservative_risk_analyst": set(),
    "neutral_risk_analyst": set(),
    "portfolio_manager": set(),
    "risk_verifier": set(),
    "data_quality_verifier": set(),
    "memory_extractor": set(),
    "leader": {
        "get_stock_data",
        "get_indicators",
        "get_news",
        "get_global_news",
        "get_insider_transactions",
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_analyst_expectations",
        "get_earnings_history",
        "get_holder_context",
        "get_company_calendar",
    },
}


def actor_for_action(action: AgentAction, default: str = "leader") -> str:
    return str(action.payload.get("agent") or action.target_agent or default)


def check_agent_tool_scope(action: AgentAction, actor: str) -> str | None:
    if action.type != ActionType.CALL_TOOL:
        return None
    if not action.tool_name:
        return "Tool action missing tool_name."
    allowed = AGENT_TOOL_ALLOWLIST.get(actor)
    if allowed is None:
        return f"Unknown agent permission scope: {actor}."
    if action.tool_name not in allowed:
        return f"{actor} is not allowed to call {action.tool_name}."
    return None


def action_risk_level(action: AgentAction, tool_risk: str | None = None) -> str:
    if action.type in {ActionType.FINALIZE_DECISION, ActionType.NOOP, ActionType.CALL_AGENT}:
        return "low"
    return tool_risk or "medium"

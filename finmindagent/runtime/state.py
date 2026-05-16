"""Runtime state models for the FinMindAgent while loop."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from finmindagent.runtime.actions import AgentAction
from finmindagent.runtime.events import EventType, RuntimeEvent


class FinMindRunState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    ticker: str
    trade_date: str
    user_request: str = ""
    step_count: int = 0
    max_steps: int = 30
    max_tool_calls: int = 30
    max_tokens: int = 120_000
    done: bool = False
    status: str = "running"
    stop_reason: str | None = None
    permission_mode: str = "safe"
    tool_call_count: int = 0
    parse_error_count: int = 0
    compact_failure_count: int = 0
    injected_memory_ids: set[str] = Field(default_factory=set)
    reports: dict[str, str] = Field(default_factory=dict)
    debate_summary: str = ""
    final_trade_decision: str = ""
    events: list[RuntimeEvent] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_event(
        self,
        event_type: EventType,
        actor: str = "runtime",
        action: AgentAction | None = None,
        observation: Any = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            run_id=self.run_id,
            step=self.step_count,
            type=event_type,
            actor=actor,
            action=action,
            observation=observation,
            message=message,
            metadata=metadata or {},
        )
        self.events.append(event)
        observer = self.metadata.get("event_observer")
        if callable(observer):
            try:
                observer(event, self)
            except Exception:
                pass
        return event

    def add_observation(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        self.add_event(EventType.OBSERVATION, observation=message, metadata=metadata)

    def as_legacy_state(self) -> dict[str, Any]:
        reports = self.reports
        investment_history = "\n\n".join(
            part
            for part in [
                reports.get("bull_researcher", ""),
                reports.get("bear_researcher", ""),
                reports.get("investment_plan", ""),
            ]
            if part
        )
        risk_history = "\n\n".join(
            part
            for part in [
                reports.get("aggressive_risk_analyst", ""),
                reports.get("conservative_risk_analyst", ""),
                reports.get("neutral_risk_analyst", ""),
                reports.get("risk_verifier", ""),
            ]
            if part
        )
        return {
            "company_of_interest": self.ticker,
            "trade_date": self.trade_date,
            "market_report": reports.get("market_report", ""),
            "sentiment_report": reports.get("sentiment_report", ""),
            "news_report": reports.get("news_report", ""),
            "fundamentals_report": reports.get("fundamentals_report", ""),
            "investment_plan": reports.get("investment_plan", ""),
            "trader_investment_plan": reports.get("trader_investment_plan", ""),
            "trader_investment_decision": reports.get("trader_investment_plan", ""),
            "final_trade_decision": self.final_trade_decision,
            "investment_debate_state": {
                "bull_history": reports.get("bull_researcher", ""),
                "bear_history": reports.get("bear_researcher", ""),
                "history": investment_history or self.debate_summary,
                "current_response": "",
                "judge_decision": reports.get("investment_plan", ""),
                "count": 0,
            },
            "risk_debate_state": {
                "aggressive_history": reports.get("aggressive_risk_analyst", ""),
                "conservative_history": reports.get("conservative_risk_analyst", ""),
                "neutral_history": reports.get("neutral_risk_analyst", ""),
                "history": risk_history or self.debate_summary,
                "latest_speaker": "Portfolio Manager" if self.final_trade_decision else "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": self.final_trade_decision,
                "count": 0,
            },
            "runtime_status": self.status,
            "runtime_stop_reason": self.stop_reason,
            "called_agents": self.metadata.get("called_agents", []),
            "missing_required_reports": self.metadata.get("missing_required_reports", []),
            "event_count": len(self.events),
            "messages": [],
            "past_context": "\n\n".join(
                str(e.observation) for e in self.events if e.type == EventType.MEMORY
            ),
        }


class FinMindRunResult(BaseModel):
    state: FinMindRunState
    final_trade_decision: str
    legacy_state: dict[str, Any]
    signal: dict[str, Any] | str | None = None
    status: str
    stop_reason: str | None = None




__all__ = [
    "FinMindRunResult",
    "FinMindRunState",
]

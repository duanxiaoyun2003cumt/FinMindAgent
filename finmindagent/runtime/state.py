"""Runtime state models for the FinMindAgent while loop."""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr

from finmindagent.runtime.actions import AgentAction
from finmindagent.runtime.events import EventType, RuntimeEvent
from finmindagent.runtime.memory.schemas import MemoryItem


class TradingRunState(BaseModel):
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
    # Canonical output language for agent generation AND report locale
    # (zh-CN / en). Legacy default zh-CN keeps old snapshots compatible.
    output_language: str = "zh-CN"
    tool_call_count: int = 0
    parse_error_count: int = 0
    compact_failure_count: int = 0
    injected_memory_ids: set[str] = Field(default_factory=set)
    recalled_memories: list[MemoryItem] = Field(default_factory=list)
    reports: dict[str, str] = Field(default_factory=dict)
    # Structured (JSON-safe) output from non-final-decision agents. Keyed by
    # agent name; values are model_dump(mode="json") dicts, never model
    # instances. Only filled on successful structured calls — never guessed
    # from prose.
    structured_reports: dict[str, dict[str, Any]] = Field(default_factory=dict)
    debate_summary: str = ""
    final_trade_decision: str = ""
    # Authoritative structured decision for HTML/machine consumers. JSON-safe
    # dict so the runtime state does not depend on agent Pydantic types.
    # None means no reliable structured decision (free-text fallback).
    portfolio_decision: dict[str, Any] | None = None
    events: list[RuntimeEvent] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Event observer (CLI reporter etc.). PrivateAttr so it never enters
    # model_dump / JSON / replay snapshots.
    _event_observer: Callable[[RuntimeEvent, "TradingRunState"], None] | None = PrivateAttr(
        default=None
    )

    def set_event_observer(
        self, observer: Callable[[RuntimeEvent, "TradingRunState"], None] | None
    ) -> None:
        self._event_observer = observer

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
        # Event must be appended before the observer is notified so the
        # observer (e.g. CLI reporter) reads a consistent state.
        self.events.append(event)
        observer = self._event_observer
        if callable(observer):
            try:
                observer(event, self)
            except Exception:
                # A reporter failure must NEVER terminate financial analysis.
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
            "artifacts": dict(self.artifacts),
            # Phase A: structured reliability diagnostics
            "report_status": self.metadata.get("report_status", "unknown"),
            "report_publishable": self.metadata.get("report_publishable", False),
            "report_completeness": self.metadata.get("report_completeness", {}),
            "structured_diagnostics": self.metadata.get("structured_diagnostics", {}),
            "messages": [],
            "past_context": "\n\n".join(
                str(e.observation) for e in self.events if e.type == EventType.MEMORY
            ),
        }


class TradingRunResult(BaseModel):
    state: TradingRunState
    final_trade_decision: str
    legacy_state: dict[str, Any]
    signal: dict[str, Any] | str | None = None
    status: str
    stop_reason: str | None = None

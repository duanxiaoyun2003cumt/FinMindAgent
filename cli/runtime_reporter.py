"""Best-effort CLI reporter for FinMindAgent runtime events."""

from __future__ import annotations

import re
from typing import Any

from finmindagent.runtime.actions import ActionType
from finmindagent.runtime.events import EventType, RuntimeEvent


AGENT_DISPLAY = {
    "market_analyst": "Market Analyst",
    "social_sentiment_analyst": "Social Analyst",
    "social_analyst": "Social Analyst",
    "news_analyst": "News Analyst",
    "fundamentals_analyst": "Fundamentals Analyst",
    "bull_researcher": "Bull Researcher",
    "bear_researcher": "Bear Researcher",
    "research_manager": "Research Manager",
    "trader": "Trader",
    "aggressive_risk_analyst": "Aggressive Analyst",
    "conservative_risk_analyst": "Conservative Analyst",
    "neutral_risk_analyst": "Neutral Analyst",
    "portfolio_manager": "Portfolio Manager",
    "risk_verifier": "Risk Verifier",
    "data_quality_verifier": "Data Quality Verifier",
}

REPORT_BY_AGENT = {
    "market_analyst": "market_report",
    "social_sentiment_analyst": "sentiment_report",
    "social_analyst": "sentiment_report",
    "news_analyst": "news_report",
    "fundamentals_analyst": "fundamentals_report",
    "research_manager": "investment_plan",
    "trader": "trader_investment_plan",
    "portfolio_manager": "final_trade_decision",
}


class CliRuntimeReporter:
    def __init__(
        self,
        message_buffer: Any,
        layout: Any,
        update_display,
        stats_handler: Any = None,
        start_time: float | None = None,
        refresh=None,
    ) -> None:
        self.message_buffer = message_buffer
        self.layout = layout
        self.update_display = update_display
        self.stats_handler = stats_handler
        self.start_time = start_time
        self.refresh = refresh
        self._seen_reports: set[str] = set()

    def on_event(self, event: RuntimeEvent, state: Any) -> None:
        try:
            self._handle_event(event, state)
            self.update_display(
                self.layout,
                stats_handler=self.stats_handler,
                start_time=self.start_time,
            )
            if callable(self.refresh):
                self.refresh()
        except Exception:
            return

    def _handle_event(self, event: RuntimeEvent, state: Any) -> None:
        if event.type == EventType.OBSERVATION and event.message == "Runtime started":
            self.message_buffer.add_progress("Runtime started")
            self.message_buffer.add_message("System", "Runtime loop started")
            return

        if event.type == EventType.MEMORY:
            msg = event.message or "Memory loaded / updated"
            self.message_buffer.add_progress(self._clean(msg))
            self.message_buffer.add_message("Memory", self._clean(msg))
            return

        if event.type == EventType.AGENT_CALL:
            display = AGENT_DISPLAY.get(event.actor, event.actor)
            self.message_buffer.add_progress(f"Start {display}")
            self.message_buffer.update_agent_status(display, "in_progress")
            return

        if event.type == EventType.TOOL_CALL:
            action = event.action
            tool_name = action.tool_name if action else "unknown_tool"
            args = self._summarize_args(action.tool_args if action else {})
            actor = AGENT_DISPLAY.get(event.actor, event.actor)
            self.message_buffer.add_progress(f"{actor} calls tool {tool_name}")
            self.message_buffer.add_tool_call(tool_name, args)
            return

        if event.type == EventType.OBSERVATION:
            self._handle_observation(event, state)
            return

        if event.type == EventType.FINAL:
            self.message_buffer.add_progress("Portfolio Manager generated final_trade_decision")
            self.message_buffer.add_message("Final", self._summarize(event.observation or event.message))
            if getattr(state, "final_trade_decision", None):
                self.message_buffer.update_report_section(
                    "final_trade_decision",
                    self._summarize_markdown(state.final_trade_decision),
                )
            return

        if event.type == EventType.ERROR:
            self.message_buffer.add_progress(f"Error: {self._summarize(event.message)}")
            self.message_buffer.add_message("Error", self._summarize(event.message))

    def _handle_observation(self, event: RuntimeEvent, state: Any) -> None:
        action = event.action
        if action and action.type == ActionType.CALL_TOOL:
            self._record_tool_result(action.tool_name or "unknown_tool", event.observation)
            return

        if action and action.type == ActionType.CALL_AGENT:
            agent_name = action.target_agent or event.actor
            display = AGENT_DISPLAY.get(agent_name, agent_name)
            report_key = REPORT_BY_AGENT.get(agent_name)
            if report_key and report_key in getattr(state, "reports", {}):
                report = state.reports[report_key]
                self.message_buffer.update_agent_status(display, "completed")
                self.message_buffer.add_progress(f"{display} report generated")
                self.message_buffer.add_message("Agent", f"{display}: {self._summarize(report)}")
                self.message_buffer.update_report_section(report_key, self._summarize_markdown(report))
                self._seen_reports.add(report_key)
            else:
                self.message_buffer.add_message(event.actor, self._summarize(event.observation))
            return

        for report_key, report in getattr(state, "reports", {}).items():
            if report_key in self._seen_reports:
                continue
            if report_key in set(REPORT_BY_AGENT.values()):
                self.message_buffer.update_report_section(report_key, self._summarize_markdown(report))
                self._seen_reports.add(report_key)

    def _record_tool_result(self, tool_name: str, observation: Any) -> None:
        if isinstance(observation, dict):
            ok = bool(observation.get("ok"))
            status = "ok" if ok else "failed"
            data = observation.get("data")
            size = f"{len(str(data)) if data is not None else 0} chars"
            truncated = ", truncated" if observation.get("truncated") else ""
            detail = observation.get("error") or self._summarize(data)
            self.message_buffer.add_message(
                "Tool Result",
                f"{tool_name} {status}; {size}{truncated}; {detail}",
            )
        else:
            self.message_buffer.add_message(
                "Tool Result",
                f"{tool_name}: {self._summarize(observation)}",
            )

    def _summarize_args(self, args: dict[str, Any], max_chars: int = 120) -> dict[str, str]:
        safe = {}
        for key, value in (args or {}).items():
            lower = str(key).lower()
            if any(secret in lower for secret in ("key", "token", "secret", "password")):
                safe[str(key)] = "<redacted>"
            else:
                safe[str(key)] = self._summarize(value, max_chars=max_chars)
        return safe

    def _summarize_markdown(self, text: Any, max_chars: int = 900) -> str:
        return self._summarize(text, max_chars=max_chars)

    def _summarize(self, value: Any, max_chars: int = 220) -> str:
        if value is None:
            return ""
        text = self._clean(str(value))
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 16].rstrip() + " ... [truncated]"

    def _clean(self, text: str) -> str:
        text = re.sub(
            r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+",
            r"\1=<redacted>",
            text,
        )
        text = text.replace("\r", " ").replace("\n", " ")
        return re.sub(r"\s+", " ", text).strip()

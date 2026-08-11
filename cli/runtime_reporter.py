"""Best-effort CLI reporter for FinMindAgent runtime events.

Producer/consumer split:
- Runtime worker thread calls ``on_event()`` — it ONLY enqueues.
- CLI main thread calls ``drain_events()`` — it converts events into the
  MessageBuffer. Rich rendering stays entirely on the main thread.
"""

from __future__ import annotations

import logging
import re
from queue import Empty, Queue
from typing import Any

from finmindagent.runtime.actions import ActionType
from finmindagent.runtime.events import EventType, RuntimeEvent

logger = logging.getLogger(__name__)


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
        event_queue: Queue | None = None,
    ) -> None:
        self.message_buffer = message_buffer
        self.layout = layout
        self.update_display = update_display
        self.stats_handler = stats_handler
        self.start_time = start_time
        self.event_queue = event_queue or Queue()

    def on_event(self, event: RuntimeEvent, state: Any = None) -> None:
        """Runtime-worker-side entry point: enqueue only.

        Runs on the runtime worker thread, so it MUST NOT touch Rich,
        update_display, live.refresh, or mutate the MessageBuffer.
        """
        try:
            self.event_queue.put_nowait(event)
        except Exception:
            # Enqueue failure must never terminate the analysis runtime.
            logger.debug("CLI reporter failed to enqueue event", exc_info=True)

    def wait_and_drain(self, timeout: float = 0.1) -> int:
        """Main-thread consumer: block until an event arrives (or timeout),
        then handle all pending events in order.

        Event-driven alternative to busy-polling: returns as soon as the
        first queued event arrives, so rendering happens promptly without a
        fixed-rate full rebuild. Returns the number of events processed.
        """
        try:
            first = self.event_queue.get(timeout=timeout)
        except Empty:
            return 0
        events = [first]
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except Empty:
                break
        for event in events:
            try:
                self._handle_event(event)
            except Exception:
                # One bad event must not kill the rest of the UI pipeline.
                logger.debug("CLI reporter failed while handling event", exc_info=True)
        return len(events)

    def drain_events(self, max_events: int | None = None) -> int:
        """Main-thread consumer: convert queued events into the MessageBuffer.

        Must be called from the CLI main thread, which owns MessageBuffer
        mutation and all Rich rendering. Returns the number of events
        processed.
        """
        processed = 0
        while max_events is None or processed < max_events:
            try:
                event = self.event_queue.get_nowait()
            except Empty:
                break
            try:
                self._handle_event(event)
            except Exception:
                # One bad event must not kill the rest of the UI pipeline.
                logger.debug("CLI reporter failed while handling event", exc_info=True)
            processed += 1
        return processed

    def _handle_event(self, event: RuntimeEvent) -> None:
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

        if event.type == EventType.LLM_CALL:
            # LLM/structured invoke blocks for 20-60s; show where we are.
            display = AGENT_DISPLAY.get(event.actor, event.actor)
            self.message_buffer.add_progress(f"{display} generating response...")
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
            self._handle_observation(event)
            return

        if event.type == EventType.FINAL:
            # The FINAL event already carries the full decision text in
            # observation (or the stop reason in message) — no state access.
            self.message_buffer.add_progress("Portfolio Manager generated final_trade_decision")
            self.message_buffer.add_message("Final", self._summarize(event.observation or event.message))
            if event.observation:
                # Authoritative section keeps the FULL final decision text.
                self.message_buffer.update_report_section(
                    "final_trade_decision",
                    str(event.observation),
                )
            return

        if event.type == EventType.ERROR:
            self.message_buffer.add_progress(f"Error: {self._summarize(event.message)}")
            self.message_buffer.add_message("Error", self._summarize(event.message))

    def _handle_observation(self, event: RuntimeEvent) -> None:
        action = event.action
        if action and action.type == ActionType.CALL_TOOL:
            self._record_tool_result(action.tool_name or "unknown_tool", event.observation)
            return

        if action and action.type == ActionType.CALL_AGENT:
            # event.observation IS the final Agent report returned by
            # _call_agent(). The reporter is fully event-driven and never
            # reads the mutable TradingRunState.
            agent_name = action.target_agent or event.actor
            display = AGENT_DISPLAY.get(agent_name, agent_name)
            report = event.observation
            if report:
                self.message_buffer.update_agent_status(display, "completed")
                self.message_buffer.add_progress(f"{display} report generated")
                self.message_buffer.add_message("Agent", f"{display}: {self._summarize(report)}")
                report_key = REPORT_BY_AGENT.get(agent_name)
                if report_key:
                    # Authoritative official section keeps the FULL report.
                    self.message_buffer.update_report_section(report_key, str(report))
                # Current Report shows THIS agent right now; the CLI preview
                # may truncate, the official section above never does.
                self.message_buffer.update_current_report(
                    display, self._summarize_markdown(report, max_chars=2000)
                )
            else:
                self.message_buffer.add_message(event.actor, self._summarize(event.observation))
            return

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

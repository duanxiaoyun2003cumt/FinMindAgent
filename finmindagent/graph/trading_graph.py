"""Compatibility wrapper over the FinMindAgent runtime loop.

The fixed LangGraph DAG has been removed. This module keeps the public
``FinMindAgentGraph`` name and ``propagate(ticker, date)`` API for existing
callers, but all execution now goes through ``FinMindAgentLoop``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from finmindagent.agents.utils.memory import TradingMemoryLog
from finmindagent.dataflows.config import set_config
from finmindagent.dataflows.utils import safe_ticker_component
from finmindagent.time_utils import US_EASTERN_TZ
from finmindagent.default_config import DEFAULT_CONFIG
from finmindagent.graph.reflection import Reflector
from finmindagent.graph.signal_processing import SignalProcessor
from finmindagent.runtime.engine import _save_replay_snapshot
from finmindagent.runtime.runner import create_runtime
from finmindagent.runtime.state import TradingRunState

logger = logging.getLogger(__name__)


class FinMindAgentGraph:
    """Backward-compatible facade for the new runtime loop."""

    def __init__(
        self,
        selected_analysts=None,
        debug: bool = False,
        config: Dict[str, Any] | None = None,
        callbacks: Optional[list[Any]] = None,
    ):
        self.debug = debug
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        self.callbacks = callbacks or []
        self.selected_analysts = selected_analysts or [
            "market",
            "social",
            "news",
            "fundamentals",
        ]
        self.config["selected_analysts"] = self.selected_analysts

        set_config(self.config)
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        self.runtime_loop = create_runtime(self.config, callbacks=self.callbacks)
        self.quick_thinking_llm = self.runtime_loop.quick_llm
        self.deep_thinking_llm = self.runtime_loop.deep_llm
        self.memory_log = TradingMemoryLog(self.config)
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        self.curr_state: dict[str, Any] | None = None
        self.ticker: str | None = None
        self.log_states_dict: dict[str, dict[str, Any]] = {}

    def propagate(self, company_name: str, trade_date: str):
        """Run FinMindAgent for a ticker/date through the runtime loop."""
        self.ticker = company_name
        self._resolve_pending_entries(company_name)

        state = TradingRunState(
            ticker=company_name,
            trade_date=str(trade_date),
            user_request=f"Analyze {company_name} for {trade_date}.",
            max_steps=int(
                self.config.get(
                    "runtime_max_steps",
                    self.config.get("max_recur_limit", 30),
                )
            ),
            max_tool_calls=int(self.config.get("runtime_max_tool_calls", 30)),
            max_tokens=int(self.config.get("runtime_max_tokens", 120_000)),
            permission_mode=str(self.config.get("permission_mode", "safe")),
        )
        result = self.runtime_loop.run(state)

        # P1 Task 11: persist replay snapshot for offline deterministic replay
        # (graph wrapper calls run() directly, so save here as well).
        _save_replay_snapshot(result.state, self.config)

        final_state = result.legacy_state
        self.curr_state = final_state
        self._log_state(trade_date, final_state)
        final_decision = str(final_state.get("final_trade_decision", "")).strip()
        if final_decision:
            self.memory_log.store_decision(
                ticker=company_name,
                trade_date=str(trade_date),
                final_trade_decision=final_decision,
            )
        else:
            logger.warning(
                "Skipping memory log write for %s on %s: final_trade_decision is empty",
                company_name,
                trade_date,
            )
        return final_state, self.process_signal(final_decision)

    def _fetch_returns(
        self,
        ticker: str,
        trade_date: str,
        holding_days: int = 5,
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Fetch raw and SPY-relative returns for deferred memory reflection.

        Point-in-time governance: the outcome window must be fully elapsed
        in the market's own timezone before it can be recorded. When the
        window end extends beyond the latest completed session (e.g. a
        same-day pending entry before the US open), the outcome is NOT
        determinable yet — defer (return None) instead of scanning into
        the future. PLTR and SPY are queried with the SAME window and the
        same as-of boundary; provider rows are defensively filtered.
        """
        try:
            import yfinance as yf

            from finmindagent.dataflows import market_asof

            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)

            # Latest completed session in the market's own timezone
            # (pre-open / weekend → previous session).
            now = market_asof.market_now(US_EASTERN_TZ)
            now_cutoff = market_asof.resolve_market_data_cutoff(
                now.date(), as_of_now=now, market_timezone=US_EASTERN_TZ
            )
            if end.date() > now_cutoff:
                # Outcome window not elapsed — defer; never forward-scan.
                return None, None, None
            end_str = end.strftime("%Y-%m-%d")

            stock = yf.Ticker(ticker).history(start=trade_date, end=end_str)
            spy = yf.Ticker("SPY").history(start=trade_date, end=end_str)

            # Defensive filter: provider is not the as-of authority. Both
            # series are capped at the same boundary (no future backfill).
            # DatetimeIndex (real yfinance output) carries market-local
            # dates; empty frames have nothing to filter.
            def _cap_at_cutoff(frame):
                if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
                    return frame
                return frame[frame.index.date <= now_cutoff]

            stock = _cap_at_cutoff(stock)
            spy = _cap_at_cutoff(spy)

            if len(stock) < 2 or len(spy) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(spy) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            spy_ret = float(
                (spy["Close"].iloc[actual_days] - spy["Close"].iloc[0])
                / spy["Close"].iloc[0]
            )
            return raw, raw - spy_ret, actual_days
        except Exception as exc:
            logger.warning(
                "Could not resolve outcome for %s on %s: %s",
                ticker,
                trade_date,
                exc,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending or self.quick_thinking_llm is None:
            return

        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(ticker, entry["date"])
            if raw is None:
                continue
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
            )
            updates.append(
                {
                    "ticker": ticker,
                    "trade_date": entry["date"],
                    "decision": entry.get("decision", ""),
                    "raw_return": raw,
                    "alpha_return": alpha,
                    "holding_days": days,
                    "reflection": reflection,
                }
            )
        if updates:
            self.memory_log.batch_update_with_outcomes(updates)
            self._sync_outcome_reflections(updates)

    def _sync_outcome_reflections(self, updates: list[dict[str, Any]]) -> None:
        """Mirror resolved legacy outcomes into the structured MemoryStore.

        The runtime loop owns the structured store; the legacy graph only
        writes to it when a runtime loop is attached. The defensive lookup
        keeps direct unit-test calls (which construct uninitialized mocks)
        working, while the real ``propagate`` path always has a runtime loop.
        """
        memory_manager = getattr(getattr(self, "runtime_loop", None), "memory_manager", None)
        write = getattr(memory_manager, "write_outcome_reflection", None)
        if write is None:
            return
        for update in updates:
            try:
                write(
                    ticker=update["ticker"],
                    trade_date=update["trade_date"],
                    decision=update.get("decision", ""),
                    raw_return=update.get("raw_return"),
                    alpha_return=update.get("alpha_return"),
                    holding_days=update.get("holding_days"),
                    reflection=update.get("reflection", ""),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to sync outcome memory for %s %s: %s",
                    update.get("ticker"),
                    update.get("trade_date"),
                    exc,
                )

    def _log_state(self, trade_date, final_state: dict[str, Any]) -> None:
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state.get("company_of_interest", self.ticker),
            "trade_date": final_state.get("trade_date", str(trade_date)),
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "investment_debate_state": final_state.get("investment_debate_state", {}),
            "trader_investment_decision": final_state.get("trader_investment_plan", ""),
            "risk_debate_state": final_state.get("risk_debate_state", {}),
            "investment_plan": final_state.get("investment_plan", ""),
            "final_trade_decision": final_state.get("final_trade_decision", ""),
            "runtime_status": final_state.get("runtime_status", ""),
            "runtime_stop_reason": final_state.get("runtime_stop_reason", ""),
            "called_agents": final_state.get("called_agents", []),
            "missing_required_reports": final_state.get("missing_required_reports", []),
            "event_count": final_state.get("event_count", 0),
            # Phase A: structured reliability diagnostics
            "report_status": final_state.get("report_status", "unknown"),
            "report_publishable": final_state.get("report_publishable", False),
            "report_completeness": final_state.get("report_completeness", {}),
            "structured_diagnostics": final_state.get("structured_diagnostics", {}),
        }

        safe_ticker = safe_ticker_component(self.ticker or "UNKNOWN")
        directory = (
            Path(self.config["results_dir"])
            / safe_ticker
            / "FinMindAgentStrategy_logs"
        )
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal: str) -> str:
        return self.signal_processor.process_signal(full_signal)

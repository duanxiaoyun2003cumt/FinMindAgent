"""P2 E2E contract tests: no-network synthetic state → ReportSchema → HTML.

Covers earnings realization, valuation context, market risk stats, ownership,
sentiment signals, provenance, conflicts, stale detection, and degraded behaviors.
"""

import json

from finmindagent.reporting.builder import build_report_schema
from finmindagent.reporting.html import render_report_html
from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.events import EventType
from finmindagent.runtime.state import TradingRunState


def _make_p2_state():
    """Complete P0+P1+P2 synthetic state."""
    state = TradingRunState(
        ticker="NVDA", trade_date="2026-08-08",
        run_id="p2-contract", status="completed",
    )
    # P0: market (60+ points) — use valid calendar dates
    from datetime import date as _date, timedelta
    start_date = _date(2026, 6, 1)
    mp = []
    for i in range(65):
        day = (start_date + timedelta(days=i)).isoformat()
        mp.append({"date": day, "open": 100.0+i, "high": 105.0+i, "low": 95.0+i, "close": 102.0+i, "volume": 1e6+i*1000})
    state.add_event(EventType.OBSERVATION, actor="market",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data"),
        observation={"ok": True, "tool_name": "get_stock_data", "structured_data": {"market_series": mp}})

    # P0: financials
    state.add_event(EventType.OBSERVATION, actor="f",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_income_statement"),
        observation={"ok": True, "tool_name": "get_income_statement", "structured_data": {
            "financial_metrics": [
                {"name": "total_revenue", "value": 100e9, "period": "2025-Q4", "source": "get_income_statement"},
                {"name": "pe_ratio_ttm", "value": 25.0, "period": "2025-Q4", "source": "get_fundamentals"},
                {"name": "forward_pe", "value": 20.0, "period": "2025-Q4", "source": "get_fundamentals"},
                {"name": "price_to_book", "value": 5.0, "period": "2025-Q4", "source": "get_fundamentals"},
                {"name": "peg_ratio", "value": 1.5, "period": "2025-Q4", "source": "get_fundamentals"},
            ],
            "financial_history": [
                {"name": "total_revenue", "period": "2024-Q1", "value": 80e9, "source": "get_income_statement"},
                {"name": "total_revenue", "period": "2024-Q2", "value": 85e9, "source": "get_income_statement"},
            ],
        }})

    # P1: analyst expectations with P2 revisions
    state.add_event(EventType.OBSERVATION, actor="f",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_analyst_expectations"),
        observation={"ok": True, "tool_name": "get_analyst_expectations", "structured_data": {"analyst_consensus": {
            "as_of": "2026-08-08",
            "price_targets": {"vendor_current": 120.0, "low": 110.0, "mean": 150.0, "median": 145.0, "high": 180.0},
            "earnings_estimates": [{"period": "current_quarter", "average": 2.5, "growth_pct": 25.0}],
            "eps_revisions": [{"period": "current_quarter", "current": 2.5, "days_7_ago": 2.4, "days_30_ago": 2.3, "up_7d": 3, "up_30d": 8, "down_7d": 1, "down_30d": 2}],
            "source": "yfinance",
        }}})

    # P2: earnings history
    state.add_event(EventType.OBSERVATION, actor="f",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_earnings_history"),
        observation={"ok": True, "tool_name": "get_earnings_history", "structured_data": {"earnings_context": {
            "as_of": "2026-08-08", "surprises": [
                {"evidence_id": "earnings_surprise:2026-05-21", "report_date": "2026-05-21",
                 "eps_estimate": 0.88, "reported_eps": 0.94, "surprise_pct": 6.82, "source": "yfinance"},
                {"evidence_id": "earnings_surprise:2026-02-15", "report_date": "2026-02-15",
                 "eps_estimate": 0.75, "reported_eps": 0.78, "surprise_pct": 4.0, "source": "yfinance"},
            ],
            "source": "yfinance",
        }}})

    # P2: ownership
    state.add_event(EventType.OBSERVATION, actor="s",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_holder_context"),
        observation={"ok": True, "tool_name": "get_holder_context", "structured_data": {"ownership_snapshot": {
            "as_of": "2026-03-31", "insiders_pct": 4.2, "institutions_pct": 68.5,
            "institutions_float_pct": 70.1, "institutions_count": 1450,
            "institutional_holders": [
                {"holder_id": "holder:institutional:vanguard-group-inc:2026-03-31",
                 "holder": "Vanguard Group Inc", "holder_type": "institutional",
                 "pct_out": 8.25, "shares": 1.2e9, "date_reported": "2026-03-31", "source": "yfinance"},
            ],
            "mutual_fund_holders": [
                {"holder_id": "holder:mutual_fund:fidelity:2026-03-31",
                 "holder": "Fidelity", "holder_type": "mutual_fund",
                 "pct_out": 5.0, "shares": 5e8, "date_reported": "2026-03-31", "source": "yfinance"},
            ],
            "source": "yfinance",
        }}})

    # P1: calendar + insider
    state.add_event(EventType.OBSERVATION, actor="news",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_company_calendar"),
        observation={"ok": True, "tool_name": "get_company_calendar", "structured_data": {"calendar_events": [{
            "event_id": "earnings:2026-08-28", "event_type": "earnings",
            "date": "2026-08-28", "title": "Earnings release",
        }]}})
    state.add_event(EventType.OBSERVATION, actor="news",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_insider_transactions"),
        observation={"ok": True, "tool_name": "get_insider_transactions", "structured_data": {"timeline": [{
            "date": "2026-06-08", "event": "Insider transaction: Jane Doe - Sale",
            "impact": "unknown", "importance": "unknown", "scope": "company", "kind": "insider",
            "source": "get_insider_transactions",
        }]}})

    # Same-period conflicting financial metrics
    state.add_event(EventType.OBSERVATION, actor="f",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_balance_sheet"),
        observation={"ok": True, "tool_name": "get_balance_sheet", "structured_data": {
            "financial_metrics": [
                {"name": "pe_ratio_ttm", "value": 26.0, "period": "2025-Q4", "source": "get_balance_sheet"},
                {"name": "total_debt", "value": 10e9, "period": "2025-Q4", "source": "get_balance_sheet"},
                {"name": "total_debt", "value": 9.5e9, "period": "2025-Q3", "source": "get_balance_sheet"},  # different period → no conflict
            ],
        }})

    # Structured reports
    state.structured_reports = {
        "market_analyst": {"analysis": "x", "score": 7.0, "score_reason": "ok"},
        "fundamentals_analyst": {"analysis": "x", "score": 7.0, "score_reason": "ok", "financial_quality_summary": "ok", "quality_warnings": [{"title": "Cash", "evidence": "FCF negative."}]},
        "news_analyst": {"analysis": "x", "score": 6.0, "score_reason": "ok", "catalysts": [
            {"title": "Earnings watch", "catalyst_type": "earnings", "direction": "mixed", "importance": "high", "trigger_condition": "Beat.", "follow_up": "Reassess.", "evidence_refs": ["earnings:2026-08-28"]},
        ]},
        "social_sentiment_analyst": {"analysis": "x", "score": 5.0, "score_reason": "ok", "signals": [
            {"title": "Institutional confidence", "signal_type": "ownership", "direction": "positive", "strength": 7.0, "reasoning": "Vanguard holds 8.25%", "evidence_refs": ["holder:institutional:vanguard-group-inc:2026-03-31"]},
            {"title": "No evidence signal", "signal_type": "other", "direction": "positive", "strength": 3.0, "reasoning": "none", "evidence_refs": ["nonexistent:ref"]},
        ]},
        "bull_researcher": {"summary": "x", "arguments": [{"title": "Growth", "argument": "Strong.", "strength": 8.0}, {"title": "AI", "argument": "Leader.", "strength": 7.0}]},
        "bear_researcher": {"summary": "x", "arguments": [{"title": "Valuation", "argument": "Rich.", "strength": 7.0}, {"title": "Competition", "argument": "Crowded.", "strength": 5.0}]},
        "research_manager": {"recommendation": "Hold", "rationale": "Balanced.", "strategic_actions": "Wait.", "resolution": {"disagreement": "Growth vs valuation."}},
        "trader": {"action": "Hold", "reasoning": "Mixed."},
        "aggressive_risk_analyst": {"summary": "x", "risk_score": 5.0, "dimensions": {"valuation": {"score": 7, "reason": "x"}, "financial_quality": {"score": 6, "reason": "x"}, "market_technical": {"score": 5, "reason": "x"}, "event_legal": {"score": 4, "reason": "x"}, "liquidity_sentiment": {"score": 3, "reason": "x"}}},
        "neutral_risk_analyst": {"summary": "x", "risk_score": 6.0, "dimensions": {"valuation": {"score": 8, "reason": "x"}, "financial_quality": {"score": 6, "reason": "x"}, "market_technical": {"score": 5, "reason": "x"}, "event_legal": {"score": 5, "reason": "x"}, "liquidity_sentiment": {"score": 4, "reason": "x"}}},
        "conservative_risk_analyst": {"summary": "x", "risk_score": 7.0, "dimensions": {"valuation": {"score": 9, "reason": "x"}, "financial_quality": {"score": 7, "reason": "x"}, "market_technical": {"score": 6, "reason": "x"}, "event_legal": {"score": 5, "reason": "x"}, "liquidity_sentiment": {"score": 4, "reason": "x"}}},
    }

    # Authoritative PM decision
    state.portfolio_decision = {
        "rating": "Hold", "executive_summary": "Wait.", "investment_thesis": "Balanced.",
        "price_target": 250.0, "time_horizon": "3-6 months", "stop_loss": 180.0,
        "position_min_pct": 3.0, "position_max_pct": 8.0,
        "confidence_level": "Medium", "risk_level": "Medium-High",
    }
    return state


class TestP2Contract:
    def test_earnings_context_present(self):
        state = _make_p2_state()
        report = build_report_schema(state)
        assert report.earnings_context is not None
        assert len(report.earnings_context.surprises) >= 1
        assert report.earnings_context.surprises[0].surprise_pct == 6.82

    def test_ownership_snapshot_present(self):
        state = _make_p2_state()
        report = build_report_schema(state)
        assert report.ownership_snapshot is not None
        assert report.ownership_snapshot.institutions_pct == 68.5
        assert len(report.ownership_snapshot.institutional_holders) >= 1

    def test_valuation_context_present(self):
        state = _make_p2_state()
        report = build_report_schema(state)
        assert report.valuation_context is not None
        assert report.valuation_context.trailing_pe is not None
        assert report.valuation_context.forward_pe == 20.0
        assert report.valuation_context.price_to_book == 5.0
        assert report.valuation_context.peg_ratio == 1.5
        # PE compression with both >0 must be present
        assert report.valuation_context.pe_compression_pct is not None
        # Analyst target from consensus, NOT PM target
        assert report.valuation_context.analyst_target_mean == 150.0

    def test_pm_authority_unchanged(self):
        """PM target (250) must not be overwritten by analyst target (150)."""
        state = _make_p2_state()
        report = build_report_schema(state)
        assert report.decision.price_target == 250.0
        assert report.decision.rating == "Hold"

    def test_market_risk_snapshot_present(self):
        state = _make_p2_state()
        report = build_report_schema(state)
        assert report.market_risk_snapshot is not None
        assert report.market_risk_snapshot.return_5d_pct is not None

    def test_forecasts_not_in_financial_series(self):
        state = _make_p2_state()
        report = build_report_schema(state)
        for fs in report.financial_series:
            for pt in fs.points:
                assert "2026E" not in pt.period

    def test_grounded_sentiment_accepted_invented_dropped(self):
        state = _make_p2_state()
        report = build_report_schema(state)
        # Vanguard ref accepted, nonexistent ref dropped
        assert len(report.sentiment_signals) == 1
        assert report.sentiment_signals[0].title == "Institutional confidence"
        assert any("Ungrounded sentiment signal dropped" in n for n in report.data_quality.notes)

    def test_conflict_recorded_same_period(self):
        """Same period pe_ratio_ttm from different sources with different values → conflict."""
        state = _make_p2_state()
        report = build_report_schema(state)
        # pe_ratio_ttm from get_fundamentals (25.0) vs get_balance_sheet (26.0), same period 2025-Q4
        conflicts = [c for c in report.data_quality.conflicts if c.field == "pe_ratio_ttm"]
        assert len(conflicts) >= 1

    def test_cross_period_not_conflict(self):
        """Different periods of total_debt are NOT conflicts."""
        state = _make_p2_state()
        report = build_report_schema(state)
        conflicts = [c for c in report.data_quality.conflicts if c.field == "total_debt"]
        assert len(conflicts) == 0

    def test_html_contains_p2_sections(self):
        state = _make_p2_state()
        report = build_report_schema(state)
        html = render_report_html(report, chart_js="")
        assert 'id="earnings-realization"' in html
        assert 'id="valuation-context"' in html
        assert 'id="market-risk-stats"' in html
        assert 'id="ownership-sentiment"' in html
        assert "None" not in html

    def test_sndk_safety_preserved(self):
        """SNDK conflicting free-text → rating stays unavailable."""
        state = TradingRunState(ticker="SNDK", trade_date="2026-08-07", run_id="sndk", status="completed")
        state.final_trade_decision = "**Rating**: Hold\n\nSome text.\n\nDetail: **Rating:** **Sell / Underweight**"
        state.portfolio_decision = None
        report = build_report_schema(state)
        assert report.decision.rating is None
        assert report.decision.structured_available is False

    def test_no_unsupported_ashare_facts(self):
        state = _make_p2_state()
        report = build_report_schema(state)
        html = render_report_html(report, chart_js="")
        for term in ("融资余额", "北向资金", "龙虎榜", "QFII", "margin financing"):
            assert term not in html

    def test_provenance_from_runtime_observation(self):
        """Provenance must flow from Runtime observation → Builder → ReportSchema."""
        state = _make_p2_state()
        report = build_report_schema(state)
        assert report.data_quality.sources, "Provenance sources must not be empty"
        tool_names = {s.tool_name for s in report.data_quality.sources}
        for expected in ("get_analyst_expectations", "get_earnings_history",
                          "get_holder_context", "get_stock_data"):
            assert expected in tool_names, f"Missing provenance for {expected}"
        # Check structured_keys populated
        for s in report.data_quality.sources:
            if s.tool_name == "get_stock_data":
                assert "market_series" in s.structured_keys
            if s.tool_name == "get_analyst_expectations":
                assert "analyst_consensus" in s.structured_keys

    def test_stale_detection_triggered(self):
        """Market data 8+ days behind trade_date → stale note."""
        from datetime import date as _date
        from finmindagent.runtime.state import TradingRunState
        from finmindagent.runtime.actions import ActionType, AgentAction
        from finmindagent.runtime.events import EventType

        state = TradingRunState(ticker="TEST", trade_date="2026-08-20", run_id="stale", status="ok")
        state.add_event(EventType.OBSERVATION, actor="m",
            action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data"),
            observation={"ok": True, "tool_name": "get_stock_data", "structured_data": {
                "market_series": [{"date": f"2026-08-{i+1:02d}", "open": 100+i, "close": 100+i, "volume": 1e6} for i in range(10)],
            }})
        report = build_report_schema(state)
        # trade_date=2026-08-20, latest market=2026-08-10, gap=10 days > 7
        stale_notes = [n for n in report.data_quality.notes if "stale" in n.lower()]
        assert stale_notes, f"Expected stale note for 10-day gap, got notes: {report.data_quality.notes}"

    def test_weekend_gap_not_stale(self):
        """Normal weekend gap (Fri→Mon) should not trigger stale."""
        from finmindagent.runtime.state import TradingRunState
        from finmindagent.runtime.actions import ActionType, AgentAction
        from finmindagent.runtime.events import EventType

        # trade_date=Monday 2026-08-10, latest market=Friday 2026-08-07 → gap=3 days
        state = TradingRunState(ticker="TEST", trade_date="2026-08-10", run_id="weekend", status="ok")
        state.add_event(EventType.OBSERVATION, actor="m",
            action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data"),
            observation={"ok": True, "tool_name": "get_stock_data", "structured_data": {
                "market_series": [{"date": "2026-08-07", "open": 100, "close": 100, "volume": 1e6}],
            }})
        report = build_report_schema(state)
        # gap=3 days (Fri→Mon) → should not be stale
        stale_notes = [n for n in report.data_quality.notes if "stale" in n.lower()]
        assert not stale_notes, f"3-day weekend gap must not be stale, got: {stale_notes}"

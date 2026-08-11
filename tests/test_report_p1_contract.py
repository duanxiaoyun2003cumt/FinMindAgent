"""P1 end-to-end contract tests: synthetic state → ReportSchema → HTML.

Covers analyst consensus, catalyst grounding, five-dim risk radar, insider
timeline, horizon-grouped risk register, and degraded behaviors.
"""

from finmindagent.reporting.builder import build_report_schema
from finmindagent.reporting.html import render_report_html
from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.events import EventType
from finmindagent.runtime.state import TradingRunState


def _make_p1_state():
    """Complete P0+P1 synthetic state."""
    state = TradingRunState(
        ticker="NVDA", trade_date="2026-08-08",
        run_id="p1-contract", status="completed",
    )
    # P0: market
    market_points = []
    for i in range(65):
        day = f"2026-06-{i+1:02d}" if i < 30 else f"2026-07-{i+1-30:02d}"
        market_points.append({
            "date": day, "open": 100.0 + i, "high": 105.0 + i,
            "low": 95.0 + i, "close": 102.0 + i, "volume": 1e6 + i * 1000,
        })
    state.add_event(EventType.OBSERVATION, actor="market_analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data"),
        observation={"ok": True, "tool_name": "get_stock_data", "structured_data": {"market_series": market_points}})

    # P0: fundamentals
    state.add_event(EventType.OBSERVATION, actor="fundamentals_analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_income_statement"),
        observation={"ok": True, "tool_name": "get_income_statement", "structured_data": {
            "financial_metrics": [{"name": "total_revenue", "value": 100e9, "period": "2025-Q4", "source": "get_income_statement"}],
            "financial_history": [
                {"name": "total_revenue", "period": "2024-Q1", "value": 80e9, "source": "get_income_statement"},
                {"name": "total_revenue", "period": "2024-Q2", "value": 85e9, "source": "get_income_statement"},
                {"name": "total_revenue", "period": "2024-Q3", "value": 90e9, "source": "get_income_statement"},
                {"name": "total_revenue", "period": "2024-Q4", "value": 95e9, "source": "get_income_statement"},
            ],
        }})

    # P1: analyst expectations
    state.add_event(EventType.OBSERVATION, actor="fundamentals_analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_analyst_expectations"),
        observation={"ok": True, "tool_name": "get_analyst_expectations", "structured_data": {"analyst_consensus": {
            "as_of": "2026-08-08",
            "price_targets": {"vendor_current": 120.0, "low": 110.0, "mean": 150.0, "median": 145.0, "high": 180.0},
            "earnings_estimates": [{"period": "current_quarter", "average": 2.5, "low": 2.3, "high": 2.8, "year_ago": 2.0, "analyst_count": 20, "growth_pct": 25.0}],
            "revenue_estimates": [{"period": "next_quarter", "average": 38e9, "low": 36e9, "high": 40e9, "year_ago": 30e9, "analyst_count": 15, "growth_pct": 26.6}],
            "recommendations": [{"period": "2026-08", "strong_buy": 15, "buy": 20, "hold": 5, "sell": 1, "strong_sell": 0}],
            "source": "yfinance",
        }}})

    # P1: calendar
    state.add_event(EventType.OBSERVATION, actor="news_analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_company_calendar"),
        observation={"ok": True, "tool_name": "get_company_calendar", "structured_data": {"calendar_events": [{
            "event_id": "earnings:2026-08-28", "event_type": "earnings",
            "date": "2026-08-28", "title": "Earnings release",
        }]}})

    # P1: insider
    state.add_event(EventType.OBSERVATION, actor="news_analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_insider_transactions"),
        observation={"ok": True, "tool_name": "get_insider_transactions", "structured_data": {"timeline": [{
            "date": "2026-06-08", "event": "Insider transaction: Jane Doe - Sale",
            "impact": "unknown", "importance": "unknown",
            "detail": "Position: Director; Shares: 100",
            "source": "get_insider_transactions", "url": "", "scope": "company", "kind": "insider",
        }]}})

    # Structured reports
    state.structured_reports = {
        "market_analyst": {"analysis": "x", "score": 7.0, "score_reason": "ok"},
        "fundamentals_analyst": {"analysis": "x", "score": 7.0, "score_reason": "ok", "financial_quality_summary": "Good but cash weak.", "quality_warnings": [{"title": "Cash", "evidence": "FCF negative."}]},
        "news_analyst": {
            "analysis": "x", "score": 6.0, "score_reason": "ok",
            "catalysts": [{"title": "Earnings watch", "catalyst_type": "earnings", "direction": "mixed", "importance": "high", "trigger_condition": "Beat consensus.", "follow_up": "Reassess.", "evidence_refs": ["earnings:2026-08-28"]}],
        },
        "social_sentiment_analyst": {"analysis": "x", "score": 5.0, "score_reason": "ok"},
        "bull_researcher": {"summary": "x", "arguments": [{"title": "Growth", "argument": "Strong.", "strength": 8.0}, {"title": "AI", "argument": "Leader.", "strength": 7.0}]},
        "bear_researcher": {"summary": "x", "arguments": [{"title": "Valuation", "argument": "Rich.", "strength": 7.0}, {"title": "Competition", "argument": "Crowded.", "strength": 5.0}]},
        "research_manager": {"recommendation": "Hold", "rationale": "Balanced.", "strategic_actions": "Wait.", "resolution": {"disagreement": "Growth vs valuation.", "key_bull_points": ["Revenue"], "key_bear_points": ["Valuation"], "decisive_factors": ["R/R not attractive"], "final_reason": "Wait."}},
        "trader": {"action": "Hold", "reasoning": "Mixed."},
        "aggressive_risk_analyst": {"summary": "x", "risk_score": 5.0, "dimensions": {"valuation": {"score": 7, "reason": "x"}, "financial_quality": {"score": 6, "reason": "x"}, "market_technical": {"score": 5, "reason": "x"}, "event_legal": {"score": 4, "reason": "x"}, "liquidity_sentiment": {"score": 3, "reason": "x"}}, "risks": [{"name": "Valuation", "severity": "high", "horizon": "3m", "evidence": "P/E high", "category": "valuation", "horizon_bucket": "short", "probability_pct": 40, "downside_impact_pct": 20, "estimate_basis": "Model estimate."}]},
        "neutral_risk_analyst": {"summary": "x", "risk_score": 6.0, "dimensions": {"valuation": {"score": 8, "reason": "x"}, "financial_quality": {"score": 6, "reason": "x"}, "market_technical": {"score": 5, "reason": "x"}, "event_legal": {"score": 5, "reason": "x"}, "liquidity_sentiment": {"score": 4, "reason": "x"}}, "risks": [{"name": "Valuation", "severity": "high", "horizon": "3m", "evidence": "P/E high", "category": "valuation", "horizon_bucket": "short", "probability_pct": 50, "downside_impact_pct": 25, "estimate_basis": "Estimate."}]},
        "conservative_risk_analyst": {"summary": "x", "risk_score": 7.0, "dimensions": {"valuation": {"score": 9, "reason": "x"}, "financial_quality": {"score": 7, "reason": "x"}, "market_technical": {"score": 6, "reason": "x"}, "event_legal": {"score": 5, "reason": "x"}, "liquidity_sentiment": {"score": 4, "reason": "x"}}, "risks": [{"name": "Valuation", "severity": "critical", "horizon": "1m", "evidence": "Correction risk", "category": "valuation", "horizon_bucket": "short", "probability_pct": 60, "downside_impact_pct": 30, "estimate_basis": "Model."}]},
    }

    # Authoritative P0 decision
    state.portfolio_decision = {
        "rating": "Hold", "executive_summary": "Wait.", "investment_thesis": "Balanced.",
        "price_target": 250.0, "time_horizon": "3-6 months", "stop_loss": 180.0,
        "position_min_pct": 3.0, "position_max_pct": 8.0,
        "confidence_level": "Medium", "risk_level": "Medium-High",
    }
    return state


class TestP1Contract:
    def test_analyst_consensus_present(self):
        state = _make_p1_state()
        report = build_report_schema(state)
        assert report.analyst_consensus is not None
        assert report.analyst_consensus.price_targets.mean == 150.0
        # mean upside: (150 - current_close) / current_close * 100
        assert report.analyst_consensus.price_targets.mean_upside_pct is not None

    def test_current_price_from_market_not_vendor(self):
        state = _make_p1_state()
        report = build_report_schema(state)
        # vendor_current = 120.0, but current_price comes from market_series
        assert report.decision.current_price is not None
        assert report.decision.current_price != 120.0  # not vendor_current

    def test_forecasts_not_in_financial_series(self):
        state = _make_p1_state()
        report = build_report_schema(state)
        # revenue_estimates are in analyst_consensus, not financial_series
        for fs in report.financial_series:
            for pt in fs.points:
                assert pt.period in ("2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"), \
                    f"Forecast period {pt.period} leaked into financial_series"

    def test_insider_survives_in_timeline(self):
        state = _make_p1_state()
        report = build_report_schema(state)
        insiders = [t for t in report.timeline if getattr(t, "kind", "news") == "insider"]
        assert len(insiders) >= 1

    def test_grounded_catalyst_accepted(self):
        state = _make_p1_state()
        report = build_report_schema(state)
        assert len(report.catalysts) >= 1
        assert report.catalysts[0].status == "scheduled"  # future calendar date
        assert report.catalysts[0].expected_date == "2026-08-28"
        assert "catalysts" not in report.data_quality.missing_sections

    def test_five_dimension_risk_radar(self):
        state = _make_p1_state()
        report = build_report_schema(state)
        assert len(report.risk_views) == 3
        for view in report.risk_views:
            assert view.dimensions is not None, f"{view.role} missing dimensions"

    def test_risk_merge_ranges(self):
        state = _make_p1_state()
        report = build_report_schema(state)
        val_risks = [r for r in report.risk_items if r.name == "Valuation"]
        assert len(val_risks) == 1
        r = val_risks[0]
        # probability: 40, 50, 60 → range 40-60
        assert r.probability_min_pct == 40
        assert r.probability_max_pct == 60
        # downside: 20, 25, 30 → range 20-30
        assert r.downside_impact_min_pct == 20
        assert r.downside_impact_max_pct == 30
        # expected loss: same-entry only: 40%*20=8, 50%*25=12.5, 60%*30=18
        assert r.expected_loss_min_pct == 8.0
        assert r.expected_loss_max_pct == 18.0
        assert r.horizon_bucket == "short"

    def test_html_contains_p1_sections(self):
        state = _make_p1_state()
        report = build_report_schema(state)
        html = render_report_html(report, chart_js="")
        assert 'id="analyst-consensus"' in html
        assert 'id="event-center"' in html  # P1: unified Event Center
        assert 'id="risk-radar-canvas"' in html
        assert "Valuation" in html  # 5-axis label (in chart config or en locale)
        assert "None" not in html

    def test_degraded_analyst_unavailable(self):
        """Analyst expectations missing: report still completes, P0 unchanged."""
        state = _make_p1_state()
        # Remove analyst expectations event
        state.events = [e for e in state.events
                        if getattr(getattr(e, "observation", {}), "get") and
                        e.observation.get("tool_name") != "get_analyst_expectations"]
        report = build_report_schema(state)
        assert report.analyst_consensus is None
        assert report.decision.rating == "Hold"  # P0 still works

    def test_ungrounded_catalyst_dropped(self):
        """Catalyst with no matching evidence_ref → dropped + data-quality note."""
        state = _make_p1_state()
        state.structured_reports["news_analyst"]["catalysts"].append({
            "title": "No evidence catalyst", "catalyst_type": "industry",
            "direction": "positive", "importance": "low",
            "trigger_condition": "N/A", "follow_up": "N/A",
            "evidence_refs": ["nonexistent:ref"],
        })
        report = build_report_schema(state)
        # Only the original grounded catalyst survives
        assert len(report.catalysts) == 1
        assert any("Ungrounded catalyst dropped" in n for n in report.data_quality.notes)

    def test_sndk_safety_preserved(self):
        """Degraded SNDK-like state: rating stays unavailable."""
        state = TradingRunState(ticker="SNDK", trade_date="2026-08-07",
                                run_id="sndk-safety", status="completed")
        state.final_trade_decision = (
            "**Rating**: Hold\n\n"
            "Some analysis...\n\n"
            "Detail: **Rating:** **Sell / Underweight**"
        )
        state.portfolio_decision = None
        report = build_report_schema(state)
        assert report.decision.rating is None
        assert report.decision.structured_available is False

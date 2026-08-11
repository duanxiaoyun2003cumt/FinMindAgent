"""Phase C P2 business contract tests (Task 1 — RED).

Tests the P2 research diagnostics, evidence registry, decision audit,
data reliability summary, PM authority preservation, and degraded-case
behaviors. These tests are RED — the P2 functionality does not exist yet.

No production code is modified in Task 1. All failures signal missing P2
features that Tasks 2-10 will implement.
"""

from __future__ import annotations

import pytest

from finmindagent.reporting.builder import build_report_schema
from finmindagent.reporting.html import render_report_html
from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.events import EventType
from finmindagent.runtime.state import TradingRunState


# ---------------------------------------------------------------------------
# Shared fixture for P2 business state
# ---------------------------------------------------------------------------

def _make_p2_business_state(**overrides) -> TradingRunState:
    """Complete P0+P1 state with structured data for P2 diagnostics."""
    from datetime import date as _date, timedelta

    state = TradingRunState(
        ticker=overrides.get("ticker", "SNDK"),
        trade_date=overrides.get("trade_date", "2026-08-09"),
        run_id=overrides.get("run_id", "p2-biz-test"),
        status="completed",
    )

    # Market series (30 points for SMA20)
    start_date = _date(2026, 7, 1)
    mp = []
    for i in range(30):
        day = (start_date + timedelta(days=i)).isoformat()
        mp.append({"date": day, "open": 1200.0 + i, "high": 1220.0 + i,
                    "low": 1180.0 + i, "close": 1210.0 + i, "volume": 1.5e6})
    state.add_event(EventType.OBSERVATION, actor="market",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data"),
        observation={"ok": True, "tool_name": "get_stock_data",
                     "structured_data": {
                         "market_series": mp,
                         "market_price_basis": {
                             "mode": "provider_adjusted", "provider": "yfinance",
                             "as_of": mp[-1]["date"],
                         },
                     }})

    # Financial metrics (core + secondary)
    state.add_event(EventType.OBSERVATION, actor="f",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_fundamentals"),
        observation={"ok": True, "tool_name": "get_fundamentals", "structured_data": {
            "financial_metrics": [
                {"name": "pe_ratio_ttm", "value": 42.94, "unit": "x", "period": "2026-Q2", "source": "get_fundamentals"},
                {"name": "forward_pe", "value": 4.6, "unit": "x", "period": "2026-Q2", "source": "get_fundamentals"},
                {"name": "price_to_book", "value": 4.78, "unit": "x", "period": "2026-Q2", "source": "get_fundamentals"},
                {"name": "profit_margin", "value": 34.19, "unit": "%", "period": "2026-Q2", "source": "get_fundamentals"},
                {"name": "market_cap", "value": 186e9, "period": "2026-Q2", "source": "get_fundamentals"},
                {"name": "current_ratio", "value": 1.5, "period": "2026-Q2", "source": "get_fundamentals"},
                {"name": "operating_margin", "value": 70.0, "unit": "%", "period": "2026-Q2", "source": "get_fundamentals"},
            ],
            "company_profile": {"name": "SanDisk Corporation", "sector": "Technology",
                                "industry": "Semiconductors", "currency": "USD"},
            "financial_history": [
                {"name": "total_revenue", "period": "FY2024", "value": 18.6, "source": "get_income_statement"},
                {"name": "total_revenue", "period": "FY2025", "value": 22.1, "source": "get_income_statement"},
            ],
        }})

    # Analyst consensus with EPS revisions, price targets, recommendations
    state.add_event(EventType.OBSERVATION, actor="f",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_analyst_expectations"),
        observation={"ok": True, "tool_name": "get_analyst_expectations",
                     "structured_data": {"analyst_consensus": {
            "as_of": "2026-08-09",
            "price_targets": {"vendor_current": 1212.21, "low": 1600.0, "mean": 2116.64,
                              "median": 2150.0, "high": 2600.0},
            "earnings_estimates": [
                {"period": "current_quarter", "average": 0.35, "growth_pct": -20.0},
                {"period": "current_year", "average": 1.40, "growth_pct": -15.0},
            ],
            "eps_revisions": [
                {"period": "current_quarter", "current": 2.5, "days_30_ago": 2.0,
                 "up_30d": 8, "down_30d": 2},
                {"period": "current_year", "current": 5.0, "days_30_ago": 6.0,
                 "up_30d": 3, "down_30d": 9},
            ],
            "recommendations": [{"period": "current_month",
                                  "strong_buy": 5, "buy": 10, "hold": 15, "sell": 3, "strong_sell": 1}],
            "source": "yfinance",
        }}})

    # Earnings history
    state.add_event(EventType.OBSERVATION, actor="f",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_earnings_history"),
        observation={"ok": True, "tool_name": "get_earnings_history",
                     "structured_data": {"earnings_context": {
            "as_of": "2026-08-09", "surprises": [
                {"evidence_id": "earnings_surprise:2026-05-21", "report_date": "2026-05-21",
                 "eps_estimate": 0.88, "reported_eps": 0.94, "surprise_pct": 6.82, "source": "yfinance"},
                {"evidence_id": "earnings_surprise:2026-02-15", "report_date": "2026-02-15",
                 "eps_estimate": 0.75, "reported_eps": 0.78, "surprise_pct": 4.0, "source": "yfinance"},
                {"evidence_id": "earnings_surprise:2025-11-10", "report_date": "2025-11-10",
                 "eps_estimate": 0.70, "reported_eps": 0.67, "surprise_pct": -4.29, "source": "yfinance"},
                {"evidence_id": "earnings_surprise:2025-08-01", "report_date": "2025-08-01",
                 "eps_estimate": 0.65, "reported_eps": 0.65, "surprise_pct": 0.0, "source": "yfinance"},
                {"evidence_id": "earnings_surprise:2025-05-01", "report_date": "2025-05-01",
                 "eps_estimate": 0.60, "reported_eps": None, "surprise_pct": None, "source": "yfinance"},
            ],
            "source": "yfinance",
        }}})

    # Ownership
    state.add_event(EventType.OBSERVATION, actor="s",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_holder_context"),
        observation={"ok": True, "tool_name": "get_holder_context",
                     "structured_data": {"ownership_snapshot": {
            "as_of": "2026-03-31", "insiders_pct": 4.2, "institutions_pct": 68.5,
            "institutions_float_pct": 70.1, "institutions_count": 1450,
            "institutional_holders": [
                {"holder_id": "holder:institutional:vanguard-group-inc:2026-03-31",
                 "holder": "Vanguard Group Inc", "holder_type": "institutional",
                 "pct_out": 8.25, "shares": 1.2e9, "date_reported": "2026-03-31", "source": "yfinance"},
                {"holder_id": "holder:institutional:fidelity:2026-03-31",
                 "holder": "Fidelity Investments", "holder_type": "institutional",
                 "pct_out": 5.0, "shares": 5e8, "date_reported": "2026-03-31", "source": "yfinance"},
                {"holder_id": "holder:institutional:blackrock:2026-03-31",
                 "holder": "BlackRock Inc", "holder_type": "institutional",
                 "pct_out": 4.0, "shares": 4e8, "date_reported": "2026-03-31", "source": "yfinance"},
            ],
            "source": "yfinance",
        }}})

    # Same-period conflicting financial metric (pe_ratio_ttm from balance sheet)
    state.add_event(EventType.OBSERVATION, actor="f",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_balance_sheet"),
        observation={"ok": True, "tool_name": "get_balance_sheet", "structured_data": {
            "financial_metrics": [
                {"name": "pe_ratio_ttm", "value": 26.0, "unit": "x", "period": "2026-Q2",
                 "source": "get_balance_sheet"},
            ],
        }})

    # Calendar
    state.add_event(EventType.OBSERVATION, actor="news",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_company_calendar"),
        observation={"ok": True, "tool_name": "get_company_calendar",
                     "structured_data": {"calendar_events": [{
                         "event_id": "earnings:2026-10-15", "event_type": "earnings",
                         "date": "2026-10-15", "title": "Q4 FY2026 Earnings",
                     }]}})

    # News/timeline
    state.add_event(EventType.OBSERVATION, actor="news",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_news"),
        observation={"ok": True, "tool_name": "get_news",
                     "structured_data": {"timeline": [
                         {"date": "2026-08-06", "event": "Sandisk lowers Q3 guidance",
                          "impact": "negative", "importance": "high",
                          "detail": "Revenue guidance cut.", "source": "Reuters",
                          "url": "https://reuters.com/sndk", "scope": "company"},
                     ]}})

    # Structured reports
    rating = overrides.get("pm_rating", "Sell")
    state.structured_reports = {
        "market_analyst": {"analysis": "x", "score": 3.0, "score_reason": "Downtrend."},
        "fundamentals_analyst": {"analysis": "x", "score": 6.0, "score_reason": "ok",
                                  "financial_quality_summary": "Adequate.",
                                  "quality_warnings": [{"title": "Cash", "evidence": "FCF negative."}]},
        "news_analyst": {"analysis": "x", "score": 4.0, "score_reason": "Mixed.",
                          "catalysts": [
                              {"title": "Q4 earnings reset", "catalyst_type": "earnings",
                               "direction": "negative", "importance": "high",
                               "trigger_condition": "Guidance below consensus",
                               "follow_up": "Review reported results",
                               "evidence_refs": ["earnings:2026-10-15"]},
                          ]},
        "social_sentiment_analyst": {"analysis": "x", "score": 3.0, "score_reason": "Bearish.",
                                      "signals": [
                                          {"title": "Institutional presence", "signal_type": "ownership",
                                           "direction": "positive", "strength": 7.0,
                                           "reasoning": "Vanguard holds 8.25%.",
                                           "evidence_refs": ["holder:institutional:vanguard-group-inc:2026-03-31"]},
                                          {"title": "Ungrounded signal", "signal_type": "other",
                                           "direction": "positive", "strength": 3.0,
                                           "reasoning": "nonexistent",
                                           "evidence_refs": ["nonexistent:ref:12345"]},
                                      ]},
    }

    # Conditionally add bull_researcher for degraded case testing
    if not overrides.get("omit_bull_researcher"):
        state.structured_reports["bull_researcher"] = {
            "summary": "AI demand cycle positive.",
            "arguments": [{"title": "AI demand", "argument": "Accelerating.", "strength": 8.0,
                            "evidence": ["Data center CapEx +45%"]}],
        }

    state.structured_reports.update({
        "bear_researcher": {"summary": "Cyclical peak.", "arguments": [
            {"title": "Cyclical peak", "argument": "Margins at cycle peak.", "strength": 9.0,
             "evidence": ["NAND pricing peaked"]},
        ]},
        "research_manager": {
            "recommendation": rating, "rationale": "Risk/reward favors exit.",
            "strategic_actions": "Reduce exposure.",
            "resolution": {"disagreement": "Cyclical peak vs AI demand.",
                           "key_bull_points": ["AI demand"],
                           "key_bear_points": ["Cyclical mean reversion"],
                           "decisive_factors": ["Guidance cut confirmed downturn"],
                           "final_reason": "Evidence supports selling."},
        },
        "trader": {"action": "Sell", "reasoning": "Exit."},
        "aggressive_risk_analyst": {"summary": "x", "risk_score": 6.0,
                                     "risks": [{"name": "valuation", "severity": "high", "horizon": "6m",
                                                 "evidence": "Multiple.", "risk_key": "valuation",
                                                 "probability_pct": 50, "downside_impact_pct": 20,
                                                 "estimate_basis": "Historical."}]},
        "neutral_risk_analyst": {"summary": "x", "risk_score": 7.0,
                                  "risks": [{"name": "cyclical", "severity": "critical", "horizon": "3-6m",
                                              "evidence": "Peak.", "risk_key": "cyclical_mean_reversion",
                                              "probability_pct": 75, "downside_impact_pct": 30,
                                              "estimate_basis": "Cycle."}]},
        "conservative_risk_analyst": {"summary": "x", "risk_score": 8.0,
                                       "risks": [{"name": "valuation", "severity": "critical", "horizon": "6m",
                                                   "evidence": "Extreme.", "risk_key": "valuation",
                                                   "probability_pct": 60, "downside_impact_pct": 25,
                                                   "estimate_basis": "Compression."}]},
    })

    target = overrides.get("pm_target", 998.0)
    stop = overrides.get("pm_stop", 1430.0)
    state.portfolio_decision = {
        "rating": rating, "executive_summary": "Exit exposure.", "investment_thesis": "Downtrend confirmed.",
        "price_target": target, "stop_loss": stop,
        "position_min_pct": 0.0, "position_max_pct": 5.0,
        "confidence_level": "Medium", "risk_level": "High",
        "time_horizon": "1-3 months",
    }

    # Data quality: one failed source, one truncated, market stale
    state.add_event(EventType.OBSERVATION, actor="dq",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_global_news"),
        observation={"ok": False, "tool_name": "get_global_news", "error": "timeout",
                     "truncated": True})
    state.add_event(EventType.OBSERVATION, actor="dq",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_insider_transactions"),
        observation={"ok": False, "tool_name": "get_insider_transactions",
                     "error": "Provider error"})

    state.metadata["report_status"] = "degraded" if overrides.get("omit_bull_researcher") else "complete"
    state.metadata["report_publishable"] = not overrides.get("omit_bull_researcher", False)

    if overrides.get("omit_bull_researcher"):
        # Task 9: real degraded diagnostics metadata (mirrors runtime contract)
        state.metadata["structured_diagnostics"] = {
            "bull_researcher": {
                "success": False,
                "schema": "DebateResearchOutput",
                "route": "unavailable",
                "failure_stage": "structured_output",
                "failure_message": "unavailable",
            },
        }
        state.metadata["report_completeness"] = {
            "decision": "complete", "scores": "complete",
            "debate": "missing", "risk_analysis": "complete",
            "research_plan": "complete", "trader_proposal": "complete",
            "catalysts": "complete", "sentiment_signals": "complete",
            "financial_quality": "complete",
        }

    return state


# ============================================================================
# Contract Group A — Research Diagnostics (RPT-21/22/23)
# ============================================================================


def _mutate_eps_revisions(state, mutation):
    """Helper: mutate a specific eps_revision row in the synthetic state."""
    for ev in state.events:
        obs = getattr(ev, "observation", None)
        if isinstance(obs, dict) and obs.get("tool_name") == "get_analyst_expectations":
            ac = obs.get("structured_data", {}).get("analyst_consensus", {})
            for r in ac.get("eps_revisions", []):
                mutation(r)


class TestEPSRevisionDiagnostics:
    """RPT-21: EPS revision table → diagnostic summary."""

    def test_eps_revision_summary_computes_direction_and_net_revisions(self):
        """current=2.5, days_30_ago=2.0 → rising +25.0%, net=6."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        diagnostics = getattr(report, "estimate_revision_diagnostics", None)
        assert diagnostics is not None, (
            "RED: estimate_revision_diagnostics not yet implemented"
        )
        assert len(diagnostics) > 0, (
            "RED: estimate_revision_diagnostics empty despite eps_revisions present"
        )

        # Verify deterministic computation
        found = False
        for d in diagnostics:
            if getattr(d, "period", "") == "current_quarter":
                found = True
                assert d.current == 2.5
                assert d.days_30_ago == 2.0
                assert d.change_30d_pct == 25.0  # (2.5 - 2.0) / 2.0 * 100
                assert d.net_revisions_30d == 6  # up_30d(8) - down_30d(2)
                assert d.direction == "rising"
        assert found, "current_quarter diagnostic row missing"

    def test_eps_revision_falling_direction(self):
        """current=5.0 < days_30_ago=6.0 → falling with -16.67%."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        diagnostics = getattr(report, "estimate_revision_diagnostics", None)
        assert diagnostics is not None, "RED: estimate_revision_diagnostics missing"
        assert len(diagnostics) > 0, "RED: empty diagnostics"

        found = False
        for d in diagnostics:
            if getattr(d, "period", "") == "current_year":
                found = True
                assert d.current == 5.0
                assert d.days_30_ago == 6.0
                assert d.direction == "falling"
                # (5.0 - 6.0) / abs(6.0) * 100 = -16.6667
                assert d.change_30d_pct is not None
                assert abs(d.change_30d_pct - (-16.666666666666668)) < 0.01
        assert found, "current_year diagnostic row missing"

    def test_eps_revision_flat_direction(self):
        """current == days_30_ago (non-zero) → flat, change 0.0."""
        state = _make_p2_business_state()
        _mutate_eps_revisions(state, lambda r: (
            r.update(current=2.5, days_30_ago=2.5)
            if r.get("period") == "current_quarter" else None
        ))

        report = build_report_schema(state)
        diagnostics = getattr(report, "estimate_revision_diagnostics", None)
        assert diagnostics is not None, "RED: estimate_revision_diagnostics missing"

        found = False
        for d in diagnostics:
            if getattr(d, "period", "") == "current_quarter":
                found = True
                assert d.direction == "flat"
                assert d.change_30d_pct == 0.0  # legal 0.0 (denominator non-zero)
        assert found, "current_quarter diagnostic row missing"

    @pytest.mark.parametrize(
        ("current", "days_30_ago"),
        [
            (None, 2.0),   # current missing
            (2.0, None),   # days_30_ago missing
        ],
    )
    def test_eps_revision_missing_trend_is_unknown(self, current, days_30_ago):
        """Either endpoint missing → direction unknown, change None."""
        state = _make_p2_business_state()
        _mutate_eps_revisions(state, lambda r: (
            r.update(current=current, days_30_ago=days_30_ago)
            if r.get("period") == "current_quarter" else None
        ))

        report = build_report_schema(state)
        diagnostics = getattr(report, "estimate_revision_diagnostics", None)
        assert diagnostics is not None, "RED: estimate_revision_diagnostics missing"

        found = False
        for d in diagnostics:
            if getattr(d, "period", "") == "current_quarter":
                found = True
                assert d.direction == "unknown"
                assert d.change_30d_pct is None
        assert found, "current_quarter diagnostic row missing"

    def test_zero_denominator_does_not_produce_false_zero(self):
        """days_30_ago=0, current=2.5 → change_30d_pct=None (direction still rising)."""
        state = _make_p2_business_state()
        _mutate_eps_revisions(state, lambda r: (
            r.update(days_30_ago=0.0, current=2.5)
            if r.get("period") == "current_quarter" else None
        ))

        report = build_report_schema(state)
        diagnostics = getattr(report, "estimate_revision_diagnostics", None)
        assert diagnostics is not None, "RED: estimate_revision_diagnostics missing"

        found = False
        for d in diagnostics:
            if getattr(d, "period", "") == "current_quarter":
                found = True
                assert d.change_30d_pct is None, (
                    "Zero denominator must produce None, not 0%."
                )
                # Direction still computable from numeric comparison
                assert d.direction == "rising"
        assert found, "current_quarter diagnostic row missing"

    def test_partial_trend_only_survives(self):
        """current + days_30_ago present, up/down missing → direction+change ok,
        net None. Row must NOT be dropped."""
        state = _make_p2_business_state()
        _mutate_eps_revisions(state, lambda r: (
            r.update(up_30d=None, down_30d=None)
            if r.get("period") == "current_quarter" else None
        ))

        report = build_report_schema(state)
        diagnostics = getattr(report, "estimate_revision_diagnostics", None)
        assert diagnostics is not None, "RED: estimate_revision_diagnostics missing"

        found = False
        for d in diagnostics:
            if getattr(d, "period", "") == "current_quarter":
                found = True
                assert d.direction == "rising"
                assert d.change_30d_pct == 25.0
                assert d.net_revisions_30d is None
        assert found, "current_quarter diagnostic row missing — partial row dropped"

    def test_partial_counts_only_survives(self):
        """up/down present, current+days_30_ago missing → direction unknown,
        change None, net computable. Row must NOT be dropped."""
        state = _make_p2_business_state()
        _mutate_eps_revisions(state, lambda r: (
            r.update(current=None, days_30_ago=None)
            if r.get("period") == "current_quarter" else None
        ))

        report = build_report_schema(state)
        diagnostics = getattr(report, "estimate_revision_diagnostics", None)
        assert diagnostics is not None, "RED: estimate_revision_diagnostics missing"

        found = False
        for d in diagnostics:
            if getattr(d, "period", "") == "current_quarter":
                found = True
                assert d.direction == "unknown"
                assert d.change_30d_pct is None
                assert d.net_revisions_30d == 6  # 8 - 2
        assert found, "current_quarter diagnostic row missing — partial row dropped"

    def test_zero_revision_counts_preserved(self):
        """up=2, down=2 → net 0, not None (0 is legal)."""
        state = _make_p2_business_state()
        _mutate_eps_revisions(state, lambda r: (
            r.update(up_30d=2, down_30d=2)
            if r.get("period") == "current_quarter" else None
        ))

        report = build_report_schema(state)
        diagnostics = getattr(report, "estimate_revision_diagnostics", None)
        assert diagnostics is not None, "RED: estimate_revision_diagnostics missing"

        found = False
        for d in diagnostics:
            if getattr(d, "period", "") == "current_quarter":
                found = True
                assert d.net_revisions_30d == 0
        assert found, "current_quarter diagnostic row missing"

    def test_no_revisions_returns_empty(self):
        """No eps_revisions in consensus → empty list, not placeholder rows."""
        state = _make_p2_business_state()
        for ev in state.events:
            obs = getattr(ev, "observation", None)
            if isinstance(obs, dict) and obs.get("tool_name") == "get_analyst_expectations":
                obs["structured_data"]["analyst_consensus"]["eps_revisions"] = []

        report = build_report_schema(state)
        diagnostics = getattr(report, "estimate_revision_diagnostics", None)
        assert diagnostics is not None, "RED: estimate_revision_diagnostics missing"
        assert diagnostics == []

    def test_pm_decision_unchanged_by_diagnostics(self):
        """Building diagnostics must not mutate PM decision fields."""
        state = _make_p2_business_state(pm_rating="Sell", pm_target=998.0, pm_stop=1430.0)
        report = build_report_schema(state)
        assert report.decision.rating == "Sell"
        assert report.decision.price_target == 998.0
        assert report.decision.stop_loss == 1430.0


def _mutate_earnings_surprises(state, mutation):
    """Helper: mutate earnings surprises in the synthetic state."""
    for ev in state.events:
        obs = getattr(ev, "observation", None)
        if isinstance(obs, dict) and obs.get("tool_name") == "get_earnings_history":
            ec = obs.get("structured_data", {}).get("earnings_context", {})
            surprises = ec.get("surprises", [])
            mutation(surprises)


class TestEarningsRealizationDiagnostics:
    """RPT-22: Earnings history → beat/miss/inline summary."""

    def test_earnings_realization_counts_beats_misses_and_mean_surprise(self):
        """Fixture: 6.82, 4.0, -4.29, 0.0, None →
        sample=4, beat=2, miss=1, inline=1, beat_rate=50.0, mean=1.6325."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        er = getattr(report, "earnings_realization", None)
        assert er is not None, "RED: earnings_realization not yet implemented"

        assert er.sample_size == 4  # None excluded
        assert er.beat_count == 2    # 6.82, 4.0
        assert er.miss_count == 1    # -4.29
        assert er.inline_count == 1  # 0.0
        assert er.beat_rate_pct == 50.0  # 2/4 * 100
        # mean_surprise: (6.82 + 4.0 - 4.29 + 0.0) / 4 = 1.6325
        assert er.mean_surprise_pct == pytest.approx(1.6325)

    def test_none_surprise_not_counted_as_miss(self):
        """None surprise_pct → not a miss, not inline, not 0%."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        er = getattr(report, "earnings_realization", None)
        assert er is not None, "RED: earnings_realization not yet implemented"
        # The None entry should have been excluded from sample
        assert er.sample_size == 4

    def test_latest_surprise_uses_report_date_not_input_order(self):
        """Latest must be determined by report_date, not input list order."""
        state = _make_p2_business_state()
        # Reverse the surprises so the newest (2026-05-21) is NOT first
        _mutate_earnings_surprises(state, lambda surprises: surprises.reverse())

        report = build_report_schema(state)
        er = getattr(report, "earnings_realization", None)
        assert er is not None, "RED: earnings_realization not yet implemented"
        # Latest report_date is 2026-05-21 with surprise_pct=6.82
        assert er.latest_report_date == "2026-05-21"
        assert er.latest_surprise_pct == pytest.approx(6.82)

    def test_empty_earnings_history_produces_no_realization_diagnostic(self):
        """Empty surprises → earnings_realization is None (no placeholder)."""
        # Schema field must exist first (RED if not implemented)
        assert hasattr(report := build_report_schema(_make_p2_business_state()),
                       "earnings_realization"), (
            "RED: earnings_realization not yet implemented"
        )

        state = _make_p2_business_state()
        _mutate_earnings_surprises(state, lambda surprises: surprises.clear())

        report = build_report_schema(state)
        # earnings_context may still exist, but realization must be None
        er = getattr(report, "earnings_realization", None)
        assert er is None, (
            "Empty earnings history must produce no realization diagnostic, "
            f"got {er!r}"
        )

    def test_latest_row_with_missing_surprise_remains_latest(self):
        """Newest row has surprise_pct=None → still latest; stats use valid rows only."""
        state = _make_p2_business_state()
        _mutate_earnings_surprises(state, lambda surprises: (
            surprises.append({
                "evidence_id": "earnings_surprise:2026-06-30",
                "report_date": "2026-06-30",
                "eps_estimate": 1.0, "reported_eps": None,
                "surprise_pct": None, "source": "yfinance",
            })
            and surprises.reverse()
        ))

        report = build_report_schema(state)
        er = getattr(report, "earnings_realization", None)
        assert er is not None, "RED: earnings_realization not yet implemented"

        # Latest is the newest date even though surprise is None
        assert er.latest_report_date == "2026-06-30"
        assert er.latest_surprise_pct is None

        # Statistics still computed from valid rows only (4 valid)
        assert er.sample_size == 4
        assert er.beat_count == 2
        assert er.miss_count == 1
        assert er.inline_count == 1
        assert er.mean_surprise_pct == pytest.approx(1.6325)

    def test_all_missing_surprises_preserve_latest_context_without_false_statistics(self):
        """All surprise_pct=None → diagnostic exists with sample_size=0 and
        no fabricated statistics, but real latest date."""
        state = _make_p2_business_state()
        _mutate_earnings_surprises(state, lambda surprises: (
            surprises.clear()
        ))
        # Add two rows with valid dates but missing surprises
        _mutate_earnings_surprises(state, lambda surprises: surprises.extend([
            {"evidence_id": "earnings_surprise:2026-05-21", "report_date": "2026-05-21",
             "eps_estimate": 0.88, "reported_eps": None, "surprise_pct": None, "source": "yfinance"},
            {"evidence_id": "earnings_surprise:2026-02-15", "report_date": "2026-02-15",
             "eps_estimate": 0.75, "reported_eps": None, "surprise_pct": None, "source": "yfinance"},
        ]))

        report = build_report_schema(state)
        er = getattr(report, "earnings_realization", None)
        assert er is not None, "RED: earnings_realization not yet implemented"

        # Real context exists → diagnostic present, but no fabricated stats
        assert er.sample_size == 0
        assert er.beat_count == 0
        assert er.miss_count == 0
        assert er.inline_count == 0
        assert er.beat_rate_pct is None
        assert er.mean_surprise_pct is None
        # Latest date from real rows
        assert er.latest_report_date == "2026-05-21"
        assert er.latest_surprise_pct is None


def _mutate_price_targets(state, mutation):
    """Helper: mutate analyst price targets in the synthetic state."""
    for ev in state.events:
        obs = getattr(ev, "observation", None)
        if isinstance(obs, dict) and obs.get("tool_name") == "get_analyst_expectations":
            ac = obs.get("structured_data", {}).get("analyst_consensus", {})
            pt = ac.get("price_targets", {})
            mutation(pt)


class TestValuationDiagnostics:
    """RPT-23: Valuation context extended with target dispersion, no valuation score."""

    def test_valuation_diagnostics_include_target_dispersion(self):
        """Exact propagation + dispersion: low=1600, mean=2116.64, high=2600
        → dispersion ≈ 47.24%."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        # No valuation score anywhere
        assert not hasattr(report, "valuation_score"), "P2 must not introduce valuation_score."
        assert not hasattr(report, "composite_valuation_score"), (
            "P2 must not introduce composite_valuation_score."
        )

        # Valuation context must exist with exact target fields
        vc = report.valuation_context
        assert vc is not None, "ValuationContext missing with analyst targets present."
        assert vc.analyst_target_low == 1600.0
        assert vc.analyst_target_mean == 2116.64
        assert vc.analyst_target_high == 2600.0

        # Production dispersion field, not test-local recomputation
        expected = (2600.0 - 1600.0) / abs(2116.64) * 100  # ≈ 47.24
        assert vc.analyst_target_dispersion_pct == pytest.approx(expected, rel=1e-6), (
            f"analyst_target_dispersion_pct {vc.analyst_target_dispersion_pct} != {expected}"
        )

    def test_no_valuation_score_fields(self):
        """ValuationContext must not carry any score-like field."""
        from finmindagent.reporting.schemas import ValuationContext
        forbidden = ["valuation_score", "composite_valuation_score",
                     "fair_value_score", "valuation_confidence_score"]
        for field in forbidden:
            assert not hasattr(ValuationContext(), field), (
                f"ValuationContext must not have '{field}'."
            )

    def test_target_dispersion_zero_mean_is_none(self):
        """mean=0 → dispersion None, no divide-by-zero, no 0%."""
        state = _make_p2_business_state()
        _mutate_price_targets(state, lambda pt: pt.update(low=100.0, mean=0.0, high=200.0))

        report = build_report_schema(state)
        vc = report.valuation_context
        assert vc is not None
        assert vc.analyst_target_low == 100.0
        assert vc.analyst_target_mean == 0.0
        assert vc.analyst_target_high == 200.0
        assert vc.analyst_target_dispersion_pct is None

    @pytest.mark.parametrize(
        ("low", "mean", "high"),
        [
            (None, 150.0, 200.0),   # low missing
            (100.0, 150.0, None),   # high missing
        ],
    )
    def test_target_dispersion_missing_endpoint_is_none(self, low, mean, high):
        """Missing low or high → dispersion None; present fields preserved."""
        state = _make_p2_business_state()
        _mutate_price_targets(state, lambda pt: pt.update(low=low, mean=mean, high=high))

        report = build_report_schema(state)
        vc = report.valuation_context
        assert vc is not None
        assert vc.analyst_target_low == low
        assert vc.analyst_target_mean == mean
        assert vc.analyst_target_high == high
        assert vc.analyst_target_dispersion_pct is None

    def test_pm_target_remains_independent_from_analyst_target(self):
        """PM target 998.0 vs analyst mean 2116.64 — neither overwrites the other."""
        state = _make_p2_business_state(pm_rating="Sell", pm_target=998.0, pm_stop=1430.0)
        report = build_report_schema(state)

        assert report.decision.price_target == 998.0
        vc = report.valuation_context
        assert vc is not None
        assert vc.analyst_target_mean == 2116.64
        # Independence both ways
        assert report.decision.price_target != vc.analyst_target_mean


# ============================================================================
# Contract Group B — Ownership Semantics (RPT-25)
# ============================================================================


def _mutate_ownership_holders(state, mutation):
    """Helper: mutate institutional holders in the synthetic state."""
    for ev in state.events:
        obs = getattr(ev, "observation", None)
        if isinstance(obs, dict) and obs.get("tool_name") == "get_holder_context":
            os_data = obs.get("structured_data", {}).get("ownership_snapshot", {})
            mutation(os_data)


class TestMarketRiskBusinessInvariant:
    """RPT-24: Market risk must derive from the authoritative market_series
    and declared price basis — no second price system."""

    def test_market_risk_uses_authoritative_market_series_basis(self):
        """risk as_of == latest market date; basis explicit; source deterministic."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        # Market series non-empty
        assert len(report.market_series) > 0
        latest_market_date = max(point.date for point in report.market_series)

        # Price basis explicit (P1 authority)
        assert report.market_price_basis.mode == "provider_adjusted"
        assert report.market_price_basis.provider == "yfinance"
        # as_of compatible with latest market date
        assert report.market_price_basis.as_of in ("", latest_market_date)

        # Risk snapshot from deterministic path, same as_of
        mrs = report.market_risk_snapshot
        assert mrs is not None, "Market risk snapshot missing with market data present."
        assert mrs.as_of == latest_market_date
        assert mrs.source == "deterministic_ohlcv"

    def test_market_risk_numeric_source_of_truth(self):
        """return_20d_pct must be reproducible from report.market_series closes."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        mrs = report.market_risk_snapshot
        assert mrs is not None
        assert mrs.return_20d_pct is not None, "return_20d_pct missing with 30 points."

        points = report.market_series
        expected = (points[-1].close / points[-21].close - 1) * 100
        assert mrs.return_20d_pct == pytest.approx(expected, rel=1e-6), (
            "return_20d_pct must match canonical market_series computation."
        )


class TestOwnershipDiagnostics:
    """RPT-25: Snapshot-only ownership, no flow/accumulation/distribution."""

    def test_ownership_snapshot_never_claims_flow(self):
        """Ownership diagnostic must not produce buy/sell/flow claims."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        od = getattr(report, "ownership_diagnostic", None)
        assert od is not None, "RED: ownership_diagnostic not yet implemented"

        # Exact snapshot semantics
        assert od.as_of == "2026-03-31"
        assert od.institutional_pct == 68.5
        assert od.insiders_pct == 4.2
        # Top holder: 8.25; top5: 8.25+5.0+4.0 = 17.25
        assert od.top_holder_pct == 8.25
        assert od.top5_institutional_pct == pytest.approx(17.25)
        # Retained count means retained records, NOT institutions_count (1450)
        assert od.retained_holder_count == 3
        assert od.retained_holder_count != 1450

        # Must NOT have flow fields
        forbidden = ["net_buy", "net_sell", "ownership_flow", "institutional_accumulation",
                     "institutional_distribution", "buy_signal", "sell_signal"]
        for field in forbidden:
            assert not hasattr(od, field), (
                f"OwnershipDiagnostic must not have field '{field}'."
            )

    def test_ownership_concentration_ignores_missing_pct(self):
        """pct_out=None holder excluded from concentration but counted as retained."""
        state = _make_p2_business_state()
        _mutate_ownership_holders(state, lambda os_data: (
            next(h for h in os_data["institutional_holders"]
                 if h["holder"] == "Fidelity Investments").update(pct_out=None)
        ))

        report = build_report_schema(state)
        od = getattr(report, "ownership_diagnostic", None)
        assert od is not None, "RED: ownership_diagnostic not yet implemented"

        # Valid pcts: 8.25, 4.0 (Fidelity None excluded)
        assert od.top_holder_pct == 8.25
        assert od.top5_institutional_pct == pytest.approx(12.25)
        # Retained count still 3 records
        assert od.retained_holder_count == 3

    def test_ownership_empty_retained_holders_does_not_fake_zero_concentration(self):
        """Empty holder list → concentration None (not 0), count 0."""
        state = _make_p2_business_state()
        _mutate_ownership_holders(state, lambda os_data: os_data.update(institutional_holders=[]))

        report = build_report_schema(state)
        od = getattr(report, "ownership_diagnostic", None)
        assert od is not None, "RED: ownership_diagnostic not yet implemented"

        # Snapshot itself is real data
        assert od.institutional_pct == 68.5
        assert od.insiders_pct == 4.2
        # No concentration data → None, never 0
        assert od.top_holder_pct is None
        assert od.top5_institutional_pct is None
        assert od.retained_holder_count == 0

    def test_no_ownership_snapshot_produces_no_diagnostic(self):
        """No ownership snapshot → no diagnostic (no fake 0% placeholder)."""
        state = _make_p2_business_state()
        for ev in state.events:
            obs = getattr(ev, "observation", None)
            if isinstance(obs, dict) and obs.get("tool_name") == "get_holder_context":
                obs["structured_data"] = {}

        report = build_report_schema(state)
        assert report.ownership_snapshot is None
        assert getattr(report, "ownership_diagnostic", None) is None

    def test_html_no_ownership_flow_language(self):
        """HTML must not use flow language (增持, 减持, 资金流入)."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="", locale="zh-CN")

        import re
        visible = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', rendered, flags=re.DOTALL)
        visible = re.sub(r'<[^>]+>', ' ', visible)
        visible = re.sub(r'\s+', ' ', visible)

        flow_terms = ["增持", "减持", "资金流入", "资金流出", "净买入", "净卖出"]
        for term in flow_terms:
            assert term not in visible, (
                f"Ownership flow language '{term}' must not appear in report."
            )


# ============================================================================
# Contract Group C — Stable Evidence Registry (RPT-26)
# ============================================================================


class TestEvidenceRegistry:
    """RPT-26: Unified structured evidence registry with stable IDs."""

    def test_evidence_registry_exists(self):
        """ReportSchema must carry an evidence_registry."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        assert hasattr(report, "evidence_registry"), (
            "ReportSchema missing evidence_registry — P2 feature not yet implemented."
        )

    def test_evidence_registry_exact_ids(self):
        """Registry must contain the exact expected evidence IDs (product path)."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        registry = getattr(report, "evidence_registry", None)
        assert registry is not None, "RED: evidence_registry not yet implemented"

        ids = [item.evidence_id for item in registry]
        expected_ids = {
            "market:return_20d:2026-07-30",
            "market:drawdown_60:2026-07-30",
            "analyst:target_mean:2026-08-09",
            "analyst:eps_revision:current_quarter:2026-08-09",
            "analyst:eps_revision:current_year:2026-08-09",
            "earnings_surprise:2026-05-21",
            "earnings_surprise:2026-02-15",
            "earnings_surprise:2025-11-10",
            "earnings_surprise:2025-08-01",
            "earnings_surprise:2025-05-01",
            "ownership:summary:2026-03-31",
            "holder:institutional:vanguard-group-inc:2026-03-31",
            "holder:institutional:fidelity:2026-03-31",
            "holder:institutional:blackrock:2026-03-31",
            "risk:valuation",
            "risk:cyclical_mean_reversion",
            "catalyst:earnings:2026-10-15",
        }
        missing = expected_ids - set(ids)
        assert not missing, (
            f"Evidence registry missing exact IDs: {sorted(missing)}"
        )

    def test_evidence_ids_unique(self):
        """All evidence IDs must be unique — one ID, one fact."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        registry = getattr(report, "evidence_registry", None)
        assert registry is not None, "RED: evidence_registry not yet implemented"

        ids = [item.evidence_id for item in registry]
        assert len(ids) == len(set(ids)), "Duplicate evidence_id in registry."

    def test_duplicate_underlying_fact_single_entry(self):
        """Duplicate earnings row with same evidence_id → exactly one entry."""
        state = _make_p2_business_state()
        _mutate_earnings_surprises(state, lambda surprises: (
            surprises.append({
                "evidence_id": "earnings_surprise:2026-05-21",  # duplicate
                "report_date": "2026-05-21", "eps_estimate": 0.88,
                "reported_eps": 0.94, "surprise_pct": 6.82, "source": "yfinance",
            })
        ))

        report = build_report_schema(state)
        registry = getattr(report, "evidence_registry", None)
        assert registry is not None, "RED: evidence_registry not yet implemented"

        ids = [item.evidence_id for item in registry]
        assert ids.count("earnings_surprise:2026-05-21") == 1, (
            "Duplicate underlying fact must produce exactly one registry entry."
        )

    def test_evidence_registry_deterministic_order(self):
        """Registry order must not depend on input list order."""
        state_1 = _make_p2_business_state()

        state_2 = _make_p2_business_state()
        # Reverse earnings and holder rows in the second state
        _mutate_earnings_surprises(state_2, lambda surprises: surprises.reverse())
        _mutate_ownership_holders(state_2, lambda os_data: (
            os_data.update(institutional_holders=list(reversed(os_data["institutional_holders"])))
        ))

        report_1 = build_report_schema(state_1)
        report_2 = build_report_schema(state_2)
        registry_1 = getattr(report_1, "evidence_registry", None)
        registry_2 = getattr(report_2, "evidence_registry", None)
        assert registry_1 is not None and registry_2 is not None, (
            "RED: evidence_registry not yet implemented"
        )

        ids_1 = [item.evidence_id for item in registry_1]
        ids_2 = [item.evidence_id for item in registry_2]
        assert ids_1 == ids_2, (
            "Registry order depends on input order — must be deterministic."
        )

    def test_evidence_directions(self):
        """Product evidence directions must match canonical rules."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        registry = getattr(report, "evidence_registry", None)
        assert registry is not None, "RED: evidence_registry not yet implemented"

        by_id = {item.evidence_id: item.direction for item in registry}

        # Market return: closes rise 1210→1239 → positive
        assert by_id.get("market:return_20d:2026-07-30") == "positive"
        # Drawdown magnitude: context, unknown
        assert by_id.get("market:drawdown_60:2026-07-30") == "unknown"
        # Analyst mean upside: (2116.64-1212.21)/1212.21 > 0 → positive
        assert by_id.get("analyst:target_mean:2026-08-09") == "positive"
        # EPS revision: rising → positive; falling → negative
        assert by_id.get("analyst:eps_revision:current_quarter:2026-08-09") == "positive"
        assert by_id.get("analyst:eps_revision:current_year:2026-08-09") == "negative"
        # Earnings surprise: >0 positive, <0 negative, ==0 neutral, None unknown
        assert by_id.get("earnings_surprise:2026-05-21") == "positive"
        assert by_id.get("earnings_surprise:2025-11-10") == "negative"
        assert by_id.get("earnings_surprise:2025-08-01") == "neutral"
        assert by_id.get("earnings_surprise:2025-05-01") == "unknown"
        # Ownership / risk: context, unknown
        assert by_id.get("ownership:summary:2026-03-31") == "unknown"
        assert by_id.get("holder:institutional:vanguard-group-inc:2026-03-31") == "unknown"
        assert by_id.get("risk:valuation") == "unknown"

    def test_invalid_evidence_is_not_fabricated(self):
        """Ungrounded evidence_refs must not enter the registry."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        registry = getattr(report, "evidence_registry", None)
        assert registry is not None, "RED: evidence_registry not yet implemented"

        # nonexistent:ref:12345 should not appear as evidence
        for item in registry:
            ref = getattr(item, "source_ref", "") if hasattr(item, "source_ref") else ""
            evidence_id = getattr(item, "evidence_id", "")
            assert "nonexistent" not in str(evidence_id).lower(), (
                "Invalid evidence_ref must not enter registry."
            )
            assert "nonexistent" not in str(ref).lower()

    def test_risk_key_produces_single_risk_entry(self):
        """Same risk_key from multiple analysts → ONE risk:<key> entry."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        registry = getattr(report, "evidence_registry", None)
        assert registry is not None, "RED: evidence_registry not yet implemented"

        ids = [item.evidence_id for item in registry]
        # valuation appears from aggressive + conservative analysts → one entry
        assert ids.count("risk:valuation") == 1
        assert ids.count("risk:cyclical_mean_reversion") == 1

    def test_catalyst_evidence_in_registry(self):
        """Grounded catalyst → catalyst:<stable-ref> with exact fields."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        registry = getattr(report, "evidence_registry", None)
        assert registry is not None, "RED: evidence_registry not yet implemented"

        by_id = {item.evidence_id: item for item in registry}
        cat = by_id.get("catalyst:earnings:2026-10-15")
        assert cat is not None, (
            "Grounded catalyst missing from registry — expected catalyst:earnings:2026-10-15"
        )
        assert cat.domain == "news"
        assert cat.direction.value == "negative"
        assert cat.source == "news_analyst"
        assert cat.source_ref == "earnings:2026-10-15"
        assert cat.as_of == "2026-10-15"

    def test_catalyst_stable_ref_dedup(self):
        """Two catalysts with same grounded stable ref → one registry fact."""
        state = _make_p2_business_state()
        # Add a second catalyst referencing the same calendar event
        state.structured_reports["news_analyst"]["catalysts"].append({
            "title": "Duplicate earnings catalyst", "catalyst_type": "earnings",
            "direction": "negative", "importance": "medium",
            "trigger_condition": "x", "follow_up": "y",
            "evidence_refs": ["earnings:2026-10-15"],
        })

        report = build_report_schema(state)
        registry = getattr(report, "evidence_registry", None)
        assert registry is not None, "RED: evidence_registry not yet implemented"

        ids = [item.evidence_id for item in registry]
        assert ids.count("catalyst:earnings:2026-10-15") == 1, (
            "Same grounded stable ref must produce one registry fact."
        )

    def test_mixed_grounded_sentiment_refs_keep_only_valid_refs(self):
        """Mixed invalid+valid sentiment refs → signal kept, only valid refs retained."""
        state = _make_p2_business_state()
        # Invalid ref FIRST, valid holder ref SECOND
        state.structured_reports["social_sentiment_analyst"]["signals"].append({
            "title": "Mixed ref signal", "signal_type": "ownership",
            "direction": "positive", "strength": 6.0,
            "reasoning": "test",
            "evidence_refs": [
                "nonexistent:ref:12345",
                "holder:institutional:vanguard-group-inc:2026-03-31",
            ],
        })

        report = build_report_schema(state)
        registry = getattr(report, "evidence_registry", None)
        assert registry is not None, "RED: evidence_registry not yet implemented"

        # Find the sentiment evidence for this signal
        by_id = {item.evidence_id: item for item in registry}
        sent = by_id.get("sentiment:mixed_ref_signal")
        assert sent is not None, "Grounded mixed-ref signal should survive."
        # Invalid ref must never become source_ref
        assert sent.source_ref == "holder:institutional:vanguard-group-inc:2026-03-31"
        assert "nonexistent" not in sent.source_ref


# ============================================================================
# Contract Group D — Decision Evidence Audit (RPT-27)
# ============================================================================


class TestDecisionEvidenceAudit:
    """RPT-27: Deterministic audit of support/conflict/context vs PM decision."""

    def test_decision_audit_exists(self):
        """ReportSchema must carry a decision_evidence_audit."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        assert hasattr(report, "decision_evidence_audit"), (
            "ReportSchema missing decision_evidence_audit — P2 feature not yet implemented."
        )

    def test_sell_decision_marks_negative_market_return_as_support(self):
        """PM=Sell + negative 20d return → exact market:return_20d ID in support."""
        state = _make_p2_business_state(pm_rating="Sell")
        # Make market return negative: drop closes so latest < 20 sessions ago
        for ev in state.events:
            obs = getattr(ev, "observation", None)
            if isinstance(obs, dict) and obs.get("tool_name") == "get_stock_data":
                mp = obs["structured_data"]["market_series"]
                for i, pt in enumerate(mp):
                    pt["close"] = 1200.0 - i  # descending → negative return

        report = build_report_schema(state)
        audit = getattr(report, "decision_evidence_audit", None)
        assert audit is not None, "RED: decision_evidence_audit not yet implemented"

        support_refs = set(getattr(audit, "support_refs", []) or [])
        assert "market:return_20d:2026-07-30" in support_refs, (
            f"market:return_20d must be in support refs, got {sorted(support_refs)}"
        )

    def test_sell_decision_marks_positive_consensus_upside_as_conflict(self):
        """PM=Sell + positive consensus upside → exact analyst:target_mean in conflict."""
        state = _make_p2_business_state(pm_rating="Sell")
        report = build_report_schema(state)
        audit = getattr(report, "decision_evidence_audit", None)
        assert audit is not None, "RED: decision_evidence_audit not yet implemented"

        conflict_refs = set(getattr(audit, "conflict_refs", []) or [])
        assert "analyst:target_mean:2026-08-09" in conflict_refs, (
            f"analyst:target_mean must be in conflict refs, got {sorted(conflict_refs)}"
        )

    @pytest.mark.parametrize("rating", ["Buy", "Overweight"])
    def test_buy_like_decision_marks_positive_revision_as_support(self, rating):
        """Buy/Overweight + rising EPS revision → exact revision ID in support."""
        state = _make_p2_business_state(pm_rating=rating)
        report = build_report_schema(state)
        audit = getattr(report, "decision_evidence_audit", None)
        assert audit is not None, "RED: decision_evidence_audit not yet implemented"

        support_refs = set(getattr(audit, "support_refs", []) or [])
        assert "analyst:eps_revision:current_quarter:2026-08-09" in support_refs, (
            f"Rising EPS revision must support {rating}, got {sorted(support_refs)}"
        )

    @pytest.mark.parametrize("rating", ["Sell", "Underweight"])
    def test_sell_like_decision_direction_negative(self, rating):
        """Sell/Underweight → decision_direction negative."""
        state = _make_p2_business_state(pm_rating=rating)
        report = build_report_schema(state)
        audit = getattr(report, "decision_evidence_audit", None)
        assert audit is not None, "RED: decision_evidence_audit not yet implemented"
        assert audit.decision_direction == "negative"

    def test_hold_decision_all_evidence_is_context(self):
        """PM=Hold → support/conflict empty; directional evidence in context."""
        state = _make_p2_business_state(pm_rating="Hold", pm_target=250.0, pm_stop=180.0)
        report = build_report_schema(state)
        audit = getattr(report, "decision_evidence_audit", None)
        assert audit is not None, "RED: decision_evidence_audit not yet implemented"

        support_refs = getattr(audit, "support_refs", []) or []
        conflict_refs = getattr(audit, "conflict_refs", []) or []
        assert len(support_refs) == 0, (
            f"Hold decision must not produce support refs, got {len(support_refs)}."
        )
        assert len(conflict_refs) == 0, (
            f"Hold decision must not produce conflict refs, got {len(conflict_refs)}."
        )
        # Directional evidence must actually appear in context_refs (not empty arrays)
        context_refs = set(getattr(audit, "context_refs", []) or [])
        assert "analyst:eps_revision:current_quarter:2026-08-09" in context_refs, (
            "Directional EPS evidence must appear in context for Hold."
        )

    def test_legacy_prose_decision_never_authoritative_for_audit(self):
        """No structured portfolio_decision → audit is None, even when P0
        legacy markdown parser accepts the prose."""
        state = _make_p2_business_state()
        state.portfolio_decision = None
        state.final_trade_decision = (
            "**Rating**: Sell\n\n"
            "**Executive Summary**: Exit exposure.\n\n"
            "**Investment Thesis**: Downtrend confirmed.\n\n"
            "**Price Target**: 998.0\n\n"
            "**Time Horizon**: 1-3 months"
        )

        report = build_report_schema(state)
        # P0 parser may accept this as a display-level structured decision
        assert report.decision.structured_available is True
        # But Decision Audit must NOT treat legacy prose as authoritative
        assert report.decision_evidence_audit is None, (
            "Decision Evidence Audit must be None when no structured portfolio_decision "
            "exists — never derive direction from prose."
        )


# ============================================================================
# Contract Group E — Data Reliability (RPT-28)
# ============================================================================


class TestDataReliabilitySummary:
    """RPT-28: Source counts, no composite trust/quality score."""

    def test_data_reliability_summary_exists(self):
        """ReportSchema must carry a data_reliability summary."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        assert hasattr(report, "data_reliability"), (
            "ReportSchema missing data_reliability — P2 feature not yet implemented."
        )

    def _reliability(self, report):
        dr = getattr(report, "data_reliability", None)
        assert dr is not None, "RED: data_reliability not yet implemented"
        return dr

    def test_reliability_all_six_counts_match_data_quality(self):
        """All six reliability fields must mirror report.data_quality exactly."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        dr = self._reliability(report)

        dq = report.data_quality
        assert dr.source_count == len(dq.sources)
        assert dr.failed_source_count == sum(1 for s in dq.sources if not s.ok)
        assert dr.truncated_source_count == sum(1 for s in dq.sources if s.truncated)
        assert dr.conflict_count == len(dq.conflicts)
        assert dr.missing_section_count == len(dq.missing_sections)
        assert dr.stale_market_data == any(
            "stale" in note.lower() for note in dq.notes
        )

    def test_stale_true_when_note_present(self):
        """stale_market_data True when data_quality has stale note."""
        state = _make_p2_business_state()
        # trade_date 2026-08-09, latest market 2026-07-30 → 10 days gap (>7)
        report = build_report_schema(state)
        dr = self._reliability(report)

        dq = report.data_quality
        stale_note = any("stale" in n.lower() for n in dq.notes)
        assert stale_note, "Fixture should produce a stale-market note."
        assert dr.stale_market_data is True

    def test_stale_false_when_dates_close(self):
        """stale_market_data False when market dates are recent enough."""
        state = _make_p2_business_state(trade_date="2026-08-01")
        report = build_report_schema(state)
        dr = self._reliability(report)

        dq = report.data_quality
        stale_note = any("stale" in n.lower() for n in dq.notes)
        assert not stale_note, "Close dates should not produce a stale note."
        assert dr.stale_market_data is False

    def test_reliability_has_counts_but_no_composite_score(self):
        """Counts only, no trust_score/reliability_score/quality_score."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        dr = self._reliability(report)

        assert hasattr(dr, "source_count")
        assert hasattr(dr, "failed_source_count")
        assert hasattr(dr, "truncated_source_count")
        assert hasattr(dr, "conflict_count")
        assert hasattr(dr, "missing_section_count")
        assert hasattr(dr, "stale_market_data")

        # Must NOT have composite scores
        forbidden = ["trust_score", "reliability_score", "quality_score",
                     "confidence_score", "composite_score"]
        for field in forbidden:
            assert not hasattr(dr, field), (
                f"Data reliability must not have '{field}'."
            )


# ============================================================================
# Contract Group F — PM Authority Preservation (RPT-27)
# ============================================================================


class TestPMAuthorityPreservation:
    """P2 diagnostics must never modify PM decision fields."""

    def test_p2_diagnostics_do_not_change_pm_decision(self):
        """PM Sell/998/1430 must survive all P2 processing."""
        state = _make_p2_business_state(pm_rating="Sell", pm_target=998.0, pm_stop=1430.0)
        report = build_report_schema(state)

        # These must remain EXACTLY as set
        assert report.decision.rating == "Sell"
        assert report.decision.price_target == 998.0
        assert report.decision.stop_loss == 1430.0
        assert report.decision.position_max_pct == 5.0
        assert report.decision.position_min_pct == 0.0

    def test_sell_with_positive_analyst_consensus_still_sell(self):
        """Even when consensus is bullish, PM Sell must survive."""
        state = _make_p2_business_state(pm_rating="Sell", pm_target=998.0)
        report = build_report_schema(state)

        assert report.decision.rating == "Sell", (
            "PM rating changed — P2 must not override PM authority."
        )
        assert report.decision.price_target == 998.0


# ============================================================================
# Contract Group G — Progressive Disclosure (RPT-29)
# ============================================================================


class TestProgressiveDisclosure:
    """P2 HTML: compact diagnostics + collapsed raw details + audit block."""

    def test_p2_diagnostics_sections_exist(self):
        """All three P2 section IDs must exist for the complete synthetic state."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="")

        assert 'id="decision-evidence-audit"' in rendered, (
            "Missing id='decision-evidence-audit' section."
        )
        assert 'id="research-diagnostics"' in rendered, (
            "Missing id='research-diagnostics' section."
        )
        assert 'id="data-reliability-summary"' in rendered, (
            "Missing id='data-reliability-summary' section."
        )

    def test_decision_evidence_audit_placement(self):
        """Audit block must appear after divergence warning, before market chart."""
        state = _make_p2_business_state(pm_rating="Sell")
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="")

        pos_divergence = rendered.find('class="divergence-warning"')
        pos_audit = rendered.find('id="decision-evidence-audit"')
        pos_market = rendered.find('id="market-chart"')
        assert pos_divergence >= 0, "Divergence warning missing."
        assert pos_audit >= 0, "Audit section missing."
        assert pos_market >= 0, "Market chart missing."
        assert pos_divergence < pos_audit < pos_market, (
            "Primary order must be: divergence → decision-evidence-audit → market-chart."
        )

    def test_audit_block_is_not_second_decision(self):
        """Audit block must not contain rating pill, decision hero, target/stop/position."""
        state = _make_p2_business_state(pm_rating="Sell")
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="")

        audit_zone = rendered.split('id="decision-evidence-audit"')[1]
        audit_zone = audit_zone.split("</section>")[0]
        assert "rating-pill" not in audit_zone
        assert "decision-hero" not in audit_zone
        assert "price_target" not in audit_zone
        assert "stop_loss" not in audit_zone
        assert "position" not in audit_zone.lower()

    def test_raw_detail_in_collapsible_details(self):
        """Raw detail rows must be inside <details class='research-raw-details'>
        without the 'open' attribute."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="")

        # Research diagnostics section contains raw details
        diag_zone = rendered.split('id="research-diagnostics"')[1]
        diag_zone = diag_zone.split("</section>")[0]

        assert '<details class="research-raw-details"' in diag_zone, (
            "Raw detail rows must be inside <details class='research-raw-details'>."
        )
        assert '<details class="research-raw-details" open' not in diag_zone, (
            "Raw details must not be open by default."
        )

    def test_stable_dom_ids_preserved_once(self):
        """P1/P2 stable section IDs must each appear exactly once."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="")

        for sid in ("analyst-consensus", "earnings-realization",
                    "valuation-context", "market-risk-stats", "ownership-sentiment"):
            assert rendered.count(f'id="{sid}"') == 1, (
                f"Stable section id='{sid}' must appear exactly once."
            )

    def test_no_duplicate_p2_panels(self):
        """Research Diagnostics must reorganize, not duplicate P2 panels."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="")

        # Count occurrences of key panel markers
        consensus_count = rendered.count('id="analyst-consensus"')
        market_risk_count = rendered.count('id="market-risk-stats"')
        ownership_count = rendered.count('id="ownership-sentiment"')
        assert consensus_count == 1 and market_risk_count == 1 and ownership_count == 1, (
            "Duplicate P2 panels detected — Research Diagnostics must reorganize, not copy."
        )

    def test_zh_cn_no_raw_enum_labels(self):
        """zh-CN must not expose raw evidence/relation enum labels in UI chrome."""
        state = _make_p2_business_state(pm_rating="Sell")
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="", locale="zh-CN")

        import re
        visible = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', rendered, flags=re.DOTALL)
        visible = re.sub(r'<[^>]+>', ' ', visible)
        visible = re.sub(r'\s+', ' ', visible)

        # Section titles must be Chinese
        assert "支持当前结论" in rendered, "Audit support bucket must be localized (支持当前结论)."
        assert "反向证据" in rendered, "Audit conflict bucket must be localized (反向证据)."
        assert "背景信息" in rendered, "Audit context bucket must be localized (背景信息)."

        # Compact diagnostics must NOT show raw period enum or '30d:'/'net:' labels
        diag_zone = rendered.split('id="research-diagnostics"')[1]
        if 'class="appendix-group"' in rendered:
            diag_zone = diag_zone.split('class="appendix-group"')[0]
        visible_diag = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', diag_zone, flags=re.DOTALL)
        visible_diag = re.sub(r'<[^>]+>', ' ', visible_diag)
        visible_diag = re.sub(r'\s+', ' ', visible_diag)

        # Raw enum period labels must not appear as visible labels
        assert "current_quarter" not in visible_diag, (
            "Raw period enum 'current_quarter' leaked into zh-CN compact diagnostics."
        )
        assert "30d:" not in visible_diag, "Raw '30d:' label leaked in zh-CN."
        assert "net:" not in visible_diag, "Raw 'net:' label leaked in zh-CN."

        # Deterministic evidence titles must be localized (not English UI)
        for en_title in ("20-day return", "60-session max drawdown",
                         "Analyst consensus mean target", "Earnings surprise",
                         "Ownership summary"):
            assert en_title not in visible, (
                f"Deterministic English title '{en_title}' leaked into zh-CN UI."
            )

    def test_zh_cn_diagnostics_localized_periods_and_labels(self):
        """zh-CN research diagnostics show localized periods (本季度) and labels."""
        state = _make_p2_business_state(pm_rating="Sell")
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="", locale="zh-CN")

        assert "本季度" in rendered, "current_quarter must render as 本季度 in zh-CN."
        assert "30日变化" in rendered, "30-day change label missing in zh-CN."
        assert "净修正" in rendered, "net revisions label missing in zh-CN."

    def test_audit_exposes_evidence_details(self):
        """Audit must expose evidence_id with useful value_text/source/as_of."""
        state = _make_p2_business_state(pm_rating="Sell")
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="")

        audit_zone = rendered.split('id="decision-evidence-audit"')[1]
        audit_zone = audit_zone.split("</section>")[0]
        # Evidence IDs appear in audit block
        assert "market:return_20d:2026-07-30" in audit_zone or \
               "analyst:target_mean:2026-08-09" in audit_zone, (
            "Audit block must expose concrete evidence IDs."
        )

    def test_reliability_summary_in_appendix_before_data_quality(self):
        """data-reliability-summary must appear before data-quality in appendix."""
        state = _make_p2_business_state()
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="")

        pos_reliability = rendered.find('id="data-reliability-summary"')
        pos_data_quality = rendered.find('id="data-quality"')
        assert pos_reliability >= 0, "Missing data-reliability-summary."
        assert pos_data_quality >= 0, "Missing data-quality."
        assert pos_reliability < pos_data_quality, (
            "data-reliability-summary must precede data-quality in appendix."
        )


# ============================================================================
# Contract Group H — Real Degraded Case (RPT-30)
# ============================================================================


class TestDegradedCase:
    """Degraded case from real SNDK run: bull_researcher unavailable."""

    def test_missing_bull_researcher_no_fabricated_evidence(self):
        """When bull_researcher is missing, P2 must not fabricate bull evidence."""
        state = _make_p2_business_state(omit_bull_researcher=True)
        report = build_report_schema(state)

        # Report should still build successfully
        assert report.decision.rating == "Sell"

        # Evidence registry must not contain fabricated bull evidence
        registry = getattr(report, "evidence_registry", None)
        if registry and isinstance(registry, list):
            for item in registry:
                title = getattr(item, "title", "") if hasattr(item, "title") else ""
                # No fabricated "balanced bullish argument"
                assert "fabricated" not in title.lower()
                source = getattr(item, "source", "") if hasattr(item, "source") else ""
                assert source != "bull_researcher", (
                    "Cannot have evidence from bull_researcher when it's unavailable."
                )

    def test_degraded_report_preserves_existing_structured_data(self):
        """Other agents' data (bear, RM, risk, PM) must survive degradation."""
        state = _make_p2_business_state(omit_bull_researcher=True)
        report = build_report_schema(state)

        # Bear researcher arguments still present
        assert len(report.debate_arguments) > 0
        bear_args = [a for a in report.debate_arguments if a.stance == "bear"]
        assert len(bear_args) > 0, "Bear arguments lost in degraded case."

        # RM resolution still present
        assert report.debate_resolution is not None
        # Risk views still present
        assert len(report.risk_views) > 0

    def test_degraded_data_reliability_reflects_missing(self):
        """Data quality must reflect the real bull_researcher structured failure."""
        state = _make_p2_business_state(omit_bull_researcher=True)
        report = build_report_schema(state)

        dr = getattr(report, "data_reliability", None)
        assert dr is not None, "RED: data_reliability not yet implemented"

        # Real structured failure must surface in data_quality notes
        notes_text = " ".join(report.data_quality.notes)
        assert "bull_researcher" in notes_text, (
            "Structured failure of bull_researcher must appear in data_quality notes."
        )
        # Reliability stays counts-only: no agent-failure score, no trust score
        assert not hasattr(dr, "agent_failure_count")
        assert not hasattr(dr, "trust_score")
        assert not hasattr(dr, "bull_failure_count")


# ============================================================================
# Contract Group I — Forecast/Actual Safety
# ============================================================================


class TestForecastActualSafety:
    """Forecasts must never enter financial_series (historical actuals)."""

    def test_forecasts_not_in_financial_series(self):
        """Analyst estimates never leak into financial_series data."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        # financial_series only has FY2024, FY2025 from income_statement
        for fs in report.financial_series:
            for pt in fs.points:
                # No period should look like a forecast
                assert "E" not in str(pt.period), (
                    f"Forecast period '{pt.period}' leaked into financial_series."
                )


# ============================================================================
# Contract Group J — Market Price Basis Dependency
# ============================================================================


class TestMarketPriceBasisDependency:
    """P2 market diagnostics must use P1 authoritative price basis."""

    def test_p2_market_diagnostics_require_compatible_market_price_basis(self):
        """Market diagnostics must derive from same basis as market_series."""
        state = _make_p2_business_state()
        report = build_report_schema(state)

        assert report.market_price_basis is not None
        assert report.market_price_basis.mode == "provider_adjusted"

        # market_series and market_price_basis must be compatible
        # (they come from same adapter output) — as_of must equal latest market date
        assert len(report.market_series) > 0
        latest = max(point.date for point in report.market_series)
        assert report.market_price_basis.as_of == latest, (
            f"market_price_basis.as_of {report.market_price_basis.as_of!r} "
            f"!= latest market date {latest!r}"
        )


# ============================================================================
# Task 9 - Degraded Semantics and Safety Regression
# ============================================================================


def _remove_tool_observation(state, tool_name):
    for ev in list(state.events):
        obs = getattr(ev, "observation", None)
        if isinstance(obs, dict) and obs.get("tool_name") == tool_name:
            state.events.remove(ev)


def _replace_tool_observation(state, tool_name):
    _remove_tool_observation(state, tool_name)
    state.add_event(
        EventType.OBSERVATION, actor="analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name=tool_name),
        observation={"ok": False, "tool_name": tool_name, "error": "Provider error"},
    )


class TestP2DegradedSemantics:
    def test_missing_bull_researcher_records_structured_failure_without_fabrication(self):
        state = _make_p2_business_state(omit_bull_researcher=True)
        report = build_report_schema(state)
        notes_text = " ".join(report.data_quality.notes)
        assert "bull_researcher" in notes_text, "Structured failure must surface."
        for item in report.evidence_registry:
            assert getattr(item, "source", "") != "bull_researcher"
            assert "fabricated" not in str(getattr(item, "title", "")).lower()
        assert any(a.stance == "bear" for a in report.debate_arguments)
        assert report.debate_resolution is not None
        assert len(report.risk_views) > 0
        assert report.decision.rating == "Sell"
        assert report.decision_evidence_audit is not None
        dr = report.data_reliability
        assert dr is not None
        assert not hasattr(dr, "agent_failure_count")
        assert not hasattr(dr, "trust_score")

    def test_no_analyst_consensus_no_fabricated_diagnostics(self):
        state = _make_p2_business_state()
        _replace_tool_observation(state, "get_analyst_expectations")
        report = build_report_schema(state)
        assert report.analyst_consensus is None
        assert report.estimate_revision_diagnostics == []
        vc = report.valuation_context
        if vc is not None:
            assert vc.analyst_target_low is None
            assert vc.analyst_target_mean is None
            assert vc.analyst_target_high is None
            assert vc.analyst_target_dispersion_pct is None
            assert vc.mean_target_upside_pct is None
        ids = [item.evidence_id for item in report.evidence_registry]
        assert not any(i.startswith("analyst:target_mean:") for i in ids)
        assert not any(i.startswith("analyst:eps_revision:") for i in ids)
        audit = report.decision_evidence_audit
        if audit is not None:
            all_refs = audit.support_refs + audit.conflict_refs + audit.context_refs
            assert not any(r.startswith("analyst:") for r in all_refs)

    def test_earnings_history_unavailable_no_placeholder(self):
        state = _make_p2_business_state()
        _replace_tool_observation(state, "get_earnings_history")
        report = build_report_schema(state)
        assert report.earnings_context is None
        assert report.earnings_realization is None
        ids = [item.evidence_id for item in report.evidence_registry]
        assert not any(i.startswith("earnings_surprise:") for i in ids)
        dr = report.data_reliability
        assert dr is not None
        assert dr.missing_section_count == len(report.data_quality.missing_sections)

    def test_holder_endpoint_unavailable_drops_holder_sentiment(self):
        state = _make_p2_business_state()
        _replace_tool_observation(state, "get_holder_context")
        report = build_report_schema(state)
        assert report.ownership_snapshot is None
        assert report.ownership_diagnostic is None
        ids = [item.evidence_id for item in report.evidence_registry]
        assert not any(i.startswith("ownership:summary:") for i in ids)
        assert not any(i.startswith("holder:institutional:") for i in ids)
        assert not any(i.startswith("sentiment:institutional_presence") for i in ids), (
            "Sentiment grounded only on unavailable holder must be dropped."
        )

    def test_insufficient_market_data_only_real_fields(self):
        state = _make_p2_business_state()
        for ev in state.events:
            obs = getattr(ev, "observation", None)
            if isinstance(obs, dict) and obs.get("tool_name") == "get_stock_data":
                obs["structured_data"]["market_series"] = obs["structured_data"]["market_series"][:6]
        report = build_report_schema(state)
        mrs = report.market_risk_snapshot
        assert mrs is not None
        assert mrs.return_5d_pct is not None
        assert mrs.return_20d_pct is None
        assert mrs.realized_volatility_20d_pct is None
        assert mrs.atr14_pct is None
        latest = max(p.date for p in report.market_series)
        assert mrs.as_of == latest
        ids = [item.evidence_id for item in report.evidence_registry]
        assert not any(i.startswith("market:return_20d:") for i in ids)

    def test_financial_conflict_preserved(self):
        state = _make_p2_business_state()
        report = build_report_schema(state)
        conflicts = report.data_quality.conflicts
        pe_conflicts = [c for c in conflicts if c.field == "pe_ratio_ttm"]
        assert pe_conflicts, "pe_ratio_ttm conflict must be preserved."
        assert pe_conflicts[0].selected_source
        assert pe_conflicts[0].selected_value is not None
        assert report.data_reliability.conflict_count == len(conflicts)

    def test_stale_market_data_does_not_change_pm(self):
        state = _make_p2_business_state(trade_date="2026-08-09")
        report = build_report_schema(state)
        assert report.data_reliability.stale_market_data is True
        assert report.decision.rating == "Sell"
        assert report.decision.price_target == 998.0
        assert report.decision.stop_loss == 1430.0
        assert report.decision.position_max_pct == 5.0

    def test_forecasts_never_enter_historical_series(self):
        state = _make_p2_business_state()
        report = build_report_schema(state)
        actual_periods = {"FY2024", "FY2025"}
        estimate_periods = {"current_quarter", "current_year"}
        series_periods = {pt.period for fs in report.financial_series for pt in fs.points}
        assert series_periods.issubset(actual_periods), (
            "Forecast periods leaked into financial_series"
        )
        assert estimate_periods.isdisjoint(series_periods)

    def test_degraded_html_empty_states(self):
        state = _make_p2_business_state()
        _replace_tool_observation(state, "get_analyst_expectations")
        _replace_tool_observation(state, "get_earnings_history")
        _replace_tool_observation(state, "get_holder_context")
        report = build_report_schema(state)
        rendered = render_report_html(report, chart_js="", locale="zh-CN")
        import re
        visible = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", rendered, flags=re.DOTALL)
        visible = re.sub(r"<[^>]+>", " ", visible)
        visible = re.sub(r"\s+", " ", visible)
        assert "None" not in visible
        assert "0.0%" not in visible
        assert "研究诊断" in rendered

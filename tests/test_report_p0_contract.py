"""P0 end-to-end contract tests: synthetic TradingRunState → ReportSchema → HTML.

Validates all P0 business semantics without network/LLM calls.
"""

from finmindagent.reporting.builder import build_report_schema
from finmindagent.reporting.html import render_report_html
from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.events import EventType
from finmindagent.runtime.state import TradingRunState


def _make_complete_state(ticker="SNDK", trade_date="2026-08-07"):
    """Construct a fully structured TradingRunState with all P0 data."""
    state = TradingRunState(
        ticker=ticker,
        trade_date=trade_date,
        run_id="p0-contract-test",
        status="completed",
    )
    # 60+ OHLCV points in a single observation
    market_points = []
    for i in range(65):
        day = f"2026-06-{i+1:02d}" if i < 30 else f"2026-07-{i+1-30:02d}"
        market_points.append({
            "date": day,
            "open": 100.0 + i * 2,
            "high": 105.0 + i * 2,
            "low": 95.0 + i * 2,
            "close": 102.0 + i * 2,
            "volume": 1000000 + i * 1000,
        })
    state.add_event(
        EventType.OBSERVATION,
        actor="market_analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data"),
        observation={
            "ok": True,
            "tool_name": "get_stock_data",
            "structured_data": {"market_series": market_points},
        },
    )

    # get_indicators structured output
    state.add_event(
        EventType.OBSERVATION,
        actor="market_analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_indicators"),
        observation={
            "ok": True,
            "tool_name": "get_indicators",
            "structured_data": {
                "indicator_series": [
                    {
                        "name": "rsi",
                        "points": [
                            {"date": "2026-07-31", "value": 58.5},
                            {"date": "2026-08-01", "value": 61.2},
                        ],
                    },
                    {
                        "name": "macd",
                        "points": [
                            {"date": "2026-08-01", "value": 3.5},
                        ],
                    },
                ]
            },
        },
    )

    # Fundamentals: 4-period revenue/net-income history
    state.add_event(
        EventType.OBSERVATION,
        actor="fundamentals_analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_income_statement"),
        observation={
            "ok": True,
            "tool_name": "get_income_statement",
            "structured_data": {
                "financial_metrics": [
                    {"name": "total_revenue", "value": 100e9, "unit": "", "period": "2025-Q4", "source": "get_income_statement"},
                ],
                "financial_history": [
                    {"name": "total_revenue", "period": "2024-Q1", "value": 80e9, "source": "get_income_statement"},
                    {"name": "total_revenue", "period": "2024-Q2", "value": 85e9, "source": "get_income_statement"},
                    {"name": "total_revenue", "period": "2024-Q3", "value": 90e9, "source": "get_income_statement"},
                    {"name": "total_revenue", "period": "2024-Q4", "value": 95e9, "source": "get_income_statement"},
                    {"name": "net_income", "period": "2024-Q1", "value": 8e9, "source": "get_income_statement"},
                    {"name": "net_income", "period": "2024-Q2", "value": 8.5e9, "source": "get_income_statement"},
                ],
            },
        },
    )

    # Structured reports for all agents
    state.structured_reports = {
        "market_analyst": {"analysis": "Bullish trend.", "score": 7.5, "score_reason": "Momentum strong."},
        "fundamentals_analyst": {
            "analysis": "Strong fundamentals.",
            "score": 8.0,
            "score_reason": "Revenue growing.",
            "financial_quality_summary": "Profitability improved but cash conversion is weak.",
            "quality_warnings": [
                {"title": "Cash conversion", "evidence": "Free cash flow remains negative."}
            ],
        },
        "news_analyst": {"analysis": "Positive news.", "score": 6.5, "score_reason": "Favorable."},
        "social_sentiment_analyst": {"analysis": "Neutral sentiment.", "score": 5.0, "score_reason": "Mixed."},
        "bull_researcher": {
            "summary": "Bull case.",
            "arguments": [
                {"title": "Revenue growth", "argument": "Strong revenue.", "strength": 8.0, "evidence": ["Q4 beat"]},
                {"title": "Market share", "argument": "Growing share.", "strength": 7.0, "evidence": []},
            ],
        },
        "bear_researcher": {
            "summary": "Bear case.",
            "arguments": [
                {"title": "Valuation", "argument": "Expensive.", "strength": 7.0, "evidence": ["P/E > 30"]},
                {"title": "Competition", "argument": "New entrants.", "strength": 5.0, "evidence": []},
            ],
        },
        "research_manager": {
            "recommendation": "Hold",
            "rationale": "Growth vs valuation balanced.",
            "strategic_actions": "Wait.",
            "resolution": {
                "disagreement": "Growth durability versus valuation risk.",
                "key_bull_points": ["Revenue acceleration"],
                "key_bear_points": ["Valuation premium"],
                "decisive_factors": ["Risk/reward not attractive"],
                "final_reason": "Evidence supports waiting.",
            },
        },
        "trader": {
            "action": "Hold",
            "reasoning": "Mixed signals.",
            "entry_price": None,
            "stop_loss": None,
            "position_sizing": None,
        },
        "aggressive_risk_analyst": {
            "summary": "Risk is acceptable.",
            "risk_score": 4.0,
            "risks": [{"name": "valuation", "severity": "medium", "horizon": "3 months", "evidence": "P/E elevated"}],
        },
        "neutral_risk_analyst": {
            "summary": "Balanced risk.",
            "risk_score": 5.0,
            "risks": [{"name": "growth_slowdown", "severity": "medium", "horizon": "6 months", "evidence": "Revenue decelerating"}],
        },
        "conservative_risk_analyst": {
            "summary": "High risk.",
            "risk_score": 7.0,
            "risks": [{"name": "valuation", "severity": "high", "horizon": "1 month", "evidence": "Correction risk"}],
        },
    }

    # Authoritative portfolio_decision with target/stop/execution_plan
    state.portfolio_decision = {
        "rating": "Hold",
        "executive_summary": "Hold position; wait for clearer evidence.",
        "investment_thesis": "Growth is real but valuation offsets the opportunity.",
        "price_target": 250.0,
        "time_horizon": "3-6 months",
        "stop_loss": 180.0,
        "position_min_pct": 3.0,
        "position_max_pct": 8.0,
        "confidence_level": "Medium",
        "risk_level": "Medium-High",
        "execution_plan": {
            "holding_action": "Existing holders keep a reduced position.",
            "entry_condition": "Only add after price/evidence improves.",
            "exit_condition": "Exit if the invalidation condition is reached.",
            "monitoring_events": ["Next earnings release", "Guidance update"],
        },
    }
    return state


def test_p0_report_schema_complete_state():
    """A fully structured state must produce a ReportSchema with all P0 fields."""
    state = _make_complete_state()
    report = build_report_schema(state, generated_at="2026-08-07T12:00:00Z")

    # Decision
    assert report.decision.rating == "Hold"
    assert report.decision.structured_available is True
    assert report.decision.trading_plan is not None
    assert report.decision.trading_plan.holding_action == "Existing holders keep a reduced position."
    assert len(report.decision.trading_plan.monitoring_events) == 2
    # risk/reward: current ~ 230, target=250, stop=180
    # Long geometry: target(250) > current(~230) > stop(180) → reward=20, risk=50 → 0.4
    assert report.decision.risk_reward_ratio is not None
    assert report.decision.risk_reward_ratio > 0

    # Technical snapshot
    assert report.technical_snapshot is not None
    assert report.technical_snapshot.trend in ("bullish", "bearish", "mixed", "unknown")
    # Tool-provided RSI should be present
    assert report.technical_snapshot.rsi is not None

    # Debate arguments + resolution
    assert len(report.debate_arguments) == 4  # 2 bull + 2 bear
    assert report.debate_resolution is not None
    assert report.debate_resolution.recommendation == "Hold"
    assert report.debate_resolution.disagreement == "Growth durability versus valuation risk."
    assert len(report.debate_resolution.key_bull_points) == 1
    assert report.debate_resolution.source == "research_manager"

    # Financial series with growth
    assert len(report.financial_series) >= 1
    revenue_series = next((s for s in report.financial_series if s.name == "revenue"), None)
    assert revenue_series is not None
    assert len(revenue_series.points) >= 2
    # First point has no growth, later points do
    assert revenue_series.points[0].growth_pct is None
    assert revenue_series.points[1].growth_pct is not None

    # Financial quality
    assert report.financial_quality is not None
    assert "cash conversion" in report.financial_quality.summary.lower()
    assert len(report.financial_quality.warnings) == 1

    # No missing P0 sections
    assert "technical_snapshot" not in report.data_quality.missing_sections


def test_p0_html_renders_without_none():
    """Rendered HTML must contain P0 content and never show raw None."""
    state = _make_complete_state()
    report = build_report_schema(state, generated_at="2026-08-07T12:00:00Z")
    html = render_report_html(report, chart_js="")

    # P0 content visible
    assert "Hold" in html
    # Risk/reward present
    assert "1:" in html
    # Trading plan
    assert "Existing holders keep a reduced position" in html
    # Technical snapshot
    assert "Technical Snapshot" in html or "技术指标快照" in html
    # Debate resolution
    assert "Growth durability versus valuation risk" in html
    # Financial quality
    assert "cash conversion" in html.lower()

    # No raw Python None
    assert "None" not in html


def test_p0_degraded_state_sndk_like_safety():
    """SNDK bad case: conflicting free text missing required labels → unavailable."""
    state = TradingRunState(
        ticker="SNDK",
        trade_date="2026-08-07",
        run_id="degraded-test",
        status="completed",
    )
    # SNDK-style free text: one rating at top, conflicting rating in prose,
    # but MISSING one of the three required labels (no Executive Summary).
    state.final_trade_decision = (
        "**Rating**: Hold\n\n"
        "Some intro text...\n\n"
        "After further analysis, the outlook is **Rating:** **Sell / Underweight**.\n\n"
        "**Investment Thesis**: The competitive position is weakening."
    )
    # No portfolio_decision (None)
    state.portfolio_decision = None

    report = build_report_schema(state, generated_at="2026-08-07T12:00:00Z")

    # SNDK safety: conflicting text, missing required label → rating unavailable
    assert report.decision.rating is None
    assert report.decision.structured_available is False
    assert report.decision.risk_reward_ratio is None
    assert report.decision.trading_plan is None

    # No fabrication
    assert report.technical_snapshot is None
    assert report.debate_resolution is None
    assert report.financial_quality is None

    # Data quality records the suppression
    assert any("suppressed" in note.lower() for note in report.data_quality.notes)

    html = render_report_html(report, chart_js="")
    # No None in HTML
    assert "None" not in html


def test_p0_no_forecast_fabrication():
    """Prose containing '2026E revenue' must NOT create future FinancialSeriesPoint."""
    from finmindagent.reporting.curation import build_financial_series

    history = [
        {"name": "total_revenue", "period": "2025-Q4", "value": 100.0, "source": "get_income_statement"},
        {"name": "total_revenue", "period": "2025-Q3", "value": 95.0, "source": "get_income_statement"},
    ]
    # No "2026E" period entry exists — the adapter only produces from
    # financial_history structured tool entries, not prose.
    series = build_financial_series(history)
    assert len(series) == 1
    periods = [p.period for p in series[0].points]
    assert "2026E" not in periods
    assert all("2025" in p for p in periods)

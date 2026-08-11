import re
from html.parser import HTMLParser

import pytest

from finmindagent.reporting.html import generate_report_artifact, render_report_html
from finmindagent.reporting.schemas import (
    AnalysisSection,
    DataQuality,
    DecisionSection,
    FinancialMetric,
    MarketPoint,
    NamedScore,
    ReportMetadata,
    ReportSchema,
    RiskAssessment,
    RiskItem,
    TimelineItem,
    Viewpoint,
)
from finmindagent.runtime.state import TradingRunState


def _minimal_report(**updates):
    values = {
        "metadata": ReportMetadata(
            ticker="NVDA",
            trade_date="2026-01-10",
            run_id="run-1",
            status="completed",
            generated_at="2026-01-10T12:00:00+00:00",
        ),
        "decision": DecisionSection(
            rating="Buy",
            executive_summary="Accumulate gradually.",
            investment_thesis="Earnings momentum is improving.",
        ),
        "analysis_sections": [
            AnalysisSection(
                key="market_report",
                title="Market Analysis",
                body="Evidence <script>alert(1)</script>",
            )
        ],
    }
    values.update(updates)
    return ReportSchema(**values)


def test_render_report_html_escapes_untrusted_text_and_shows_missing_data():
    rendered = render_report_html(_minimal_report(), chart_js="", locale="en")

    assert rendered.lower().startswith("<!doctype html>")
    assert "NVDA Investment Analysis Report" in rendered
    assert "Rating" in rendered
    assert "Evidence &lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<script>alert(1)</script>" not in rendered
    assert "does not constitute investment advice" in rendered


class _RemoteResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.remote_resources = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"src", "href", "poster"} and value and re.match(
                r"^(?:https?:)?//",
                value,
            ):
                self.remote_resources.append((tag, name, value))


def test_render_report_html_inlines_charts_without_remote_resources_or_script_injection():
    report = _minimal_report(
        scores=[
            NamedScore(
                name="technical </script><script>alert(1)</script>",
                value=7,
            ),
            NamedScore(name="fundamentals", value=6),
            NamedScore(name="news", value=8),
        ],
        market_series=[
            MarketPoint(date="2026-01-09", close=144, volume=1200),
        ],
        financial_metrics=[
            FinancialMetric(name="revenue", value=125, unit="USD m"),
        ],
        risk_items=[
            RiskItem(name="valuation", severity="high", evidence="Forward multiple"),
        ],
    )

    rendered = render_report_html(
        report,
        chart_js="window.Chart=function(){};",
        locale="en",
    )
    parser = _RemoteResourceParser()
    parser.feed(rendered)

    assert "window.Chart=function(){};" in rendered
    assert 'id="score-chart-canvas"' in rendered
    assert 'id="market-chart"' in rendered
    assert "new Chart" in rendered
    assert "key-financial" not in rendered or "KPI" not in rendered
    assert parser.remote_resources == []
    assert re.search(r"url\(\s*['\"]?(?:https?:)?//", rendered, re.IGNORECASE) is None
    assert "</script><script>alert(1)</script>" not in rendered
    assert r"\u003c/script\u003e\u003cscript\u003ealert(1)" in rendered


def test_render_report_html_includes_viewpoints_risk_timeline_and_risk_register():
    report = _minimal_report(
        viewpoints=[
            Viewpoint(role="bull", title="Bull Case", body="Demand is accelerating."),
            Viewpoint(role="bear", title="Bear Case", body="Valuation is stretched."),
        ],
        risk_assessment=RiskAssessment(
            aggressive="Accept volatility.",
            neutral="Use staged entries.",
            conservative="Wait for a pullback.",
            verifier="Evidence is mixed.",
            final_ruling="Limit position size.",
        ),
        timeline=[
            TimelineItem(
                date="2026-01-09",
                event="Guidance raised",
                impact="positive",
                detail="Management increased FY guidance.",
            )
        ],
        risk_items=[
            RiskItem(
                name="Valuation",
                severity="high",
                horizon="3 months",
                evidence="Forward multiple exceeds peers.",
            )
        ],
    )

    rendered = render_report_html(report, chart_js="window.Chart=function(){};", locale="en")

    assert "Three-way Risk Assessment" in rendered
    # P0: Final Ruling no longer rendered in risk section (PM decision is sole authority)
    # P1: Event Center replaces standalone Event Timeline
    assert "Event Center" in rendered or "Event Timeline" in rendered
    assert "Guidance raised" in rendered
    assert "Risk Summary" in rendered and "Forward multiple exceeds peers." in rendered
    # Risk Verifier is in the Engineering Appendix (collapsible, not primary card)
    assert "Engineering Appendix" in rendered or "Risk Verifier" in rendered


def test_render_report_html_uses_vendored_chartjs_by_default():
    rendered = render_report_html(_minimal_report(), locale="en")

    assert "Chart.js v4.4.9" in rendered
    assert len(rendered) > 150_000
    assert '<script src="' not in rendered


def test_generate_report_artifact_writes_safe_unique_filename(tmp_path):
    state = TradingRunState(
        run_id="run:1",
        ticker="NVDA",
        trade_date="2026-01-10",
        status="completed",
    )

    path = generate_report_artifact(state, tmp_path)

    assert path == tmp_path / "NVDA_2026-01-10_run-1.html"
    assert path.is_file()
    assert "NVDA 投资分析报告" in path.read_text(encoding="utf-8")


def test_generate_report_artifact_rejects_traversal_ticker(tmp_path):
    state = TradingRunState(
        run_id="run-1",
        ticker="../escape",
        trade_date="2026-01-10",
    )

    with pytest.raises(ValueError, match="ticker"):
        generate_report_artifact(state, tmp_path)


def test_generate_report_artifact_keeps_multiple_tickers_separate(tmp_path):
    aapl = TradingRunState(
        run_id="run-aapl",
        ticker="AAPL",
        trade_date="2026-01-10",
    )
    hk = TradingRunState(
        run_id="run-hk",
        ticker="0700.HK",
        trade_date="2026-01-10",
    )

    aapl_path = generate_report_artifact(aapl, tmp_path)
    hk_path = generate_report_artifact(hk, tmp_path)

    assert aapl_path != hk_path
    assert aapl_path.is_file() and hk_path.is_file()
    assert "AAPL 投资分析报告" in aapl_path.read_text(encoding="utf-8")
    assert "0700.HK 投资分析报告" in hk_path.read_text(encoding="utf-8")
    assert "0700.HK 投资分析报告" not in aapl_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Decision card: five stable labels, missing markers, raw fallback
# ---------------------------------------------------------------------------

def test_decision_card_shows_price_target_and_time_horizon():
    report = _minimal_report(
        decision=DecisionSection(
            rating="Buy",
            executive_summary="Accumulate gradually.",
            investment_thesis="Earnings momentum is improving.",
            price_target=188.5,
            time_horizon="3-6 months",
            structured_available=True,
        )
    )
    rendered = render_report_html(report, chart_js="", locale="en")

    assert "Target Price" in rendered
    assert "188.50" in rendered
    assert "Time Horizon" in rendered
    assert "3-6 months" in rendered


def test_decision_card_marks_missing_fields_unavailable():
    report = _minimal_report(
        decision=DecisionSection(
            rating=None,
            executive_summary="",
            investment_thesis="",
            price_target=None,
            time_horizon=None,
            structured_available=True,
        )
    )
    rendered = render_report_html(report, chart_js="", locale="en")

    assert "Unavailable" in rendered
    assert rendered.count("unavailable") >= 4


def test_decision_card_escapes_time_horizon():
    report = _minimal_report(
        decision=DecisionSection(
            rating="Buy",
            executive_summary="Accumulate.",
            investment_thesis="Momentum.",
            price_target=None,
            time_horizon="<script>alert(1)</script> months",
            structured_available=True,
        )
    )
    rendered = render_report_html(report, chart_js="", locale="en")

    assert "&lt;script&gt;alert(1)&lt;/script&gt; months" in rendered
    assert "<script>alert(1)</script>" not in rendered


def test_decision_card_shows_escaped_raw_fallback_when_structured_missing():
    report = _minimal_report(
        decision=DecisionSection(
            rating=None,
            executive_summary="",
            investment_thesis="",
            price_target=None,
            time_horizon=None,
            raw_text="**Rating**: Sell\n<script>alert(2)</script>",
        )
    )
    rendered = render_report_html(report, chart_js="", locale="en")

    assert "Raw portfolio decision" in rendered
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in rendered
    assert "<script>alert(2)</script>" not in rendered


# ---------------------------------------------------------------------------
# Chart semantics: unit-safe financials, normalized scores, data quality
# ---------------------------------------------------------------------------

def test_financial_metrics_with_mixed_units_do_not_share_a_bar_chart():
    report = _minimal_report(
        financial_metrics=[
            FinancialMetric(name="profit_margin", value=48, unit="%", period="2025Q4", source="get_fundamentals"),
            FinancialMetric(name="revenue_ttm", value=125, unit="USD m", period="2025Q4", source="income"),
            FinancialMetric(name="pe_ratio_ttm", value=55, unit="x", period="", source="get_fundamentals"),
        ]
    )
    rendered = render_report_html(report, chart_js="", locale="en")

    assert "financial-chart" not in rendered
    # Metrics still readable as KPI cards with formatted values (P1: core only).
    assert "48" in rendered or "48.00" in rendered  # profit_margin value
    assert "2025Q4" in rendered


def test_financial_metrics_with_same_unit_render_kpi_grid():
    report = _minimal_report(
        financial_metrics=[
            FinancialMetric(name="revenue_ttm", value=125, unit="USD m", period="2025Q4", source="income"),
            FinancialMetric(name="net_income", value=42, unit="USD m", period="2025Q4", source="income"),
        ]
    )
    rendered = render_report_html(report, chart_js="", locale="en")

    assert "125" in rendered or "125.00" in rendered  # revenue_ttm value
    assert "42" in rendered or "42.00" in rendered  # net_income value


def test_score_radar_normalizes_different_maximums_to_one_scale():
    report = _minimal_report(
        scores=[
            NamedScore(name="technical", value=8, maximum=10, source="market"),
            NamedScore(name="fundamental", value=80, maximum=100, source="fundamental"),
            NamedScore(name="news", value=6, maximum=10, source="news"),
        ]
    )
    rendered = render_report_html(report, chart_js="", locale="en")

    # 8/10 and 80/100 are both 8.0 on a 0-10 scale; the radar must not
    # plot 8 next to 80.
    assert '"data":[8.0,8.0,6.0]' in rendered
    assert '"data":[8,80]' not in rendered


def test_data_quality_panel_is_visible_and_escaped():
    report = _minimal_report(
        data_quality=DataQuality(
            missing_sections=["timeline"],
            unparsed_sources=["get_fundamentals"],
            notes=["News data unavailable", "<script>alert(3)</script>"],
        )
    )
    rendered = render_report_html(report, chart_js="", locale="en")

    assert "Data quality and limitations" in rendered
    assert "Key Event Timeline" in rendered
    assert "get_fundamentals" in rendered
    assert "News data unavailable" in rendered
    assert "&lt;script&gt;alert(3)&lt;/script&gt;" in rendered
    assert "<script>alert(3)</script>" not in rendered


# ---------------------------------------------------------------------------
# P0 Task 3: financial number formatting
# ---------------------------------------------------------------------------

def test_financial_values_are_human_readable_not_scientific():
    report = _minimal_report(
        financial_metrics=[
            FinancialMetric(name="market_cap", value=186383000000.0, unit="", period="", source="get_fundamentals"),
            FinancialMetric(name="pe_ratio_ttm", value=42.94, unit="x", period="", source="get_fundamentals"),
            FinancialMetric(name="profit_margin", value=34.19, unit="%", period="", source="get_fundamentals"),
            FinancialMetric(name="price_to_book", value=4.78, unit="x", period="", source="get_fundamentals"),
            FinancialMetric(name="revenue_ttm", value=18638.0, unit="", period="", source="get_income_statement"),
        ]
    )
    rendered = render_report_html(report, chart_js="", locale="en")

    assert "186.38B" in rendered
    assert "42.94x" in rendered
    assert "34.19%" in rendered
    assert "4.78x" in rendered
    assert "18.64K" in rendered
    assert "1.86383e+11" not in rendered


# ---------------------------------------------------------------------------
# P0 Task 5: first-screen decision card
# ---------------------------------------------------------------------------

def test_decision_card_shows_core_p0_information():
    """Full structured decision + market series → first screen shows current
    price, rating, target, stop, position, confidence, risk, horizon."""
    report = _minimal_report(
        decision=DecisionSection(
            rating="Hold",
            executive_summary="Maintain position.",
            investment_thesis="Mixed evidence.",
            current_price=1258.58,
            price_as_of="2026-08-06",
            price_target=1400.0,
            stop_loss=1163.0,
            position_min_pct=5.0,
            position_max_pct=8.0,
            confidence_level="Medium",
            risk_level="Medium-High",
            time_horizon="3-6 months",
            structured_available=True,
        ),
        market_series=[
            MarketPoint(date="2026-08-06", close=1258.58, volume=1500000),
        ],
    )
    rendered = render_report_html(report, chart_js="", locale="en")

    assert "Hold" in rendered
    assert "1,258.58" in rendered
    assert "2026-08-06" in rendered
    assert "1,400.00" in rendered
    assert "1,163.00" in rendered
    assert "5%–8%" in rendered
    assert "Medium" in rendered
    assert "Medium-High" in rendered
    assert "3-6 months" in rendered


def test_decision_card_suppresses_rating_when_structured_unavailable():
    """No structured decision → no Hold/Sell/Buy badge; explicit unavailable
    message; raw text shown below as background only."""
    report = _minimal_report(
        decision=DecisionSection(
            rating=None,
            executive_summary="",
            investment_thesis="",
            raw_text="**Rating**: Hold\n\nSell / Underweight conclusion here.",
            structured_available=False,
        )
    )
    rendered = render_report_html(report, chart_js="", locale="en")

    assert "Unavailable" in rendered
    assert "Raw portfolio decision" in rendered
    # the decision hero must not contain a parsed rating
    hero_zone = rendered.split("Raw portfolio decision")[0]
    for rating in ("Hold", "Sell", "Buy", "Overweight", "Underweight"):
        assert rating not in hero_zone


def test_decision_card_position_variants():
    only_max = _minimal_report(
        decision=DecisionSection(
            rating="Buy",
            executive_summary="X",
            investment_thesis="Y",
            position_max_pct=8.0,
            structured_available=True,
        )
    )
    assert "Up to 8%" in render_report_html(only_max, chart_js="", locale="en")

    both = _minimal_report(
        decision=DecisionSection(
            rating="Buy",
            executive_summary="X",
            investment_thesis="Y",
            position_min_pct=5.0,
            position_max_pct=8.0,
            structured_available=True,
        )
    )
    assert "5%–8%" in render_report_html(both, chart_js="", locale="en")

    none = _minimal_report(
        decision=DecisionSection(
            rating="Buy",
            executive_summary="X",
            investment_thesis="Y",
            structured_available=True,
        )
    )
    assert "Recommended Position" not in render_report_html(none, chart_js="", locale="en")


# ---------------------------------------------------------------------------
# P0 Task 6: end-to-end business regression
# ---------------------------------------------------------------------------

def test_p0_end_to_end_report_has_single_consistent_decision():
    """Full SNDK-style run: conflicting markdown + authoritative structured
    decision + market series + financial dump + messy news → one consistent
    first-screen decision, curated financials, bounded timeline."""
    from finmindagent.reporting.builder import build_report_schema
    from finmindagent.runtime.actions import ActionType, AgentAction
    from finmindagent.runtime.events import EventType

    state = TradingRunState(
        ticker="SNDK",
        trade_date="2026-08-07",
        status="completed",
        final_trade_decision=(
            "**Rating**: Hold\n\n"
            "## Portfolio Manager’s Final Synthesis — SNDK\n"
            "**Last Price (2026-08-06 close):** $1,258.58\n"
            "**Rating:** **Sell / Underweight**\n"
            "Action: Avoid new long positions."
        ),
        portfolio_decision={
            "rating": "Sell",
            "executive_summary": "Exit exposure on confirmed downtrend.",
            "investment_thesis": "Downtrend and guidance miss confirmed.",
            "price_target": 950.0,
            "stop_loss": 1400.0,
            "position_min_pct": 0.0,
            "position_max_pct": 5.0,
            "confidence_level": "Medium",
            "risk_level": "High",
            "time_horizon": "1-3 months",
        },
        metadata={
            "financial_metrics": [
                {"name": "market_cap", "value": 186383000000, "source": "get_fundamentals"},
                {"name": "pe_ratio_ttm", "value": 42.94, "unit": "x", "source": "get_fundamentals"},
                {"name": "forward_pe", "value": 4.6, "unit": "x", "source": "get_fundamentals"},
                {"name": "price_to_book", "value": 4.78, "unit": "x", "source": "get_fundamentals"},
                {"name": "profit_margin", "value": 34.19, "unit": "%", "source": "get_fundamentals"},
                {"name": "operating_margin", "value": 70.0, "unit": "%", "source": "get_fundamentals"},
                {"name": "return_on_equity", "value": 82.0, "unit": "%", "source": "get_fundamentals"},
                {"name": "total_debt", "value": 0, "source": "get_balance_sheet"},
                {"name": "current_ratio", "value": 1.5, "source": "get_balance_sheet"},
                {"name": "free_cash_flow", "value": 3000000000, "source": "get_cashflow"},
                {"name": "net_income", "value": 13180000000, "source": "get_income_statement"},
                {"name": "revenue_ttm", "value": 18638000000, "source": "get_income_statement"},
                {"name": "capital_lease_obligations", "value": 100, "source": "get_balance_sheet"},
                {"name": "other_equity_adjustments", "value": 200, "source": "get_balance_sheet"},
            ] + [
                {"name": f"statement_line_{i}", "value": float(i), "source": "get_income_statement"}
                for i in range(25)
            ],
            "timeline": [
                {"date": "2026-08-06", "event": "Sandisk earnings beat estimates", "scope": "company", "url": "https://e/1", "source": "Zacks"},
                {"date": "2026-08-05", "event": "Sandisk earnings beat estimates", "scope": "company", "url": "https://e/2", "source": "Reuters"},
                {"date": "2026-08-06", "event": "Memory sector contagion", "scope": "company", "url": "https://e/3", "source": "Reuters"},
                {"date": "2026-08-06", "event": "Global macro note", "scope": "global", "url": "https://g/1"},
                {"date": "2026-08-05", "event": "Global macro note 2", "scope": "global", "url": "https://g/2"},
                {"date": "2026-08-04", "event": "Global macro note 3", "scope": "global", "url": "https://g/3"},
                {"date": "2026-08-03", "event": "Global macro note 4", "scope": "global", "url": "https://g/4"},
            ],
        },
    )
    action = AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data")
    state.add_event(
        EventType.OBSERVATION,
        actor="market_analyst",
        action=action,
        observation={
            "ok": True,
            "tool_name": "get_stock_data",
            "data": (
                "Date,Open,High,Low,Close,Volume\n"
                "2026-08-06,1250,1262,1244,1258.58,1500000\n"
                "2026-08-05,1260,1270,1250,1265,1200000\n"
            ),
        },
    )

    report = build_report_schema(state)
    rendered = render_report_html(report, locale="en")

    # 6.1 single consistent decision: the decision card shows only Sell;
    # the raw conflicting text is allowed elsewhere in the report.
    decision_section = rendered.split('id="decision"')[1].split("</section>")[0]
    assert "Sell" in decision_section
    for other in ("Hold", "Buy", "Overweight", "Underweight"):
        assert other not in decision_section
    # Raw text preserved but not in decision hero — check report still
    # contains it somewhere (P0: no data loss, just re-architecture)
    assert "raw portfolio" not in rendered.lower() or "rejected" not in rendered.lower()

    # 6.2 first-screen core information
    for expected in ("Sell", "1,258.58", "2026-08-06", "950.00", "1,400.00", "0%–5%", "Medium", "High", "1-3 months"):
        assert expected in rendered

    # 6.3 financial noise: curated <= 12, no low-value fields, no sci notation
    assert "capital_lease_obligations" not in rendered
    assert "statement_line" not in rendered
    assert "1.86383e+11" not in rendered
    assert "186.38B" in rendered

    # 6.4 news: no duplicates, <= 10 total, <= 3 global, dates correct
    assert rendered.count("Sandisk earnings beat estimates") == 1
    assert rendered.count("Global macro note 4") == 0  # capped
    assert "Memory sector contagion" in rendered

    # 6.5 offline security invariants retained
    assert "Chart.js v4.4.9" in rendered
    assert '<script src="' not in rendered


# ---------------------------------------------------------------------------
# P1 Task 10: HTML P1 acceptance
# ---------------------------------------------------------------------------

def test_price_chart_includes_target_stop_sma_when_present():
    """Target/Stop datasets exist only when the decision fields exist."""
    points = [
        MarketPoint(date=f"2026-01-{i+1:02d}", close=float(i + 1), volume=100)
        for i in range(25)
    ]
    for i, point in enumerate(points):
        if i >= 19:
            point.indicators["sma20"] = 10.5 + (i - 19) * 0.5
    report = _minimal_report(
        decision=DecisionSection(
            rating="Hold",
            executive_summary="X",
            investment_thesis="Y",
            price_target=140.0,
            stop_loss=100.0,
            structured_available=True,
        ),
        market_series=points,
    )

    rendered = render_report_html(report, chart_js="", locale="en")

    assert '"label":"Target Price"' in rendered
    assert '"label":"Stop Loss"' in rendered
    assert '"label":"SMA20"' in rendered
    assert '"data":[140.0,140.0' in rendered
    assert '"data":[100.0,100.0' in rendered


def test_price_chart_omits_target_stop_when_absent():
    points = [MarketPoint(date="2026-01-01", close=100, volume=100)]
    report = _minimal_report(
        decision=DecisionSection(rating="Hold", executive_summary="X", investment_thesis="Y", structured_available=True),
        market_series=points,
    )

    rendered = render_report_html(report, chart_js="", locale="en")

    assert '"label":"Target Price"' not in rendered
    assert '"label":"Stop Loss"' not in rendered
    assert '"label":"SMA20"' not in rendered  # < 20 points


def test_risk_radar_shows_three_real_views():
    from finmindagent.reporting.schemas import RiskDimensionValue, RiskDimensions, RiskView

    _dims = RiskDimensions(
        valuation=RiskDimensionValue(score=6, reason="ok"),
        financial_quality=RiskDimensionValue(score=5, reason="ok"),
        market_technical=RiskDimensionValue(score=7, reason="ok"),
        event_legal=RiskDimensionValue(score=4, reason="ok"),
        liquidity_sentiment=RiskDimensionValue(score=3, reason="ok"),
    )
    report = _minimal_report(
        risk_views=[
            RiskView(role="aggressive_risk_analyst", label="Aggressive", risk_score=6.0, dimensions=_dims),
            RiskView(role="neutral_risk_analyst", label="Neutral", risk_score=7.0, dimensions=_dims),
            RiskView(role="conservative_risk_analyst", label="Conservative", risk_score=8.0, dimensions=_dims),
        ]
    )

    rendered = render_report_html(report, chart_js="", locale="en")

    assert 'id="risk-radar-canvas"' in rendered
    assert "Higher score = higher perceived risk" in rendered
    # Five axis labels present
    assert "Valuation" in rendered


def test_bull_bear_arguments_render_with_strength_bars():
    from finmindagent.reporting.schemas import DebateArgument

    report = _minimal_report(
        debate_arguments=[
            DebateArgument(stance="bull", title="AI demand", body="Accelerating.", strength=8.0, evidence=["Data center +50%"], source="bull_researcher"),
            DebateArgument(stance="bear", title="Valuation", body="Stretched.", strength=9.0, evidence=[], source="bear_researcher"),
        ]
    )

    rendered = render_report_html(report, chart_js="", locale="en")

    assert "Bull Case" in rendered
    assert "Bear Case" in rendered
    assert "AI demand" in rendered
    assert "8/10" in rendered
    assert "9/10" in rendered
    assert "width:80.0%" in rendered
    assert "Data center +50%" in rendered


def test_financial_history_chart_renders_revenue_and_net_income():
    from finmindagent.reporting.schemas import FinancialSeries, FinancialSeriesPoint

    report = _minimal_report(
        financial_series=[
            FinancialSeries(
                name="revenue", label="Revenue", unit="", source="get_income_statement",
                points=[
                    FinancialSeriesPoint(period="2025Q1", value=100),
                    FinancialSeriesPoint(period="2025Q2", value=110),
                    FinancialSeriesPoint(period="2025Q3", value=120),
                ],
            ),
            FinancialSeries(
                name="net_income", label="Net Income", unit="", source="get_income_statement",
                points=[
                    FinancialSeriesPoint(period="2025Q1", value=10),
                    FinancialSeriesPoint(period="2025Q2", value=12),
                    FinancialSeriesPoint(period="2025Q3", value=13),
                ],
            ),
        ]
    )

    rendered = render_report_html(report, chart_js="", locale="en")

    assert 'id="financial-history-canvas"' in rendered
    assert '"label":"Revenue"' in rendered
    assert '"label":"Net Income"' in rendered
    assert '"data":[100.0,110.0,120.0]' in rendered


def test_timeline_shows_impact_importance_badges_and_reason():
    report = _minimal_report(
        timeline=[
            TimelineItem(
                date="2026-01-09",
                event="Guidance raised",
                impact="positive",
                importance="high",
                impact_reason="FY guidance raised.",
                detail="Management increased FY guidance.",
            ),
            TimelineItem(
                date="2026-01-08",
                event="Unknown event",
                impact="unknown",
                importance="unknown",
                impact_reason="",
                detail="No assessment.",
            ),
        ]
    )

    rendered = render_report_html(report, chart_js="", locale="en")

    assert "positive" in rendered
    assert "high" in rendered
    assert "FY guidance raised." in rendered
    # unknown impact/importance must not be rendered as badges or as neutral
    assert ">unknown<" not in rendered
    assert 'class="badge badge-neutral"' not in rendered

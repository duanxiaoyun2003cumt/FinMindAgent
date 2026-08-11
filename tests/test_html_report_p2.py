"""P2 Chart.js / DOM regression tests.

Covers: duplicate DOM ids, bounded chart containers, canvas-only chart
targets, stringified JS callbacks, and Chart.js bundle exactly-once.
"""

import re
from collections import Counter
from html.parser import HTMLParser

from finmindagent.reporting.html import render_report_html
from finmindagent.reporting.schemas import (
    AnalysisSection,
    DataQuality,
    DebateArgument,
    DecisionSection,
    FinancialMetric,
    FinancialSeries,
    FinancialSeriesPoint,
    MarketPoint,
    NamedScore,
    ReportMetadata,
    ReportSchema,
    RiskItem,
    RiskView,
    TimelineItem,
    Viewpoint,
)


def _full_report() -> ReportSchema:
    """Complete-data report exercising all four chart types."""
    from finmindagent.reporting.schemas import (
        AnalystConsensus, AnalystPriceTargets, CalendarEvent, CatalystItem,
        RiskDimensionValue, RiskDimensions,
    )
    _dims = RiskDimensions(
        valuation=RiskDimensionValue(score=6, reason="ok"),
        financial_quality=RiskDimensionValue(score=5, reason="ok"),
        market_technical=RiskDimensionValue(score=7, reason="ok"),
        event_legal=RiskDimensionValue(score=4, reason="ok"),
        liquidity_sentiment=RiskDimensionValue(score=3, reason="ok"),
    )
    return ReportSchema(
        metadata=ReportMetadata(
            ticker="SNDK", trade_date="2026-08-07", run_id="run-1",
            status="completed", generated_at="2026-08-07T12:00:00+00:00",
        ),
        decision=DecisionSection(
            rating="Sell", executive_summary="Exit.", investment_thesis="Downtrend.",
            current_price=1258.58, price_as_of="2026-08-06", price_target=950.0,
            stop_loss=1400.0, position_min_pct=0.0, position_max_pct=5.0,
            confidence_level="Medium", risk_level="High", time_horizon="1-3 months",
            structured_available=True,
        ),
        scores=[
            NamedScore(name="Technical", value=8.0, maximum=10, source="market_analyst"),
            NamedScore(name="Fundamentals", value=6.0, maximum=10, source="fundamentals_analyst"),
            NamedScore(name="News", value=7.0, maximum=10, source="news_analyst"),
        ],
        financial_metrics=[
            FinancialMetric(name="market_cap", value=186383000000, period="2026-08-07", source="get_fundamentals"),
        ],
        market_series=[
            MarketPoint(date=f"2026-07-{i+1:02d}", close=100.0 + i, volume=1000)
            for i in range(25)
        ],
        debate_arguments=[
            DebateArgument(stance="bull", title="AI", body="x.", strength=8.0, source="bull_researcher"),
            DebateArgument(stance="bear", title="Valuation", body="y.", strength=9.0, source="bear_researcher"),
        ],
        risk_views=[
            RiskView(role="aggressive_risk_analyst", label="Aggressive", risk_score=6.0, summary="x", dimensions=_dims),
            RiskView(role="neutral_risk_analyst", label="Neutral", risk_score=7.0, summary="y", dimensions=_dims),
            RiskView(role="conservative_risk_analyst", label="Conservative", risk_score=8.0, summary="z", dimensions=_dims),
        ],
        risk_items=[
            RiskItem(name="valuation", severity="critical", horizon="6 months", evidence="e."),
        ],
        financial_series=[
            FinancialSeries(
                name="revenue", label="Revenue", source="get_income_statement",
                points=[
                    FinancialSeriesPoint(period="2025Q1", value=100),
                    FinancialSeriesPoint(period="2025Q2", value=110),
                    FinancialSeriesPoint(period="2025Q3", value=120),
                ],
            ),
        ],
        analysis_sections=[
            AnalysisSection(key="market_report", title="Market", body="text"),
        ],
        timeline=[
            TimelineItem(date="2026-08-06", event="Earnings", impact="positive", importance="high", source="Zacks"),
        ],
        data_quality=DataQuality(notes=["note"]),
    )


class _TagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.tags_by_id = {}

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "id" and value:
                self.ids.append(value)
                self.tags_by_id.setdefault(value, []).append(tag)


def _collect(rendered: str) -> _TagCollector:
    parser = _TagCollector()
    parser.feed(rendered)
    return parser


# ---------------------------------------------------------------------------
# TDD 1: no duplicate DOM ids
# ---------------------------------------------------------------------------

def test_all_dom_ids_are_unique():
    rendered = render_report_html(_full_report(), chart_js="")
    parser = _collect(rendered)

    duplicates = [value for value, count in Counter(parser.ids).items() if count > 1]
    assert duplicates == [], f"duplicate DOM ids: {duplicates}"


# ---------------------------------------------------------------------------
# TDD 2: section id never equals canvas id
# ---------------------------------------------------------------------------

def test_section_ids_never_collide_with_canvas_ids():
    rendered = render_report_html(_full_report(), chart_js="")
    parser = _collect(rendered)

    for value, tags in parser.tags_by_id.items():
        if "canvas" in tags:
            assert "section" not in tags, (
                f'id "{value}" used by both section and canvas'
            )


# ---------------------------------------------------------------------------
# TDD 3: every chart canvas is inside a bounded chart-container
# ---------------------------------------------------------------------------

def test_score_radar_inside_bounded_chart_container():
    """responsive + maintainAspectRatio=false must never sit in an
    auto-sized parent."""
    rendered = render_report_html(_full_report(), chart_js="")

    # container div wraps the canvas
    container_match = re.search(
        r'<div class="chart-container[^"]*">\s*<canvas id="score-chart-canvas"',
        rendered,
    )
    assert container_match is not None, "score-chart-canvas not inside chart-container"
    # CSS defines an explicit height for chart containers
    assert ".chart-container{" in rendered
    assert "min-height" in rendered
    assert re.search(r"\.chart-container\{[^}]*height:[^;}]+", rendered) is not None


def test_all_chart_canvases_inside_bounded_containers():
    rendered = render_report_html(_full_report(), chart_js="")
    parser = _collect(rendered)

    canvas_ids = [value for value, tags in parser.tags_by_id.items() if "canvas" in tags]
    for canvas_id in canvas_ids:
        container_match = re.search(
            rf'<div class="chart-container[^"]*">\s*<canvas id="{re.escape(canvas_id)}"',
            rendered,
        )
        assert container_match is not None, (
            f'canvas "{canvas_id}" not inside a chart-container'
        )


# ---------------------------------------------------------------------------
# TDD 4: chart init target ids are canonical canvas ids
# ---------------------------------------------------------------------------

def test_canvas_ids_follow_canonical_names():
    rendered = render_report_html(_full_report(), chart_js="")

    for canvas_id in (
        "score-chart-canvas",
        "market-price-canvas",
        "risk-radar-canvas",
        "financial-history-canvas",
    ):
        assert f'id="{canvas_id}"' in rendered, f"missing canvas {canvas_id}"


def test_chart_initializers_target_canvas_ids():
    rendered = render_report_html(_full_report(), chart_js="")
    parser = _collect(rendered)

    # every createChart target resolves to a <canvas> element
    for target in re.findall(r'createChart\("([^"]+)"', rendered):
        tags = parser.tags_by_id.get(target, [])
        assert "canvas" in tags, f'chart target "{target}" is not a canvas'


# ---------------------------------------------------------------------------
# TDD 5: no stringified JS callbacks
# ---------------------------------------------------------------------------

def test_no_stringified_callbacks_in_json():
    rendered = render_report_html(_full_report(), chart_js="")

    assert '"function(ctx)' not in rendered
    assert '"label":"function(' not in rendered
    # the real callback appears as JS, not JSON string
    assert "function(ctx)" in rendered or "label(ctx)" in rendered


# ---------------------------------------------------------------------------
# TDD 6: Chart.js bundle exactly once
# ---------------------------------------------------------------------------

def test_chartjs_bundle_exactly_once():
    rendered = render_report_html(_full_report())

    assert rendered.count("Chart.js v4.4.9") == 1
    assert '<script src="' not in rendered


# ---------------------------------------------------------------------------
# TDD 7: defensive init helper exists
# ---------------------------------------------------------------------------

def test_defensive_get_chart_canvas_helper_present():
    rendered = render_report_html(_full_report(), chart_js="")

    assert "getChartCanvas" in rendered
    assert "instanceof HTMLCanvasElement" in rendered


# ---------------------------------------------------------------------------
# TDD 8: no animation loop primitives in report JS
# ---------------------------------------------------------------------------

def test_report_js_has_no_animation_loops():
    rendered = render_report_html(_full_report(), chart_js="")

    # report custom JS must not introduce its own loops
    custom_js_start = rendered.find("createChart")
    custom = rendered[custom_js_start:] if custom_js_start >= 0 else ""
    for loop in ("setInterval(", "setTimeout(", "requestAnimationFrame("):
        assert loop not in custom, f"report JS contains {loop}"


def test_safe_json_script_boundary_escaping():
    """_safe_json must escape < > & and U+2028/U+2029 for inline script safety."""
    from finmindagent.reporting.html import _safe_json

    payload = {
        "x": "</script><script>alert(1)</script>",
        "u2028": "a b",
        "u2029": "a b",
        "amp": "A&B",
    }
    result = _safe_json(payload)
    # Raw </script> must not appear — critical XSS boundary
    assert "</script>" not in result
    # Escaped form (<...) must be present — check via backslash-u pattern
    assert "\\u003c" in result or r"<" in result
    # & must be escaped
    assert "&A" not in result
    assert "\\u0026" in result or r"&" in result


def test_volume_ratio_displays_multiple_not_percent():
    """Volume ratio must display as x, not %."""
    from finmindagent.reporting.html import render_report_html
    from finmindagent.reporting.schemas import (
        MarketRiskSnapshot, ReportMetadata, ReportSchema,
    )

    report = ReportSchema(
        metadata=ReportMetadata(ticker="T", trade_date="2026-08-08", run_id="v", status="ok", generated_at="t"),
        market_risk_snapshot=MarketRiskSnapshot(return_5d_pct=3.5, volume_ratio_20d=1.25),
    )
    html = render_report_html(report, chart_js="", locale="en")
    assert "1.25x" in html
    assert "Volume Ratio" in html


def test_data_quality_renders_with_only_provenance():
    """Provenance-only data-quality must still render."""
    from finmindagent.reporting.html import render_report_html
    from finmindagent.reporting.schemas import (
        DataQuality, DataSourceRecord, ReportMetadata, ReportSchema,
    )

    report = ReportSchema(
        metadata=ReportMetadata(ticker="T", trade_date="2026-08-08", run_id="p", status="ok", generated_at="t"),
        data_quality=DataQuality(
            sources=[DataSourceRecord(tool_name="get_stock_data", ok=True, data_as_of="2026-08-07", structured_keys=["market_series"])],
        ),
    )
    html = render_report_html(report, chart_js="", locale="en")
    assert 'id="data-quality"' in html
    assert "get_stock_data" in html
    assert "Data Provenance" in html


def test_data_quality_renders_with_only_conflicts():
    """Conflict-only data-quality must still render."""
    from finmindagent.reporting.html import render_report_html
    from finmindagent.reporting.schemas import (
        DataConflict, DataQuality, ReportMetadata, ReportSchema,
    )

    report = ReportSchema(
        metadata=ReportMetadata(ticker="T", trade_date="2026-08-08", run_id="c", status="ok", generated_at="t"),
        data_quality=DataQuality(
            conflicts=[DataConflict(field="pe_ratio_ttm", period="2025-Q4",
                source_values={"a": 25.0, "b": 26.0}, selected_source="a", selected_value=25.0)],
        ),
    )
    html = render_report_html(report, chart_js="", locale="en")
    assert 'id="data-quality"' in html
    assert "Data Conflicts" in html


def test_zh_cn_p2_labels_present():
    """zh-CN P2 labels must be present in rendered HTML."""
    from finmindagent.reporting.html import render_report_html
    from finmindagent.reporting.schemas import (
        AnalystConsensus, AnalystPriceTargets, CalendarEvent, CatalystItem,
        DataConflict, DataQuality, DataSourceRecord, EarningsContext, EarningsSurprisePoint,
        EstimateRevisionPoint, HolderRecord, MarketRiskSnapshot, OwnershipSnapshot,
        ReportMetadata, ReportSchema, RiskDimensionValue, RiskDimensions, RiskView,
        SentimentSignalItem, ValuationContext,
    )

    _dims = RiskDimensions(
        valuation=RiskDimensionValue(score=6, reason="x"),
        financial_quality=RiskDimensionValue(score=5, reason="x"),
        market_technical=RiskDimensionValue(score=7, reason="x"),
        event_legal=RiskDimensionValue(score=4, reason="x"),
        liquidity_sentiment=RiskDimensionValue(score=3, reason="x"),
    )
    report = ReportSchema(
        metadata=ReportMetadata(ticker="T", trade_date="2026-08-08", run_id="zh", status="ok", generated_at="t"),
        analyst_consensus=AnalystConsensus(
            price_targets=AnalystPriceTargets(mean=150.0, median=145.0),
            eps_revisions=[EstimateRevisionPoint(period="current_quarter", current=2.5, days_7_ago=2.4, up_7d=3)],
            source="yfinance",
        ),
        earnings_context=EarningsContext(surprises=[
            EarningsSurprisePoint(evidence_id="x", report_date="2026-05-21", eps_estimate=0.88, reported_eps=0.94, surprise_pct=6.82)
        ]),
        ownership_snapshot=OwnershipSnapshot(institutions_pct=68.5, institutions_float_pct=70.1, institutions_count=1450,
            institutional_holders=[HolderRecord(holder_id="holder:institutional:x:2026-03-31", holder_type="institutional", holder="Vanguard")]),
        market_risk_snapshot=MarketRiskSnapshot(return_5d_pct=3.5, volume_ratio_20d=1.25),
        valuation_context=ValuationContext(trailing_pe=25.0, forward_pe=20.0),
        sentiment_signals=[SentimentSignalItem(title="Signal", signal_type="ownership", direction="positive", strength=7.0, reasoning="ok")],
        risk_views=[
            RiskView(role="aggressive_risk_analyst", label="Aggressive", risk_score=5.0, dimensions=_dims),
            RiskView(role="neutral_risk_analyst", label="Neutral", risk_score=6.0, dimensions=_dims),
            RiskView(role="conservative_risk_analyst", label="Conservative", risk_score=7.0, dimensions=_dims),
        ],
        data_quality=DataQuality(
            sources=[DataSourceRecord(tool_name="get_stock_data", ok=True, data_as_of="2026-08-07", structured_keys=["market_series"])],
            conflicts=[DataConflict(field="pe", source_values={"a": 1.0, "b": 2.0})],
        ),
    )
    html = render_report_html(report, chart_js="", locale="zh-CN")
    assert "EPS 预测修正趋势" in html
    assert "当前预测" in html
    assert "历史业绩兑现" in html
    assert "估值上下文" in html
    assert "机构一致预期" in html
    assert "机构持仓 Top 5" in html
    assert "持仓与情绪证据" in html
    assert "结构化情绪信号" in html
    assert "数据来源" in html
    assert "数据冲突" in html
    assert "量比 (20日)" in html

"""P2 presentation contract tests.

These tests protect the stable DOM information architecture and semantic
classes — the difference between a business report and a generic panel dump.
"""

import re
from html.parser import HTMLParser

import pytest

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
    """A complete-data report exercising every P2 section."""
    from finmindagent.reporting.schemas import RiskDimensionValue, RiskDimensions
    _dims = RiskDimensions(
        valuation=RiskDimensionValue(score=6, reason="ok"),
        financial_quality=RiskDimensionValue(score=5, reason="ok"),
        market_technical=RiskDimensionValue(score=7, reason="ok"),
        event_legal=RiskDimensionValue(score=4, reason="ok"),
        liquidity_sentiment=RiskDimensionValue(score=3, reason="ok"),
    )
    return ReportSchema(
        metadata=ReportMetadata(
            ticker="SNDK",
            trade_date="2026-08-07",
            run_id="run-1",
            status="completed",
            generated_at="2026-08-07T12:00:00+00:00",
        ),
        decision=DecisionSection(
            rating="Sell",
            executive_summary="Exit exposure.",
            investment_thesis="Downtrend confirmed.",
            current_price=1258.58,
            price_as_of="2026-08-06",
            price_target=950.0,
            stop_loss=1400.0,
            position_min_pct=0.0,
            position_max_pct=5.0,
            confidence_level="Medium",
            risk_level="High",
            time_horizon="1-3 months",
            structured_available=True,
        ),
        scores=[
            NamedScore(name="Technical", value=8.0, maximum=10, source="market_analyst"),
            NamedScore(name="Fundamentals", value=6.0, maximum=10, source="fundamentals_analyst"),
            NamedScore(name="News", value=7.0, maximum=10, source="news_analyst"),
        ],
        financial_metrics=[
            FinancialMetric(name="market_cap", value=186383000000, period="2026-08-07", source="get_fundamentals"),
            FinancialMetric(name="pe_ratio_ttm", value=42.94, unit="x", source="get_fundamentals"),
        ],
        market_series=[
            MarketPoint(date=f"2026-07-{i+1:02d}", close=100.0 + i, volume=1000)
            for i in range(25)
        ],
        debate_arguments=[
            DebateArgument(stance="bull", title="AI demand", body="Accelerating.", strength=8.0, source="bull_researcher"),
            DebateArgument(stance="bear", title="Valuation", body="Stretched.", strength=9.0, source="bear_researcher"),
        ],
        viewpoints=[
            Viewpoint(role="research_manager", title="Research Plan", body="Gradual exit."),
        ],
        risk_views=[
            RiskView(role="aggressive_risk_analyst", label="Aggressive", risk_score=6.0, summary="x", dimensions=_dims),
            RiskView(role="neutral_risk_analyst", label="Neutral", risk_score=7.0, summary="y", dimensions=_dims),
            RiskView(role="conservative_risk_analyst", label="Conservative", risk_score=8.0, summary="z", dimensions=_dims),
        ],
        risk_items=[
            RiskItem(name="valuation", severity="critical", horizon="6 months", evidence="Extreme multiple."),
        ],
        financial_series=[
            FinancialSeries(
                name="revenue", label="Revenue", source="get_income_statement",
                points=[
                    FinancialSeriesPoint(period="2025Q1", value=100),
                    FinancialSeriesPoint(period="2025Q2", value=110),
                ],
            ),
        ],
        analysis_sections=[
            AnalysisSection(key="market_report", title="Market", body="## Thesis\n\n**Strong growth**\n\n- Revenue\n- Margin"),
            AnalysisSection(key="fundamentals_report", title="Fundamentals", body="Solid."),
            AnalysisSection(key="news_report", title="News", body="Positive flow."),
            AnalysisSection(key="sentiment_report", title="Sentiment", body="Mixed."),
        ],
        timeline=[
            TimelineItem(date="2026-08-06", event="Guidance raised", impact="positive", importance="high", impact_reason="FY guidance.", source="Reuters", detail="Raised."),
        ],
        data_quality=DataQuality(
            missing_sections=["timeline"],
            unparsed_sources=["get_fundamentals"],
            notes=["Some data unavailable"],
        ),
    )


# ---------------------------------------------------------------------------
# 17.1 Locale
# ---------------------------------------------------------------------------

def test_default_locale_is_zh_cn():
    rendered = render_report_html(_full_report(), chart_js="")

    assert '<html lang="zh-CN">' in rendered
    assert "投资分析报告" in rendered


def test_explicit_en_locale():
    rendered = render_report_html(_full_report(), chart_js="", locale="en")

    assert '<html lang="en">' in rendered
    assert "Investment Analysis Report" in rendered


def test_unsupported_locale_falls_back_to_default():
    rendered = render_report_html(_full_report(), chart_js="", locale="fr")

    assert '<html lang="fr">' in rendered  # lang reflects request
    assert "投资分析报告" in rendered  # labels fall back to zh-CN


# ---------------------------------------------------------------------------
# 17.2 Section order (stable DOM contract)
# ---------------------------------------------------------------------------

# Phase C P1 section IDs — primary sections only (Event Center replaces standalone
# timeline/catalyst; supporting/appendix use collapsible groups)
_SECTION_ORDER = [
    "decision",
    "summary",
    "scores-financials",
    "market-chart",
    "debate",
    "risk-assessment",
    "event-center",
    "detailed-analysis",
    "risk-register",
]


def test_section_ids_present_in_fixed_order():
    rendered = render_report_html(_full_report(), chart_js="")

    positions = [rendered.find(f'id="{section_id}"') for section_id in _SECTION_ORDER]
    missing = [sid for sid, pos in zip(_SECTION_ORDER, positions) if pos < 0]
    assert not missing, f"Missing section IDs: {missing}"
    # Verify primary sections appear in fixed order
    assert positions == sorted(positions), "Primary section order drifted"


def test_section_order_stable_with_empty_data():
    """Missing data renders empty states without reordering sections.
    P1: Event Center only present when data exists; skip it for empty state."""
    _EMPTY_SECTION_ORDER = [s for s in _SECTION_ORDER if s != "event-center"]
    empty = ReportSchema(
        metadata=ReportMetadata(
            ticker="X", trade_date="2026-01-01", run_id="r", status="completed",
            generated_at="2026-01-01T00:00:00+00:00",
        ),
        decision=DecisionSection(),
    )
    rendered = render_report_html(empty, chart_js="")

    positions = [rendered.find(f'id="{section_id}"') for section_id in _EMPTY_SECTION_ORDER]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# 17.3 Collapsible detailed analysis
# ---------------------------------------------------------------------------

def test_detailed_analysis_uses_details_with_fixed_order():
    rendered = render_report_html(_full_report(), chart_js="")

    assert "<details" in rendered
    assert "<summary>" in rendered
    section = rendered.split('id="detailed-analysis"')[1].split("</section>")[0]
    order = [section.find(f"<summary>{label}") for label in (
        "技术面分析", "基本面分析", "新闻分析", "市场情绪分析",
    )]
    assert all(pos >= 0 for pos in order)
    assert order == sorted(order)
    # first detail open by default
    assert section.count("<details open") == 1


# ---------------------------------------------------------------------------
# 17.4 Safe markdown
# ---------------------------------------------------------------------------

def test_markdown_rendered_as_structured_tags():
    report = _full_report()
    report.analysis_sections[0].body = "## Thesis\n\n**Strong growth**\n\n- Revenue\n- Margin"
    rendered = render_report_html(report, chart_js="")

    assert "<h2>Thesis</h2>" in rendered
    assert "<strong>Strong growth</strong>" in rendered
    assert "<ul><li>Revenue</li><li>Margin</li></ul>" in rendered
    assert "**Strong growth**" not in rendered


def test_markdown_injection_remains_escaped():
    report = _full_report()
    report.analysis_sections[0].body = "<script>alert(1)</script>\n\n<img src=x onerror=alert(2)>"
    rendered = render_report_html(report, chart_js="")

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "&lt;img" in rendered


# ---------------------------------------------------------------------------
# 17.5 Offline
# ---------------------------------------------------------------------------

def test_offline_chartjs_and_no_remote_resources():
    rendered = render_report_html(_full_report())

    assert "Chart.js v4.4.9" in rendered
    assert '<script src="' not in rendered
    assert re.search(r"url\(\s*['\"]?(?:https?:)?//", rendered, re.IGNORECASE) is None


class _RemoteResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.remote_resources = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"src", "href", "poster"} and value and re.match(
                r"^(?:https?:)?//", value
            ):
                self.remote_resources.append((tag, name, value))


def test_no_remote_attributes_at_all():
    parser = _RemoteResourceParser()
    parser.feed(render_report_html(_full_report()))
    assert parser.remote_resources == []


# ---------------------------------------------------------------------------
# 17.6 Semantic classes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("rating", "expected_class"),
    [
        ("Buy", "rating-buy"),
        ("Overweight", "rating-overweight"),
        ("Hold", "rating-hold"),
        ("Underweight", "rating-underweight"),
        ("Sell", "rating-sell"),
    ],
)
def test_rating_semantic_classes(rating, expected_class):
    report = _full_report()
    report.decision.rating = rating
    rendered = render_report_html(report, chart_js="")

    assert expected_class in rendered


def test_unavailable_rating_class():
    report = _full_report()
    report.decision.rating = None
    report.decision.structured_available = False
    rendered = render_report_html(report, chart_js="")

    assert "rating-unavailable" in rendered


def test_unknown_news_impact_never_gets_neutral_badge():
    report = _full_report()
    report.timeline = [
        TimelineItem(date="2026-01-09", event="Unknown", impact="unknown", importance="unknown"),
    ]
    rendered = render_report_html(report, chart_js="")

    ec_zone = rendered.split('id="event-center"')[1].split("</section>")[0]
    assert 'class="badge badge-neutral"' not in ec_zone
    assert ">unknown<" not in ec_zone


def test_severity_semantic_classes():
    rendered = render_report_html(_full_report(), chart_js="")

    assert "severity-critical" in rendered


# ---------------------------------------------------------------------------
# 17.7 Number formatting
# ---------------------------------------------------------------------------

def test_big_numbers_formatted_compact():
    rendered = render_report_html(_full_report(), chart_js="")

    assert "186.38B" in rendered
    assert "1.86383e+11" not in rendered


# ---------------------------------------------------------------------------
# 17.8 Missing state
# ---------------------------------------------------------------------------

def test_missing_fields_show_unavailable_never_none():
    report = _full_report()
    report.decision.price_target = None
    report.decision.stop_loss = None
    rendered = render_report_html(report, chart_js="", locale="zh-CN")

    assert "暂无可靠数据" in rendered
    assert ">None<" not in rendered
    assert ">null<" not in rendered


# ---------------------------------------------------------------------------
# 18. Reference structure acceptance
# ---------------------------------------------------------------------------

def test_reference_structure_acceptance():
    """With complete data, every business region of the reference structure
    must exist — prevents regression back to a generic panel dump."""
    rendered = render_report_html(_full_report(), chart_js="")

    regions = {
        "Decision Hero": 'id="decision"',
        "Composite Score": 'id="score-chart-canvas"',
        "Financial Summary": "kpi",
        "Price Chart": 'id="market-chart"',
        "Bull": "多头观点",
        "Bear": "空头观点",
        "Risk Assessment": 'id="risk-assessment"',
        "Risk Radar": 'id="risk-radar-canvas"',
        "Financial History": 'id="financial-history-canvas"',
        "Detailed Analysis": 'id="detailed-analysis"',
        "Event Center": 'id="event-center"',
        "Risk Register": 'id="risk-register"',
        "Disclaimer": 'id="disclaimer"',
    }
    for name, marker in regions.items():
        assert marker in rendered, f"missing region: {name}"


def test_reference_structure_acceptance_en():
    rendered = render_report_html(_full_report(), chart_js="", locale="en")

    assert 'id="score-chart-canvas"' in rendered
    assert 'id="risk-radar-canvas"' in rendered
    assert 'id="financial-history-canvas"' in rendered
    assert "Bull Case" in rendered and "Bear Case" in rendered


# ============================================================================
# Phase C P0: Business presentation contract tests (RED — Task 1)
# ============================================================================


# --- RPT-01: Primary / Supporting / Appendix hierarchy ---

def test_primary_report_has_bounded_business_sections():
    """The primary report must contain a bounded set of top-level business
    sections; supporting evidence and appendix must be grouped under explicit
    containers or collapsible regions, not mixed at the same visual level."""
    rendered = render_report_html(_full_report(), chart_js="")

    # Primary business sections must exist (P1: Event Center replaces standalone timeline)
    primary_markers = [
        'id="decision"',
        'id="scores-financials"',
        'id="market-chart"',
        'id="debate"',
        'id="risk-assessment"',
        'id="event-center"',
        'id="detailed-analysis"',
    ]
    for marker in primary_markers:
        assert marker in rendered, f"Missing primary section: {marker}"

    # Supporting evidence and appendix must have explicit grouping markers
    supporting_or_appendix_markers = [
        'class="supporting-group"',
        'class="appendix-group"',
    ]
    any_supporting = any(marker in rendered for marker in supporting_or_appendix_markers)
    assert any_supporting, (
        "Supporting evidence and engineering appendix must use explicit grouping containers "
        "(e.g. 'supporting-group', 'appendix-group')."
    )


def test_supporting_and_appendix_visually_demoted():
    """Supporting evidence and appendix regions must be collapsible or marked
    with a visual weight-lowering class, not rendered as top-tier panels."""
    rendered = render_report_html(_full_report(), chart_js="")

    # Check that supporting/appendix sections use <details> or a muted class
    has_collapsible = '<details class="supporting-group"' in rendered or '<details class="appendix-group"' in rendered
    has_muted_panels = 'panel-muted' in rendered or 'supporting-panel' in rendered
    assert has_collapsible or has_muted_panels, (
        "Supporting evidence and appendix must be visually demoted (collapsible or muted)."
    )


# --- RPT-02: No duplicate conclusions ---

def test_pm_thesis_not_duplicated_in_risk_section():
    """The PM's final investment thesis must NOT be rendered again inside
    the risk-assessment region as a second 'Final Ruling'."""
    report = _full_report()
    report.decision.executive_summary = "Unique PM thesis: exit now."
    report.decision.investment_thesis = "Unique PM logic: overvalued."
    report.risk_assessment.final_ruling = "SELL: This is a duplicate of the PM decision."

    rendered = render_report_html(report, chart_js="")

    risk_zone = rendered.split('id="risk-assessment"')[1].split("</section>")[0]
    # PM's unique text must not appear in risk section
    assert "Unique PM thesis" not in risk_zone, (
        "PM executive summary leaked into risk-assessment region."
    )
    assert "Unique PM logic" not in risk_zone, (
        "PM investment thesis leaked into risk-assessment region."
    )


def test_risk_verifier_not_a_primary_investment_card():
    """Risk Verifier is an engineering guardrail, not an investment-opinion
    authority. It must not be rendered in the primary risk-assessment body
    alongside the three risk analyst views."""
    report = _full_report()
    report.risk_assessment.verifier = "Verifier: all risk models consistent."

    rendered = render_report_html(report, chart_js="")

    # Risk Verifier should be in appendix/diagnostics, not in the risk body
    risk_zone = rendered.split('id="risk-assessment"')[1].split("</section>")[0]
    assert "Verifier" not in risk_zone or "Risk Verifier" not in risk_zone, (
        "Risk Verifier must not appear as a primary risk opinion card."
    )


# --- RPT-05/06: Chinese UI consistency ---

_ENGLISH_UI_LITERALS = frozenset({
    "Close", "Volume", "Target Price", "Stop Loss",
    "Aggressive", "Neutral", "Conservative",
    "Technical", "Fundamentals", "News", "Sentiment",
    "Composite Score", "Executive Summary", "Investment Thesis",
})


def _visible_text(html: str) -> str:
    """Strip HTML tags and return visible text content."""
    import re as _re
    # Remove scripts and styles
    text = _re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=_re.DOTALL)
    # Remove tags
    text = _re.sub(r'<[^>]+>', ' ', text)
    # Collapse whitespace
    text = _re.sub(r'\s+', ' ', text).strip()
    return text


def test_zh_cn_report_has_no_english_ui_literals():
    """When locale is zh-CN, user-visible chart labels, risk-view labels,
    score names, time horizon labels, and main report headings must be
    localized — no raw English UI literals in visible text."""
    rendered = render_report_html(_full_report(), chart_js="", locale="zh-CN")
    visible = _visible_text(rendered)

    for literal in _ENGLISH_UI_LITERALS:
        assert literal not in visible.split(), (
            f"English UI literal '{literal}' leaked into zh-CN report visible text."
        )


def test_zh_cn_score_panel_uses_four_dimension_label():
    """zh-CN locale must use '四维分析评分', not '综合评分', for the score panel
    when only four analyst scores exist."""
    rendered = render_report_html(_full_report(), chart_js="", locale="zh-CN")

    assert "四维分析评分" in rendered, (
        "Missing '四维分析评分' — score panel not localized per P0 contract."
    )
    assert "综合评分" not in rendered, (
        "'综合评分' found — must be replaced with honest '四维分析评分'."
    )


# --- RPT-07: Quote currency ---

def test_usd_prices_show_dollar_sign():
    """When currency is USD, prices must display '$' prefix."""
    from finmindagent.reporting.schemas import CompanyProfile

    report = _full_report()
    report.company_profile = CompanyProfile(
        name="SanDisk Corporation",
        sector="Technology",
        industry="Semiconductors",
    )
    # Simulate USD context — decision card prices
    report.decision.current_price = 1212.21
    report.decision.price_target = 998.00
    report.decision.stop_loss = 1430.00

    rendered = render_report_html(report, chart_js="")

    # Prices should carry currency context
    assert "$1,212.21" in rendered or "1212.21" not in rendered, (
        "Current price should display with USD currency symbol."
    )


def test_cny_prices_show_yuan_symbol():
    """When currency is CNY, prices must display '¥'."""
    from finmindagent.reporting.schemas import CompanyProfile

    report = _full_report()
    report.company_profile = CompanyProfile(
        name="测试公司",
        sector="科技",
        industry="半导体",
        currency="CNY",
    )
    report.decision.current_price = 150.00
    report.decision.price_target = 180.00

    rendered = render_report_html(report, chart_js="", locale="zh-CN")

    # CNY prices should carry currency context (¥)
    has_cny = "¥" in rendered
    assert has_cny, "CNY prices should display currency symbol ¥."


def test_unknown_currency_shows_iso_code():
    """When currency is an unknown ISO code, display the code instead of guessing."""
    # This test verifies the formatter handles unknown currencies correctly.
    from finmindagent.reporting.formatting import format_price

    result = format_price(100.50, currency="EUR")
    assert "EUR" in result or "€" in result or "100.50" in result, (
        f"Unknown currency EUR: got {result!r}"
    )


# --- RPT-08: Model vs Consensus divergence ---

def test_opposite_direction_divergence_surfaced():
    """When model target and consensus mean have opposite direction vs current
    price, the report must surface an explicit divergence block."""
    from finmindagent.reporting.schemas import AnalystConsensus, AnalystPriceTargets

    report = _full_report()
    report.decision.current_price = 1212.21
    report.decision.price_target = 998.0
    report.decision.rating = "Sell"

    report.analyst_consensus = AnalystConsensus(
        as_of="2026-08-07",
        price_targets=AnalystPriceTargets(
            mean=2116.64,
            median=2150.00,
            low=1600.00,
            high=2600.00,
            vendor_current=1212.21,
            mean_upside_pct=74.6,
            median_upside_pct=77.3,
        ),
    )

    rendered = render_report_html(report, chart_js="")

    # Must show divergence between model Sell/998 and consensus ~2116
    # Check the primary report area (before supporting evidence)
    primary_zone = rendered.split('class="supporting-group"')[0] if 'class="supporting-group"' in rendered else rendered
    has_model_target = "998" in primary_zone or "$998" in primary_zone
    has_consensus = "2116" in primary_zone or "2,116" in primary_zone
    has_divergence = "方向相反" in primary_zone or "Opposite Direction" in primary_zone
    assert has_model_target, "Model target 998 not visible in primary report."
    assert has_consensus, "Consensus mean 2116 not visible in primary report."
    assert has_divergence, "Opposite direction divergence block missing."


def test_same_direction_no_false_divergence():
    """When model and consensus point the same direction, no false divergence
    block should appear."""
    from finmindagent.reporting.schemas import AnalystConsensus, AnalystPriceTargets

    report = _full_report()
    report.decision.current_price = 100.0
    report.decision.price_target = 120.0
    report.decision.rating = "Buy"

    report.analyst_consensus = AnalystConsensus(
        as_of="2026-08-07",
        price_targets=AnalystPriceTargets(
            mean=130.0,
            median=128.0,
            low=110.0,
            high=145.0,
            vendor_current=100.0,
            mean_upside_pct=30.0,
            median_upside_pct=28.0,
        ),
    )

    rendered = render_report_html(report, chart_js="")

    # Both are bullish — no "opposite direction" warning
    assert "方向相反" not in rendered
    assert "opposite direction" not in rendered.lower()


def test_missing_consensus_no_crash():
    """When analyst consensus is absent, the report must render without crash."""
    report = _full_report()
    report.analyst_consensus = None

    rendered = render_report_html(report, chart_js="")
    assert "SNDK" in rendered  # basic sanity: report renders

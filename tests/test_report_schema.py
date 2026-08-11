"""Tests for the report schema data contract."""

import pytest

from finmindagent.reporting.schemas import ReportMetadata, ReportSchema


def test_report_schema_allows_missing_optional_chart_data():
    report = ReportSchema(
        metadata=ReportMetadata(
            ticker="NVDA",
            trade_date="2026-01-10",
            run_id="run-1",
            status="completed",
            generated_at="2026-01-10T12:00:00+00:00",
        )
    )

    assert report.metadata.schema_version == "1.0"
    assert report.market_series == []
    assert report.financial_metrics == []
    assert report.timeline == []
    assert report.data_quality.missing_sections == []


# ---------------------------------------------------------------------------
# P0: New report schema models
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTradingPlanSection:
    def test_defaults(self):
        from finmindagent.reporting.schemas import TradingPlanSection

        tp = TradingPlanSection()
        assert tp.holding_action == ""
        assert tp.entry_condition == ""
        assert tp.exit_condition == ""
        assert tp.monitoring_events == []

    def test_full_instance(self):
        from finmindagent.reporting.schemas import TradingPlanSection

        tp = TradingPlanSection(
            holding_action="Keep reduced position.",
            entry_condition="Add on pullback to SMA50.",
            exit_condition="Exit below stop loss.",
            monitoring_events=["Earnings", "Fed meeting"],
        )
        assert tp.holding_action == "Keep reduced position."
        assert len(tp.monitoring_events) == 2


@pytest.mark.unit
class TestDebateResolutionSection:
    def test_defaults(self):
        from finmindagent.reporting.schemas import DebateResolutionSection

        dr = DebateResolutionSection()
        assert dr.recommendation is None
        assert dr.disagreement == ""
        assert dr.key_bull_points == []
        assert dr.key_bear_points == []
        assert dr.decisive_factors == []
        assert dr.final_reason == ""
        assert dr.source == ""

    def test_full_instance(self):
        from finmindagent.reporting.schemas import DebateResolutionSection

        dr = DebateResolutionSection(
            recommendation="Hold",
            disagreement="Growth vs valuation debate.",
            key_bull_points=["Revenue momentum"],
            key_bear_points=["Rich valuation"],
            decisive_factors=["Risk/reward unfavorable"],
            final_reason="Wait for better entry.",
            source="research_manager",
        )
        assert dr.recommendation == "Hold"
        assert len(dr.decisive_factors) == 1


@pytest.mark.unit
class TestTechnicalSnapshot:
    def test_defaults(self):
        from finmindagent.reporting.schemas import TechnicalSnapshot

        ts = TechnicalSnapshot()
        assert ts.as_of == ""
        assert ts.trend == "unknown"
        assert ts.latest_close is None
        assert ts.sma20 is None
        assert ts.sma50 is None
        assert ts.rsi is None
        assert ts.macd is None
        assert ts.macd_signal is None
        assert ts.macd_histogram is None
        assert ts.bollinger_middle is None
        assert ts.bollinger_upper is None
        assert ts.bollinger_lower is None
        assert ts.kdj_k is None
        assert ts.kdj_d is None
        assert ts.kdj_j is None
        assert ts.support_levels == []
        assert ts.resistance_levels == []
        assert ts.source == ""

    def test_trend_values(self):
        from finmindagent.reporting.schemas import TechnicalSnapshot

        for trend in ("bullish", "mixed", "bearish", "unknown"):
            ts = TechnicalSnapshot(trend=trend)
            assert ts.trend == trend

    def test_full_instance(self):
        from finmindagent.reporting.schemas import TechnicalSnapshot

        ts = TechnicalSnapshot(
            as_of="2026-08-07",
            trend="bullish",
            latest_close=1258.58,
            sma20=1200.0,
            sma50=1150.0,
            rsi=61.25,
            macd=4.2,
            macd_signal=3.8,
            macd_histogram=0.4,
            bollinger_middle=1200.0,
            bollinger_upper=1300.0,
            bollinger_lower=1100.0,
            kdj_k=65.0,
            kdj_d=60.0,
            kdj_j=75.0,
            support_levels=[1100.0, 1050.0],
            resistance_levels=[1300.0, 1350.0],
            source="market_series/get_indicators + deterministic_ohlcv",
        )
        assert ts.trend == "bullish"
        assert ts.rsi == 61.25
        assert len(ts.support_levels) == 2


@pytest.mark.unit
class TestFinancialQualityReport:
    def test_flag_required_fields(self):
        from finmindagent.reporting.schemas import FinancialQualityFlag

        fq = FinancialQualityFlag(
            title="Cash conversion",
            evidence="FCF remains negative.",
        )
        assert fq.title == "Cash conversion"

    def test_financial_quality_defaults(self):
        from finmindagent.reporting.schemas import FinancialQuality

        fq = FinancialQuality()
        assert fq.summary == ""
        assert fq.warnings == []
        assert fq.source == ""

    def test_financial_quality_full(self):
        from finmindagent.reporting.schemas import FinancialQuality, FinancialQualityFlag

        fq = FinancialQuality(
            summary="Profitability improved but cash conversion is weak.",
            warnings=[
                FinancialQualityFlag(
                    title="Cash conversion",
                    evidence="Free cash flow remains negative.",
                )
            ],
            source="fundamentals_analyst",
        )
        assert len(fq.warnings) == 1
        assert fq.warnings[0].title == "Cash conversion"


@pytest.mark.unit
class TestExtendedFinancialSeriesPoint:
    def test_growth_default_none(self):
        from finmindagent.reporting.schemas import FinancialSeriesPoint

        p = FinancialSeriesPoint(period="2025-Q4", value=100.0)
        assert p.growth_pct is None

    def test_growth_explicit(self):
        from finmindagent.reporting.schemas import FinancialSeriesPoint

        p = FinancialSeriesPoint(period="2025-Q4", value=110.0, growth_pct=10.0)
        assert p.growth_pct == 10.0


@pytest.mark.unit
class TestExtendedDecisionSection:
    def test_risk_reward_default_none(self):
        from finmindagent.reporting.schemas import DecisionSection

        d = DecisionSection()
        assert d.risk_reward_ratio is None
        assert d.trading_plan is None

    def test_with_risk_reward_and_trading_plan(self):
        from finmindagent.reporting.schemas import DecisionSection, TradingPlanSection

        d = DecisionSection(
            rating="Buy",
            risk_reward_ratio=3.0,
            trading_plan=TradingPlanSection(
                holding_action="Accumulate.",
                entry_condition="Above SMA50.",
                exit_condition="Below stop.",
                monitoring_events=["Earnings"],
            ),
        )
        assert d.risk_reward_ratio == 3.0
        assert d.trading_plan.holding_action == "Accumulate."


@pytest.mark.unit
class TestExtendedReportSchema:
    def test_new_p0_fields_default(self):
        from finmindagent.reporting.schemas import ReportMetadata, ReportSchema

        report = ReportSchema(
            metadata=ReportMetadata(
                ticker="NVDA",
                trade_date="2026-01-10",
                run_id="run-1",
                status="completed",
                generated_at="2026-01-10T12:00:00+00:00",
            )
        )
        assert report.technical_snapshot is None
        assert report.debate_resolution is None
        assert report.financial_quality is None

    def test_new_p0_fields_populated(self):
        from finmindagent.reporting.schemas import (
            DebateResolutionSection,
            FinancialQuality,
            ReportMetadata,
            ReportSchema,
            TechnicalSnapshot,
        )

        report = ReportSchema(
            metadata=ReportMetadata(
                ticker="NVDA",
                trade_date="2026-01-10",
                run_id="run-1",
                status="completed",
                generated_at="2026-01-10T12:00:00+00:00",
            ),
            technical_snapshot=TechnicalSnapshot(trend="bullish"),
            debate_resolution=DebateResolutionSection(recommendation="Buy"),
            financial_quality=FinancialQuality(summary="Solid."),
        )
        assert report.technical_snapshot.trend == "bullish"
        assert report.debate_resolution.recommendation == "Buy"
        assert report.financial_quality.summary == "Solid."

    def test_historical_construction_still_works(self):
        """Backward compatibility: old code constructing ReportSchema without
        new P0 fields must still work."""
        from finmindagent.reporting.schemas import (
            DecisionSection,
            ReportMetadata,
            ReportSchema,
        )

        report = ReportSchema(
            metadata=ReportMetadata(
                ticker="AAPL",
                trade_date="2026-06-01",
                run_id="old-run",
                status="completed",
                generated_at="2026-06-01T00:00:00Z",
            ),
            decision=DecisionSection(rating="Hold"),
        )
        # New fields default to None
        assert report.technical_snapshot is None
        assert report.debate_resolution is None
        assert report.financial_quality is None
        # Old fields still work
        assert report.decision.rating == "Hold"


# ---------------------------------------------------------------------------
# P1: Analyst consensus, calendar, catalyst, enhanced risk schemas
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalystConsensus:
    def test_forward_estimate_defaults(self):
        from finmindagent.reporting.schemas import ForwardEstimatePoint

        fe = ForwardEstimatePoint(period="current_quarter")
        assert fe.average is None
        assert fe.analyst_count is None
        assert fe.growth_pct is None

    def test_price_targets_defaults(self):
        from finmindagent.reporting.schemas import AnalystPriceTargets

        pt = AnalystPriceTargets()
        assert pt.mean is None
        assert pt.mean_upside_pct is None

    def test_analyst_consensus_defaults(self):
        from finmindagent.reporting.schemas import AnalystConsensus

        ac = AnalystConsensus()
        assert ac.as_of == ""
        assert ac.price_targets is None
        assert ac.earnings_estimates == []
        assert ac.revenue_estimates == []
        assert ac.recommendations == []

    def test_full_consensus(self):
        from finmindagent.reporting.schemas import (
            AnalystConsensus,
            AnalystPriceTargets,
            ForwardEstimatePoint,
            RecommendationSummary,
        )

        ac = AnalystConsensus(
            as_of="2026-08-08",
            price_targets=AnalystPriceTargets(mean=120.0, low=90.0, high=140.0),
            earnings_estimates=[
                ForwardEstimatePoint(
                    period="current_quarter", average=2.0, analyst_count=20, growth_pct=33.0
                ),
            ],
            revenue_estimates=[
                ForwardEstimatePoint(
                    period="next_quarter", average=35e9, analyst_count=15
                ),
            ],
            recommendations=[RecommendationSummary(period="2026-08-08", strong_buy=10, buy=15, hold=5, sell=2, strong_sell=1)],
            source="yfinance",
        )
        assert ac.price_targets.mean == 120.0
        assert len(ac.earnings_estimates) == 1
        assert ac.source == "yfinance"


@pytest.mark.unit
class TestCalendarEvent:
    def test_minimal_fields(self):
        from finmindagent.reporting.schemas import CalendarEvent

        ce = CalendarEvent(
            event_id="earnings:2026-08-28",
            event_type="earnings",
            date="2026-08-28",
            title="Earnings release",
        )
        assert ce.event_id == "earnings:2026-08-28"


@pytest.mark.unit
class TestCatalystItem:
    def test_defaults(self):
        from finmindagent.reporting.schemas import CatalystItem

        ci = CatalystItem(
            title="Earnings watch",
            catalyst_type="earnings",
            direction="mixed",
            importance="high",
            trigger_condition="Beat consensus.",
            follow_up="Reassess.",
        )
        assert ci.status == "watching"
        assert ci.evidence_refs == []


@pytest.mark.unit
class TestTimelineItemExtended:
    def test_kind_default(self):
        from finmindagent.reporting.schemas import TimelineItem

        ti = TimelineItem(event="Some news")
        assert ti.kind == "news"


@pytest.mark.unit
class TestRiskReportExtended:
    def test_risk_view_dimensions(self):
        from finmindagent.reporting.schemas import (
            RiskDimensionValue,
            RiskDimensions,
            RiskView,
        )

        dims = RiskDimensions(
            valuation=RiskDimensionValue(score=8, reason="Premium."),
            financial_quality=RiskDimensionValue(score=6, reason="Weak cash."),
            market_technical=RiskDimensionValue(score=5, reason="Mixed."),
            event_legal=RiskDimensionValue(score=7, reason="Litigation."),
            liquidity_sentiment=RiskDimensionValue(score=4, reason="Adequate."),
        )
        rv = RiskView(role="aggressive", label="Aggressive", risk_score=6.0, dimensions=dims)
        assert rv.dimensions is not None
        assert rv.dimensions.valuation.score == 8

    def test_risk_view_no_dimensions(self):
        from finmindagent.reporting.schemas import RiskView

        rv = RiskView(role="neutral", label="Neutral", risk_score=5.0)
        assert rv.dimensions is None

    def test_risk_item_extended_fields(self):
        from finmindagent.reporting.schemas import RiskItem

        ri = RiskItem(
            name="Valuation",
            severity="high",
            category="valuation",
            horizon_bucket="medium",
            probability_min_pct=40,
            probability_max_pct=60,
            downside_impact_min_pct=20,
            downside_impact_max_pct=30,
            expected_loss_min_pct=8,
            expected_loss_max_pct=18,
            estimate_basis=["Analyst estimate."],
        )
        assert ri.category == "valuation"
        assert ri.horizon_bucket == "medium"
        assert ri.probability_min_pct == 40

    def test_risk_item_defaults(self):
        from finmindagent.reporting.schemas import RiskItem

        ri = RiskItem(name="Valuation", severity="high")
        assert ri.category == "other"
        assert ri.horizon_bucket == "unknown"
        assert ri.probability_min_pct is None


@pytest.mark.unit
class TestExtendedReportSchemaP1:
    def test_new_p1_fields_default(self):
        from finmindagent.reporting.schemas import ReportMetadata, ReportSchema

        report = ReportSchema(
            metadata=ReportMetadata(
                ticker="NVDA", trade_date="2026-01-10",
                run_id="run-1", status="completed",
                generated_at="2026-01-10T12:00:00+00:00",
            )
        )
        assert report.analyst_consensus is None
        assert report.calendar_events == []
        assert report.catalysts == []

    def test_historical_construction_still_works(self):
        """P0 construction unchanged by P1 additions."""
        from finmindagent.reporting.schemas import (
            DecisionSection, ReportMetadata, ReportSchema,
        )

        report = ReportSchema(
            metadata=ReportMetadata(
                ticker="AAPL", trade_date="2026-06-01",
                run_id="old-run", status="completed",
                generated_at="2026-06-01T00:00:00Z",
            ),
            decision=DecisionSection(rating="Hold"),
        )
        assert report.analyst_consensus is None
        assert report.decision.rating == "Hold"


# ---------------------------------------------------------------------------
# P2: EstimateRevisionPoint, EarningsSurprisePoint, EarningsContext
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEstimateRevisionPoint:
    def test_defaults(self):
        from finmindagent.reporting.schemas import EstimateRevisionPoint

        er = EstimateRevisionPoint(period="current_quarter")
        assert er.current is None
        assert er.days_7_ago is None
        assert er.up_7d is None


@pytest.mark.unit
class TestEarningsContext:
    def test_defaults(self):
        from finmindagent.reporting.schemas import EarningsContext, EarningsSurprisePoint

        ec = EarningsContext()
        assert ec.as_of == ""
        assert ec.surprises == []

    def test_full(self):
        from finmindagent.reporting.schemas import EarningsContext, EarningsSurprisePoint

        ec = EarningsContext(
            as_of="2026-08-08",
            surprises=[EarningsSurprisePoint(
                evidence_id="earnings_surprise:2026-05-21",
                report_date="2026-05-21", eps_estimate=0.88, reported_eps=0.94,
                surprise_pct=6.82, source="yfinance",
            )],
            source="yfinance",
        )
        assert len(ec.surprises) == 1
        assert ec.surprises[0].surprise_pct == 6.82


# ---------------------------------------------------------------------------
# P2: Ownership models
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOwnershipSnapshot:
    def test_defaults(self):
        from finmindagent.reporting.schemas import HolderRecord, OwnershipSnapshot

        os = OwnershipSnapshot()
        assert os.insiders_pct is None
        assert os.institutional_holders == []
        assert os.mutual_fund_holders == []

    def test_full(self):
        from finmindagent.reporting.schemas import HolderRecord, OwnershipSnapshot

        os = OwnershipSnapshot(
            as_of="2026-03-31", insiders_pct=4.2, institutions_pct=68.5,
            institutions_float_pct=70.1, institutions_count=1450,
            institutional_holders=[HolderRecord(
                holder_id="holder:institutional:vanguard:2026-03-31",
                holder_type="institutional", holder="Vanguard",
                shares=123456.0, date_reported="2026-03-31",
                pct_out=8.25, value=1e7, source="yfinance",
            )],
            source="yfinance",
        )
        assert os.insiders_pct == 4.2
        assert len(os.institutional_holders) == 1
        assert os.institutional_holders[0].holder_id.startswith("holder:")


# ---------------------------------------------------------------------------
# P2: MarketRiskSnapshot
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMarketRiskSnapshot:
    def test_defaults(self):
        from finmindagent.reporting.schemas import MarketRiskSnapshot

        ms = MarketRiskSnapshot()
        assert ms.return_5d_pct is None
        assert ms.source == "deterministic_ohlcv"

    def test_partial(self):
        from finmindagent.reporting.schemas import MarketRiskSnapshot

        ms = MarketRiskSnapshot(return_5d_pct=3.5, max_drawdown_60_sessions_pct=-5.0)
        assert ms.return_5d_pct == 3.5
        assert ms.realized_volatility_20d_pct is None  # not set


# ---------------------------------------------------------------------------
# P2: ValuationContext
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestValuationContext:
    def test_defaults(self):
        from finmindagent.reporting.schemas import ValuationContext

        vc = ValuationContext()
        assert vc.trailing_pe is None
        assert vc.pe_compression_pct is None

    def test_full(self):
        from finmindagent.reporting.schemas import ValuationContext

        vc = ValuationContext(
            trailing_pe=25.0, forward_pe=20.0, price_to_book=5.0,
            peg_ratio=1.5, pe_compression_pct=20.0,
            analyst_target_mean=150.0, mean_target_upside_pct=25.0,
            current_year_eps_growth_pct=33.0, sources=["yfinance"],
        )
        assert vc.trailing_pe == 25.0
        assert vc.pe_compression_pct == 20.0


# ---------------------------------------------------------------------------
# P2: SentimentSignalItem, DataSourceRecord, DataConflict
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSentimentSignalItem:
    def test_defaults(self):
        from finmindagent.reporting.schemas import SentimentSignalItem

        si = SentimentSignalItem(title="x", signal_type="news",
                                 direction="positive", strength=7.0, reasoning="ok")
        assert si.source == "social_sentiment_analyst"
        assert si.evidence_refs == []


@pytest.mark.unit
class TestDataSourceRecord:
    def test_defaults(self):
        from finmindagent.reporting.schemas import DataSourceRecord

        ds = DataSourceRecord(tool_name="get_stock_data", ok=True)
        assert ds.tool_name == "get_stock_data"
        assert ds.truncated is False


@pytest.mark.unit
class TestDataConflict:
    def test_defaults(self):
        from finmindagent.reporting.schemas import DataConflict

        dc = DataConflict(field="pe_ratio_ttm", source_values={"a": 25.0, "b": 26.0},
                          selected_source="a", selected_value=25.0)
        assert dc.field == "pe_ratio_ttm"
        assert len(dc.source_values) == 2


# ---------------------------------------------------------------------------
# P2: Extended ReportSchema
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtendedReportSchemaP2:
    def test_p2_fields_default(self):
        from finmindagent.reporting.schemas import ReportMetadata, ReportSchema

        report = ReportSchema(
            metadata=ReportMetadata(
                ticker="NVDA", trade_date="2026-01-10",
                run_id="r1", status="ok", generated_at="t",
            )
        )
        assert report.earnings_context is None
        assert report.ownership_snapshot is None
        assert report.market_risk_snapshot is None
        assert report.valuation_context is None
        assert report.sentiment_signals == []

    def test_historical_p1_construction_still_works(self):
        from finmindagent.reporting.schemas import DecisionSection, ReportMetadata, ReportSchema

        report = ReportSchema(
            metadata=ReportMetadata(
                ticker="AAPL", trade_date="2026-06-01",
                run_id="old", status="ok", generated_at="t",
            ),
            decision=DecisionSection(rating="Hold"),
        )
        assert report.earnings_context is None
        assert report.decision.rating == "Hold"
        # P1 fields still default correctly
        assert report.analyst_consensus is None

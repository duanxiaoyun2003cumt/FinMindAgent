"""Stable data contract for generated investment reports."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ReportMetadata(BaseModel):
    schema_version: str = "1.0"
    ticker: str
    trade_date: str
    run_id: str
    status: str
    generated_at: str


class TradingPlanSection(BaseModel):
    holding_action: str = ""
    entry_condition: str = ""
    exit_condition: str = ""
    monitoring_events: list[str] = Field(default_factory=list)


class DecisionSection(BaseModel):
    rating: str | None = None
    executive_summary: str = ""
    investment_thesis: str = ""

    current_price: float | None = None
    price_as_of: str = ""

    price_target: float | None = None
    stop_loss: float | None = None

    position_min_pct: float | None = None
    position_max_pct: float | None = None

    confidence_level: str | None = None
    risk_level: str | None = None

    time_horizon: str | None = None

    # P0: deterministic risk/reward from Builder (reward / risk), never LLM.
    risk_reward_ratio: float | None = None
    trading_plan: TradingPlanSection | None = None

    raw_text: str = ""
    structured_available: bool = False


class MarketPoint(BaseModel):
    date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    volume: float | None = None
    indicators: dict[str, float] = Field(default_factory=dict)


class FinancialMetric(BaseModel):
    name: str
    value: float
    unit: str = ""
    period: str = ""
    source: str = ""


class NamedScore(BaseModel):
    name: str
    value: float
    maximum: float = 10.0
    source: str = ""


class Viewpoint(BaseModel):
    role: str
    title: str
    body: str


class RiskAssessment(BaseModel):
    aggressive: str = ""
    neutral: str = ""
    conservative: str = ""
    verifier: str = ""
    final_ruling: str = ""


class AnalysisSection(BaseModel):
    key: str
    title: str
    body: str


class ForwardEstimatePoint(BaseModel):
    period: Literal[
        "current_quarter", "next_quarter", "current_year", "next_year"
    ]
    average: float | None = None
    low: float | None = None
    high: float | None = None
    year_ago: float | None = None
    growth_pct: float | None = None
    analyst_count: int | None = None


class AnalystPriceTargets(BaseModel):
    low: float | None = None
    mean: float | None = None
    median: float | None = None
    high: float | None = None
    vendor_current: float | None = None
    mean_upside_pct: float | None = None
    median_upside_pct: float | None = None


class RecommendationSummary(BaseModel):
    period: str
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0


class EstimateRevisionPoint(BaseModel):
    period: Literal["current_quarter", "next_quarter", "current_year", "next_year"]
    current: float | None = None
    days_7_ago: float | None = None
    days_30_ago: float | None = None
    days_60_ago: float | None = None
    days_90_ago: float | None = None
    up_7d: int | None = None
    up_30d: int | None = None
    down_7d: int | None = None
    down_30d: int | None = None


class AnalystConsensus(BaseModel):
    as_of: str = ""
    price_targets: AnalystPriceTargets | None = None
    earnings_estimates: list[ForwardEstimatePoint] = Field(default_factory=list)
    revenue_estimates: list[ForwardEstimatePoint] = Field(default_factory=list)
    recommendations: list[RecommendationSummary] = Field(default_factory=list)
    eps_revisions: list[EstimateRevisionPoint] = Field(default_factory=list)
    source: str = ""


class EstimateRevisionDiagnostic(BaseModel):
    """P2 (RPT-21): Deterministic derived diagnostic from an EPS revision row.

    Only consumes ``AnalystConsensus.eps_revisions``; never parses prose.
    Partial data is preserved: a row with trend but no counts (or vice versa)
    still produces a diagnostic row, with missing fields left None.
    """

    period: str
    current: float | None = None
    days_30_ago: float | None = None
    change_30d_pct: float | None = None
    up_30d: int | None = None
    down_30d: int | None = None
    net_revisions_30d: int | None = None
    direction: Literal["rising", "falling", "flat", "unknown"] = "unknown"
    source: str = ""


class CalendarEvent(BaseModel):
    event_id: str
    event_type: Literal["earnings"]
    date: str
    title: str
    eps_estimate: float | None = None
    revenue_estimate: float | None = None
    source: str = ""


class CatalystItem(BaseModel):
    title: str
    catalyst_type: str
    direction: str
    importance: str
    trigger_condition: str
    invalidation_condition: str = ""
    follow_up: str
    evidence_refs: list[str] = Field(default_factory=list)
    expected_date: str = ""
    status: Literal["scheduled", "watching"] = "watching"
    source: str = ""


class TimelineItem(BaseModel):
    date: str = ""
    event: str
    # Only set when the data source explicitly provides sentiment; never
    # guessed from keywords. "unknown" is the honest default.
    impact: str = "unknown"
    importance: str = "unknown"
    impact_reason: str = ""
    detail: str = ""
    source: str = ""
    url: str = ""
    scope: str = "company"
    kind: Literal["news", "insider", "calendar", "other"] = "news"


class CompanyProfile(BaseModel):
    name: str = ""
    sector: str = ""
    industry: str = ""
    currency: str = ""
    exchange: str = ""


class DebateResolutionSection(BaseModel):
    recommendation: str | None = None
    disagreement: str = ""
    key_bull_points: list[str] = Field(default_factory=list)
    key_bear_points: list[str] = Field(default_factory=list)
    decisive_factors: list[str] = Field(default_factory=list)
    final_reason: str = ""
    source: str = ""


class TechnicalSnapshot(BaseModel):
    as_of: str = ""
    trend: Literal["bullish", "mixed", "bearish", "unknown"] = "unknown"
    latest_close: float | None = None
    sma20: float | None = None
    sma50: float | None = None
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bollinger_middle: float | None = None
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None
    kdj_k: float | None = None
    kdj_d: float | None = None
    kdj_j: float | None = None
    support_levels: list[float] = Field(default_factory=list)
    resistance_levels: list[float] = Field(default_factory=list)
    source: str = ""


class FinancialQualityFlag(BaseModel):
    title: str
    evidence: str


class FinancialQuality(BaseModel):
    summary: str = ""
    warnings: list[FinancialQualityFlag] = Field(default_factory=list)
    source: str = ""


class DebateArgument(BaseModel):
    stance: str  # bull | bear (runtime fact)
    title: str
    body: str
    strength: float
    evidence: list[str] = Field(default_factory=list)
    source: str = ""


class FinancialSeriesPoint(BaseModel):
    period: str
    value: float
    growth_pct: float | None = None


class FinancialSeries(BaseModel):
    name: str
    label: str
    unit: str = ""
    source: str
    points: list[FinancialSeriesPoint]


class RiskDimensionValue(BaseModel):
    score: float = Field(ge=0, le=10)
    reason: str


class RiskDimensions(BaseModel):
    valuation: RiskDimensionValue
    financial_quality: RiskDimensionValue
    market_technical: RiskDimensionValue
    event_legal: RiskDimensionValue
    liquidity_sentiment: RiskDimensionValue


class RiskItem(BaseModel):
    name: str
    severity: Literal["low", "medium", "high", "critical"]
    horizon: str = ""
    evidence: str = ""
    sources: list[str] = Field(default_factory=list)
    category: str = "other"
    horizon_bucket: str = "unknown"
    # P2 (RPT-26): canonical risk identity preserved through merge.
    # Empty when the entry had no structured risk_key (legacy path).
    risk_key: str = ""
    probability_min_pct: float | None = None
    probability_max_pct: float | None = None
    downside_impact_min_pct: float | None = None
    downside_impact_max_pct: float | None = None
    expected_loss_min_pct: float | None = None
    expected_loss_max_pct: float | None = None
    estimate_basis: list[str] = Field(default_factory=list)


class RiskView(BaseModel):
    role: str  # aggressive | neutral | conservative
    label: str  # Aggressive | Neutral | Conservative
    risk_score: float
    summary: str = ""
    dimensions: RiskDimensions | None = None


class EarningsSurprisePoint(BaseModel):
    evidence_id: str
    report_date: str
    eps_estimate: float | None = None
    reported_eps: float | None = None
    surprise_pct: float | None = None
    source: str = ""


class EarningsContext(BaseModel):
    as_of: str = ""
    surprises: list[EarningsSurprisePoint] = Field(default_factory=list)
    source: str = ""


class EarningsRealizationDiagnostic(BaseModel):
    """P2 (RPT-22): Deterministic earnings realization summary.

    Only consumes ``EarningsContext.surprises``; never parses prose.
    Rows with ``surprise_pct is None`` are excluded from sample statistics
    but the latest row is chosen by ``report_date`` over ALL rows.
    """

    sample_size: int = 0
    beat_count: int = 0
    miss_count: int = 0
    inline_count: int = 0
    beat_rate_pct: float | None = None
    mean_surprise_pct: float | None = None
    latest_surprise_pct: float | None = None
    latest_report_date: str = ""
    source: str = ""


class HolderRecord(BaseModel):
    holder_id: str
    holder_type: Literal["institutional", "mutual_fund"]
    holder: str
    shares: float | None = None
    date_reported: str = ""
    pct_out: float | None = None
    value: float | None = None
    source: str = ""


class OwnershipSnapshot(BaseModel):
    as_of: str = ""
    insiders_pct: float | None = None
    institutions_pct: float | None = None
    institutions_float_pct: float | None = None
    institutions_count: int | None = None
    institutional_holders: list[HolderRecord] = Field(default_factory=list)
    mutual_fund_holders: list[HolderRecord] = Field(default_factory=list)
    source: str = ""


class OwnershipDiagnostic(BaseModel):
    """P2 (RPT-25): Deterministic point-in-time ownership snapshot diagnostic.

    Only consumes the normalized ``OwnershipSnapshot``; never infers flow,
    accumulation, distribution, or buy/sell signals. ``retained_holder_count``
    is the number of retained institutional holder records, NOT
    ``institutions_count`` (provider total).
    """

    as_of: str = ""
    institutional_pct: float | None = None
    insiders_pct: float | None = None
    top_holder_pct: float | None = None
    top5_institutional_pct: float | None = None
    retained_holder_count: int = 0
    source: str = ""


class MarketRiskSnapshot(BaseModel):
    as_of: str = ""
    return_5d_pct: float | None = None
    return_20d_pct: float | None = None
    realized_volatility_20d_pct: float | None = None
    max_drawdown_60_sessions_pct: float | None = None
    atr14_pct: float | None = None
    volume_ratio_20d: float | None = None
    source: str = "deterministic_ohlcv"


class ValuationContext(BaseModel):
    trailing_pe: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    peg_ratio: float | None = None
    pe_compression_pct: float | None = None
    analyst_target_mean: float | None = None
    analyst_target_median: float | None = None
    analyst_target_low: float | None = None
    analyst_target_high: float | None = None
    analyst_target_dispersion_pct: float | None = None
    mean_target_upside_pct: float | None = None
    median_target_upside_pct: float | None = None
    current_year_eps_growth_pct: float | None = None
    next_year_eps_growth_pct: float | None = None
    sources: list[str] = Field(default_factory=list)


class SentimentSignalItem(BaseModel):
    title: str
    signal_type: str
    direction: str
    strength: float
    reasoning: str
    evidence_refs: list[str] = Field(default_factory=list)
    source: str = "social_sentiment_analyst"


class DataSourceRecord(BaseModel):
    tool_name: str
    ok: bool
    retrieved_at: str = ""
    data_as_of: str = ""
    structured_keys: list[str] = Field(default_factory=list)
    truncated: bool = False
    error: str = ""


class DataConflict(BaseModel):
    field: str
    period: str = ""
    unit: str = ""
    source_values: dict[str, float] = Field(default_factory=dict)
    selected_source: str = ""
    selected_value: float | None = None


class DataQuality(BaseModel):
    missing_sections: list[str] = Field(default_factory=list)
    unparsed_sources: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    sources: list[DataSourceRecord] = Field(default_factory=list)
    conflicts: list[DataConflict] = Field(default_factory=list)


class EvidenceDirection(str, Enum):
    """Deterministic evidence direction — only assigned when the underlying
    fact itself has an explicit direction. Context facts stay UNKNOWN."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ResearchEvidenceItem(BaseModel):
    """P2 (RPT-26): One stable structured evidence fact for the decision audit.

    ``evidence_id`` is a deterministic stable identity (reusing structured
    IDs such as earnings evidence_id, holder_id, risk_key, calendar event_id).
    Direction is derived from the fact's numeric/structured sign only;
    volatility, drawdown magnitude, ownership concentration stay UNKNOWN.
    """

    evidence_id: str
    domain: Literal[
        "market",
        "valuation",
        "earnings",
        "estimates",
        "ownership",
        "news",
        "sentiment",
        "risk",
        "financial_quality",
    ]
    title: str
    direction: EvidenceDirection = EvidenceDirection.UNKNOWN
    value_text: str = ""
    as_of: str = ""
    source: str = ""
    source_ref: str = ""


class EvidenceRelation(str, Enum):
    """Relation of an evidence item to the authoritative PM decision."""

    SUPPORT = "support"
    CONFLICT = "conflict"
    CONTEXT = "context"


class DecisionEvidenceAudit(BaseModel):
    """P2 (RPT-27): Deterministic audit of which evidence supports, conflicts,
    or is context for the authoritative PM decision.

    An audit layer only — it never modifies rating/target/stop/position.
    Requires a validated structured ``portfolio_decision``; legacy prose is
    never treated as authoritative direction.
    """

    decision_direction: Literal["positive", "negative", "neutral", "unknown"]
    support_refs: list[str] = Field(default_factory=list)
    conflict_refs: list[str] = Field(default_factory=list)
    context_refs: list[str] = Field(default_factory=list)


class DataReliabilitySummary(BaseModel):
    """P2 (RPT-28): Deterministic summary of DataQuality.

    Counts only — no trust/reliability/quality score, no weighted formula.
    ``stale_market_data`` mirrors the existence of a stale note in
    ``data_quality.notes``.
    """

    source_count: int = 0
    failed_source_count: int = 0
    truncated_source_count: int = 0
    conflict_count: int = 0
    missing_section_count: int = 0
    stale_market_data: bool = False


class MarketPriceBasis(BaseModel):
    """Explicit market price adjustment metadata.

    Declares the price-basis contract shared by current price, technical
    indicators, support/resistance, chart OHLCV, and all target/stop prices.
    """

    mode: Literal["provider_adjusted", "raw", "unknown"] = "unknown"
    provider: str = ""
    currency: str = ""
    exchange: str = ""
    as_of: str = ""


class CorporateAction(BaseModel):
    """Provider-reported corporate action event from yfinance history metadata."""

    date: str = ""
    action_type: Literal["stock_split", "dividend", "capital_gain", "other"] = "other"
    value: float = 0.0
    source: str = ""


class ReportSchema(BaseModel):
    metadata: ReportMetadata
    decision: DecisionSection = Field(default_factory=DecisionSection)
    market_price_basis: MarketPriceBasis = Field(default_factory=MarketPriceBasis)
    market_series: list[MarketPoint] = Field(default_factory=list)
    financial_metrics: list[FinancialMetric] = Field(default_factory=list)
    scores: list[NamedScore] = Field(default_factory=list)
    viewpoints: list[Viewpoint] = Field(default_factory=list)
    risk_assessment: RiskAssessment = Field(default_factory=RiskAssessment)
    analysis_sections: list[AnalysisSection] = Field(default_factory=list)
    timeline: list[TimelineItem] = Field(default_factory=list)
    risk_items: list[RiskItem] = Field(default_factory=list)
    risk_views: list[RiskView] = Field(default_factory=list)
    debate_arguments: list[DebateArgument] = Field(default_factory=list)
    financial_series: list[FinancialSeries] = Field(default_factory=list)
    company_profile: CompanyProfile | None = None
    technical_snapshot: TechnicalSnapshot | None = None
    debate_resolution: DebateResolutionSection | None = None
    financial_quality: FinancialQuality | None = None
    analyst_consensus: AnalystConsensus | None = None
    estimate_revision_diagnostics: list[EstimateRevisionDiagnostic] = Field(default_factory=list)
    calendar_events: list[CalendarEvent] = Field(default_factory=list)
    catalysts: list[CatalystItem] = Field(default_factory=list)
    earnings_context: EarningsContext | None = None
    earnings_realization: EarningsRealizationDiagnostic | None = None
    ownership_snapshot: OwnershipSnapshot | None = None
    ownership_diagnostic: OwnershipDiagnostic | None = None
    market_risk_snapshot: MarketRiskSnapshot | None = None
    valuation_context: ValuationContext | None = None
    sentiment_signals: list[SentimentSignalItem] = Field(default_factory=list)
    corporate_actions: list[CorporateAction] = Field(default_factory=list)
    evidence_registry: list[ResearchEvidenceItem] = Field(default_factory=list)
    decision_evidence_audit: DecisionEvidenceAudit | None = None
    data_reliability: DataReliabilitySummary | None = None
    data_quality: DataQuality = Field(default_factory=DataQuality)

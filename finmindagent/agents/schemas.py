"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class DebateResolution(BaseModel):
    """Structured verdict produced by the Research Manager after weighing
    the Bull and Bear arguments.

    This is the Research Manager's own judgement, not a re-statement of
    the raw debate. All fields may be empty when evidence is insufficient;
    the field descriptions instruct the model not to invent facts.
    """

    disagreement: str = Field(
        default="",
        description="Core point of disagreement between the bull and bear sides.",
    )
    key_bull_points: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three most compelling bull points accepted by the manager.",
    )
    key_bear_points: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three most concerning bear points accepted by the manager.",
    )
    decisive_factors: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three factors that ultimately decided the recommendation.",
    )
    final_reason: str = Field(
        default="",
        description="Concise explanation of why the recommendation was reached.",
    )


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )
    resolution: DebateResolution = Field(
        default_factory=DebateResolution,
        description="Structured resolution of the bull/bear debate.",
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    parts = [
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ]
    resolution = plan.resolution
    if resolution.disagreement:
        parts.extend(["", f"**Disagreement**: {resolution.disagreement}"])
    if resolution.key_bull_points:
        parts.append("**Key Bull Points**:")
        parts.extend(f"- {point}" for point in resolution.key_bull_points)
    if resolution.key_bear_points:
        parts.append("**Key Bear Points**:")
        parts.extend(f"- {point}" for point in resolution.key_bear_points)
    if resolution.decisive_factors:
        parts.append("**Decisive Factors**:")
        parts.extend(f"- {point}" for point in resolution.decisive_factors)
    if resolution.final_reason:
        parts.extend(["", f"**Final Reason**: {resolution.final_reason}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class ConfidenceLevel(str, Enum):
    """Model confidence in the final decision."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# ---------------------------------------------------------------------------
# Scored analysis (market / fundamentals / news / sentiment analysts)
# ---------------------------------------------------------------------------


class ScoredAnalysisOutput(BaseModel):
    """Structured output for the four analysis agents.

    ``analysis`` is the natural-language report consumed by downstream agents;
    ``score`` is the investment favorability of this dimension only.  Higher
    is always more favorable (0 strongly negative, 5 neutral, 10 strongly
    positive) so all dimensions can share one radar.
    """

    analysis: str = Field(
        description=(
            "Full natural-language analysis consumed by downstream agents."
        ),
    )
    score: float = Field(
        ge=0,
        le=10,
        description=(
            "Investment favorability score for this analysis dimension. "
            "0 is strongly negative, 5 is neutral, 10 is strongly positive."
        ),
    )
    score_reason: str = Field(
        description="Short explanation for the numerical score.",
    )


class NewsImpact(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EventImportance(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NewsEventAssessment(BaseModel):
    """Per-event impact/importance produced by the News Analyst."""

    title: str = Field(
        description=(
            "Exact event title as it appears in the supplied news data."
        ),
    )
    impact: NewsImpact = Field(
        description="Market impact direction of this event.",
    )
    importance: EventImportance = Field(
        description="How important this event is for the investment case.",
    )
    reason: str = Field(
        description="Short justification for the impact/importance.",
    )


class CatalystType(str, Enum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    CORPORATE_ACTION = "corporate_action"
    LEGAL = "legal"
    INSIDER = "insider"
    INDUSTRY = "industry"
    OTHER = "other"


class CatalystDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"


class CatalystCandidate(BaseModel):
    """A future catalyst or event to watch, with evidence anchoring."""

    title: str = Field(description="Short title of the catalyst.")
    catalyst_type: CatalystType = Field(description="Category of the catalyst.")
    direction: CatalystDirection = Field(description="Expected market impact direction.")
    importance: EventImportance = Field(description="How important this catalyst is.")
    trigger_condition: str = Field(description="What must happen for the catalyst to fire.")
    invalidation_condition: str = Field(
        default="",
        description="What would invalidate this catalyst.",
    )
    follow_up: str = Field(description="What to do when the catalyst triggers or invalidates.")
    evidence_refs: list[str] = Field(
        min_length=1,
        max_length=3,
        description=(
            "Evidence references. Each must be an exact supplied news/insider "
            "event title or an exact company-calendar event_id. Do not invent refs."
        ),
    )


class NewsAnalysisOutput(ScoredAnalysisOutput):
    """News Analyst output: favorability score plus per-event assessments."""

    events: list[NewsEventAssessment] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Assessments for events actually present in the supplied news "
            "data. Do not invent events; each title must correspond to an "
            "event in the news."
        ),
    )
    catalysts: list[CatalystCandidate] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Future catalysts or events to watch. Every evidence_ref must "
            "be an exact supplied event title or calendar event_id."
        ),
    )


# ---------------------------------------------------------------------------
# Bull / Bear debate research
# ---------------------------------------------------------------------------


class ScoredArgument(BaseModel):
    """A single debate argument with its own persuasiveness strength.

    ``strength`` is the argument's own strength (0-10), not the stock rating.
    """

    title: str = Field(
        description="Short title of the argument.",
    )
    argument: str = Field(
        description="The argument body.",
    )
    strength: float = Field(
        ge=0,
        le=10,
        description=(
            "Persuasiveness of this individual argument on a 0-10 scale."
        ),
    )
    evidence: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three concrete evidence points for the argument.",
    )


class DebateResearchOutput(BaseModel):
    """Structured output for Bull/Bear researchers.

    The stance (bull/bear) is a runtime fact derived from the agent role, not
    returned by the model.
    """

    summary: str = Field(
        description="One-paragraph summary of the case.",
    )
    arguments: list[ScoredArgument] = Field(
        min_length=2,
        max_length=4,
        description="Two to four arguments supporting the stance.",
    )


# --- Binding variants for provider tolerance -------------------------------
# The canonical schema above stays strict (evidence max 3). When binding the
# provider schema, LangChain validates inside ``invoke`` and raises
# ValidationError on overflow — so the pipeline binds with these generous
# caps, then applies ``normalize_to_schema_max`` (runtime/structured_normalize)
# to truncate deterministically back to the canonical maximums. This only
# absorbs collection-length overflow; everything else still fails validation.

_BIND_MAX_EVIDENCE = 12
_BIND_MAX_ARGUMENTS = 8


class ScoredArgumentBind(BaseModel):
    """Binding variant of ``ScoredArgument`` with tolerant evidence cap."""

    title: str = Field(description="Short title of the argument.")
    argument: str = Field(description="The argument body.")
    strength: float = Field(
        ge=0,
        le=10,
        description="Persuasiveness of this individual argument on a 0-10 scale.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        max_length=_BIND_MAX_EVIDENCE,
        description="Up to twelve evidence points; runtime truncates to three.",
    )


class DebateResearchOutputBind(BaseModel):
    """Binding variant of ``DebateResearchOutput`` for provider tolerance.

    Never stored as the report's authoritative structured data — the engine
    normalizes it to the canonical ``DebateResearchOutput`` before use.
    """

    summary: str = Field(description="One-paragraph summary of the case.")
    arguments: list[ScoredArgumentBind] = Field(
        min_length=2,
        max_length=_BIND_MAX_ARGUMENTS,
        description="Two to eight arguments; runtime truncates to four.",
    )


# ---------------------------------------------------------------------------
# Risk analysts (aggressive / neutral / conservative)
# ---------------------------------------------------------------------------


class RiskCanonicalKey(str, Enum):
    """Canonical business risk identity used for merging the same underlying
    risk across analysts. Choose by economic meaning, not wording.

    P0 taxonomy covers the most common risk categories. The free-form ``name``
    field remains the user-facing display label; ``risk_key`` is the stable
    system identity for deduplication and aggregation.
    """

    CYCLICAL_MEAN_REVERSION = "cyclical_mean_reversion"
    FORWARD_ESTIMATE_REVISION = "forward_estimate_revision"
    VALUATION = "valuation"
    TECHNICAL_DOWNTREND = "technical_downtrend"
    VOLATILITY_LIQUIDITY = "volatility_liquidity"
    EVENT_CATALYST = "event_catalyst"
    INDUSTRY_SUPPLY_DEMAND = "industry_supply_demand"
    FINANCIAL_QUALITY = "financial_quality"
    BALANCE_SHEET = "balance_sheet"
    OWNERSHIP_INSIDER = "ownership_insider"
    LEGAL_REGULATORY = "legal_regulatory"
    OTHER = "other"


class RiskCategory(str, Enum):
    VALUATION = "valuation"
    FINANCIAL_QUALITY = "financial_quality"
    MARKET_TECHNICAL = "market_technical"
    EVENT_LEGAL = "event_legal"
    LIQUIDITY_SENTIMENT = "liquidity_sentiment"
    OTHER = "other"


class RiskHorizon(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    UNKNOWN = "unknown"


class RiskSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskDimensionValue(BaseModel):
    score: float = Field(ge=0, le=10, description="Dimension score 0-10, higher = more risk.")
    reason: str = Field(description="Brief justification for the score.")


class RiskDimensions(BaseModel):
    valuation: RiskDimensionValue
    financial_quality: RiskDimensionValue
    market_technical: RiskDimensionValue
    event_legal: RiskDimensionValue
    liquidity_sentiment: RiskDimensionValue


class StructuredRisk(BaseModel):
    name: str = Field(
        description="Short risk name, e.g. 'valuation' or 'liquidity'.",
    )
    risk_key: RiskCanonicalKey = Field(
        description=(
            "Canonical business risk identity used to merge the same underlying "
            "risk across analysts. Choose by economic meaning, not wording. "
            "Three Risk Analysts describing the same economic risk must use the "
            "same risk_key."
        ),
    )
    severity: RiskSeverity = Field(
        description="Severity of the risk.",
    )
    horizon: str = Field(
        description="Time horizon over which the risk matters.",
    )
    evidence: str = Field(
        description="Concrete evidence for the risk.",
    )
    category: RiskCategory = Field(
        default=RiskCategory.OTHER,
        description="Risk category for dimensional grouping.",
    )
    horizon_bucket: RiskHorizon = Field(
        default=RiskHorizon.UNKNOWN,
        description="Time-horizon bucket for grouping.",
    )
    probability_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Model-estimated probability (0-100). Requires estimate_basis.",
    )
    downside_impact_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Model-estimated downside impact percentage. Requires estimate_basis.",
    )
    estimate_basis: str = Field(
        default="",
        description="Explicit basis for any numeric probability/impact estimates.",
    )

    @model_validator(mode="after")
    def _require_basis_for_numeric(self) -> "StructuredRisk":
        if (
            self.probability_pct is not None
            or self.downside_impact_pct is not None
        ) and not self.estimate_basis.strip():
            raise ValueError("numeric risk estimates require estimate_basis")
        return self


class RiskAnalystOutput(BaseModel):
    """Structured output for the three risk analysts.

    ``risk_score`` is risk *intensity* — 0 is very low risk, 10 is extremely
    high risk. This is intentionally opposite to the favorability scores in
    ``ScoredAnalysisOutput`` and must never share a collection with them.
    """

    summary: str = Field(
        description="One-paragraph risk assessment.",
    )
    risk_score: float = Field(
        ge=0,
        le=10,
        description=(
            "Overall risk intensity on a 0-10 scale. 0 is very low risk; "
            "10 is extremely high risk."
        ),
    )
    risks: list[StructuredRisk] = Field(
        default_factory=list,
        max_length=5,
        description="Up to five concrete risks with severity.",
    )
    dimensions: RiskDimensions | None = Field(
        default=None,
        description="Optional five-dimension risk breakdown.",
    )


def render_risk_analyst(output: RiskAnalystOutput) -> str:
    """Render a risk analyst output back to markdown for prose consumers."""
    parts = [output.summary, f"Risk score: {output.risk_score:g}/10"]
    for risk in output.risks:
        parts.append(
            f"- {risk.name} [{risk.severity.value}] ({risk.horizon}): "
            f"{risk.evidence}"
        )
    return "\n\n".join(parts)


def render_debate_research(output: DebateResearchOutput) -> str:
    """Render debate research back to markdown for prose consumers.

    Strength values are not re-parsed anywhere downstream; the structured
    copy in ``state.structured_reports`` is the only source for scores.
    """
    parts = [f"## Summary\n\n{output.summary}", "## Arguments"]
    for argument in output.arguments:
        parts.extend(
            [
                f"### {argument.title}",
                "",
                argument.argument,
                "",
                f"Strength: {argument.strength:g}/10",
            ]
        )
        if argument.evidence:
            parts.extend(["Evidence:", *[f"- {point}" for point in argument.evidence]])
    return "\n\n".join(parts)


class RiskLevel(str, Enum):
    """Overall portfolio risk level of the decision."""

    LOW = "Low"
    MEDIUM = "Medium"
    MEDIUM_HIGH = "Medium-High"
    HIGH = "High"


class PortfolioExecutionPlan(BaseModel):
    """Structured execution plan for the portfolio decision.

    Describes the concrete actions, entry/exit conditions, and events to
    monitor. All fields may be empty when evidence is insufficient; the
    field descriptions instruct the model not to invent prices or events
    not present in the supplied evidence.
    """

    holding_action: str = Field(
        default="",
        description=(
            "What existing holders should do: e.g. 'Keep reduced position', "
            "'Accumulate gradually', 'Exit fully'. Do not invent a price or "
            "event not present in supplied evidence."
        ),
    )
    entry_condition: str = Field(
        default="",
        description=(
            "Condition for new entry or adding to position. Do not invent a "
            "price or event not present in supplied evidence."
        ),
    )
    exit_condition: str = Field(
        default="",
        description=(
            "Condition for reducing or closing the position. Do not invent a "
            "price or event not present in supplied evidence."
        ),
    )
    monitoring_events: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Up to five events or catalysts to monitor.",
    )


class FinancialQualityFlag(BaseModel):
    """A single financial quality concern or observation."""

    title: str = Field(description="Short label for the quality observation.")
    evidence: str = Field(description="Concrete evidence supporting the observation.")


class SentimentSignalType(str, Enum):
    NEWS = "news"
    OWNERSHIP = "ownership"
    OTHER = "other"


class SentimentDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SentimentSignal(BaseModel):
    title: str = Field(description="Short signal title.")
    signal_type: SentimentSignalType = Field(description="Source type of the signal.")
    direction: SentimentDirection = Field(description="Market impact direction.")
    strength: float = Field(ge=0, le=10, description="Signal strength 0-10.")
    reasoning: str = Field(description="Brief reasoning for the signal.")
    evidence_refs: list[str] = Field(
        min_length=1, max_length=3,
        description="Evidence references. NEWS = exact news title; OWNERSHIP = exact holder_id.",
    )


class SentimentAnalysisOutput(ScoredAnalysisOutput):
    signals: list[SentimentSignal] = Field(
        default_factory=list, max_length=8,
        description="Grounded sentiment signals. Evidence_refs must be exact references.",
    )


class FundamentalsAnalysisOutput(ScoredAnalysisOutput):
    """Fundamentals analyst output with structured financial-quality fields.

    Extends the shared favorability score with quality assessment that the
    Builder can display directly — no prose parsing required.
    """

    financial_quality_summary: str = Field(
        min_length=1,
        description="One-paragraph summary of financial quality and key concerns. Required — "
                    "must be non-empty; absence causes schema validation failure.",
    )
    quality_warnings: list[FinancialQualityFlag] = Field(
        default_factory=list,
        max_length=5,
        description="Up to five concrete financial quality warnings or observations.",
    )


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.

    All optional numeric fields stay ``None`` when the model has no reliable
    basis for them; Python enforces the position/stop-loss sanity constraints.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description=(
            "Target price in the instrument's quote currency. MUST be filled "
            "when evidence supports a concrete level (analyst targets, technical "
            "support/resistance, trading-plan entry or exit conditions); MUST "
            "be null ONLY when no reliable price can be formed. Direction-"
            "consistent: Buy/Overweight targets are usually above the current "
            "price; Underweight/Sell targets are usually below the current "
            "price. Never fabricate a number to fill the field."
        ),
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description=(
            "Stop-loss / invalidation price level in the instrument's quote "
            "currency. MUST be filled when a concrete invalidation level "
            "exists; MUST be null ONLY when no reliable level can be formed. "
            "Direction-consistent: Buy/Overweight stops are usually below the "
            "current price; Underweight/Sell invalidation levels are usually "
            "above the current price. Must be greater than 0. Never fabricate "
            "a number to fill the field."
        ),
    )
    position_min_pct: Optional[float] = Field(
        default=None,
        description=(
            "Optional minimum recommended position size as a percentage of the "
            "portfolio (0-100)."
        ),
    )
    position_max_pct: Optional[float] = Field(
        default=None,
        description=(
            "Optional maximum recommended position size as a percentage of the "
            "portfolio (0-100). Must be >= position_min_pct."
        ),
    )
    confidence_level: Optional[ConfidenceLevel] = Field(
        default=None,
        description=(
            "Optional confidence in this decision: exactly one of Low / Medium / "
            "High."
        ),
    )
    risk_level: Optional[RiskLevel] = Field(
        default=None,
        description=(
            "Optional overall risk level of this decision: exactly one of "
            "Low / Medium / Medium-High / High."
        ),
    )
    execution_plan: PortfolioExecutionPlan = Field(
        default_factory=PortfolioExecutionPlan,
        description="Structured execution plan. May be empty when evidence is insufficient.",
    )

    @model_validator(mode="after")
    def _validate_position_and_stop(self) -> "PortfolioDecision":
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("stop_loss must be greater than 0")
        position_values = [
            value
            for value in (self.position_min_pct, self.position_max_pct)
            if value is not None
        ]
        if any(value < 0 or value > 100 for value in position_values):
            raise ValueError("position_min_pct/position_max_pct must be within 0-100")
        if (
            self.position_min_pct is not None
            and self.position_max_pct is not None
            and self.position_min_pct > self.position_max_pct
        ):
            raise ValueError("position_min_pct must not exceed position_max_pct")
        return self


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    if decision.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {decision.stop_loss}"])
    if decision.position_min_pct is not None or decision.position_max_pct is not None:
        if decision.position_min_pct is not None and decision.position_max_pct is not None:
            position = f"{decision.position_min_pct:g}%-{decision.position_max_pct:g}%"
        elif decision.position_max_pct is not None:
            position = f"Up to {decision.position_max_pct:g}%"
        else:
            position = f"At least {decision.position_min_pct:g}%"
        parts.extend(["", f"**Position Size**: {position}"])
    if decision.confidence_level is not None:
        parts.extend(["", f"**Confidence Level**: {decision.confidence_level.value}"])
    if decision.risk_level is not None:
        parts.extend(["", f"**Risk Level**: {decision.risk_level.value}"])
    ep = decision.execution_plan
    if ep.holding_action:
        parts.extend(["", f"**Holding Action**: {ep.holding_action}"])
    if ep.entry_condition:
        parts.extend(["", f"**Entry Condition**: {ep.entry_condition}"])
    if ep.exit_condition:
        parts.extend(["", f"**Exit Condition**: {ep.exit_condition}"])
    if ep.monitoring_events:
        parts.append("**Monitoring Events**:")
        parts.extend(f"- {event}" for event in ep.monitoring_events)
    return "\n".join(parts)

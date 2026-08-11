"""Deterministic adapters from Runtime state to the report contract."""

from __future__ import annotations

import csv
import math
import re
from io import StringIO
from typing import Any

from pydantic import BaseModel, ValidationError

from finmindagent.agents.schemas import PortfolioDecision
from finmindagent.time_utils import now_system
from finmindagent.reporting.curation import (
    attach_derived_indicators,
    attach_moving_average,
    build_data_reliability_summary,
    build_decision_evidence_audit,
    build_earnings_realization_diagnostic,
    build_estimate_revision_diagnostics,
    build_evidence_registry,
    build_financial_series,
    build_market_risk_snapshot,
    build_ownership_diagnostic,
    build_technical_snapshot,
    curate_financial_metrics,
    curate_timeline,
    merge_risk_items,
    normalize_event_title,
)
from finmindagent.reporting.schemas import (

    AnalysisSection,
    AnalystConsensus,
    CalendarEvent,
    CorporateAction,
    EarningsContext,
    OwnershipSnapshot,
    CompanyProfile,
    DataQuality,
    DebateArgument,
    DebateResolutionSection,
    DecisionSection,
    FinancialMetric,
    FinancialQuality,
    FinancialQualityFlag,
    MarketPoint,
    MarketPriceBasis,
    MarketRiskSnapshot,
    NamedScore,
    ReportMetadata,
    ReportSchema,
    RiskItem,
    RiskAssessment,
    RiskView,
    TechnicalSnapshot,
    TimelineItem,
    TradingPlanSection,
    Viewpoint,
)


_ANALYSIS_REPORTS = (
    ("market_report", "Technical and Market Analysis"),
    ("sentiment_report", "Social Sentiment Analysis"),
    ("news_report", "News and Events Analysis"),
    ("fundamentals_report", "Fundamental Analysis"),
)

_VIEWPOINT_REPORTS = (
    ("bull_researcher", "bull", "Bull Case"),
    ("bear_researcher", "bear", "Bear Case"),
    ("investment_plan", "research_manager", "Research Manager Plan"),
    ("trader_investment_plan", "trader", "Trading Plan"),
)

_DECISION_LABELS = (
    "Rating",
    "Executive Summary",
    "Investment Thesis",
    "Price Target",
    "Time Horizon",
    "Stop Loss",
    "Position Size",
    "Confidence Level",
    "Risk Level",
    "Holding Action",
    "Entry Condition",
    "Exit Condition",
    "Monitoring Events",
)

_DECISION_PATTERN = re.compile(
    rf"^\*\*({'|'.join(re.escape(label) for label in _DECISION_LABELS)})\*\*:\s*"
    rf"(.*?)(?=^\*\*(?:{'|'.join(re.escape(label) for label in _DECISION_LABELS)})\*\*:|\Z)",
    re.MULTILINE | re.DOTALL,
)

_RATING_VALUES = frozenset({"Buy", "Overweight", "Hold", "Underweight", "Sell"})

_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def build_report_schema(
    state: Any,
    *,
    generated_at: str | None = None,
) -> ReportSchema:
    """Build a stable report without asking an LLM to reinterpret the run."""
    reports = dict(getattr(state, "reports", {}) or {})
    analysis_sections = [
        AnalysisSection(key=key, title=title, body=str(reports[key]))
        for key, title in _ANALYSIS_REPORTS
        if str(reports.get(key, "")).strip()
    ]
    viewpoints = [
        Viewpoint(role=role, title=title, body=str(reports[key]))
        for key, role, title in _VIEWPOINT_REPORTS
        if str(reports.get(key, "")).strip()
    ]

    risk = RiskAssessment(
        aggressive=str(reports.get("aggressive_risk_analyst", "")),
        neutral=str(reports.get("neutral_risk_analyst", "")),
        conservative=str(reports.get("conservative_risk_analyst", "")),
        verifier=str(reports.get("risk_verifier", "")),
        final_ruling=str(getattr(state, "final_trade_decision", "") or ""),
    )

    decision = _build_decision(state)
    report = ReportSchema(
        metadata=ReportMetadata(
            ticker=str(getattr(state, "ticker", "")),
            trade_date=str(getattr(state, "trade_date", "")),
            run_id=str(getattr(state, "run_id", "")),
            status=str(getattr(state, "status", "")),
            generated_at=generated_at or now_system().isoformat(),
        ),
        decision=decision,
        viewpoints=viewpoints,
        risk_assessment=risk,
        analysis_sections=analysis_sections,
        data_quality=DataQuality(),
    )
    metadata = getattr(state, "metadata", {}) or {}
    metadata_scores = _normalize_collection(
        metadata.get("report_scores"),
        NamedScore,
        "metadata:report_scores",
        report.data_quality,
    )
    report.scores = _build_scores(state, metadata_scores)
    report.financial_metrics = _normalize_collection(
        metadata.get("financial_metrics"),
        FinancialMetric,
        "metadata:financial_metrics",
        report.data_quality,
    )
    report.timeline = _normalize_collection(
        metadata.get("timeline"),
        TimelineItem,
        "metadata:timeline",
        report.data_quality,
    )
    report.risk_items = _normalize_collection(
        metadata.get("risk_items"),
        RiskItem,
        "metadata:risk_items",
        report.data_quality,
    )
    # P1: structured three-way risk views and merged risk register — the
    # structured outputs replace any metadata placeholder when present.
    risk_views, structured_risk_items = _build_risk_views(state)
    report.risk_views = risk_views
    if structured_risk_items:
        report.risk_items = structured_risk_items
    collected_history: list[dict[str, Any]] = []
    indicator_series: list[dict[str, Any]] = []
    for tool_result in _iter_tool_results(state):
        tool_name = str(tool_result.get("tool_name") or "")
        structured = tool_result.get("structured_data")
        if isinstance(structured, dict):
            # New path: deterministic structured payload attached by the
            # tool adapters. No prose guessing happens here.
            if tool_name == "get_stock_data":
                points = _normalize_collection(
                    structured.get("market_series"),
                    MarketPoint,
                    tool_name,
                    report.data_quality,
                )
                if points:
                    report.market_series = points
                else:
                    _record_unparsed(report.data_quality, tool_name)
                # Task 9: MarketPriceBasis from adapter metadata
                basis_raw = structured.get("market_price_basis")
                if isinstance(basis_raw, dict):
                    try:
                        report.market_price_basis = MarketPriceBasis.model_validate(basis_raw)
                    except ValidationError:
                        _record_unparsed(report.data_quality, f"{tool_name}:market_price_basis")
                # Task 10: Corporate actions from provider-native metadata
                ca_raw = structured.get("corporate_actions")
                if isinstance(ca_raw, list):
                    from finmindagent.reporting.schemas import CorporateAction
                    for ca in ca_raw:
                        if isinstance(ca, dict):
                            try:
                                report.corporate_actions.append(
                                    CorporateAction.model_validate(ca)
                                )
                            except ValidationError:
                                pass
                # P2: consume structured market_risk_snapshot from adapter
                risk_raw = structured.get("market_risk_snapshot")
                if isinstance(risk_raw, dict):
                    try:
                        report.market_risk_snapshot = MarketRiskSnapshot.model_validate(risk_raw)
                    except ValidationError:
                        _record_unparsed(report.data_quality, f"{tool_name}:market_risk_snapshot")
                # Drop-through: if no structured snapshot, fallback recomputation happens later
            elif tool_name == "get_indicators":
                series = structured.get("indicator_series")
                if isinstance(series, list):
                    indicator_series.extend(series)
                else:
                    _record_unparsed(report.data_quality, tool_name)
            elif tool_name in {
                "get_fundamentals",
                "get_balance_sheet",
                "get_cashflow",
                "get_income_statement",
            }:
                _extend_tool_collection(
                    report.financial_metrics,
                    structured,
                    "financial_metrics",
                    FinancialMetric,
                    tool_name,
                    report.data_quality,
                )
                history = structured.get("financial_history")
                if isinstance(history, list):
                    collected_history.extend(history)
            elif tool_name in {"get_news", "get_global_news", "get_insider_transactions"}:
                _extend_tool_collection(
                    report.timeline,
                    structured,
                    "timeline",
                    TimelineItem,
                    tool_name,
                    report.data_quality,
                )
            elif tool_name == "get_company_calendar":
                events = structured.get("calendar_events")
                if isinstance(events, list):
                    cal_items = _normalize_collection(
                        events, CalendarEvent, tool_name, report.data_quality
                    )
                    report.calendar_events.extend(cal_items)
                else:
                    _record_unparsed(report.data_quality, tool_name)
            elif tool_name == "get_analyst_expectations":
                consensus = structured.get("analyst_consensus")
                if isinstance(consensus, dict):
                    try:
                        report.analyst_consensus = AnalystConsensus.model_validate(consensus)
                    except ValidationError:
                        _record_unparsed(report.data_quality, tool_name)
                else:
                    _record_unparsed(report.data_quality, tool_name)
            elif tool_name == "get_earnings_history":
                ec = structured.get("earnings_context")
                if isinstance(ec, dict):
                    try:
                        report.earnings_context = EarningsContext.model_validate(ec)
                    except ValidationError:
                        _record_unparsed(report.data_quality, tool_name)
                else:
                    _record_unparsed(report.data_quality, tool_name)
            elif tool_name == "get_holder_context":
                os_data = structured.get("ownership_snapshot")
                if isinstance(os_data, dict):
                    try:
                        report.ownership_snapshot = OwnershipSnapshot.model_validate(os_data)
                    except ValidationError:
                        _record_unparsed(report.data_quality, tool_name)
                else:
                    _record_unparsed(report.data_quality, tool_name)
            if (
                tool_name == "get_fundamentals"
                and report.company_profile is None
                and isinstance(structured.get("company_profile"), dict)
            ):
                try:
                    report.company_profile = CompanyProfile.model_validate(
                        structured["company_profile"]
                    )
                except ValidationError:
                    _record_unparsed(report.data_quality, f"{tool_name}:company_profile")
            continue
        # Legacy path: raw observation dicts hand-assembled by callers.
        if tool_name == "get_stock_data":
            points = _parse_market_csv(tool_result.get("data"))
            if points:
                report.market_series = points
            else:
                _record_unparsed(report.data_quality, tool_name)
        elif tool_name in {
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
        }:
            _extend_tool_collection(
                report.financial_metrics,
                tool_result.get("data"),
                "financial_metrics",
                FinancialMetric,
                tool_name,
                report.data_quality,
            )
        elif tool_name in {"get_news", "get_global_news"}:
            _extend_tool_collection(
                report.timeline,
                tool_result.get("data"),
                "timeline",
                TimelineItem,
                tool_name,
                report.data_quality,
            )
    # P2: snapshot raw financial metrics before curation (for ValuationContext + conflicts).
    _raw_metrics_snapshot = list(report.financial_metrics)

    # Business curation: keep only core whitelisted metrics, deduplicated by
    # canonical name with fixed source priority.
    report.financial_metrics = curate_financial_metrics(report.financial_metrics)

    # P1: multi-period history → Revenue / Net Income series (>= 2 points).
    report.financial_series = build_financial_series(collected_history)

    # P1: deterministic trailing SMAs on market data (no look-ahead).
    attach_moving_average(report.market_series, window=20, key="sma20")
    attach_moving_average(report.market_series, window=50, key="sma50")

    # P0: merge tool-provided indicator values by exact date, then fill missing
    # indicators deterministically from OHLCV (EMA/MACD/RSI/Bollinger/KDJ).
    # Tool values always win on the same date and key.
    if report.market_series:
        attach_derived_indicators(report.market_series, indicator_series)
        tech_source = (
            "market_series/get_indicators + deterministic_ohlcv"
            if indicator_series
            else "deterministic_ohlcv"
        )
        report.technical_snapshot = build_technical_snapshot(
            report.market_series, source=tech_source
        )

    # Deterministic company-news relevance filter: when a company profile is
    # available, company-scope items must mention the ticker or a core company
    # name (corporate suffixes stripped). Without a profile, nothing is
    # filtered aggressively.
    if report.company_profile and report.company_profile.name:
        core_names = _core_company_names(report.company_profile.name)
        ticker = str(getattr(state, "ticker", "")).lower()
        relevant = []
        for item in report.timeline:
            if item.scope == "global":
                relevant.append(item)
                continue
            # P1: insider items come from ticker-specific tools and are always
            # scoped to the company. Do not filter them just because the event
            # text lacks the ticker or company name.
            if getattr(item, "kind", "news") == "insider":
                relevant.append(item)
                continue
            haystack = f"{item.event} {item.detail}".lower()
            if ticker and ticker in haystack:
                relevant.append(item)
            elif any(name and name in haystack for name in core_names):
                relevant.append(item)
        report.timeline = relevant

    # P1: attach five risk dimensions to RiskViews (only from structured output).
    _attach_risk_dimensions(report.risk_views, state)

    # P2: deterministic market risk snapshot — only if not already from structured data.
    if report.market_series and report.market_risk_snapshot is None:
        report.market_risk_snapshot = build_market_risk_snapshot(report.market_series)

    # Current price comes only from market data, never from model prose.
    if report.market_series:
        latest = max(report.market_series, key=lambda point: point.date)
        report.decision.current_price = latest.close
        report.decision.price_as_of = latest.date

    # P0: analyst consensus → target upside (current market price must exist).
    # Must run BEFORE ValuationContext so upside fields are populated there.
    if report.analyst_consensus and report.analyst_consensus.price_targets:
        _attach_target_upside(report.analyst_consensus.price_targets, report.decision.current_price)

    # P2: ValuationContext from RAW structured metrics + analyst consensus.
    report.valuation_context = _build_valuation_context(report, _raw_metrics_snapshot)

    # P2: Provenance from tool observations.
    _build_provenance(report, state)

    # P2: Financial conflict detection using RAW metrics (before curation).
    _build_conflicts(report, _raw_metrics_snapshot)

    # P2: Market stale detection.
    _build_stale_note(report, state)

    # P2: Grounded sentiment signals from social_sentiment_analyst.
    report.sentiment_signals = _build_sentiment_signals(report, state)

    # Structured Bull/Bear arguments — only from structured_reports, never
    # parsed from prose.
    report.debate_arguments = _build_debate_arguments(state)

    # Enrich timeline items with the News Analyst's structured per-event
    # impact/importance (exact title match only; unmatched stays unknown).
    _enrich_timeline_from_news_assessments(report, state)

    # News curation: dedupe by URL/title, company first, dated newest-first,
    # global capped, total capped.
    report.timeline = curate_timeline(report.timeline)

    # P2 (RPT-21): deterministic EPS revision diagnostics from structured
    # AnalystConsensus.eps_revisions — never parses prose, never calls LLM.
    report.estimate_revision_diagnostics = build_estimate_revision_diagnostics(
        report.analyst_consensus
    )

    # P2 (RPT-22): deterministic earnings realization from structured
    # EarningsContext.surprises — never parses prose, never calls LLM.
    report.earnings_realization = build_earnings_realization_diagnostic(
        report.earnings_context
    )

    # P2 (RPT-25): deterministic ownership snapshot diagnostic from
    # normalized OwnershipSnapshot — never infers flow or buy/sell signals.
    report.ownership_diagnostic = build_ownership_diagnostic(
        report.ownership_snapshot
    )

    # P1: catalyst grounding (evidence_refs matched against calendar + timeline).
    report.catalysts = _build_catalysts(state, report)

    # P0: deterministic risk/reward from current/target/stop, direction-aware
    # (a bearish rating never gets a long-formula ratio).
    report.decision.risk_reward_ratio = _calculate_risk_reward(
        report.decision.current_price,
        report.decision.price_target,
        report.decision.stop_loss,
        rating=report.decision.rating,
    )

    # P0: Research Manager resolution from structured_reports.
    report.debate_resolution = _build_debate_resolution(state)

    # P0: Financial quality from fundamentals_analyst structured output.
    report.financial_quality = _build_financial_quality(state)

    # P1: Final missing_sections — computed AFTER all P0/P1 fields are built.
    report.data_quality.missing_sections = [
        field_name
        for field_name in (
            "market_series",
            "financial_metrics",
            "scores",
            "timeline",
            "risk_items",
            "risk_views",
            "debate_arguments",
            "financial_series",
            "technical_snapshot",
            "analyst_consensus",
            "catalysts",
        )
        if not getattr(report, field_name)
    ]

    if not report.decision.structured_available:
        report.data_quality.notes.append(
            "Structured portfolio decision unavailable; decision card fields "
            "were suppressed to avoid conflicting ratings."
        )

    # Phase A: Structured output diagnostics integration
    _attach_structured_diagnostics_notes(report, state)

    # P2: Conditional missing/data-quality semantics (tool-attempted checks)
    _apply_p2_missing_semantics(report, state)

    # P2 (RPT-26): Stable evidence registry — built LAST after all diagnostics,
    # risk, catalysts, and data-quality finalization.
    report.evidence_registry = build_evidence_registry(report)

    # P2 (RPT-27): Decision Evidence Audit — authoritative structured PM only.
    # Legacy prose is never used to derive audit direction.
    authoritative = _authoritative_portfolio_decision(state)
    if authoritative is not None:
        report.decision_evidence_audit = build_decision_evidence_audit(
            report, _rating_direction(authoritative.rating.value)
        )
    else:
        report.decision_evidence_audit = None

    # P2 (RPT-28): Data reliability summary — MUST be built last, after
    # data-quality finalization and evidence validation notes.
    report.data_reliability = build_data_reliability_summary(report.data_quality)

    return report


def _attach_structured_diagnostics_notes(report: ReportSchema, state: Any) -> None:
    """Append Phase A structured-output diagnostics to data-quality notes.

    Shows per-agent structured status (route, success/failure stage) without
    exposing full exception stacks or secrets.
    """
    from finmindagent.llm_clients.capabilities import (
        completeness_is_ok,
        completeness_allows_empty,
        completeness_allows_not_applicable,
    )

    diag = (getattr(state, "metadata", {}) or {}).get("structured_diagnostics")
    completeness = (getattr(state, "metadata", {}) or {}).get("report_completeness", {})
    report_status = (getattr(state, "metadata", {}) or {}).get("report_status", "unknown")
    report_publishable = (getattr(state, "metadata", {}) or {}).get("report_publishable", False)

    # Overall completeness with 5-state semantics (P0-3, P0-7)
    if completeness:
        missing = [k for k, v in completeness.items() if v == "missing"]
        invalid = [k for k, v in completeness.items() if v == "invalid"]
        empty_valid = [k for k, v in completeness.items() if v == "empty_valid"]
        not_applicable = [k for k, v in completeness.items() if v == "not_applicable"]

        parts = [f"Report: {report_status}" + (", publishable" if report_publishable else ", not publishable")]
        if missing:
            parts.append(f"Missing: {', '.join(missing)}")
        if invalid:
            parts.append(f"Invalid: {', '.join(invalid)}")
        if empty_valid:
            parts.append(f"Valid but empty: {', '.join(empty_valid)}")
        if not_applicable:
            parts.append(f"Not applicable (unselected): {', '.join(not_applicable)}")
        report.data_quality.notes.append("Report completeness: " + "; ".join(parts) + ".")

    # Per-agent structured status (non-success entries only)
    if isinstance(diag, dict):
        for agent_name, entry in sorted(diag.items()):
            if not isinstance(entry, dict):
                continue
            if entry.get("success"):
                continue
            schema = entry.get("schema", "?")
            route = entry.get("route", "?")
            stage = entry.get("failure_stage", "unknown")
            msg = entry.get("failure_message", "")[:120]
            report.data_quality.notes.append(
                f"Structured Output: {agent_name} (schema={schema}, route={route}) "
                f"— {stage}" + (f": {msg}" if msg else "")
            )


def _normalize_collection(
    value: Any,
    model: type[BaseModel],
    source: str,
    data_quality: DataQuality,
) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        _record_unparsed(data_quality, source)
        return []
    normalized = []
    for item in value:
        try:
            normalized.append(model.model_validate(item))
        except ValidationError:
            _record_unparsed(data_quality, source)
    return normalized


def _record_unparsed(data_quality: DataQuality, source: str) -> None:
    if source not in data_quality.unparsed_sources:
        data_quality.unparsed_sources.append(source)


_CORPORATE_SUFFIXES = frozenset(
    {
        "corporation", "corp", "inc", "incorporated", "co", "ltd",
        "limited", "plc", "group", "holdings", "holding",
    }
)


_STANCE_BY_RESEARCHER = {
    "bull_researcher": "bull",
    "bear_researcher": "bear",
}


# Fixed display order and names for the four analysis scores; names,
# maximums, and sources are runtime facts, never returned by the model.
REPORT_SCORE_NAMES = {
    "market_analyst": "Technical",
    "fundamentals_analyst": "Fundamentals",
    "news_analyst": "News",
    "social_sentiment_analyst": "Sentiment",
}


def _build_scores(state: Any, metadata_scores: list[NamedScore]) -> list[NamedScore]:
    """Analysis scores come from structured_reports only. Missing agents are
    simply absent — never padded with a neutral 5.0."""
    structured = getattr(state, "structured_reports", {}) or {}
    scores: list[NamedScore] = []
    for agent, name in REPORT_SCORE_NAMES.items():
        output = structured.get(agent)
        score = output.get("score") if isinstance(output, dict) else None
        if score is None:
            continue
        scores.append(
            NamedScore(name=name, value=float(score), maximum=10, source=agent)
        )
    if not scores:
        # Legacy metadata path (hand-assembled test fixtures only).
        return metadata_scores
    return scores


_RISK_ROLE_LABELS = {
    "aggressive_risk_analyst": "Aggressive",
    "neutral_risk_analyst": "Neutral",
    "conservative_risk_analyst": "Conservative",
}


def _build_risk_views(state: Any) -> tuple[list[Any], list[RiskItem]]:
    """Build risk views (one per analyst with a structured risk_score) and
    merge all structured risks into the register."""
    structured = getattr(state, "structured_reports", {}) or {}
    views = []
    entries: list[dict[str, Any]] = []
    for role, label in _RISK_ROLE_LABELS.items():
        output = structured.get(role)
        if not isinstance(output, dict) or output.get("risk_score") is None:
            continue
        views.append(
            RiskView(
                role=role,
                label=label,
                risk_score=float(output["risk_score"]),
                summary=str(output.get("summary") or ""),
            )
        )
        for raw_risk in output.get("risks") or []:
            if isinstance(raw_risk, dict):
                entries.append({**raw_risk, "role": role})
    return views, merge_risk_items(entries)


def _build_debate_arguments(state: Any) -> list[DebateArgument]:
    """Assemble structured bull/bear arguments from the runtime state."""
    structured = getattr(state, "structured_reports", {}) or {}
    arguments: list[DebateArgument] = []
    for agent_name, stance in _STANCE_BY_RESEARCHER.items():
        output = structured.get(agent_name)
        if not isinstance(output, dict):
            continue
        for raw in output.get("arguments") or []:
            try:
                item = DebateArgument(
                    stance=stance,
                    title=str(raw["title"]),
                    body=str(raw["argument"]),
                    strength=float(raw["strength"]),
                    evidence=[str(point) for point in (raw.get("evidence") or [])],
                    source=agent_name,
                )
            except (KeyError, TypeError, ValueError):
                continue
            arguments.append(item)
    return arguments


def _enrich_timeline_from_news_assessments(report: Any, state: Any) -> None:
    """Apply News Analyst event assessments to matching timeline items.

    Matching is exact on the normalized title; anything unmatched keeps the
    honest ``unknown`` values. Never guesses sentiment from keywords.
    """
    news = getattr(state, "structured_reports", {}).get("news_analyst") or {}
    events = news.get("events") or []
    if not events:
        return
    by_title = {normalize_event_title(event["title"]): event for event in events}
    for item in report.timeline:
        assessment = by_title.get(normalize_event_title(item.event))
        if assessment is None:
            continue
        item.impact = str(assessment.get("impact") or "unknown")
        item.importance = str(assessment.get("importance") or "unknown")
        item.impact_reason = str(assessment.get("reason") or "")


def _core_company_names(name: str) -> list[str]:
    """Full company name plus the name with corporate suffixes stripped."""
    words = name.lower().split()
    core = list(words)
    while core and core[-1].strip(".,") in _CORPORATE_SUFFIXES:
        core.pop()
    names = [" ".join(words)]
    if core and core != words:
        names.append(" ".join(core))
    return names


def _extend_tool_collection(
    destination: list[Any],
    data: Any,
    collection_key: str,
    model: type[BaseModel],
    source: str,
    data_quality: DataQuality,
) -> None:
    if not isinstance(data, dict) or not isinstance(data.get(collection_key), list):
        _record_unparsed(data_quality, source)
        return
    before = len(destination)
    destination.extend(
        _normalize_collection(data[collection_key], model, source, data_quality)
    )
    if data[collection_key] and len(destination) == before:
        _record_unparsed(data_quality, source)


def _iter_tool_results(state: Any):
    for event in getattr(state, "events", []) or []:
        observation = getattr(event, "observation", None)
        if not isinstance(observation, dict):
            continue
        if observation.get("ok") is True and observation.get("tool_name"):
            yield observation


def _parse_market_csv(data: Any) -> list[MarketPoint]:
    if not isinstance(data, str):
        return []
    lines = [line for line in data.splitlines() if line.strip()]
    header_index = None
    for index, line in enumerate(lines):
        columns = [column.strip().lower() for column in next(csv.reader([line]))]
        if "date" in columns and "close" in columns:
            header_index = index
            break
    if header_index is None:
        return []

    points = []
    for row in csv.DictReader(StringIO("\n".join(lines[header_index:]))):
        normalized = {
            str(key).strip().lower(): value
            for key, value in row.items()
            if key is not None
        }
        date = str(normalized.get("date") or "").strip()
        close = _finite_float(normalized.get("close"))
        if not date or close is None:
            continue
        points.append(
            MarketPoint(
                date=date,
                open=_finite_float(normalized.get("open")),
                high=_finite_float(normalized.get("high")),
                low=_finite_float(normalized.get("low")),
                close=close,
                volume=_finite_float(normalized.get("volume")),
            )
        )
    return sorted(points, key=lambda point: point.date)[-60:]


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _authoritative_portfolio_decision(state: Any) -> "PortfolioDecision | None":
    """Return the validated structured PortfolioDecision, or None.

    The Decision Evidence Audit authority gate: only a structured
    ``state.portfolio_decision`` validated against ``PortfolioDecision``
    counts. Legacy strict markdown parsing (P0 display contract) is never
    treated as an authoritative audit direction.
    """
    raw = getattr(state, "portfolio_decision", None)
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        return PortfolioDecision.model_validate(raw)
    except ValidationError:
        return None


_RATING_DIRECTION = {
    "Buy": "positive",
    "Overweight": "positive",
    "Hold": "neutral",
    "Underweight": "negative",
    "Sell": "negative",
}


def _rating_direction(rating: str) -> str:
    """Exact PortfolioRating → audit direction (no guessed aliases)."""
    return _RATING_DIRECTION.get(rating, "unknown")


def _build_decision(state: Any) -> DecisionSection:
    """Build the decision with a fixed priority:

    1. ``state.portfolio_decision`` (authoritative structured decision).
    2. A complete, strict legacy markdown protocol (all three stable labels
       present and a valid five-tier rating) for historical states.
    3. No structured decision — fields stay None, ``structured_available``
       is False, and the raw text is kept only for display below the card.

    The real SNDK bad case (top ``**Rating**: Hold`` followed by a prose
    ``**Rating:** **Sell / Underweight**``) fails the strict protocol because
    the three-label contract is not satisfied, so no rating is trusted.
    """
    raw_text = str(getattr(state, "final_trade_decision", "") or "")

    portfolio_decision = getattr(state, "portfolio_decision", None)
    if isinstance(portfolio_decision, dict) and portfolio_decision:
        try:
            decision = PortfolioDecision.model_validate(portfolio_decision)
        except ValidationError:
            decision = None
        if decision is not None:
            ep = decision.execution_plan
            trading_plan = (
                TradingPlanSection(
                    holding_action=ep.holding_action,
                    entry_condition=ep.entry_condition,
                    exit_condition=ep.exit_condition,
                    monitoring_events=list(ep.monitoring_events),
                )
                if any((ep.holding_action, ep.entry_condition, ep.exit_condition, ep.monitoring_events))
                else None
            )
            return DecisionSection(
                rating=decision.rating.value,
                executive_summary=decision.executive_summary,
                investment_thesis=decision.investment_thesis,
                price_target=decision.price_target,
                stop_loss=decision.stop_loss,
                position_min_pct=decision.position_min_pct,
                position_max_pct=decision.position_max_pct,
                confidence_level=(
                    decision.confidence_level.value
                    if decision.confidence_level is not None
                    else None
                ),
                risk_level=(
                    decision.risk_level.value if decision.risk_level is not None else None
                ),
                time_horizon=decision.time_horizon,
                raw_text=raw_text,
                structured_available=True,
                trading_plan=trading_plan,
            )

    parsed = _parse_decision_strict(raw_text)
    if parsed is not None:
        parsed.raw_text = raw_text
        parsed.structured_available = True
        return parsed

    return DecisionSection(raw_text=raw_text, structured_available=False)


def _parse_decision_strict(text: str) -> DecisionSection | None:
    """Strict legacy markdown decision protocol.

    A legal historical structured decision requires all of:
    - ``**Rating**: <one of the five tiers>``
    - ``**Executive Summary**: ...``
    - ``**Investment Thesis**: ...``

    Anything short of that (prose, conflicting ratings, missing labels) is
    rejected so the report never trusts a rating from arbitrary free text.
    """
    if not text:
        return None
    fields = {
        label: value.strip()
        for label, value in _DECISION_PATTERN.findall(text)
    }
    rating = fields.get("Rating") or None
    if rating not in _RATING_VALUES:
        return None
    if not fields.get("Executive Summary") or not fields.get("Investment Thesis"):
        return None
    price_target = None
    price_text = fields.get("Price Target", "")
    if _NUMBER_PATTERN.fullmatch(price_text):
        price_target = float(price_text)
    stop_loss = None
    stop_text = fields.get("Stop Loss", "")
    if _NUMBER_PATTERN.fullmatch(stop_text):
        stop_loss = float(stop_text)
    position_min = position_max = None
    position_text = fields.get("Position Size", "")
    position_match = re.fullmatch(r"([0-9.]+)%\s*-\s*([0-9.]+)%", position_text)
    if position_match:
        position_min = float(position_match.group(1))
        position_max = float(position_match.group(2))
    elif re.fullmatch(r"Up to\s+([0-9.]+)%", position_text):
        position_max = float(re.fullmatch(r"Up to\s+([0-9.]+)%", position_text).group(1))
    return DecisionSection(
        rating=rating,
        executive_summary=fields.get("Executive Summary", ""),
        investment_thesis=fields.get("Investment Thesis", ""),
        price_target=price_target,
        stop_loss=stop_loss,
        position_min_pct=position_min,
        position_max_pct=position_max,
        confidence_level=fields.get("Confidence Level") or None,
        risk_level=fields.get("Risk Level") or None,
        time_horizon=fields.get("Time Horizon") or None,
        raw_text=text,
    )


# ---------------------------------------------------------------------------
# P0: risk/reward, debate resolution, financial quality
# ---------------------------------------------------------------------------


def _calculate_risk_reward(
    current: float | None,
    target: float | None,
    stop: float | None,
    rating: str | None = None,
) -> float | None:
    """Deterministic risk/reward = reward / risk, direction-aware.

    Bullish (Buy/Overweight): long geometry target > current > stop.
    Bearish (Underweight/Sell): short geometry target < current < stop.
    Neutral/unknown rating: either legal geometry is accepted (the geometry
    itself defines the direction of the trade).

    A bearish rating with long-only geometry is rejected: computing the
    long formula would produce a positive-looking ratio that contradicts
    the rating. Any missing input, non-finite value, invalid geometry, or
    geometry contradicting the rating → None.
    """
    if current is None or target is None or stop is None:
        return None
    if not all(math.isfinite(v) for v in (current, target, stop)):
        return None
    if rating in ("Underweight", "Sell"):
        if not (target < current < stop):
            return None
        reward = current - target
        risk = stop - current
    elif rating in ("Buy", "Overweight"):
        if not (target > current > stop):
            return None
        reward = target - current
        risk = current - stop
    else:
        # Neutral / unknown rating: geometry alone defines the direction.
        if target > current > stop:
            reward = target - current
            risk = current - stop
        elif target < current < stop:
            reward = current - target
            risk = stop - current
        else:
            return None
    return reward / risk if risk > 0 else None


def _build_debate_resolution(state: Any) -> DebateResolutionSection | None:
    """Build Research Manager resolution from structured_reports.

    Only consumes structured JSON; never parses investment_plan prose.
    If only legacy prose exists, returns None.
    """
    structured = getattr(state, "structured_reports", {}) or {}
    rm_output = structured.get("research_manager")
    if not isinstance(rm_output, dict):
        return None
    resolution = rm_output.get("resolution")
    if not isinstance(resolution, dict):
        return None
    recommendation = rm_output.get("recommendation")
    return DebateResolutionSection(
        recommendation=str(recommendation) if recommendation else None,
        disagreement=str(resolution.get("disagreement") or ""),
        key_bull_points=[str(p) for p in (resolution.get("key_bull_points") or [])],
        key_bear_points=[str(p) for p in (resolution.get("key_bear_points") or [])],
        decisive_factors=[str(p) for p in (resolution.get("decisive_factors") or [])],
        final_reason=str(resolution.get("final_reason") or ""),
        source="research_manager",
    )


def _build_financial_quality(state: Any) -> FinancialQuality | None:
    """Build financial quality from fundamentals_analyst structured output.

    Never parses the natural-language analysis field for warnings.
    """
    structured = getattr(state, "structured_reports", {}) or {}
    fa_output = structured.get("fundamentals_analyst")
    if not isinstance(fa_output, dict):
        return None
    summary = str(fa_output.get("financial_quality_summary") or "")
    raw_warnings = fa_output.get("quality_warnings") or []
    if not summary and not raw_warnings:
        return None
    warnings = []
    for w in raw_warnings:
        if isinstance(w, dict) and w.get("title"):
            warnings.append(
                FinancialQualityFlag(
                    title=str(w["title"]),
                    evidence=str(w.get("evidence") or ""),
                )
            )
    return FinancialQuality(
        summary=summary,
        warnings=warnings,
        source="fundamentals_analyst",
    )


# ---------------------------------------------------------------------------
# P1: Analyst consensus target upside
# ---------------------------------------------------------------------------


def _attach_target_upside(
    targets: Any, current_price: float | None
) -> None:
    """Compute mean/median upside from current market price.

    ``(target - current_price) / current_price * 100``.
    Only when current_price > 0 and target exists.
    """
    if current_price is None or current_price <= 0:
        return
    if targets.mean is not None:
        targets.mean_upside_pct = (targets.mean - current_price) / current_price * 100
    if targets.median is not None:
        targets.median_upside_pct = (targets.median - current_price) / current_price * 100


# ---------------------------------------------------------------------------
# P1: Catalyst grounding
# ---------------------------------------------------------------------------


_IMPORTANCE_ORDER = {"high": 0, "medium": 1, "low": 2}


def _build_catalysts(state: Any, report: Any) -> list[Any]:
    """Ground News Analyst catalyst candidates against calendar + timeline evidence.

    Only accepted if at least one evidence_ref matches a known reference.
    Ungrounded candidates are dropped with a data-quality note.
    """
    from finmindagent.reporting.schemas import CatalystItem

    structured = getattr(state, "structured_reports", {}) or {}
    na = structured.get("news_analyst")
    if not isinstance(na, dict):
        return []
    candidates = na.get("catalysts") or []
    if not candidates:
        return []

    # Build allowed reference sets
    calendar_ids: set[str] = set()
    calendar_date: dict[str, str] = {}
    for ce in report.calendar_events:
        if ce.event_id:
            calendar_ids.add(ce.event_id)
            if ce.date:
                calendar_date[ce.event_id] = ce.date

    timeline_titles: set[str] = set()
    for ti in report.timeline:
        title = normalize_event_title(ti.event)
        if title:
            timeline_titles.add(title)

    accepted: list[dict] = []
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        refs = raw.get("evidence_refs") or []
        matched_calendar = None
        matched_timeline = False
        for ref in refs:
            ref_str = str(ref)
            if ref_str in calendar_ids:
                matched_calendar = ref_str
            if ref_str in timeline_titles or normalize_event_title(ref_str) in timeline_titles:
                matched_timeline = True

        if not matched_calendar and not matched_timeline:
            title = raw.get("title", "unknown")
            if "Ungrounded catalyst" not in " ".join(report.data_quality.notes):
                report.data_quality.notes.append(
                    f"Ungrounded catalyst dropped: {title}"
                )
            continue

        expected_date = ""
        status = "watching"
        if matched_calendar:
            expected_date = calendar_date.get(matched_calendar, "")
            trade_date = str(getattr(state, "trade_date", ""))
            # P1: scheduled only when strictly future (> trade_date);
            # same-day is watching, not scheduled.
            if expected_date and expected_date > trade_date:
                status = "scheduled"

        accepted.append({
            "title": str(raw.get("title", "")),
            "catalyst_type": str(raw.get("catalyst_type", "other")),
            "direction": str(raw.get("direction", "mixed")),
            "importance": str(raw.get("importance", "low")),
            "trigger_condition": str(raw.get("trigger_condition", "")),
            "invalidation_condition": str(raw.get("invalidation_condition", "")),
            "follow_up": str(raw.get("follow_up", "")),
            "evidence_refs": [str(r) for r in refs],
            "expected_date": expected_date,
            "status": status,
            "source": "news_analyst",
        })

    # Dedup by normalized title: higher importance replaces lower;
    # equal importance → first wins (stable deterministic).
    deduped: dict[str, dict] = {}
    for item in accepted:
        key = normalize_event_title(item["title"])
        current = deduped.get(key)
        if current is None:
            deduped[key] = item
        else:
            new_rank = _IMPORTANCE_ORDER.get(item.get("importance", "low"), 3)
            cur_rank = _IMPORTANCE_ORDER.get(current.get("importance", "low"), 3)
            if new_rank < cur_rank:  # lower rank number = higher importance
                deduped[key] = item
            # equal → first wins (don't replace)
    deduped_list = list(deduped.values())

    # Sort: scheduled by date asc, then watching by importance
    scheduled = sorted(
        [i for i in deduped_list if i["status"] == "scheduled"],
        key=lambda i: i["expected_date"],
    )
    watching = sorted(
        [i for i in deduped_list if i["status"] != "scheduled"],
        key=lambda i: _IMPORTANCE_ORDER.get(i["importance"], 3),
    )
    ordered = (scheduled + watching)[:8]

    result = []
    for item in ordered:
        try:
            result.append(CatalystItem.model_validate(item))
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# P1: Risk dimension merge (five dimensions → RiskView)
# ---------------------------------------------------------------------------


def _attach_risk_dimensions(
    views: list[Any], state: Any
) -> None:
    """Attach RiskDimensions to RiskViews from structured_reports.

    Uses strict Pydantic validation — if any axis is missing or has an
    invalid score, dimensions stays None. Never defaults missing axes to 0.
    """
    from finmindagent.reporting.schemas import RiskDimensions

    struct = getattr(state, "structured_reports", {}) or {}
    for view in views:
        output = struct.get(view.role)
        if not isinstance(output, dict):
            continue
        dims_raw = output.get("dimensions")
        if not isinstance(dims_raw, dict):
            continue
        try:
            view.dimensions = RiskDimensions.model_validate(dims_raw)
        except Exception:
            view.dimensions = None


# ---------------------------------------------------------------------------
# P2: ValuationContext
# ---------------------------------------------------------------------------


def _build_valuation_context(report: Any, raw_metrics: list) -> Any | None:
    from finmindagent.reporting.schemas import ValuationContext

    vc = ValuationContext()

    # Map from raw financial metrics (before whitelist drops PEG)
    for m in raw_metrics:
        name = getattr(m, "name", "")
        val = getattr(m, "value", None)
        if name == "pe_ratio_ttm" and val is not None:
            vc.trailing_pe = float(val)
        elif name == "forward_pe" and val is not None:
            vc.forward_pe = float(val)
        elif name == "price_to_book" and val is not None:
            vc.price_to_book = float(val)
        elif name == "peg_ratio" and val is not None:
            vc.peg_ratio = float(val)

    # Analyst context (NOT PM target)
    ac = getattr(report, "analyst_consensus", None)
    if ac and ac.price_targets:
        vc.analyst_target_mean = ac.price_targets.mean
        vc.analyst_target_median = ac.price_targets.median
        # P2 (RPT-23): low/high propagation + deterministic dispersion
        vc.analyst_target_low = ac.price_targets.low
        vc.analyst_target_high = ac.price_targets.high
        low, high, mean = vc.analyst_target_low, vc.analyst_target_high, vc.analyst_target_mean
        if low is not None and high is not None and mean is not None and mean != 0:
            vc.analyst_target_dispersion_pct = (high - low) / abs(mean) * 100
        else:
            vc.analyst_target_dispersion_pct = None
        vc.mean_target_upside_pct = ac.price_targets.mean_upside_pct
        vc.median_target_upside_pct = ac.price_targets.median_upside_pct
    if ac:
        for est in ac.earnings_estimates:
            if est.period == "current_year" and est.growth_pct is not None:
                vc.current_year_eps_growth_pct = est.growth_pct
            if est.period == "next_year" and est.growth_pct is not None:
                vc.next_year_eps_growth_pct = est.growth_pct

    # PE compression — only when both trailing and forward PE > 0
    if vc.trailing_pe is not None and vc.trailing_pe > 0 and vc.forward_pe is not None and vc.forward_pe > 0:
        vc.pe_compression_pct = (vc.trailing_pe - vc.forward_pe) / abs(vc.trailing_pe) * 100

    if any(v is not None for v in (vc.trailing_pe, vc.forward_pe, vc.analyst_target_mean)):
        srcs = set()
        for m in raw_metrics:
            s = getattr(m, "source", "")
            if s and getattr(m, "name", "") in ("pe_ratio_ttm", "forward_pe", "price_to_book", "peg_ratio"):
                srcs.add(s)
        if report.analyst_consensus:
            srcs.add(report.analyst_consensus.source)
        vc.sources = sorted(srcs)
        return vc
    return None


# ---------------------------------------------------------------------------
# P2: Provenance / Conflict / Stale
# ---------------------------------------------------------------------------


def _build_provenance(report: Any, state: Any) -> None:
    from finmindagent.reporting.schemas import DataSourceRecord

    for event in getattr(state, "events", []) or []:
        obs = getattr(event, "observation", None)
        if not isinstance(obs, dict) or not obs.get("tool_name"):
            continue
        tn = str(obs.get("tool_name") or "")
        sd = obs.get("structured_data") or {}
        data_as_of = ""
        if tn == "get_stock_data" and isinstance(sd.get("market_series"), list) and sd["market_series"]:
            data_as_of = max(p.get("date", "") for p in sd["market_series"] if isinstance(p, dict))
        elif tn == "get_analyst_expectations" and isinstance(sd.get("analyst_consensus"), dict):
            data_as_of = str(sd["analyst_consensus"].get("as_of", ""))
        elif tn == "get_earnings_history" and isinstance(sd.get("earnings_context"), dict):
            data_as_of = str(sd["earnings_context"].get("as_of", ""))
        elif tn == "get_holder_context" and isinstance(sd.get("ownership_snapshot"), dict):
            data_as_of = str(sd["ownership_snapshot"].get("as_of", ""))
        report.data_quality.sources.append(DataSourceRecord(
            tool_name=tn, ok=bool(obs.get("ok", False)),
            retrieved_at=str(obs.get("timestamp", "")),
            data_as_of=data_as_of,
            structured_keys=sorted(sd.keys()) if isinstance(sd, dict) else [],
            truncated=bool(obs.get("truncated", False)),
            error=str(obs.get("error", "") or ""),
        ))


def _build_conflicts(report: Any, raw_metrics: list) -> None:
    from finmindagent.reporting.schemas import DataConflict

    groups: dict[tuple, list] = {}
    for m in raw_metrics:
        key = (getattr(m, "name", ""), getattr(m, "period", ""), getattr(m, "unit", ""))
        groups.setdefault(key, []).append(m)
    for (name, period, unit), items in groups.items():
        if len(items) < 2:
            continue
        by_source: dict[str, float] = {}
        for m in items:
            src = getattr(m, "source", "")
            val = getattr(m, "value", None)
            if src and val is not None:
                by_source[src] = float(val)
        if len(by_source) < 2:
            continue
        vals = list(by_source.values())
        if all(abs(v - vals[0]) < 1e-6 for v in vals):
            continue
        # Find selected source via curation priority
        from finmindagent.reporting.curation import curate_financial_metrics

        curated = curate_financial_metrics(list(items))
        selected_src = getattr(curated[0], "source", "") if curated else ""
        selected_val = getattr(curated[0], "value", None) if curated else None
        report.data_quality.conflicts.append(DataConflict(
            field=name, period=period, unit=unit,
            source_values=by_source,
            selected_source=selected_src,
            selected_value=selected_val,
        ))


def _build_stale_note(report: Any, state: Any) -> None:
    trade_date_str = str(getattr(state, "trade_date", ""))
    if not trade_date_str or not report.market_series:
        return
    try:
        from datetime import date, timedelta
        trade_d = date.fromisoformat(trade_date_str)
        latest_date_str = max(p.date for p in report.market_series)
        latest_d = date.fromisoformat(latest_date_str)
        gap = (trade_d - latest_d).days
        if gap > 7:
            report.data_quality.notes.append(
                f"Market data may be stale: latest market date {latest_date_str} "
                f"is {gap} days before trade date."
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# P2: Sentiment signals grounding
# ---------------------------------------------------------------------------


def _build_sentiment_signals(report: Any, state: Any) -> list[Any]:
    from finmindagent.reporting.schemas import SentimentSignalItem

    struct = getattr(state, "structured_reports", {}) or {}
    sa = struct.get("social_sentiment_analyst")
    if not isinstance(sa, dict):
        return []
    signals_raw = sa.get("signals") or []
    if not signals_raw:
        return []

    # Build allowed ref sets
    holder_ids: set[str] = set()
    if report.ownership_snapshot:
        for h in (report.ownership_snapshot.institutional_holders or []):
            holder_ids.add(h.holder_id)
        for h in (report.ownership_snapshot.mutual_fund_holders or []):
            holder_ids.add(h.holder_id)
    timeline_titles: set[str] = set()
    for ti in (report.timeline or []):
        title = normalize_event_title(ti.event)
        if title:
            timeline_titles.add(title)

    accepted: list[dict] = []
    for raw in signals_raw:
        if not isinstance(raw, dict):
            continue
        refs = raw.get("evidence_refs") or []
        # RF-2: keep ONLY exact grounded refs — an invalid ref must never leak
        # merely because another ref happened to be valid.
        grounded_refs: list[str] = []
        for ref in refs:
            ref_str = str(ref)
            if (ref_str in holder_ids
                    or ref_str in timeline_titles
                    or normalize_event_title(ref_str) in timeline_titles):
                grounded_refs.append(ref_str)
        if not grounded_refs:
            report.data_quality.notes.append(
                f"Ungrounded sentiment signal dropped: {raw.get('title', 'unknown')}"
            )
            continue
        accepted.append({
            "title": str(raw.get("title", "")),
            "signal_type": str(raw.get("signal_type", "other")),
            "direction": str(raw.get("direction", "neutral")),
            "strength": float(raw.get("strength", 5)),
            "reasoning": str(raw.get("reasoning", "")),
            "evidence_refs": grounded_refs,
        })

    # Dedup + sort
    deduped: dict[str, dict] = {}
    for item in accepted:
        key = normalize_event_title(item["title"])
        cur = deduped.get(key)
        if cur is None or item["strength"] > cur["strength"]:
            deduped[key] = item
    result = []
    for item in sorted(deduped.values(), key=lambda i: -i["strength"])[:8]:
        try:
            result.append(SentimentSignalItem.model_validate(item))
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# P2: Conditional P2 missing/data-quality semantics (6R.8)
# ---------------------------------------------------------------------------


def _tool_was_attempted(state: Any, tool_name: str) -> bool:
    for event in getattr(state, "events", []) or []:
        obs = getattr(event, "observation", None)
        if isinstance(obs, dict) and obs.get("tool_name") == tool_name:
            return True
    return False


def _apply_p2_missing_semantics(report: Any, state: Any) -> None:
    dq = report.data_quality

    # earnings_context: only missing if tool attempted but no valid context
    if "earnings_context" not in dq.missing_sections:
        pass  # not missing by basic check
    elif not _tool_was_attempted(state, "get_earnings_history"):
        dq.missing_sections.remove("earnings_context")

    # ownership_snapshot: only missing if tool attempted but no valid snapshot
    if "ownership_snapshot" not in dq.missing_sections:
        pass
    # Not adding ownership_snapshot to basic missing list since it's optional

    # market_risk_snapshot: missing when market exists but no risk stats
    if report.market_series and report.market_risk_snapshot is None:
        if "market_risk_snapshot" not in dq.missing_sections:
            dq.missing_sections.append("market_risk_snapshot")

    # valuation_context: missing only if neither multiple nor analyst target
    if report.valuation_context is None:
        if "valuation_context" not in dq.missing_sections:
            dq.missing_sections.append("valuation_context")

    # sentiment_signals: only if structured social output exists but no groundedsignal
    struct = getattr(state, "structured_reports", {}) or {}
    sa = struct.get("social_sentiment_analyst")
    if isinstance(sa, dict) and sa.get("signals"):
        if not report.sentiment_signals:
            if "sentiment_signals" not in dq.missing_sections:
                dq.missing_sections.append("sentiment_signals")
            dq.notes.append("Structured sentiment signals unavailable: all signals were ungrounded.")

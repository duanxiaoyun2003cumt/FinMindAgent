"""Safe single-file HTML rendering for investment reports.

Renders the P2 stable DOM contract: a fixed business section order with
stable section IDs, semantic rating/badge classes, localized static labels,
safe markdown for agent prose, and collapsible detailed analysis.

``ReportSchema`` is the only business input; this module never re-interprets
agent prose or derives business fields.
"""

from __future__ import annotations

import json
import re
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import Any

from finmindagent.dataflows.utils import safe_ticker_component
from finmindagent.reporting.builder import build_report_schema
from finmindagent.reporting.curation import normalize_event_title
from finmindagent.reporting.formatting import (
    format_metric_value,
    format_optional,
    format_position,
    format_price,
)
from finmindagent.reporting.localization import DEFAULT_LOCALE, SUPPORTED_LOCALES, text
from finmindagent.reporting.markdown import render_safe_markdown
from finmindagent.reporting.schemas import ReportSchema
from finmindagent.time_utils import format_system_timestamp


# ---------------------------------------------------------------------------
# Semantic mapping (fixed, never business values as class names)
# ---------------------------------------------------------------------------

RATING_CLASS = {
    "Buy": "rating-buy",
    "Overweight": "rating-overweight",
    "Hold": "rating-hold",
    "Underweight": "rating-underweight",
    "Sell": "rating-sell",
}

SEVERITY_WIDTH = {"low": 25, "medium": 50, "high": 75, "critical": 100}

# Fixed detailed-analysis order (not dictionary insertion order).
_ANALYSIS_KEYS = (
    ("market_report", "technical_analysis"),
    ("fundamentals_report", "fundamentals_analysis"),
    ("news_report", "news_analysis"),
    ("sentiment_report", "sentiment_analysis"),
)

MISSING_SECTION_LABELS = {
    "market_series": "market_series",
    "financial_metrics": "financial_metrics",
    "scores": "composite_score",
    "timeline": "event_timeline",
    "risk_items": "risk_register",
    "risk_views": "risk_assessment",
    "debate_arguments": "debate",
    "financial_series": "financial_history",
}


# ---------------------------------------------------------------------------
# Styles (token-based, responsive)
# ---------------------------------------------------------------------------

_STYLES = """
:root{color-scheme:light;--surface:#fff;--surface-muted:#f8fafc;--ink:#172033;--ink-2:#64748b;--border:#dbe3ef;--accent:#2563eb;--positive:#16a34a;--negative:#dc2626;--neutral:#64748b;--warning:#b45309;--bg:#f4f7fb;--surface-inverse:#0f172a;--ink-inverse:#f8fafc;--ink-inverse-muted:#cbd5e1;--positive-soft:#dcfce7;--negative-soft:#fee2e2;--warning-soft:#fef3c7;--accent-soft:#dbeafe;--mixed:#7c3aed}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
main{width:min(1120px,calc(100% - 32px));margin:32px auto}
section.panel{background:var(--surface);border:1px solid var(--border);border-radius:16px;box-shadow:0 8px 30px rgba(30,41,59,.06);padding:22px;margin:16px 0;overflow-x:auto}
section.panel h2{margin:0 0 14px;font-size:20px}
.muted{color:var(--ink-2)}.unavailable{border:1px dashed #94a3b8;border-radius:12px;padding:18px;text-align:center;color:var(--ink-2)}
/* panel modifiers: always compound `.panel.XXX` so base `section.panel`
   (0,1,1) can never silently override a single-class modifier (0,1,0) */
.panel.panel-muted{background:var(--surface-muted)}
.panel.disclaimer{font-size:12px;color:var(--ink-2);background:var(--surface-muted)}
.grid{display:grid;gap:16px}.cols-2{grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.card{padding:16px;border:1px solid var(--border);border-radius:12px;background:var(--surface-muted)}.card h3{margin:0 0 8px;font-size:16px}
table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top;overflow-wrap:break-word}th{color:var(--ink-2);font-size:12px;text-transform:uppercase}
details{border:1px solid var(--border);border-radius:12px;margin:10px 0;background:var(--surface-muted);overflow:hidden}details summary{cursor:pointer;padding:14px 16px;font-weight:700;list-style:none}details summary::before{content:"▸ ";color:var(--accent)}details[open] summary::before{content:"▾ "}details .detail-body{padding:4px 16px 16px}
details summary:hover{background:rgba(37,99,235,.06);color:var(--accent)}
details summary:focus-visible{outline:2px solid var(--accent);outline-offset:-2px;border-radius:12px}
details[open]>summary{border-bottom:1px solid var(--border)}
/* header */
header.hero{padding:28px;border-radius:16px;color:#fff;background:linear-gradient(135deg,#172554,#2563eb);margin:0 0 16px}.hero h1{margin:0 0 6px;font-size:28px}.hero .meta{opacity:.85}
/* decision hero: dark surface, explicit inverse text; all conflicting props
   live on the compound `.panel.decision-hero` so the base panel cannot win */
.panel.decision-hero{background:linear-gradient(135deg,var(--surface-inverse),#1e293b);color:var(--ink-inverse);border:1px solid #1e293b;margin:0 0 16px;padding:28px}
.panel.decision-hero h2{margin:0 0 12px;font-size:18px;opacity:.9;font-weight:600;color:var(--ink-inverse)}
.rating-pill{display:inline-block;padding:10px 22px;border-radius:999px;font-size:22px;font-weight:800;letter-spacing:.5px;color:#fff}
.rating-buy,.rating-overweight{background:var(--positive)}.rating-hold{background:var(--warning)}.rating-underweight,.rating-sell{background:var(--negative)}.rating-unavailable{background:var(--neutral)}
.hero-badges{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 18px}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.metric-card{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);border-radius:12px;padding:12px 14px}.metric-card .label{font-size:12px;opacity:.75}.metric-card .value{font-size:17px;font-weight:700;margin-top:2px}.metric-card .sub{font-size:11px;opacity:.6}
/* trading plan: a light action card inside the dark hero — never let it
   inherit the hero's inverse colors */
.panel.decision-hero .trading-plan{background:var(--surface);color:var(--ink);border:1px solid var(--border);margin:18px 0 0}
.panel.decision-hero .trading-plan summary{color:var(--ink)}
.trading-plan .tp-row{display:grid;grid-template-columns:190px 1fr;gap:6px 16px;padding:10px 2px;border-top:1px solid var(--border)}
.trading-plan .tp-row:first-of-type{border-top:none}
.trading-plan .tp-label{color:var(--ink-2);font-size:13px;font-weight:600;padding-top:1px}
.trading-plan .tp-value{color:var(--ink);font-size:14px}
.trading-plan .tp-events{list-style:disc;margin:2px 0 0;padding-left:20px;line-height:1.7}
/* badges */
.badge{display:inline-block;padding:3px 10px;border-radius:999px;color:#fff;font-size:11px;font-weight:700;text-transform:capitalize;margin:1px}
.badge-positive{background:var(--positive)}.badge-negative{background:var(--negative)}.badge-neutral{background:var(--neutral)}.badge-unknown{background:#94a3b8}.badge-mixed{background:var(--mixed)}
.badge-insider{background:#4f46e5}
.badge.importance-high{background:var(--warning-soft);color:#92400e}
.badge.importance-medium{background:var(--accent-soft);color:#1e40af}
.badge.importance-low{background:#e2e8f0;color:#334155}
.severity-low{color:#15803d}.severity-medium{color:#b45309}.severity-high{color:#dc2626}.severity-critical{color:#7f1d1d;font-weight:800}
.importance-low{color:var(--ink-2)}.importance-medium{color:#1d4ed8}.importance-high{color:#b45309;font-weight:700}
/* strength + severity bars */
.strength-bar,.severity-bar{height:8px;border-radius:999px;background:#e2e8f0;margin:6px 0 2px;overflow:hidden}.strength-fill{height:100%;background:var(--accent)}.severity-fill{height:100%;background:var(--negative)}
/* score + financial grid */
.score-financial{display:grid;grid-template-columns:minmax(280px,420px) 1fr;gap:16px;align-items:start}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}.kpi{padding:12px;border:1px solid var(--border);border-radius:10px;background:var(--surface-muted)}.kpi .label{font-size:12px;color:var(--ink-2)}.kpi .value{font-size:16px;font-weight:700;margin-top:2px}.kpi .period{font-size:11px;color:var(--ink-2)}
/* charts: every responsive canvas lives in a container with an explicit height,
   so Chart.js resizing never feeds back into parent layout */
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.chart-container{position:relative;width:100%;height:320px;min-height:320px}
.chart-container canvas{display:block;width:100%!important;height:100%!important}
/* debate */
.debate-cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}.argument{margin-bottom:14px}.argument-evidence{font-size:13px;color:var(--ink-2);margin:4px 0;padding-left:10px;border-left:2px solid var(--border)}
/* timeline */
.timeline-list{list-style:none;margin:0;padding:0}.timeline-item{padding:12px 0;border-bottom:1px solid var(--border);display:grid;grid-template-columns:110px 1fr;gap:8px 14px}.timeline-item:last-child{border-bottom:none}.timeline-date{color:var(--ink-2);font-size:13px}
/* technical snapshot */
.technical-snapshot .ts-grid{display:grid;gap:8px}
.technical-snapshot .ts-table{width:100%;max-width:560px}
.ts-source{font-size:11px;color:var(--ink-2);margin-top:10px;text-align:right}
.empty-state{color:var(--ink-2);font-size:14px;margin:8px 0}
/* debate resolution */
.debate-resolution .dr-field{margin:12px 0}
.debate-resolution .dr-field strong{display:block;margin-bottom:4px}
.debate-resolution .dr-field ul{margin:4px 0 0;padding-left:22px}
.debate-resolution .dr-field p{margin:4px 0 0}
/* analyst consensus */
.analyst-consensus .ac-table{width:100%}
.ac-source{font-size:11px;color:var(--ink-2);margin-top:10px;text-align:right}
/* catalysts (legacy standalone renderer kept styled) */
.catalyst-item{padding:12px 0;border-bottom:1px solid var(--border)}
.catalyst-item:last-child{border-bottom:none}
.cat-header{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px}
.cat-date{color:var(--ink-2);font-size:13px}
.cat-status{font-size:12px;font-weight:700;color:#1d4ed8}
.cat-status.cat-scheduled{color:#15803d}
.cat-direction,.cat-importance{font-size:12px;color:var(--ink-2)}
/* financial quality */
.fq-summary{margin:0 0 12px;color:var(--ink)}
.fq-warning{background:var(--warning-soft);border-left:3px solid #f59e0b;border-radius:8px;padding:10px 14px;margin:8px 0}
.fq-warning strong{display:block;color:#92400e;margin-bottom:2px}
/* decision evidence audit */
.audit-bucket{margin:10px 0}
.audit-bucket h4{margin:0 0 6px;font-size:14px}
.audit-bucket ul{margin:0;padding-left:20px}
.audit-bucket li{margin:4px 0;font-size:13px}
.audit-bucket code{font-size:12px;background:#eef2f7;border-radius:4px;padding:1px 5px}
/* research diagnostics */
.diagnostic-grid{display:grid;gap:16px}
.research-raw-details{margin-top:14px}
/* supporting evidence / appendix group containers */
details.supporting-group,details.appendix-group{background:var(--surface)}
/* prose */
.prose p{margin:0 0 12px}.prose ul,.prose ol{margin:0 0 12px;padding-left:22px}.prose h2,.prose h3,.prose h4{margin:14px 0 8px}.prose blockquote{border-left:3px solid var(--border);margin:0 0 12px;padding:2px 14px;color:var(--ink-2)}.prose code{background:#eef2f7;border-radius:4px;padding:1px 6px;font-size:13px}
/* responsive */
@media(max-width:900px){.score-financial{grid-template-columns:1fr}.debate-cols{grid-template-columns:1fr}}
@media(max-width:640px){main{width:min(100% - 20px,1120px);margin:10px auto}section.panel,.decision-hero,.hero{border-radius:12px;padding:18px}.hero h1{font-size:23px}.metric-grid{grid-template-columns:repeat(2,1fr)}.timeline-item{grid-template-columns:1fr}.trading-plan .tp-row{grid-template-columns:1fr;gap:2px 0}}
"""


def render_report_html(
    report: ReportSchema,
    *,
    chart_js: str | None = None,
    locale: str = DEFAULT_LOCALE,
) -> str:
    """Render a complete document while treating all report content as text."""
    if chart_js is None:
        chart_js = (
            files("finmindagent.reporting")
            .joinpath("vendor/chart.umd.js")
            .read_text(encoding="utf-8")
        )
    chart_script = f"<script>{chart_js}</script>" if chart_js else ""

    sections, chart_targets = _render_sections(report, locale)
    chart_runtime = (
        _chart_runtime_script(chart_targets) if chart_targets else ""
    )

    return (
        "<!doctype html>\n"
        f'<html lang="{locale}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(report.metadata.ticker)} {escape(text('investment_report', locale))}</title>"
        f"<style>{_STYLES}</style>{chart_script}</head><body><main>"
        f"{sections}"
        f"</main>"
        f"{chart_runtime}"
        + "</body></html>"
    )


def _render_sections(report: ReportSchema, locale: str) -> tuple[str, dict[str, Any]]:
    """Phase C P1 DOM order — Primary / Supporting Evidence / Engineering Appendix."""
    chart_targets: dict[str, Any] = {}

    # === Primary Business Report ===
    has_event_center = bool(report.catalysts or report.timeline)
    primary = [
        _render_header(report, locale),
        _render_decision_hero(report, locale),
        _render_summary(report, locale),
        *_render_score_financial_grid(report, locale, chart_targets),
        _render_model_consensus_divergence(report, locale),
        # P2 (RPT-27): Decision Evidence Audit — after divergence, before market
        _render_decision_evidence_audit(report, locale),
        _render_market_chart(report, locale, chart_targets),
        _render_technical_snapshot(report, locale),
        _render_debate_section(report, locale),
        _render_debate_resolution(report, locale),
        _render_risk_section(report, locale),
        _render_event_center(report, locale) if has_event_center else "",
        _render_analysis_details(report, locale),
        _render_risk_register(report, locale),
    ]

    # === Supporting Evidence (collapsible, visually demoted) ===
    supporting_parts = []
    # P1: Supporting financial metrics (non-core KPIs)
    sfm = _render_supporting_financial_metrics(report, locale)
    if sfm.strip():
        supporting_parts.append(sfm)
    # P2 (RPT-29): Research Diagnostics — compact diagnostics + collapsed raw
    # details. Replaces the five separate default-visible P2 panels.
    rd = _render_research_diagnostics(report, locale)
    if rd.strip():
        supporting_parts.append(rd)
    # Financial quality
    fq = _render_financial_quality(report, locale)
    if fq.strip():
        supporting_parts.append(fq)
    # Secondary charts (risk radar, financial history)
    sc = _render_secondary_charts(report, locale, chart_targets)
    if sc.strip():
        supporting_parts.append(sc)

    supporting = ""
    if supporting_parts:
        supporting = (
            '<details class="supporting-group" open>'
            f'<summary>{escape(_text_supporting_evidence(locale))}</summary>'
            f'<div class="detail-body">{"".join(supporting_parts)}</div>'
            "</details>"
        )

    # === Engineering Appendix (collapsed by default) ===
    appendix_parts = []
    # P2 (RPT-28): Data Reliability Summary — before full data quality
    drs = _render_data_reliability_summary(report, locale)
    if drs.strip():
        appendix_parts.append(drs)
    # Data quality
    dq = _render_data_quality(report, locale)
    if dq.strip():
        appendix_parts.append(dq)
    # Risk verifier (engineering guardrail)
    verifier = _render_risk_verifier(report, locale)
    if verifier.strip():
        appendix_parts.append(verifier)
    # Disclaimer
    appendix_parts.append(_render_disclaimer(report, locale))

    appendix = ""
    if appendix_parts:
        appendix = (
            '<details class="appendix-group">'
            f'<summary>{escape(_text_engineering_appendix(locale))}</summary>'
            f'<div class="detail-body">{"".join(appendix_parts)}</div>'
            "</details>"
        )

    return "".join(primary + [supporting, appendix]), chart_targets


def _text_supporting_evidence(locale: str) -> str:
    if locale == "en":
        return "Supporting Evidence"
    return "补充证据"


def _text_engineering_appendix(locale: str) -> str:
    if locale == "en":
        return "Engineering Appendix"
    return "工程附录"


# ---------------------------------------------------------------------------
# 1. Report Header
# ---------------------------------------------------------------------------

def _render_header(report: ReportSchema, locale: str) -> str:
    metadata = report.metadata
    profile = report.company_profile
    meta_parts = [
        f"{text('trade_date', locale)}: {escape(metadata.trade_date)}",
        f"{text('status', locale)}: {escape(metadata.status)}",
    ]
    if profile and profile.name:
        meta_parts.insert(0, f"{text('company', locale)}: {escape(profile.name)}")
    if profile and profile.industry:
        meta_parts.append(f"{text('industry', locale)}: {escape(profile.industry)}")
    return (
        '<header class="hero">'
        f"<h1>{escape(metadata.ticker)} {escape(text('investment_report', locale))}</h1>"
        f'<div class="meta">{" · ".join(meta_parts)}</div></header>'
    )


# ---------------------------------------------------------------------------
# 2. Decision Hero
# ---------------------------------------------------------------------------

def _render_decision_hero(report: ReportSchema, locale: str) -> str:
    decision = report.decision
    currency = report.company_profile.currency if report.company_profile else None
    if decision.structured_available and decision.rating:
        rating_class = RATING_CLASS.get(decision.rating, "rating-unavailable")
        rating_text = decision.rating
    else:
        rating_class = "rating-unavailable"
        rating_text = text("unavailable", locale)

    badges = []
    if decision.confidence_level:
        badges.append(_badge(text("confidence", locale), decision.confidence_level, "badge-neutral"))
    if decision.risk_level:
        badges.append(_badge(text("risk_level", locale), decision.risk_level, "badge-neutral"))
    position = format_position(
        decision.position_min_pct, decision.position_max_pct, locale
    )
    if position:
        badges.append(_badge(text("position", locale), position, "badge-neutral"))
    badges_html = (
        f'<div class="hero-badges">{"".join(badges)}</div>' if badges else ""
    )

    metrics = []
    current_price = format_price(decision.current_price, currency=currency, locale=locale)
    if current_price:
        metrics.append(_metric_card(
            text("current_price", locale), current_price,
            f"{text('price_as_of', locale)} {escape(decision.price_as_of)}" if decision.price_as_of else "",
        ))
    else:
        metrics.append(_metric_card(text("current_price", locale), text("unavailable", locale), ""))
    metrics.append(_metric_card(
        text("target_price", locale),
        format_price(decision.price_target, currency=currency, locale=locale) or text("unavailable", locale),
        "",
    ))
    metrics.append(_metric_card(
        text("stop_loss", locale),
        format_price(decision.stop_loss, currency=currency, locale=locale) or text("unavailable", locale),
        "",
    ))
    metrics.append(_metric_card(
        text("time_horizon", locale),
        decision.time_horizon or text("unavailable", locale),
        "",
    ))
    # P0: risk/reward ratio
    if decision.risk_reward_ratio is not None:
        rr_text = f"1:{decision.risk_reward_ratio:.2f}"
    else:
        rr_text = text("unavailable", locale)
    metrics.append(_metric_card(text("risk_reward", locale), rr_text, ""))

    # P0: trading plan
    trading_plan_html = ""
    tp = decision.trading_plan
    if tp and any((tp.holding_action, tp.entry_condition, tp.exit_condition, tp.monitoring_events)):
        plan_parts = []
        if tp.holding_action:
            plan_parts.append(
                f'<div class="tp-row"><span class="tp-label">{escape(text("holding_action", locale))}</span>'
                f'<span class="tp-value">{escape(tp.holding_action)}</span></div>'
            )
        if tp.entry_condition:
            plan_parts.append(
                f'<div class="tp-row"><span class="tp-label">{escape(text("entry_condition", locale))}</span>'
                f'<span class="tp-value">{escape(tp.entry_condition)}</span></div>'
            )
        if tp.exit_condition:
            plan_parts.append(
                f'<div class="tp-row"><span class="tp-label">{escape(text("exit_condition", locale))}</span>'
                f'<span class="tp-value">{escape(tp.exit_condition)}</span></div>'
            )
        if tp.monitoring_events:
            events_html = "".join(
                f"<li>{escape(event)}</li>" for event in tp.monitoring_events
            )
            plan_parts.append(
                f'<div class="tp-row"><span class="tp-label">{escape(text("monitoring_events", locale))}</span>'
                f'<ul class="tp-events">{events_html}</ul></div>'
            )
        if plan_parts:
            trading_plan_html = (
                f'<details class="trading-plan" open>'
                f'<summary>{escape(text("trading_plan", locale))}</summary>'
                f'{"".join(plan_parts)}</details>'
            )

    return (
        '<section class="panel decision-hero" id="decision">'
        f'<h2>{text("rating", locale)}</h2>'
        f'<span class="rating-pill {rating_class}">{escape(rating_text)}</span>'
        f"{badges_html}"
        f'<div class="metric-grid">{"".join(metrics)}</div>'
        f"{trading_plan_html}"
        '</section>'
    )


def _metric_card(label: str, value: str, sub: str) -> str:
    sub_html = f'<div class="sub">{escape(sub)}</div>' if sub else ""
    return (
        f'<div class="metric-card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>{sub_html}</div>'
    )


def _badge(label: str, value: str, cls: str) -> str:
    return (
        f'<span class="badge {cls}" title="{escape(label)}">{escape(value)}</span>'
    )


# ---------------------------------------------------------------------------
# 3. Executive Summary / Investment Thesis
# ---------------------------------------------------------------------------

def _render_summary(report: ReportSchema, locale: str) -> str:
    decision = report.decision
    blocks = []
    if decision.executive_summary:
        blocks.append(f"<h3>{escape(text('executive_summary', locale))}</h3>"
                      f"{render_safe_markdown(decision.executive_summary)}")
    if decision.investment_thesis:
        blocks.append(f"<h3>{escape(text('investment_thesis', locale))}</h3>"
                      f"{render_safe_markdown(decision.investment_thesis)}")
    if not blocks and decision.raw_text:
        # No structured summary: show the raw decision text as background
        # material, clearly labelled, never as an authoritative rating.
        blocks.append(
            f"<h3>{escape(text('raw_portfolio_decision', locale))}</h3>"
            f"{render_safe_markdown(decision.raw_text)}"
        )
    if not blocks:
        blocks.append(
            f'<div class="unavailable">{escape(text("unavailable", locale))}</div>'
        )
    return (
        '<section class="panel" id="summary">'
        f'<h2>{escape(text("executive_summary", locale))}</h2>'
        f'<div class="prose">{"".join(blocks)}</div></section>'
    )


# ---------------------------------------------------------------------------
# 4. Composite Score + Key Financial Metrics
# ---------------------------------------------------------------------------

def _render_score_financial_grid(
    report: ReportSchema, locale: str, targets: dict[str, Any]
) -> tuple[str, str]:
    score_panel = _composite_score_panel(report, locale, targets)
    financial_panel = _financial_kpi_panel(report, locale)
    if score_panel is None and financial_panel is None:
        return (
            f'<section class="panel" id="scores-financials">'
            f'<div class="unavailable">{escape(text("unavailable", locale))}</div></section>',
            "",
        )
    return (
        f'<section class="panel" id="scores-financials">'
        f'<div class="score-financial">{score_panel or ""}{financial_panel or ""}</div>'
        f"</section>",
        "",
    )


def _composite_score_panel(
    report: ReportSchema, locale: str, targets: dict[str, Any]
) -> str | None:
    if len(report.scores) < 3:
        return None
    valid = [score for score in report.scores if score.maximum > 0]
    if not valid:
        return None
    targets["score"] = {
        "type": "radar",
        "data": {
            "labels": [score.name for score in valid],
            "datasets": [{
                "label": text("composite_score", locale),
                "data": [round(score.value / score.maximum * 10, 2) for score in valid],
                "borderColor": "#2563eb",
                "backgroundColor": "rgba(37,99,235,.18)",
            }],
        },
        "options": {
            "scales": {"r": {"min": 0, "max": 10}},
        },
    }
    return (
        f'<div><h2>{escape(text("composite_score", locale))}</h2>'
        '<div class="chart-container chart-container-radar">'
        '<canvas id="score-chart-canvas"></canvas></div></div>'
    )


# P1: Core KPI priority order (max 10 in primary grid)
_CORE_KPI_ORDER = (
    "market_cap",
    "pe_ratio_ttm",
    "forward_pe",
    "price_to_book",
    "revenue_ttm",
    "net_income",
    "profit_margin",
    "return_on_equity",
    "total_debt",
    "free_cash_flow",
)


def _financial_kpi_panel(report: ReportSchema, locale: str) -> str | None:
    if not report.financial_metrics:
        return None
    # P1: split core (max 10) from supporting
    by_name = {m.name: m for m in report.financial_metrics}
    core = [by_name[k] for k in _CORE_KPI_ORDER if k in by_name]
    cards = "".join(
        '<div class="kpi">'
        f'<div class="label">{escape(metric_display_name(metric.name, locale))}</div>'
        f'<div class="value">{format_metric_value(metric.value, metric.unit)}</div>'
        f'<div class="period">{escape(metric.period)}</div></div>'
        for metric in core[:10]
    )
    return (
        f'<div><h2>{escape(text("financial_summary", locale))}</h2>'
        f'<div class="kpi-grid">{cards}</div></div>'
    )


# P1: KPI display name → localization key
_KPI_LOCALE_KEY = {
    "market_cap": "kpi_market_cap",
    "pe_ratio_ttm": "kpi_pe_ttm",
    "forward_pe": "kpi_forward_pe",
    "price_to_book": "kpi_price_to_book",
    "revenue_ttm": "kpi_revenue_ttm",
    "net_income": "kpi_net_income",
    "profit_margin": "kpi_profit_margin",
    "return_on_equity": "kpi_return_on_equity",
    "total_debt": "kpi_total_debt",
    "free_cash_flow": "kpi_free_cash_flow",
    "current_ratio": "kpi_current_ratio",
    "operating_margin": "kpi_operating_margin",
}


def metric_display_name(name: str, locale: str = "zh-CN") -> str:
    """Localized KPI display name. Falls back to canonical English name."""
    locale_key = _KPI_LOCALE_KEY.get(name, "")
    if locale_key:
        return text(locale_key, locale)
    from finmindagent.reporting.curation import FINANCIAL_METRIC_LABELS

    return FINANCIAL_METRIC_LABELS.get(name, name)


def _render_supporting_financial_metrics(report, locale: str) -> str:
    """Render non-core KPIs in supporting evidence."""
    by_name = {m.name: m for m in report.financial_metrics}
    core_names = set(_CORE_KPI_ORDER)
    secondary = [
        (n, by_name[n]) for n in
        ["current_ratio", "operating_margin"]
        if n in by_name and n not in core_names
    ]
    # Also include any whitelisted metric not in core
    for n, m in by_name.items():
        if n not in core_names and (n, m) not in secondary:
            secondary.append((n, m))
    secondary = secondary[:12]  # reasonable bound

    if not secondary:
        return ""

    cards = "".join(
        '<div class="kpi">'
        f'<div class="label">{escape(metric_display_name(name, locale))}</div>'
        f'<div class="value">{format_metric_value(metric.value, metric.unit)}</div>'
        f'<div class="period">{escape(metric.period)}</div></div>'
        for name, metric in secondary
    )
    return (
        f'<section class="panel panel-muted">'
        f'<h3>{escape(text("supporting_financial_metrics", locale))}</h3>'
        f'<div class="kpi-grid">{cards}</div></section>'
    )


# ---------------------------------------------------------------------------
# 5. Price & Technical Chart
# ---------------------------------------------------------------------------

def _render_market_chart(
    report: ReportSchema, locale: str, targets: dict[str, Any]
) -> str:
    if not report.market_series:
        return (
            f'<section class="panel" id="market-chart">'
            f'<h2>{escape(text("price_chart", locale))}</h2>'
            f'<div class="unavailable">{escape(text("unavailable", locale))}</div></section>'
        )
    labels = [point.date for point in report.market_series]
    # P1: localized chart labels
    _close_label = "收盘" if locale != "en" else "Close"
    _volume_label = "成交量" if locale != "en" else "Volume"
    _target_label = "目标价" if locale != "en" else "Target Price"
    _stop_label = "止损价" if locale != "en" else "Stop Loss"
    datasets: list[dict[str, Any]] = [
        {
            "type": "line",
            "label": _close_label,
            "data": [point.close for point in report.market_series],
            "borderColor": "#2563eb",
            "yAxisID": "price",
        },
        {
            "type": "bar",
            "label": _volume_label,
            "data": [point.volume for point in report.market_series],
            "backgroundColor": "rgba(100,116,139,.28)",
            "yAxisID": "volume",
        },
    ]
    for window, key, color in ((20, "sma20", "#f59e0b"), (50, "sma50", "#8b5cf6")):
        values = [point.indicators.get(key) for point in report.market_series]
        if any(value is not None for value in values):
            datasets.append({
                "type": "line",
                "label": f"SMA{window}",
                "data": values,
                "borderColor": color,
                "pointRadius": 0,
                "yAxisID": "price",
            })
    if report.decision.price_target is not None:
        datasets.append(_horizontal_line(_target_label, report.decision.price_target, labels, "#16a34a"))
    if report.decision.stop_loss is not None:
        datasets.append(_horizontal_line(_stop_label, report.decision.stop_loss, labels, "#dc2626"))
    targets["market"] = {
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "plugins": {
                "legend": {"labels": {"boxWidth": 12}},
            },
            "scales": {
                "price": {"position": "left"},
                "volume": {"position": "right", "grid": {"drawOnChartArea": False}},
            },
        },
    }
    return (
        f'<section class="panel" id="market-chart">'
        f'<h2>{escape(text("price_chart", locale))}</h2>'
        '<div class="chart-container chart-container-market">'
        '<canvas id="market-price-canvas"></canvas>'
        f'<div class="muted" style="font-size:12px;margin-top:4px">{_price_basis_note(report, locale)}</div>'
        '</div></section>'
    )


def _horizontal_line(label: str, value: float, labels: list[str], color: str) -> dict[str, Any]:
    return {
        "type": "line",
        "label": label,
        "data": [value] * len(labels),
        "borderColor": color,
        "borderDash": [6, 6],
        "pointRadius": 0,
        "yAxisID": "price",
    }


# ---------------------------------------------------------------------------
# 6. Bull vs Bear + Research Manager Conclusion
# ---------------------------------------------------------------------------

def _render_debate_section(report: ReportSchema, locale: str) -> str:
    arguments = report.debate_arguments
    # P1: only show research_manager viewpoint when structured resolution
    # does NOT exist (degraded/historical states). Structured
    # debate_resolution is the sole authority for modern runs.
    has_structured_resolution = (
        report.debate_resolution is not None
        and report.debate_resolution.final_reason
    )
    research_plan = (
        None if has_structured_resolution
        else _find_viewpoint(report, "research_manager")
    )
    if not arguments and research_plan is None:
        return (
            f'<section class="panel" id="debate">'
            f'<h2>{escape(text("bull_case", locale))} / {escape(text("bear_case", locale))}</h2>'
            f'<div class="unavailable">{escape(text("unavailable", locale))}</div></section>'
        )
    columns = []
    for stance, label_key in (("bull", "bull_case"), ("bear", "bear_case")):
        stance_args = [a for a in arguments if a.stance == stance]
        cards = "".join(_argument_card(a, locale) for a in stance_args)
        empty = (
            f'<div class="unavailable">{escape(text("unavailable", locale))}</div>'
            if not stance_args else ""
        )
        columns.append(
            f'<div><h2>{escape(text(label_key, locale))}</h2>{cards}{empty}</div>'
        )
    research_html = ""
    if research_plan is not None:
        research_html = (
            f'<h3>{escape(text("research_manager_conclusion", locale))}</h3>'
            f'<div class="prose">{render_safe_markdown(research_plan)}</div>'
        )
    return (
        f'<section class="panel" id="debate">'
        f'<div class="debate-cols">{"".join(columns)}</div>'
        f"{research_html}</section>"
    )


def _find_viewpoint(report: ReportSchema, role: str) -> str | None:
    for item in report.viewpoints:
        if item.role == role and item.body.strip():
            return item.body
    return None


def _compact_evidence_excerpt(argument, max_chars: int = 160) -> str:
    """Deterministic compact evidence: first evidence item, or bounded
    body excerpt. Never calls an LLM."""
    evidence_list = argument.evidence if isinstance(argument.evidence, list) else []
    if evidence_list and str(evidence_list[0]).strip():
        return escape(str(evidence_list[0]))
    # Fallback: bounded plain-text excerpt from body
    body = str(getattr(argument, "body", ""))
    if body:
        return escape(body[:max_chars].rsplit(" ", 1)[0] + "…")
    return ""


def _argument_card(argument, locale: str) -> str:
    strength_pct = min(100.0, max(0.0, argument.strength / 10.0 * 100))
    evidence_list = argument.evidence if isinstance(argument.evidence, list) else []
    compact = _compact_evidence_excerpt(argument)

    # Default visible card: title + strength + one compact evidence
    card = (
        '<div class="argument">'
        f"<h3>{escape(argument.title)}</h3>"
        '<div class="strength-bar">'
        f'<div class="strength-fill" style="width:{strength_pct:.1f}%"></div></div>'
        f'<div class="muted">{argument.strength:g}/10</div>'
    )
    if compact:
        card += f'<div class="argument-evidence">{compact}</div>'

    # Collapsible detail: full body + all evidence
    full_evidence = "".join(
        f"<li>{escape(point)}</li>" for point in evidence_list
    )
    detail_label = "详细论证" if locale != "en" else "Full argument"
    detail = (
        f'<details class="argument-details">'
        f"<summary>{escape(detail_label)}</summary>"
        f'<div class="detail-body">'
        f'<div class="prose">{render_safe_markdown(argument.body)}</div>'
        + (f"<ul>{full_evidence}</ul>" if full_evidence else "")
        + "</div></details>"
    )

    return card + detail + "</div>"


# ---------------------------------------------------------------------------
# 7. Three-way Risk Assessment + Final Ruling
# ---------------------------------------------------------------------------

_RISK_ROLE_LOCALIZED: dict[str, dict[str, str]] = {
    "Aggressive": {"zh-CN": "积极型", "en": "Aggressive"},
    "Neutral": {"zh-CN": "中性", "en": "Neutral"},
    "Conservative": {"zh-CN": "保守型", "en": "Conservative"},
}


def _render_risk_section(report: ReportSchema, locale: str) -> str:
    views = report.risk_views
    cards = "".join(
        '<div class="card">'
        f"<h3>{escape(_RISK_ROLE_LOCALIZED.get(view.label, {}).get(locale, view.label))}</h3>"
        f'<div class="muted">{escape(text("risk_score", locale))}: {view.risk_score:g}/10</div>'
        f'<div class="prose">{render_safe_markdown(view.summary)}</div></div>'
        for view in views
    )
    # Risk Verifier is an engineering guardrail — moved to appendix.
    # Final Ruling belongs to the PM decision — not duplicated here.
    if not cards:
        cards = f'<div class="unavailable">{escape(text("unavailable", locale))}</div>'
    return (
        f'<section class="panel" id="risk-assessment">'
        f'<h2>{escape(text("risk_assessment", locale))}</h2>'
        f'<div class="cards">{cards}</div></section>'
    )


# ---------------------------------------------------------------------------
# 8. Risk Radar + Historical Financials
# ---------------------------------------------------------------------------

def _render_secondary_charts(
    report: ReportSchema, locale: str, targets: dict[str, Any]
) -> str:
    panels: list[str] = []
    # P1: Five-dimension risk radar — localized axes and dataset labels
    _FIVE_AXES = ["valuation", "financial_quality", "market_technical", "event_legal", "liquidity_sentiment"]
    _AXIS_LABELS_ZH = ["估值", "财务质量", "市场/技术", "事件/法律", "流动性/情绪"]
    _AXIS_LABELS_EN = ["Valuation", "Financial Quality", "Market / Technical", "Event / Legal", "Liquidity / Sentiment"]
    _AXIS_LABELS = _AXIS_LABELS_ZH if locale != "en" else _AXIS_LABELS_EN
    _DATASET_COLORS = [
        ("aggressive", "#dc2626", "rgba(220,38,38,.14)"),
        ("neutral", "#6366f1", "rgba(99,102,241,.14)"),
        ("conservative", "#0d9488", "rgba(13,148,136,.14)"),
    ]

    all_have_dims = (
        len(report.risk_views) == 3
        and all(view.dimensions is not None for view in report.risk_views)
    )
    if all_have_dims:
        datasets = []
        for view, (role, border, bg) in zip(report.risk_views, _DATASET_COLORS):
            dims = view.dimensions
            role_label = _RISK_ROLE_LOCALIZED.get(view.label, {}).get(locale, view.label)
            datasets.append({
                "label": role_label,
                "data": [
                    dims.valuation.score,
                    dims.financial_quality.score,
                    dims.market_technical.score,
                    dims.event_legal.score,
                    dims.liquidity_sentiment.score,
                ],
                "borderColor": border,
                "backgroundColor": bg,
            })
        targets["risk"] = {
            "type": "radar",
            "data": {"labels": _AXIS_LABELS, "datasets": datasets},
            "options": {
                "plugins": {
                    "title": {"display": True, "text": text("higher_risk_note", locale)},
                },
                "scales": {"r": {"min": 0, "max": 10}},
            },
        }
        panels.append(
            '<div class="chart-container chart-container-radar">'
            '<canvas id="risk-radar-canvas"></canvas></div>'
        )
    elif report.risk_views:
        # Dimensions incomplete: show unavailable, keep risk_score cards visible
        panels.append(
            f'<div class="unavailable">{escape(text("unavailable", locale))}'
            f' — {escape(text("higher_risk_note", locale))}</div>'
        )

    if report.financial_series:
        labels = sorted(
            {point.period for series in report.financial_series for point in series.points}
        )
        # P1: localize series labels
        _SERIES_LABELS_ZH = {"Revenue": "营业收入", "Net Income": "净利润"}
        _SERIES_LABELS_EN = {"Revenue": "Revenue", "Net Income": "Net Income"}
        _series_labels = _SERIES_LABELS_ZH if locale != "en" else _SERIES_LABELS_EN
        datasets = []
        axes: dict[str, Any] = {}
        axis_by_unit: dict[str, str] = {}
        for series in report.financial_series:
            unit_key = series.unit or "_"
            if unit_key not in axis_by_unit:
                axis_by_unit[unit_key] = f"axis{len(axis_by_unit) + 1}"
            y_axis = axis_by_unit[unit_key]
            axes[y_axis] = {"position": "left" if len(axes) == 0 else "right"}
            by_period = {point.period: point.value for point in series.points}
            by_period_growth = {point.period: point.growth_pct for point in series.points}
            display_label = _series_labels.get(series.label, series.label)
            datasets.append({
                "type": "line",
                "label": f"{display_label} ({series.unit})" if series.unit else display_label,
                "data": [by_period.get(period) for period in labels],
                "borderColor": "#0d9488" if series.name == "revenue" else "#6366f1",
                "yAxisID": y_axis,
                "_rawValues": [by_period.get(period) for period in labels],
                "_growthPct": [
                    by_period_growth.get(period) for period in labels
                ],
            })
        targets["financial"] = {
            "data": {"labels": labels, "datasets": datasets},
            "options": {
                "plugins": {
                    # Marker consumed by the JS template: a real function
                    # callback is attached there, never stringified here.
                    "tooltip": {"_compact": True},
                },
                "scales": axes,
            },
        }
        panels.append(
            '<div class="chart-container chart-container-financial">'
            '<canvas id="financial-history-canvas"></canvas></div>'
        )

    if not panels:
        return (
            f'<section class="panel" id="secondary-charts">'
            f'<h2>{escape(text("supporting_charts", locale))}</h2>'
            f'<div class="unavailable">{escape(text("unavailable", locale))}</div></section>'
        )
    # Split into risk radar panel + financial history panel with separate titles
    radar_panel = ""
    history_panel = ""
    for p in panels:
        if "risk-radar-canvas" in p:
            radar_panel = p
        else:
            history_panel = p

    sections = []
    if radar_panel:
        sections.append(
            f'<div><h3>{escape(text("risk_radar_title", locale))}</h3>{radar_panel}</div>'
        )
    if history_panel:
        sections.append(
            f'<div><h3>{escape(_history_title(locale))}</h3>{history_panel}</div>'
        )

    return (
        f'<section class="panel" id="secondary-charts">'
        f'<h2>{escape(text("supporting_charts", locale))}</h2>'
        f'<div class="chart-grid">{"".join(sections)}</div></section>'
    )


def _price_basis_note(report, locale: str) -> str:
    basis = report.market_price_basis
    provider = basis.provider or "yfinance"
    mode = basis.mode
    if mode == "provider_adjusted":
        if locale == "en":
            return f"Price basis: {provider} adjusted-close"
        return f"价格口径：{provider} 调整后价格"
    if mode == "raw":
        if locale == "en":
            return f"Price basis: {provider} raw (unadjusted)"
        return f"价格口径：{provider} 未调整价格"
    if locale == "en":
        return f"Price basis: {provider} (mode unknown)"
    return f"价格口径：{provider}（模式未知）"


def _history_title(locale: str) -> str:
    if locale == "en":
        return "Historical Financials (Actual)"
    return "历史财务趋势（实际）"


# ---------------------------------------------------------------------------
# 9. Collapsible Detailed Analysis
# ---------------------------------------------------------------------------

def _render_analysis_details(report: ReportSchema, locale: str) -> str:
    by_key = {section.key: section for section in report.analysis_sections}
    details = []
    for index, (key, label_key) in enumerate(_ANALYSIS_KEYS):
        section = by_key.get(key)
        body = section.body if section and section.body.strip() else ""
        summary = escape(text(label_key, locale))
        if body:
            content = (
                f'<details{" open" if index == 0 else ""}>'
                f"<summary>{summary}</summary>"
                f'<div class="detail-body prose">{render_safe_markdown(body)}</div></details>'
            )
        else:
            content = (
                f"<details><summary>{summary}</summary>"
                f'<div class="detail-body"><div class="unavailable">'
                f'{escape(text("unavailable", locale))}</div></div></details>'
            )
        details.append(content)
    return (
        f'<section class="panel" id="detailed-analysis">'
        f'<h2>{escape(text("detailed_analysis", locale))}</h2>'
        f'{"".join(details)}</section>'
    )


# ---------------------------------------------------------------------------
# 10. Key Event Timeline
# ---------------------------------------------------------------------------

def _render_timeline(report: ReportSchema, locale: str) -> str:
    if not report.timeline:
        return (
            f'<section class="panel" id="timeline">'
            f'<h2>{escape(text("event_timeline", locale))}</h2>'
            f'<div class="unavailable">{escape(text("unavailable", locale))}</div></section>'
        )
    items = "".join(_timeline_item(item, locale) for item in report.timeline)
    return (
        f'<section class="panel" id="timeline">'
        f'<h2>{escape(text("event_timeline", locale))}</h2>'
        f'<ul class="timeline-list">{items}</ul></section>'
    )


_IMPACT_LOCALIZED: dict[str, dict[str, str]] = {
    "positive": {"zh-CN": "利好", "en": "positive"},
    "negative": {"zh-CN": "利空", "en": "negative"},
    "neutral": {"zh-CN": "中性", "en": "neutral"},
}

_IMPORTANCE_LOCALIZED: dict[str, dict[str, str]] = {
    "high": {"zh-CN": "高重要度", "en": "high"},
    "medium": {"zh-CN": "中重要度", "en": "medium"},
    "low": {"zh-CN": "低重要度", "en": "low"},
}


def _timeline_item(item, locale: str) -> str:
    badges = []
    kind = getattr(item, "kind", "news") or "news"
    if kind == "insider":
        badges.append(f'<span class="badge badge-insider">{escape(text("insider_kind", locale))}</span>')
    if item.impact != "unknown":
        impact_label = _IMPACT_LOCALIZED.get(item.impact, {}).get(locale, item.impact)
        badges.append(f'<span class="badge badge-{escape(item.impact)}">{escape(impact_label)}</span>')
    if item.importance != "unknown":
        imp_label = _IMPORTANCE_LOCALIZED.get(item.importance, {}).get(locale, item.importance)
        badges.append(f'<span class="badge importance-{escape(item.importance)}">{escape(imp_label)}</span>')
    badges_html = "".join(badges)
    source = escape(item.source)
    link = _safe_external_link(item.url, label=source) if item.url else ""
    detail = escape(item.detail).replace("\n", "<br>")
    reason = escape(item.impact_reason) if item.impact_reason else ""
    return (
        '<li class="timeline-item">'
        f'<div class="timeline-date">{escape(item.date)}</div>'
        f'<div><strong>{escape(item.event)}</strong> {badges_html}'
        f'<div class="muted">{source}{link}</div>'
        + (f'<div class="muted">{reason}</div>' if reason else "")
        + (f"<div>{detail}</div>" if detail else "")
        + "</div></li>"
    )


def _safe_external_link(url: str, label: str = "", locale: str = "zh-CN") -> str:
    """Only http/https hrefs; visible text = label or localized fallback, never raw URL.
    Returns empty string for invalid schemes (javascript:, file:, data:, ftp:, etc.)."""
    if not re.match(r"^https?://", url):
        return ""
    if not label:
        fallback = "查看来源" if locale != "en" else "View source"
        label = fallback
    return f' <a href="{escape(url, quote=True)}" rel="noopener noreferrer">{escape(label)}</a>'
    return ""


# ---------------------------------------------------------------------------
# Phase C P1: Event Center (unified Catalyst + Timeline)
# ---------------------------------------------------------------------------

def _render_event_center(report: ReportSchema, locale: str) -> str:
    """One unified Event Center merging catalysts and timeline events.

    Order: scheduled catalysts → watching catalysts → recent company events →
    recent global/other events. Dedup by evidence_refs/calendar IDs then
    date + normalized title.
    """
    entries: list[dict] = []

    # Collect catalyst entries with future/pending semantics
    for cat in report.catalysts:
        direction_label = _IMPACT_LOCALIZED.get(cat.direction, {}).get(locale, cat.direction)
        entries.append({
            "date": cat.expected_date or "",
            "title": cat.title,
            "kind": "catalyst",
            "status": cat.status,
            "direction": cat.direction,
            "direction_label": direction_label,
            "importance": cat.importance,
            "importance_label": _IMPORTANCE_LOCALIZED.get(cat.importance, {}).get(locale, cat.importance),
            "detail": cat.trigger_condition or "",
            "follow_up": cat.follow_up or "",
            "invalidation": cat.invalidation_condition or "",
            "source": getattr(cat, "source", "") or "",
            "catalyst_type": cat.catalyst_type,
            "sort_group": 0 if cat.status == "scheduled" else 1,
        })

    # Collect timeline entries
    for item in report.timeline:
        direction_label = _IMPACT_LOCALIZED.get(item.impact, {}).get(locale, item.impact)
        is_global = getattr(item, "scope", "company") == "global"
        is_insider = getattr(item, "kind", "news") == "insider"
        entries.append({
            "date": item.date or "",
            "title": item.event,
            "kind": "insider" if is_insider else "event",
            "status": "past",
            "direction": item.impact,
            "direction_label": direction_label,
            "importance": item.importance,
            "importance_label": _IMPORTANCE_LOCALIZED.get(item.importance, {}).get(locale, item.importance),
            "detail": getattr(item, "detail", "") or "",
            "source": item.source or "",
            "url": item.url or "",
            "scope": "global" if is_global else "company",
            "sort_group": 4 if is_global else (3 if is_insider else 2),
            "impact_reason": getattr(item, "impact_reason", "") or "",
        })

    # Dedup: priority = evidence_refs/calendar IDs > date + normalized title
    # Collect catalyst evidence_refs for dedup against timeline items
    cat_refs: set[str] = set()
    for c in report.catalysts:
        for ref in getattr(c, "evidence_refs", []) or []:
            cat_refs.add(str(ref))
    cat_titles = {
        normalize_event_title(c.title) for c in report.catalysts if c.title
    }
    # Build calendar event date map from catalyst evidence_refs
    cat_calendar_dates: dict[str, str] = {}
    for ce in (report.calendar_events or []):
        if ce.event_id in cat_refs and ce.date:
            cat_calendar_dates[ce.event_id] = ce.date

    deduped: list[dict] = []
    seen_ids: set[str] = set()
    for entry in entries:
        # Build identity
        eid_parts = []
        if entry.get("kind") == "catalyst":
            eid_parts.append(f"cat:{normalize_event_title(entry['title'])}")
        else:
            norm = normalize_event_title(entry["title"])
            eid_parts.append(norm)
        if entry.get("date"):
            eid_parts.append(entry["date"])
        eid = "::".join(eid_parts)

        # Dedup timeline items against catalysts
        if entry["kind"] != "catalyst":
            norm = normalize_event_title(entry["title"])
            # Covered by catalyst evidence_refs + date match?
            entry_date = entry.get("date", "")
            if entry_date and any(
                entry_date == cal_date for cal_date in cat_calendar_dates.values()
            ):
                continue
            # Covered by catalyst title match (exact or substring)?
            if norm in cat_titles:
                continue
            # Deterministic substring: catalyst title contained in timeline title or vice versa
            for ct in cat_titles:
                if ct and len(ct) >= 8 and (norm.startswith(ct) or norm.find(ct) >= 0):
                    continue

        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        deduped.append(entry)

    # Sort: catalysts first (scheduled→watching), then company events, then global.
    # Within same group: high → medium → low → unknown (importance rank ascending).
    _importance_rank = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    deduped.sort(key=lambda e: (
        e["sort_group"],
        _importance_rank.get(e.get("importance", "unknown"), 3),
        e["date"] if e["date"] else "",
    ))

    if not deduped:
        return ""

    rows = []
    for entry in deduped:
        badges = []
        if entry["kind"] == "catalyst":
            status_badge = "已排期" if locale != "en" else "Scheduled"
            if entry["status"] == "watching":
                status_badge = "观察中" if locale != "en" else "Watching"
            badges.append(f'<span class="badge badge-neutral">{escape(status_badge)}</span>')
        if entry.get("kind") == "insider":
            badges.append(f'<span class="badge badge-insider">{escape(text("insider_kind", locale))}</span>')
        if entry.get("direction") and entry["direction"] != "unknown":
            badges.append(
                f'<span class="badge badge-{escape(entry["direction"])}">{escape(entry.get("direction_label", entry["direction"]))}</span>'
            )
        if entry.get("importance") and entry["importance"] != "unknown":
            badges.append(
                f'<span class="badge importance-{escape(entry["importance"])}">{escape(entry.get("importance_label", entry["importance"]))}</span>'
            )

        detail = ""
        if entry.get("detail"):
            detail = f'<div class="muted">{escape(str(entry["detail"])[:200])}</div>'
        follow_up = ""
        if entry.get("follow_up"):
            follow_up_label = "跟进" if locale != "en" else "Follow-up"
            follow_up = (
                f'<details class="event-detail"><summary>{escape(follow_up_label)}</summary>'
                f'<div class="detail-body">{escape(entry["follow_up"])}</div></details>'
            )
        invalidation = ""
        if entry.get("invalidation"):
            inv_label = "失效条件" if locale != "en" else "Invalidation"
            invalidation = (
                f'<details class="event-detail"><summary>{escape(inv_label)}</summary>'
                f'<div class="detail-body">{escape(entry["invalidation"])}</div></details>'
            )

        source = entry.get("source", "")
        has_url = bool(entry.get("url") and re.match(r"^https?://", entry.get("url", "")))
        # When URL is present, source is only the link label (not duplicated as plain text).
        # When no URL, source is plain text.
        if has_url:
            link = _safe_external_link(entry["url"], label=source, locale=locale)
            source_line = link
        else:
            link = ""
            source_line = escape(source) if source else ""
        impact_reason = escape(entry.get("impact_reason", "")) if entry.get("impact_reason") else ""

        rows.append(
            '<li class="timeline-item">'
            f'<div class="timeline-date">{escape(entry.get("date", ""))}</div>'
            f'<div><strong>{escape(entry["title"])}</strong> {"".join(badges)}'
            + (f'<div class="muted">{source_line}</div>' if source_line else "")
            + (f'<div class="muted">{impact_reason}</div>' if impact_reason else "")
            + detail + follow_up + invalidation
            + "</div></li>"
        )

    title = "事件中心" if locale != "en" else "Event Center"
    return (
        f'<section class="panel" id="event-center">'
        f"<h2>{escape(title)}</h2>"
        f'<ul class="timeline-list">{"".join(rows)}</ul></section>'
    )


# ---------------------------------------------------------------------------
# 11. Risk Register
# ---------------------------------------------------------------------------

def _render_risk_register(report: ReportSchema, locale: str) -> str:
    if not report.risk_items:
        return (
            f'<section class="panel" id="risk-register">'
            f'<h2>{escape(text("risk_summary", locale))}</h2>'
            f'<div class="unavailable">{escape(text("unavailable", locale))}</div></section>'
        )

    # P1: group by horizon_bucket
    horizon_order = {"short": 0, "medium": 1, "long": 2, "unknown": 3}
    horizon_labels = {
        "short": text("short_term", locale),
        "medium": text("medium_term", locale),
        "long": text("long_term", locale),
        "unknown": text("unclassified", locale),
    }
    grouped: dict[str, list] = {}
    for item in report.risk_items:
        bucket = getattr(item, "horizon_bucket", "unknown") or "unknown"
        grouped.setdefault(bucket, []).append(item)

    sections = []
    for bucket in sorted(grouped.keys(), key=lambda b: horizon_order.get(b, 3)):
        label = horizon_labels.get(bucket, bucket)
        cards = "".join(_risk_item_card(item, locale) for item in grouped[bucket])
        sections.append(f'<h3>{escape(label)}</h3><div class="cards">{cards}</div>')

    return (
        f'<section class="panel" id="risk-register">'
        f'<h2>{escape(text("risk_summary", locale))}</h2>'
        f'{"".join(sections)}</section>'
    )


def _risk_item_card(item, locale: str) -> str:
    width = SEVERITY_WIDTH.get(item.severity, 0)

    # P1: model-estimate fields moved into collapsible details
    estimate_parts = []
    prob_min = getattr(item, "probability_min_pct", None)
    prob_max = getattr(item, "probability_max_pct", None)
    if prob_min is not None and prob_max is not None:
        pct = f"{prob_min:g}%" if prob_min == prob_max else f"{prob_min:g}%–{prob_max:g}%"
        estimate_parts.append(
            f'<div class="muted">{escape(text("model_probability", locale))}: {pct}</div>'
        )
    down_min = getattr(item, "downside_impact_min_pct", None)
    down_max = getattr(item, "downside_impact_max_pct", None)
    if down_min is not None and down_max is not None:
        pct = f"{down_min:g}%" if down_min == down_max else f"{down_min:g}%–{down_max:g}%"
        estimate_parts.append(
            f'<div class="muted">{escape(text("downside_impact", locale))}: {pct}</div>'
        )
    el_min = getattr(item, "expected_loss_min_pct", None)
    el_max = getattr(item, "expected_loss_max_pct", None)
    if el_min is not None and el_max is not None:
        pct = f"{el_min:g}%" if el_min == el_max else f"{el_min:g}%–{el_max:g}%"
        estimate_parts.append(
            f'<div class="muted">{escape(text("expected_loss", locale))}: {pct}</div>'
        )
    if getattr(item, "estimate_basis", None):
        bases_text = ", ".join(escape(b) for b in item.estimate_basis)
        estimate_parts.append(f'<div class="muted">{escape(_basis_label(locale))}: {bases_text}</div>')

    sources_text = " · ".join(escape(source) for source in item.sources) if item.sources else ""

    # Only build details when there is metadata to hide
    details = ""
    if estimate_parts or (item.sources and len(item.sources) > 1):
        detail_label = "模型估算详情" if locale != "en" else "Model estimate details"
        source_line = (
            f'<div class="muted">{escape(_sources_label(locale))}: {sources_text}</div>'
            if sources_text else ""
        )
        details = (
            f'<details class="risk-model-details">'
            f"<summary>{escape(detail_label)}</summary>"
            f'<div class="detail-body">'
            + "".join(estimate_parts)
            + source_line
            + "</div></details>"
        )

    return (
        '<div class="card">'
        f"<h3>{escape(item.name)}</h3>"
        f'<div class="severity-{escape(item.severity)}">{escape(_severity_label(item.severity, locale))}</div>'
        f'<div class="severity-bar"><div class="severity-fill" style="width:{width}%"></div></div>'
        f'<div class="muted">{escape(text("horizon", locale))}: {escape(item.horizon)}</div>'
        f'<div class="prose">{escape(item.evidence)}</div>'
        + (f'<div class="muted">{escape(_sources_label(locale))}: {sources_text}</div>' if sources_text and not details else "")
        + details
        + "</div>"
    )


def _severity_label(severity: str, locale: str) -> str:
    _map = {
        "zh-CN": {"low": "低", "medium": "中", "high": "高", "critical": "严重"},
        "en": {"low": "low", "medium": "medium", "high": "high", "critical": "critical"},
    }
    return _map.get(locale, _map["en"]).get(severity, severity)


def _basis_label(locale: str) -> str:
    return "估算依据" if locale != "en" else "Basis"


def _sources_label(locale: str) -> str:
    return "来源" if locale != "en" else "Sources"


# ---------------------------------------------------------------------------
# 12. Data Quality & Limitations
# ---------------------------------------------------------------------------

def _render_data_quality(report: ReportSchema, locale: str) -> str:
    dq = report.data_quality
    if not any((dq.missing_sections, dq.unparsed_sources, dq.notes, dq.sources, dq.conflicts)):
        return (
            f'<section class="panel" id="data-quality">'
            f'<h2>{escape(text("data_quality", locale))}</h2>'
            f'<div class="muted">{escape(text("notes", locale))}: —</div></section>'
        )
    entries = []
    if dq.missing_sections:
        labels = ", ".join(
            text(MISSING_SECTION_LABELS.get(key, key), locale)
            for key in dq.missing_sections
        )
        entries.append(f"<li>{escape(text('missing_sections', locale))}: {escape(labels)}</li>")
    if dq.unparsed_sources:
        entries.append(
            f"<li>{escape(text('unparsed_sources', locale))}: "
            f"{escape(', '.join(dq.unparsed_sources))}</li>"
        )
    for note in dq.notes:
        entries.append(f"<li>{escape(note)}</li>")
    # P2: Provenance table
    if dq.sources:
        src_rows = []
        for s in dq.sources:
            src_rows.append(
                f"<tr><td>{escape(s.tool_name)}</td>"
                f"<td>{'✓' if s.ok else '✗'}</td>"
                f"<td class=\"muted\">{escape(s.data_as_of)}</td>"
                # retrieved_at is a system timestamp → display in Asia/Shanghai
                # with explicit +08:00 (never a truncated offset-less string).
                f"<td class=\"muted\">{escape(format_system_timestamp(s.retrieved_at) if s.retrieved_at else '')}</td>"
                f"<td class=\"muted\">{escape(', '.join(s.structured_keys))}</td>"
                f"<td>{escape(str(s.truncated))}</td></tr>"
            )
        entries.append(
            f"<li>{escape(text('provenance', locale))}:<table><thead><tr>"
            f"<th>{escape(text('tool', locale))}</th><th>{escape(text('ok', locale))}</th><th>{escape(text('data_as_of', locale))}</th>"
            f"<th>{escape(text('retrieved_at', locale))}</th>"
            f"<th>{escape(text('structured_keys', locale))}</th>"
            f"<th>{escape(text('truncated_v', locale))}</th>"
            f"</tr></thead><tbody>{''.join(src_rows)}</tbody></table></li>"
        )
    # P2: Conflicts table
    if dq.conflicts:
        conf_rows = []
        for c in dq.conflicts:
            src_vals = ", ".join(f"{k}: {v:.2f}" for k, v in c.source_values.items())
            conf_rows.append(
                f"<tr><td>{escape(c.field)}</td><td>{escape(c.period)}</td>"
                f"<td class=\"muted\">{escape(src_vals)}</td>"
                f"<td>{escape(c.selected_source)} ({c.selected_value:.2f})</td></tr>"
                if c.selected_value is not None else
                f"<tr><td>{escape(c.field)}</td><td>{escape(c.period)}</td>"
                f"<td class=\"muted\">{escape(src_vals)}</td><td>{escape(c.selected_source)}</td></tr>"
            )
        entries.append(
            f"<li>{escape(text('conflicts', locale))}:<table><thead><tr>"
            f"<th>{escape(text('metric', locale))}</th><th>{escape(text('period', locale))}</th><th>{escape(text('source', locale))}</th><th>{escape(text('selected', locale))}</th>"
            f"</tr></thead><tbody>{''.join(conf_rows)}</tbody></table></li>"
        )
    return (
        f'<section class="panel" id="data-quality">'
        f'<h2>{escape(text("data_quality", locale))}</h2>'
        f"<ul>{''.join(entries)}</ul></section>"
    )


# ---------------------------------------------------------------------------
# P2: EPS revisions table (extend analyst-consensus)
# ---------------------------------------------------------------------------


def _render_eps_revisions(ac, locale: str) -> str:
    """EPS revision momentum table for analyst-consensus section."""
    if not ac or not ac.eps_revisions:
        return ""
    p_labels = _PERIOD_LABELS.get(locale, _PERIOD_LABELS["en"])
    hdrs = [
        text("period", locale),
        text("current_estimate", locale), text("days_7_ago", locale),
        text("days_30_ago", locale), text("days_60_ago", locale), text("days_90_ago", locale),
        text("up_revisions_7d", locale), text("down_revisions_7d", locale),
        text("up_revisions_30d", locale), text("down_revisions_30d", locale),
    ]
    rows = []
    for rev in ac.eps_revisions:
        period_label = p_labels.get(rev.period, rev.period)
        cells = [f"<td>{escape(period_label)}</td>"]
        for attr in ("current", "days_7_ago", "days_30_ago", "days_60_ago", "days_90_ago"):
            v = getattr(rev, attr, None)
            cells.append(f"<td>{f'{v:.2f}' if v is not None else '—'}</td>")
        for attr in ("up_7d", "down_7d", "up_30d", "down_30d"):
            v = getattr(rev, attr, None)
            cells.append(f"<td>{v if v is not None else '—'}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<h3>{escape(text("eps_revisions_title", locale))}</h3>'
        f'<table><thead><tr>'
        + "".join(f"<th>{escape(h)}</th>" for h in hdrs)
        + f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 12b. P0: Technical Snapshot
# ---------------------------------------------------------------------------


def _render_technical_snapshot(report: ReportSchema, locale: str) -> str:
    ts = report.technical_snapshot
    if ts is None:
        return (
            '<section class="panel technical-snapshot" id="technical">'
            f'<h2>{escape(text("technical_snapshot", locale))}</h2>'
            f'<p class="empty-state">{escape(text("unavailable", locale))}</p></section>'
        )

    def _fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else escape(text("unavailable", locale))

    # P1: localized trend labels
    _trend_keys = {"bullish": "trend_bullish", "bearish": "trend_bearish",
                   "mixed": "trend_mixed", "unknown": "trend_unknown"}
    _trend_arrows = {"bullish": "↑ ", "bearish": "↓ ", "mixed": "↔ ", "unknown": ""}
    trend_key = _trend_keys.get(ts.trend, "trend_unknown")
    trend_label = _trend_arrows.get(ts.trend, "") + escape(text(trend_key, locale))

    rows = [
        f'<tr><td>{escape(text("trend", locale))}</td><td>{escape(trend_label)}</td></tr>',
        f'<tr><td>{escape(text("current_price", locale))}</td><td>{_fmt(ts.latest_close)}</td></tr>',
        f'<tr><td>SMA20</td><td>{_fmt(ts.sma20)}</td></tr>',
        f'<tr><td>SMA50</td><td>{_fmt(ts.sma50)}</td></tr>',
    ]
    if ts.rsi is not None:
        rows.append(f'<tr><td>{escape(text("rsi_label", locale))}</td><td>{_fmt(ts.rsi)}</td></tr>')
    if ts.macd is not None:
        rows.append(f'<tr><td>{escape(text("macd_label", locale))}</td><td>{_fmt(ts.macd)}</td></tr>')
    if ts.macd_signal is not None:
        rows.append(f'<tr><td>{escape(text("macd_signal_label", locale))}</td><td>{_fmt(ts.macd_signal)}</td></tr>')
    if ts.macd_histogram is not None:
        rows.append(f'<tr><td>{escape(text("macd_histogram_label", locale))}</td><td>{_fmt(ts.macd_histogram)}</td></tr>')
    if ts.bollinger_middle is not None:
        rows.append(f'<tr><td>{escape(text("bollinger_label", locale))}</td><td>{_fmt(ts.bollinger_middle)} (U:{_fmt(ts.bollinger_upper)} / L:{_fmt(ts.bollinger_lower)})</td></tr>')
    if ts.kdj_k is not None:
        rows.append(f'<tr><td>{escape(text("kdj_label", locale))}</td><td>K:{_fmt(ts.kdj_k)} D:{_fmt(ts.kdj_d)} J:{_fmt(ts.kdj_j)}</td></tr>')
    if ts.support_levels:
        supports = ", ".join(f"{s:.2f}" for s in ts.support_levels)
        rows.append(f'<tr><td>{escape(text("support", locale))}</td><td>{escape(supports)}</td></tr>')
    if ts.resistance_levels:
        resistances = ", ".join(f"{r:.2f}" for r in ts.resistance_levels)
        rows.append(f'<tr><td>{escape(text("resistance", locale))}</td><td>{escape(resistances)}</td></tr>')

    return (
        '<section class="panel technical-snapshot" id="technical">'
        f'<h2>{escape(text("technical_snapshot", locale))}</h2>'
        '<div class="ts-grid">'
        '<table class="ts-table">'
        f'{"".join(rows)}'
        '</table></div>'
        f'<div class="ts-source">source: {escape(ts.source)}</div>'
        '</section>'
    )


# ---------------------------------------------------------------------------
# 9b. P0: Debate Resolution (Research Manager verdict)
# ---------------------------------------------------------------------------


def _render_debate_resolution(report: ReportSchema, locale: str) -> str:
    dr = report.debate_resolution
    if dr is None:
        return ""
    rec_html = (
        f'<span class="rating-pill rating-{dr.recommendation.lower() if dr.recommendation else "unavailable"}">'
        f'{escape(dr.recommendation or text("unavailable", locale))}</span>'
    )
    parts = [f'<h3>{escape(text("debate_resolution", locale))}</h3>', rec_html]
    if dr.disagreement:
        parts.append(
            f'<div class="dr-field"><strong>{escape(text("core_disagreement", locale))}</strong>'
            f'<p>{escape(dr.disagreement)}</p></div>'
        )
    for label_key, items in [
        ("key_bull_points", dr.key_bull_points),
        ("key_bear_points", dr.key_bear_points),
        ("decisive_factors", dr.decisive_factors),
    ]:
        if items:
            items_html = "".join(f"<li>{escape(item)}</li>" for item in items)
            parts.append(
                f'<div class="dr-field"><strong>{escape(text(label_key, locale))}</strong>'
                f'<ul>{items_html}</ul></div>'
            )
    if dr.final_reason:
        parts.append(
            f'<div class="dr-field"><strong>{escape(text("final_reason", locale))}</strong>'
            f'<p>{escape(dr.final_reason)}</p></div>'
        )
    return (
        '<section class="panel debate-resolution" id="debate-resolution">'
        f'{"".join(parts)}</section>'
    )


# ---------------------------------------------------------------------------
# P1: Analyst Consensus
# ---------------------------------------------------------------------------

_PERIOD_LABELS = {
    "zh-CN": {
        "current_quarter": "本季度", "next_quarter": "下一季度",
        "current_year": "本财年", "next_year": "下一财年",
    },
    "en": {
        "current_quarter": "Current quarter", "next_quarter": "Next quarter",
        "current_year": "Current year", "next_year": "Next year",
    },
}


def _render_analyst_consensus(report: ReportSchema, locale: str) -> str:
    ac = report.analyst_consensus
    if ac is None:
        return (
            '<section class="panel analyst-consensus" id="analyst-consensus">'
            f'<h2>{escape(text("analyst_consensus", locale))}</h2>'
            f'<p class="empty-state">{escape(text("unavailable", locale))}</p></section>'
        )

    parts = [f'<h2>{escape(text("analyst_consensus", locale))}</h2>']

    # Price targets
    pt = ac.price_targets
    if pt:
        target_rows = []
        for key, label in [("low", "target_low"), ("mean", "target_mean"), ("median", "target_median"), ("high", "target_high")]:
            val = getattr(pt, key, None)
            if val is not None:
                target_rows.append(f"<tr><td>{escape(text(label, locale))}</td><td>{val:.2f}</td></tr>")
        if pt.mean_upside_pct is not None:
            sign = "+" if pt.mean_upside_pct >= 0 else ""
            target_rows.append(f"<tr><td>{escape(text('implied_upside', locale))} (mean)</td><td>{sign}{pt.mean_upside_pct:.2f}%</td></tr>")
        if pt.median_upside_pct is not None:
            sign = "+" if pt.median_upside_pct >= 0 else ""
            target_rows.append(f"<tr><td>{escape(text('implied_upside', locale))} (median)</td><td>{sign}{pt.median_upside_pct:.2f}%</td></tr>")
        if target_rows:
            parts.append(f'<h3>{escape(text("target_price", locale))}</h3><table class="ac-table"><tbody>{"".join(target_rows)}</tbody></table>')

    # EPS / Revenue estimates
    for est_key, est_label in [("earnings_estimates", "eps_estimate"), ("revenue_estimates", "revenue_estimate")]:
        estimates = getattr(ac, est_key, [])
        if estimates:
            p_labels = _PERIOD_LABELS.get(locale, _PERIOD_LABELS["en"])
            rows = []
            for est in estimates:
                period_label = p_labels.get(est.period, est.period)
                cells = [f"<td>{escape(period_label)}</td>"]
                for attr in ("average", "low", "high"):
                    v = getattr(est, attr, None)
                    cells.append(f"<td>{f'{v:.2f}' if v is not None else '—'}</td>")
                growth = f"{est.growth_pct:+.2f}%" if est.growth_pct is not None else "—"
                cells.append(f"<td>{growth}</td>")
                count = str(est.analyst_count) if est.analyst_count is not None else "—"
                cells.append(f"<td>{count}</td>")
                rows.append(f"<tr>{''.join(cells)}</tr>")
            header = (
                f"<tr><th>{escape(text('period', locale))}</th>"
                f"<th>{escape(text('avg', locale))}</th>"
                f"<th>{escape(text('low', locale))}</th>"
                f"<th>{escape(text('high', locale))}</th>"
                f"<th>{escape(text('growth_pct', locale))}</th>"
                f"<th>{escape(text('analyst_count', locale))}</th></tr>"
            )
            parts.append(f'<h3>{escape(text(est_label, locale))}</h3><table class="ac-table"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>')

    # Recommendations
    recs = ac.recommendations
    if recs:
        rows = []
        for r in recs:
            rows.append(
                f"<tr><td>{escape(r.period)}</td><td>{r.strong_buy}</td><td>{r.buy}</td>"
                f"<td>{r.hold}</td><td>{r.sell}</td><td>{r.strong_sell}</td></tr>"
            )
        header = (
            "<tr>"
            f"<th>{escape(text('period', locale))}</th>"
            f"<th>{escape(text('strong_buy', locale))}</th>"
            f"<th>{escape(text('buy', locale))}</th>"
            f"<th>{escape(text('hold', locale))}</th>"
            f"<th>{escape(text('sell', locale))}</th>"
            f"<th>{escape(text('strong_sell', locale))}</th>"
            "</tr>"
        )
        parts.append(f'<h3>{escape(text("recommendation_summary", locale))}</h3><table class="ac-table"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>')

    if ac.source:
        parts.append(f'<div class="ac-source">source: {escape(ac.source)}</div>')

    # P2: EPS revisions table
    parts.append(_render_eps_revisions(ac, locale))

    return f'<section class="panel analyst-consensus" id="analyst-consensus">{"".join(parts)}</section>'


# ---------------------------------------------------------------------------
# P1: Catalyst Watchlist
# ---------------------------------------------------------------------------


def _render_catalyst_watchlist(report: ReportSchema, locale: str) -> str:
    catalysts = report.catalysts
    if not catalysts:
        return ""

    items = []
    for c in catalysts:
        fields = []
        if c.expected_date:
            fields.append(f'<span class="cat-date">{escape(c.expected_date)}</span>')
        status_cls = "cat-scheduled" if c.status == "scheduled" else "cat-watching"
        fields.append(f'<span class="cat-status {status_cls}">{escape(text(c.status, locale))}</span>')
        fields.append(f'<span class="cat-direction">[{escape(c.direction)}]</span>')
        fields.append(f'<span class="cat-importance">({escape(c.importance)})</span>')
        detail = []
        if c.trigger_condition:
            detail.append(f'<div><strong>{escape(text("trigger", locale))}:</strong> {escape(c.trigger_condition)}</div>')
        if c.invalidation_condition:
            detail.append(f'<div><strong>{escape(text("invalidation", locale))}:</strong> {escape(c.invalidation_condition)}</div>')
        if c.follow_up:
            detail.append(f'<div><strong>{escape(text("follow_up", locale))}:</strong> {escape(c.follow_up)}</div>')
        if c.evidence_refs:
            refs = ", ".join(escape(r) for r in c.evidence_refs)
            detail.append(f'<div><strong>{escape(text("evidence", locale))}:</strong> {refs}</div>')
        items.append(
            f'<div class="catalyst-item">'
            f'<div class="cat-header"><strong>{escape(c.title)}</strong> {"".join(fields)}</div>'
            f'{"".join(detail)}</div>'
        )

    return (
        '<section class="panel catalysts" id="catalysts">'
        f'<h2>{escape(text("catalyst_watchlist", locale))}</h2>'
        f'{"".join(items)}</section>'
    )


# ---------------------------------------------------------------------------
# P2: Earnings Realization
# ---------------------------------------------------------------------------


def _render_earnings_realization(report: ReportSchema, locale: str) -> str:
    ec = report.earnings_context
    if ec is None or not ec.surprises:
        return ""
    rows = []
    for s in ec.surprises[:8]:
        surprise_str = f"{s.surprise_pct:+.2f}%" if s.surprise_pct is not None else "—"
        rows.append(
            f"<tr><td>{escape(s.report_date)}</td>"
            f"<td>{escape(f'{s.eps_estimate:.2f}' if s.eps_estimate is not None else '—')}</td>"
            f"<td>{escape(f'{s.reported_eps:.2f}' if s.reported_eps is not None else '—')}</td>"
            f"<td>{surprise_str}</td></tr>"
        )
    return (
        '<section class="panel earnings-realization" id="earnings-realization">'
        f'<h2>{escape(text("earnings_realization", locale))}</h2>'
        '<table><thead><tr>'
        f'<th>{escape(text("report_date", locale))}</th>'
        f'<th>{escape(text("eps_estimate_v", locale))}</th>'
        f'<th>{escape(text("reported_eps_v", locale))}</th>'
        f'<th>{escape(text("surprise", locale))}</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></section>'
    )


# ---------------------------------------------------------------------------
# P2: Valuation Context
# ---------------------------------------------------------------------------


def _render_valuation_context(report: ReportSchema, locale: str) -> str:
    vc = report.valuation_context
    if vc is None:
        return ""

    parts = [f'<h2>{escape(text("valuation_context", locale))}</h2>']

    # Market multiples
    mult_rows = []
    for key, label in [("trailing_pe", "pe_ttm"), ("forward_pe", "forward_pe_v"),
                        ("price_to_book", "pb_v"), ("peg_ratio", "peg_v")]:
        val = getattr(vc, key, None)
        if val is not None:
            mult_rows.append(f"<tr><td>{escape(text(label, locale))}</td><td>{val:.2f}</td></tr>")
    if vc.pe_compression_pct is not None:
        sign = "+" if vc.pe_compression_pct >= 0 else ""
        mult_rows.append(f"<tr><td>{escape(text('pe_compression', locale))}</td><td>{sign}{vc.pe_compression_pct:.2f}%</td></tr>")
    if mult_rows:
        parts.append(f'<h3>{escape(text("market_multiples", locale))}</h3><table><tbody>{"".join(mult_rows)}</tbody></table>')

    # Analyst consensus
    ac_rows = []
    if vc.analyst_target_mean is not None:
        ac_rows.append(f"<tr><td>{escape(text('target_mean_v', locale))}</td><td>{vc.analyst_target_mean:.2f}</td></tr>")
    if vc.analyst_target_median is not None:
        ac_rows.append(f"<tr><td>{escape(text('target_median_v', locale))}</td><td>{vc.analyst_target_median:.2f}</td></tr>")
    if vc.mean_target_upside_pct is not None:
        sign = "+" if vc.mean_target_upside_pct >= 0 else ""
        ac_rows.append(f"<tr><td>{escape(text('upside_mean_v', locale))}</td><td>{sign}{vc.mean_target_upside_pct:.2f}%</td></tr>")
    if vc.current_year_eps_growth_pct is not None:
        ac_rows.append(f"<tr><td>{escape(text('eps_growth_cy', locale))}</td><td>{vc.current_year_eps_growth_pct:+.2f}%</td></tr>")
    if vc.next_year_eps_growth_pct is not None:
        ac_rows.append(f"<tr><td>{escape(text('eps_growth_ny', locale))}</td><td>{vc.next_year_eps_growth_pct:+.2f}%</td></tr>")
    if ac_rows:
        parts.append(f'<h3>{escape(text("analyst_consensus_v", locale))}</h3><table><tbody>{"".join(ac_rows)}</tbody></table>')
        parts.append(f'<p class="muted">{escape(text("pm_target_note", locale))}</p>')

    return f'<section class="panel valuation-context" id="valuation-context">{"".join(parts)}</section>'


# ---------------------------------------------------------------------------
# P2: Market Risk Stats
# ---------------------------------------------------------------------------


def _render_market_risk_stats(report: ReportSchema, locale: str) -> str:
    ms = report.market_risk_snapshot
    if ms is None:
        return ""
    rows = []
    _MR_METRICS = [
        ("return_5d_pct", "return_5d", "%"),
        ("return_20d_pct", "return_20d", "%"),
        ("realized_volatility_20d_pct", "realized_vol_20d", "%"),
        ("max_drawdown_60_sessions_pct", "max_drawdown_60", "%"),
        ("atr14_pct", "atr14_pct_v", "%"),
        ("volume_ratio_20d", "volume_ratio_20d_v", "x"),
    ]
    for key, label, unit in _MR_METRICS:
        val = getattr(ms, key, None)
        if val is not None:
            display = f"{val:.2f}{unit}"
            rows.append(f"<tr><td>{escape(text(label, locale))}</td><td>{display}</td></tr>")
    if not rows:
        return ""
    return (
        '<section class="panel market-risk-stats" id="market-risk-stats">'
        f'<h2>{escape(text("market_risk_stats", locale))}</h2>'
        f'<table><tbody>{"".join(rows)}</tbody></table></section>'
    )


# ---------------------------------------------------------------------------
# P2: Ownership & Sentiment
# ---------------------------------------------------------------------------


def _render_ownership_sentiment(report: ReportSchema, locale: str) -> str:
    snap = report.ownership_snapshot
    signals = report.sentiment_signals
    if snap is None and not signals:
        return ""

    parts = [f'<h2>{escape(text("ownership_sentiment", locale))}</h2>']

    if snap:
        summary_rows = []
        for key, label in [("insiders_pct", "insiders_pct_v"), ("institutions_pct", "institutions_pct_v"),
                            ("institutions_float_pct", "institutions_float_v"), ("institutions_count", "institutions_count_v")]:
            val = getattr(snap, key, None)
            if val is not None:
                summary_rows.append(f"<tr><td>{escape(text(label, locale))}</td><td>{val:.2f}%</td></tr>" if "count" not in key else f"<tr><td>{escape(text(label, locale))}</td><td>{int(val)}</td></tr>")
        if summary_rows:
            parts.append(f'<h3>{escape(text("ownership_summary", locale))}</h3><table><tbody>{"".join(summary_rows)}</tbody></table>')

        for holder_key, holder_label in [("institutional_holders", "institutional_holders_v"), ("mutual_fund_holders", "mutual_fund_holders_v")]:
            holders = getattr(snap, holder_key, [])[:5]
            if holders:
                h_rows = []
                for h in holders:
                    h_rows.append(f"<tr><td>{escape(h.holder)}</td><td>{escape(f'{h.shares:,.0f}' if h.shares else '—')}</td>"
                                  f"<td>{escape(f'{h.pct_out:.2f}%' if h.pct_out is not None else '—')}</td>"
                                  f"<td>{escape(h.date_reported)}</td></tr>")
                parts.append(f'<h3>{escape(text(holder_label, locale))}</h3><table><thead><tr>'
                             f'<th>{escape(text("holder_v", locale))}</th>'
                             f'<th>{escape(text("shares_v", locale))}</th>'
                             f'<th>{escape(text("pct_out_v", locale))}</th>'
                             f'<th>{escape(text("report_date", locale))}</th>'
                             f'</tr></thead><tbody>{"".join(h_rows)}</tbody></table>')
        parts.append(f'<p class="muted">{escape(text("ownership_disclaimer", locale))}</p>')

    # Sentiment signals
    if signals:
        sig_rows = []
        for s in signals:
            refs = ", ".join(escape(r) for r in s.evidence_refs) if s.evidence_refs else "—"
            sig_rows.append(f"<tr><td>{escape(s.title)}</td><td>{escape(s.direction)}</td>"
                            f"<td>{s.strength:.1f}</td><td>{escape(s.reasoning[:200])}</td>"
                            f"<td class=\"muted\">{refs}</td></tr>")
        parts.append(f'<h3>{escape(text("grounded_signals", locale))}</h3><table><thead><tr>'
                     f'<th>{escape(text("title", locale))}</th>'
                     f'<th>{escape(text("direction_v", locale))}</th>'
                     f'<th>{escape(text("strength_v", locale))}</th>'
                     f'<th>{escape(text("reasoning_v", locale))}</th>'
                     f'<th>{escape(text("evidence_v", locale))}</th>'
                     f'</tr></thead><tbody>{"".join(sig_rows)}</tbody></table>')

    return f'<section class="panel ownership-sentiment" id="ownership-sentiment">{"".join(parts)}</section>'


# ---------------------------------------------------------------------------
# 11b. P0: Financial Quality
# ---------------------------------------------------------------------------


def _render_financial_quality(report: ReportSchema, locale: str) -> str:
    fq = report.financial_quality
    if fq is None or (not fq.summary and not fq.warnings):
        return ""
    parts = [f'<h2>{escape(text("financial_quality", locale))}</h2>']
    if fq.summary:
        parts.append(f'<p class="fq-summary">{escape(fq.summary)}</p>')
    if fq.warnings:
        parts.append(f'<h3>{escape(text("quality_warnings", locale))}</h3>')
        for w in fq.warnings:
            parts.append(
                f'<div class="fq-warning"><strong>{escape(w.title)}</strong>'
                f'<p>{escape(w.evidence)}</p></div>'
            )
    return (
        '<section class="panel financial-quality" id="financial-quality">'
        f'{"".join(parts)}</section>'
    )


# ---------------------------------------------------------------------------
# 13. Disclaimer
# ---------------------------------------------------------------------------

def _render_disclaimer(report: ReportSchema, locale: str) -> str:
    return (
        '<section class="panel disclaimer" id="disclaimer">'
        f"<p>{escape(text('disclaimer', locale))}</p></section>"
    )


# ---------------------------------------------------------------------------
# Phase C P0: Model vs Consensus divergence block
# ---------------------------------------------------------------------------

def _render_model_consensus_divergence(report: ReportSchema, locale: str) -> str:
    """Render a compact divergence block when model target and consensus
    mean have opposite directions vs current price."""
    decision = report.decision
    consensus = report.analyst_consensus
    currency = report.company_profile.currency if report.company_profile else None
    if consensus is None or consensus.price_targets is None:
        return ""
    current = decision.current_price
    model_target = decision.price_target
    consensus_mean = consensus.price_targets.mean
    if current is None or model_target is None or consensus_mean is None or current <= 0:
        return ""

    model_direction = model_target > current
    consensus_direction = consensus_mean > current
    if model_direction == consensus_direction:
        return ""  # same direction — no divergence to highlight

    # Opposite direction — surface prominently
    model_sign = "+" if model_direction else "-"
    consensus_sign = "+" if consensus_direction else "-"
    model_pct = (model_target / current - 1) * 100
    consensus_pct = (consensus_mean / current - 1) * 100

    model_price_str = format_price(model_target, currency=currency, locale=locale)
    consensus_price_str = format_price(consensus_mean, currency=currency, locale=locale)

    if locale == "en":
        block = (
            '<div class="divergence-warning" style="background:#fff3cd;border:1px solid #ffc107;'
            'border-radius:12px;padding:16px;margin:12px 0">'
            "<strong>⚠ Model vs Consensus: Opposite Direction</strong><br>"
            f"<span>PM Target: {model_price_str} ({model_sign}{abs(model_pct):.1f}%)</span> · "
            f"<span>Consensus Mean: {consensus_price_str} ({consensus_sign}{abs(consensus_pct):.1f}%)</span>"
        )
    else:
        block = (
            '<div class="divergence-warning" style="background:#fff3cd;border:1px solid #ffc107;'
            'border-radius:12px;padding:16px;margin:12px 0">'
            "<strong>⚠ 模型与机构一致预期方向相反</strong><br>"
            f"<span>投资组合经理目标价: {model_price_str} ({model_sign}{abs(model_pct):.1f}%)</span> · "
            f"<span>机构一致预期均值: {consensus_price_str} ({consensus_sign}{abs(consensus_pct):.1f}%)</span>"
        )

    # Append decisive factors from debate resolution if available
    resolution = report.debate_resolution
    if resolution and resolution.decisive_factors:
        factors = resolution.decisive_factors[:3]
        block += (
            "<div style='margin-top:8px;font-size:13px;color:#856404'>"
            + ("".join(f"<div>• {escape(f)}</div>" for f in factors))
            + "</div>"
        )

    block += "</div>"
    return block


# ---------------------------------------------------------------------------
# Phase C P0: Risk Verifier — engineering appendix
# ---------------------------------------------------------------------------

def _render_risk_verifier(report: ReportSchema, locale: str) -> str:
    """Risk Verifier is an engineering guardrail, not an investment opinion.
    Rendered only in the engineering appendix."""
    verifier_text = report.risk_assessment.verifier.strip() if report.risk_assessment else ""
    if not verifier_text:
        return ""
    title = "Risk Verifier Diagnostics" if locale == "en" else "风险验证器诊断"
    return (
        f'<section class="panel panel-muted" id="risk-verifier">'
        f"<h2>{escape(title)}</h2>"
        f'<div class="prose">{render_safe_markdown(verifier_text)}</div></section>'
    )


# ---------------------------------------------------------------------------
# P2 (RPT-27): Decision Evidence Audit renderer
# ---------------------------------------------------------------------------

_DIR_LOCALIZED = {
    "positive": "dir_positive", "negative": "dir_negative",
    "neutral": "dir_neutral", "mixed": "dir_mixed", "unknown": "dir_unknown",
    "rising": "dir_rising", "falling": "dir_falling", "flat": "dir_flat",
    "support": "rel_support", "conflict": "rel_conflict", "context": "rel_context",
}

# Deterministic period enum → localization key (period_current_quarter etc.)
_PERIOD_LOCALIZE_KEYS = {
    "current_quarter": "period_current_quarter",
    "next_quarter": "period_next_quarter",
    "current_year": "period_current_year",
    "next_year": "period_next_year",
    "current_month": "period_current_month",
}


def _localize_direction(label: str, locale: str) -> str:
    key = _DIR_LOCALIZED.get(label)
    return text(key, locale) if key else label


def _localize_evidence_title(item, locale: str) -> str:
    """Localize Python-deterministic evidence titles; keep provider/model
    supplied natural-language content untranslated.

    Canonical evidence_id is NEVER translated (audit identity).
    """
    eid = str(getattr(item, "evidence_id", "") or "")
    if eid.startswith("market:return_20d:"):
        return text("evidence_20d_return", locale)
    if eid.startswith("market:drawdown_60:"):
        return text("evidence_drawdown_60", locale)
    if eid.startswith("analyst:target_mean:"):
        return text("evidence_target_mean", locale)
    if eid.startswith("analyst:eps_revision:"):
        return text("evidence_eps_revision", locale)
    if eid.startswith("earnings_surprise:"):
        return text("evidence_earnings_surprise", locale)
    if eid.startswith("ownership:summary:"):
        return text("evidence_ownership_summary", locale)
    if eid.startswith("holder:institutional:"):
        holder_name = str(getattr(item, "title", "") or "")
        if holder_name.startswith("Holder: "):
            holder_name = holder_name[len("Holder: "):]
        elif holder_name.startswith("机构持有人："):
            holder_name = holder_name[len("机构持有人："):]
        return f"{text('evidence_holder', locale)}：{holder_name}" if locale != "en" \
            else f"{text('evidence_holder', locale)}: {holder_name}"
    # Everything else is provider/model content (catalyst/sentiment/risk titles)
    return str(getattr(item, "title", "") or "")


def _render_decision_evidence_audit(report: ReportSchema, locale: str) -> str:
    """Compact audit of evidence support/conflict/context vs the PM decision.

    Never a second investment decision: shows no rating pill, no target/stop/
    position. Lookup of evidence details via the Evidence Registry by ID.
    """
    audit = report.decision_evidence_audit
    if audit is None:
        return ""
    if not (audit.support_refs or audit.conflict_refs or audit.context_refs):
        return ""

    registry_by_id = {item.evidence_id: item for item in report.evidence_registry}

    def _bucket(refs: list[str], label_key: str, max_show: int = 5) -> str:
        if not refs:
            return ""
        rows = []
        for ref in refs[:max_show]:
            item = registry_by_id.get(ref)
            if item is None:
                rows.append(
                    f'<li><code>{escape(ref)}</code></li>'
                )
                continue
            # RF-3C: localize deterministic evidence titles; canonical ID stays raw
            detail = escape(_localize_evidence_title(item, locale) or ref)
            if item.value_text:
                detail += f" — {escape(item.value_text)}"
            asof = f" ({escape(item.as_of)})" if item.as_of else ""
            rows.append(
                f'<li><code>{escape(ref)}</code> {detail}{asof}'
                + (f" · {escape(item.source)}" if item.source else "")
                + "</li>"
            )
        return (
            f'<div class="audit-bucket"><h4>{escape(text(label_key, locale))}</h4>'
            f'<ul>{"".join(rows)}</ul></div>'
        )

    buckets = (
        _bucket(audit.support_refs, "supporting_current_decision")
        + _bucket(audit.conflict_refs, "conflicting_evidence")
        + _bucket(audit.context_refs, "context_evidence")
    )
    if not buckets:
        return ""

    note = escape(text("audit_not_second_decision", locale))
    return (
        f'<section class="panel panel-muted" id="decision-evidence-audit">'
        f'<h2>{escape(text("decision_evidence_audit", locale))}</h2>'
        f'{buckets}'
        f'<div class="muted" style="font-size:12px">{note}</div>'
        f"</section>"
    )


# ---------------------------------------------------------------------------
# P2 (RPT-29): Research Diagnostics — compact + collapsed raw details
# ---------------------------------------------------------------------------


def _compact_kpi_card(label: str, value: str) -> str:
    return (
        '<div class="kpi">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div></div>'
    )


def _render_research_diagnostics(report: ReportSchema, locale: str) -> str:
    """One parent section with compact diagnostics + collapsed raw details.

    Reorganizes existing P2 panels (analyst consensus, earnings realization,
    valuation context, market risk stats, ownership) — does NOT duplicate them.
    """
    panels: list[str] = []

    # --- Compact diagnostic cards ---
    compact = []

    # Estimate revision
    if report.estimate_revision_diagnostics:
        cards = []
        for d in report.estimate_revision_diagnostics:
            direction = _localize_direction(d.direction, locale)
            change = f"{d.change_30d_pct:.2f}%" if d.change_30d_pct is not None else "—"
            net = str(d.net_revisions_30d) if d.net_revisions_30d is not None else "—"
            period_key = _PERIOD_LOCALIZE_KEYS.get(d.period)
            period_label = text(period_key, locale) if period_key else d.period
            cards.append(
                '<div class="card">'
                f"<h4>{escape(period_label)}</h4>"
                f'<div class="muted">{escape(text("change_30d", locale))}: {escape(change)} · {escape(direction)}</div>'
                f'<div class="muted">{escape(text("net_revisions", locale))}: {escape(net)}</div></div>'
            )
        compact.append(
            f'<div><h3>{escape(text("estimate_revision_diagnostic", locale))}</h3>'
            f'<div class="cards">{"".join(cards)}</div></div>'
        )

    # Earnings realization
    er = report.earnings_realization
    if er is not None:
        beat = f"{er.beat_rate_pct:.1f}%" if er.beat_rate_pct is not None else "—"
        mean = f"{er.mean_surprise_pct:.2f}%" if er.mean_surprise_pct is not None else "—"
        latest = f"{er.latest_surprise_pct:.2f}%" if er.latest_surprise_pct is not None else "—"
        cards = (
            _compact_kpi_card(text("sample_size", locale), str(er.sample_size))
            + _compact_kpi_card(text("beat_rate", locale), beat)
            + _compact_kpi_card(text("mean_surprise", locale), mean)
            + _compact_kpi_card(text("latest_surprise", locale), latest)
        )
        compact.append(
            f'<div><h3>{escape(text("earnings_realization", locale))}</h3>'
            f'<div class="kpi-grid">{cards}</div></div>'
        )

    # Valuation
    vc = report.valuation_context
    if vc is not None:
        disp = f"{vc.analyst_target_dispersion_pct:.2f}%" if vc.analyst_target_dispersion_pct is not None else "—"
        comp = f"{vc.pe_compression_pct:.2f}%" if vc.pe_compression_pct is not None else "—"
        mean_t = f"{vc.analyst_target_mean:.2f}" if vc.analyst_target_mean is not None else "—"
        cards = (
            _compact_kpi_card(text("pe_ttm", locale), f"{vc.trailing_pe:.2f}" if vc.trailing_pe is not None else "—")
            + _compact_kpi_card(text("forward_pe_v", locale), f"{vc.forward_pe:.2f}" if vc.forward_pe is not None else "—")
            + _compact_kpi_card(text("pe_compression", locale), comp)
            + _compact_kpi_card(text("analyst_consensus_v", locale), mean_t)
            + _compact_kpi_card(text("target_dispersion", locale), disp)
        )
        compact.append(
            f'<div><h3>{escape(text("valuation_consensus_diagnostic", locale))}</h3>'
            f'<div class="kpi-grid">{cards}</div></div>'
        )

    # Market risk
    mrs = report.market_risk_snapshot
    if mrs is not None:
        cards = ""
        if mrs.return_5d_pct is not None:
            cards += _compact_kpi_card(text("return_5d", locale), f"{mrs.return_5d_pct:.2f}%")
        if mrs.return_20d_pct is not None:
            cards += _compact_kpi_card(text("return_20d", locale), f"{mrs.return_20d_pct:.2f}%")
        if mrs.realized_volatility_20d_pct is not None:
            cards += _compact_kpi_card(text("realized_vol_20d", locale), f"{mrs.realized_volatility_20d_pct:.2f}%")
        if mrs.max_drawdown_60_sessions_pct is not None:
            cards += _compact_kpi_card(text("max_drawdown_60", locale), f"{mrs.max_drawdown_60_sessions_pct:.2f}%")
        compact.append(
            f'<div><h3>{escape(text("market_risk_stats", locale))}</h3>'
            f'<div class="kpi-grid">{cards}</div></div>'
        )

    # Ownership
    od = report.ownership_diagnostic
    if od is not None:
        top = f"{od.top_holder_pct:.2f}%" if od.top_holder_pct is not None else "—"
        top5 = f"{od.top5_institutional_pct:.2f}%" if od.top5_institutional_pct is not None else "—"
        cards = (
            _compact_kpi_card(text("institutions_pct_v", locale), f"{od.institutional_pct}%" if od.institutional_pct is not None else "—")
            + _compact_kpi_card(text("insiders_pct_v", locale), f"{od.insiders_pct}%" if od.insiders_pct is not None else "—")
            + _compact_kpi_card(text("top_holder", locale), top)
            + _compact_kpi_card(text("top5_concentration", locale), top5)
            + _compact_kpi_card(text("retained_holder_count", locale), str(od.retained_holder_count))
        )
        compact.append(
            f'<div><h3>{escape(text("ownership_structure", locale))}</h3>'
            f'<div class="kpi-grid">{cards}</div></div>'
        )

    if compact:
        panels.append(f'<div class="diagnostic-grid">{"".join(compact)}</div>')

    # --- Raw structured details (collapsed) ---
    raw_parts = []
    for renderer in (
        _render_analyst_consensus,
        _render_earnings_realization,
        _render_valuation_context,
        _render_market_risk_stats,
        _render_ownership_sentiment,
    ):
        part = renderer(report, locale)
        if part.strip():
            raw_parts.append(part)
    if raw_parts:
        panels.append(
            f'<details class="research-raw-details">'
            f'<summary>{escape(text("raw_details", locale))}</summary>'
            f'<div class="detail-body">{"".join(raw_parts)}</div></details>'
        )

    if not panels:
        return ""
    return (
        f'<section class="panel" id="research-diagnostics">'
        f'<h2>{escape(text("research_diagnostics", locale))}</h2>'
        f'{"".join(panels)}</section>'
    )


# ---------------------------------------------------------------------------
# P2 (RPT-28): Data Reliability Summary renderer
# ---------------------------------------------------------------------------


def _render_data_reliability_summary(report: ReportSchema, locale: str) -> str:
    """Six deterministic counts mirroring data_quality — no score/grade."""
    dr = report.data_reliability
    if dr is None:
        return ""
    stale = text("yes", locale) if dr.stale_market_data else text("no", locale)
    cards = (
        _compact_kpi_card(text("source_count", locale), str(dr.source_count))
        + _compact_kpi_card(text("failed_source_count", locale), str(dr.failed_source_count))
        + _compact_kpi_card(text("truncated_source_count", locale), str(dr.truncated_source_count))
        + _compact_kpi_card(text("conflict_count", locale), str(dr.conflict_count))
        + _compact_kpi_card(text("missing_section_count", locale), str(dr.missing_section_count))
        + _compact_kpi_card(text("stale_market_data", locale), stale)
    )
    return (
        f'<section class="panel panel-muted" id="data-reliability-summary">'
        f'<h2>{escape(text("data_reliability_summary", locale))}</h2>'
        f'<div class="kpi-grid">{cards}</div></section>'
    )


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

# Chart targets: canonical canvas ids keyed by target name.
_CHART_CANVAS_IDS = {
    "score": "score-chart-canvas",
    "market": "market-price-canvas",
    "risk": "risk-radar-canvas",
    "financial": "financial-history-canvas",
}

# Fixed JS template: Python injects plain business data as JSON only; all
# functions/callbacks live here as real JavaScript. Never stringify functions.
_CHART_RUNTIME_TEMPLATE = """<script>
function fmtCompact(v) {
    const a = Math.abs(v);
    if (a >= 1e12) return (v / 1e12).toFixed(2) + "T";
    if (a >= 1e9) return (v / 1e9).toFixed(2) + "B";
    if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (a >= 1e3) return (v / 1e3).toFixed(2) + "K";
    return String(Math.round(v * 100) / 100);
}
function getChartCanvas(id) {
    const el = document.getElementById(id);
    if (!el) {
        console.error('Chart target "' + id + '" not found.');
        return null;
    }
    if (!(el instanceof HTMLCanvasElement)) {
        console.error('Chart target "' + id + '" is not a canvas element.');
        return null;
    }
    return el;
}
function createChart(id, config) {
    const canvas = getChartCanvas(id);
    if (!canvas) {
        return;
    }
    config.options = config.options || {};
    config.options.responsive = true;
    config.options.maintainAspectRatio = false;
    config.options.animation = false;
    const tooltip = config.options.plugins && config.options.plugins.tooltip;
    if (tooltip && tooltip._compact) {
        tooltip.callbacks = tooltip.callbacks || {};
        tooltip.callbacks.label = function(ctx) {
            const raw = ctx.dataset._rawValues ? ctx.dataset._rawValues[ctx.dataIndex] : null;
            const value = raw != null ? fmtCompact(raw) : ctx.parsed.y;
            let label = ctx.dataset.label + ": " + value;
            const growth = ctx.dataset._growthPct ? ctx.dataset._growthPct[ctx.dataIndex] : null;
            if (growth != null) {
                const sign = growth >= 0 ? "+" : "";
                label += " (" + sign + growth.toFixed(2) + "%)";
            }
            return label;
        };
    }
    new Chart(canvas, config);
}
const chartTargets = __TARGETS__;
__CALLS__
</script>"""


def _chart_runtime_script(targets: dict[str, Any]) -> str:
    """Emit the single chart initializer block: data JSON + fixed JS calls.

    Only targets that actually exist get a createChart call, so absent charts
    produce no console errors.
    """
    calls = "".join(
        f'if (chartTargets.{key}) {{ createChart("{canvas_id}", chartTargets.{key}); }}\n'
        for key, canvas_id in _CHART_CANVAS_IDS.items()
        if key in targets
    )
    return (
        _CHART_RUNTIME_TEMPLATE.replace("__TARGETS__", _safe_json(targets))
        .replace("__CALLS__", calls)
    )


def _safe_json(value) -> str:
    """Serialize JSON so script boundaries (`<`, `>`, `&`) cannot break out."""
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        serialized.replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace(" ", r" ")
        .replace(" ", r" ")
    )


def _safe_artifact_component(value: str, *, fallback: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-")
    return (sanitized or fallback)[:64]


def write_report_html(
    report: ReportSchema,
    output_path: str | Path,
    *,
    locale: str = DEFAULT_LOCALE,
) -> Path:
    """Write a report as UTF-8 and return its concrete path."""
    path = Path(output_path)
    path.write_text(render_report_html(report, locale=locale), encoding="utf-8")
    return path


def _artifact_locale(state) -> str:
    """Report locale from the runtime language source of truth.

    ``state.output_language`` is the same canonical value that drove agent
    generation (zh-CN / en); falls back to the legacy default for old
    states that predate the field.
    """
    value = str(getattr(state, "output_language", "") or "")
    if value in SUPPORTED_LOCALES:
        return value
    return DEFAULT_LOCALE


def generate_report_artifact(state, output_dir: str | Path) -> Path:
    """Build and write one safely named offline report artifact."""
    ticker = safe_ticker_component(str(getattr(state, "ticker", "")))
    trade_date = _safe_artifact_component(
        str(getattr(state, "trade_date", "")),
        fallback="unknown-date",
    )
    run_id = _safe_artifact_component(
        str(getattr(state, "run_id", "")),
        fallback="run",
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = build_report_schema(state)
    return write_report_html(
        report,
        directory / f"{ticker}_{trade_date}_{run_id}.html",
        locale=_artifact_locale(state),
    )

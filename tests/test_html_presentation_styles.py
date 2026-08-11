"""Style-layer presentation regression tests.

Protect the CSS architecture contract of the offline report:

1. Panel modifiers must be expressed as compound selectors (``.panel.XXX``)
   with specificity >= the base ``section.panel`` rule — a bare ``.XXX``
   modifier silently loses to ``section.panel`` and produces white-on-white
   or grey-on-white text.
2. The Trading Plan inside the decision hero must carry an explicit
   light-surface card contract (never inherit the hero's inverse colors).
3. Every semantic component class must have at least a minimal visual
   contract — no component that renders entirely on browser defaults.
4. Badge variants must never produce "transparent background + white text".
5. Tables may scroll inside their panel but never blow the layout; details
   keep working collapse + visible focus.

These tests inspect the generated stylesheet text and the rendered DOM;
they do not change any business data or section structure.
"""

import re

import pytest

from finmindagent.reporting.html import _STYLES, render_report_html
from tests.test_html_presentation_contract import _full_report


def _rule(css: str, selector: str) -> str | None:
    """Return the declaration block of the first rule whose selector
    contains ``selector`` as a substring (a trailing ``{`` in the query is
    ignored — the regex splits selector and declaration block at the brace).
    Media queries collapse naturally: their inner rules match individually."""
    selector = selector.rstrip("{").strip()
    for block in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        sel, decls = block
        if selector in sel:
            return decls.strip()
    return None


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


# ---------------------------------------------------------------------------
# RED 1: Decision Hero specificity (root cause: section.panel vs .decision-hero)
# ---------------------------------------------------------------------------

def test_decision_hero_selector_beats_base_panel():
    """The hero background/color must live on a compound selector with
    specificity >= ``section.panel`` (0,1,1), never on a bare
    ``.decision-hero`` (0,1,0) that silently loses to the base panel."""
    decls = _rule(_STYLES, ".panel.decision-hero") or _rule(
        _STYLES, "section.panel.decision-hero"
    )
    assert decls is not None, (
        "missing compound selector `.panel.decision-hero` / "
        "`section.panel.decision-hero` in stylesheet"
    )
    assert re.search(
        r"background\s*:\s*(?:linear-gradient|var\(--surface-inverse)", decls
    ), "decision hero background must be an explicit dark surface/gradient"
    assert "color:var(--ink-inverse)" in _compact(decls), (
        "decision hero text color must be explicitly inverse-light"
    )
    # base panel border must not be silently inherited into the dark hero
    assert re.search(r"border\s*:", decls), (
        "decision hero must coordinate its own border with the dark theme"
    )


def test_decision_hero_h2_beats_section_panel_h2():
    decls = _rule(_STYLES, ".panel.decision-hero h2")
    assert decls is not None, "missing `.panel.decision-hero h2` selector"
    assert "color:var(--ink-inverse)" in _compact(decls), (
        "hero h2 must set an explicit inverse color"
    )


def test_no_important_anywhere():
    """No `!important` outside the single sanctioned chart-canvas rule.

    The one exception is `.chart-container canvas { width:100%!important; ... }`
    — part of the explicit-height mechanism that holds canvas size against
    Chart.js inline styles (never reintroduce a resize feedback loop).
    """
    stripped = re.sub(r"\.chart-container\s+canvas\{[^}]*\}", "", _STYLES)
    assert "!important" not in stripped


def test_decision_hero_dom_class_pair_present():
    """The DOM keeps the existing class pair so the compound selector
    actually applies (DOM contract unchanged)."""
    rendered = render_report_html(_full_report(), chart_js="")
    assert 'class="panel decision-hero"' in rendered
    assert 'id="decision"' in rendered


# ---------------------------------------------------------------------------
# RED 2: Trading Plan contrast (inherited white text on light card)
# ---------------------------------------------------------------------------

def test_trading_plan_explicit_light_surface():
    decls = _rule(_STYLES, ".trading-plan{")
    assert decls is not None, "missing rule for `.trading-plan`"
    compact = _compact(decls)
    assert "background:var(--surface)" in compact, (
        "trading plan must be an explicit light card on the dark hero"
    )
    assert "color:var(--ink)" in compact, (
        "trading plan text must be explicit dark ink, never inherited"
    )
    assert re.search(r"border[^-]*:", decls), "trading plan must set its own border"


def test_trading_plan_sub_components_styled():
    for selector in (".tp-row{", ".tp-label{", ".tp-value{", ".tp-events{"):
        assert _rule(_STYLES, selector) is not None, (
            f"missing CSS rule for {selector[:-1]}"
        )


def test_trading_plan_label_value_semantics():
    label = _compact(_rule(_STYLES, ".tp-label{") or "")
    assert "color:var(--ink-2)" in label, "tp-label must be secondary ink"
    assert "font-weight" in label, "tp-label must carry label weight"
    value = _compact(_rule(_STYLES, ".tp-value{") or "")
    assert "color:var(--ink)" in value, "tp-value must be primary ink"


def test_trading_plan_rows_separated():
    row = _compact(_rule(_STYLES, ".tp-row{") or "")
    assert "grid-template-columns" in row, "tp-row must be a label/value two-column row"
    assert "border-top" in row or "border-bottom" in row, (
        "tp rows must be separated by a light border"
    )


def test_trading_plan_summary_explicit_color():
    decls = _rule(_STYLES, ".trading-plan summary")
    assert decls is not None, "missing `.trading-plan summary` selector"
    assert "color:var(--ink)" in _compact(decls), (
        "trading plan summary must not depend on ancestor color inheritance"
    )


def test_trading_plan_events_legible_list():
    events = _compact(_rule(_STYLES, ".tp-events{") or "")
    assert "padding-left" in events or "margin" in events, (
        "tp-events must have real list padding"
    )
    assert "line-height" in events, "tp-events must set a readable line-height"


def test_trading_plan_single_column_on_mobile():
    mobile = _STYLES.split("@media(max-width:640px)")[1]
    assert re.search(r"\.tp-row\s*\{[^}]*grid-template-columns:1fr", mobile), (
        "trading plan must collapse to a single column on mobile"
    )


def test_hero_contains_trading_plan_dom():
    """The hero DOM keeps the trading plan inside it (structure unchanged)."""
    from finmindagent.reporting.schemas import TradingPlanSection

    report = _full_report()
    report.decision.trading_plan = TradingPlanSection(
        holding_action="Trim 20%",
        entry_condition="Entry below support",
        exit_condition="Exit on stop",
        monitoring_events=["Earnings 2026-10-15", "Volume check"],
    )
    rendered = render_report_html(report, chart_js="")
    hero = rendered.split('class="panel decision-hero"')[1].split("</section>")[0]
    assert '<details class="trading-plan" open>' in hero
    assert 'class="tp-row"' in hero
    assert 'class="tp-label"' in hero
    assert 'class="tp-events"' in hero


# ---------------------------------------------------------------------------
# Panel modifier specificity model (uniform base + modifier rule)
# ---------------------------------------------------------------------------

def test_panel_muted_beats_base_panel():
    decls = _rule(_STYLES, ".panel.panel-muted") or _rule(
        _STYLES, "section.panel.panel-muted"
    )
    assert decls is not None, "missing compound `.panel.panel-muted` selector"
    assert "background:var(--surface-muted)" in _compact(decls)


def test_disclaimer_beats_base_panel():
    decls = _rule(_STYLES, ".panel.disclaimer") or _rule(
        _STYLES, "section.panel.disclaimer"
    )
    assert decls is not None, "missing compound `.panel.disclaimer` selector"
    assert "background:var(--surface-muted)" in _compact(decls)


# ---------------------------------------------------------------------------
# Badges: no transparent-background + white-text variants
# ---------------------------------------------------------------------------

def test_badge_insider_has_background():
    decls = _rule(_STYLES, ".badge-insider")
    assert decls is not None, "`.badge-insider` variant used by renderers but unstyled"
    assert re.search(r"background\s*:", decls)


def test_badge_mixed_has_background():
    """Event Center emits `badge-mixed` for mixed-direction catalysts —
    must carry an explicit background, never white-on-transparent."""
    decls = _rule(_STYLES, ".badge-mixed")
    assert decls is not None, "`.badge-mixed` variant used by renderers but unstyled"
    assert re.search(r"background\s*:", decls)


def test_every_rendered_badge_variant_has_background():
    """Holistic guard: every `badge-*` class the renderers emit must have an
    explicit background in the stylesheet — `.badge` alone gives white-on-
    transparent on light surfaces."""
    from finmindagent.reporting.schemas import TimelineItem

    report = _full_report()
    report.timeline.append(TimelineItem(
        date="2026-08-08", event="Mixed signal", impact="mixed",
        importance="medium", source="S", detail="d",
    ))
    rendered = render_report_html(report, chart_js="")
    variants = set(re.findall(r'class="badge\s+([a-z-]+)"', rendered))
    variants = {v for v in variants if v.startswith("badge-")}
    assert variants, "expected at least one badge variant in rendered DOM"
    for variant in variants:
        decls = _rule(_STYLES, f".{variant}")
        assert decls is not None, f".{variant} used in DOM but unstyled"
        assert re.search(r"background\s*:", decls), (
            f".{variant} has no background (white-on-transparent risk)"
        )


def test_badge_importance_soft_variants():
    for cls in (".badge.importance-high", ".badge.importance-medium", ".badge.importance-low"):
        decls = _rule(_STYLES, cls)
        assert decls is not None, f"missing compound badge variant {cls}"
        assert re.search(r"background\s*:", decls), (
            f"{cls} must carry a soft background (`.badge` alone would give white text)"
        )
        assert re.search(r"color\s*:", decls), f"{cls} must set an explicit dark text color"


def test_plain_severity_and_importance_are_dark_text():
    """Text-only variants used on light surfaces must stay dark ink."""
    for cls in (".severity-low", ".severity-medium", ".severity-high",
                ".severity-critical", ".importance-high", ".importance-medium",
                ".importance-low"):
        decls = _rule(_STYLES, cls)
        assert decls is not None, f"missing rule for {cls}"
        assert re.search(r"color\s*:", decls) and "#fff" not in decls, (
            f"{cls} must be dark text, never white on light surface"
        )


# ---------------------------------------------------------------------------
# Semantic class coverage (no component left to browser defaults)
# ---------------------------------------------------------------------------

_SEMANTIC_CONTRACTS = {
    "ts-grid": r"display\s*:\s*grid",
    "ts-table": r"width\s*:\s*100%",
    "ts-source": r"color\s*:\s*var\(--ink-2\)",
    "empty-state": r"color\s*:\s*var\(--ink-2\)",
    "dr-field": r"margin",
    "ac-table": r"width\s*:\s*100%",
    "ac-source": r"color\s*:\s*var\(--ink-2\)",
    "catalyst-item": r"border-bottom",
    "cat-header": r"display\s*:\s*flex",
    "cat-date": r"color\s*:\s*var\(--ink-2\)",
    "cat-status": r"color\s*:",
    "cat-direction": r"color\s*:",
    "cat-importance": r"color\s*:",
    "fq-summary": r"color\s*:\s*var\(--ink\)",
    "fq-warning": r"background\s*:\s*var\(--warning-soft\)",
    "audit-bucket": r"margin",
    "diagnostic-grid": r"display\s*:\s*grid",
    "research-raw-details": r"margin-top",
    "supporting-group": r"background\s*:\s*var\(--surface\)",
    "appendix-group": r"background\s*:\s*var\(--surface\)",
}


@pytest.mark.parametrize(
    ("cls", "pattern"), sorted(_SEMANTIC_CONTRACTS.items())
)
def test_semantic_class_has_visual_contract(cls, pattern):
    decls = _rule(_STYLES, f".{cls}")
    assert decls is not None, f"missing CSS rule for .{cls}"
    assert re.search(pattern, decls), (
        f".{cls} rule lacks {pattern!r}: {decls}"
    )


# ---------------------------------------------------------------------------
# Details / summary affordances
# ---------------------------------------------------------------------------

def test_details_hover_and_focus_defined():
    assert _rule(_STYLES, "details summary:hover") is not None
    assert _rule(_STYLES, "details summary:focus-visible") is not None


def test_details_open_summary_separator():
    assert _rule(_STYLES, "details[open]>summary") is not None


# ---------------------------------------------------------------------------
# Tables / panels: scroll inside panel, never blow the layout
# ---------------------------------------------------------------------------

def test_panels_allow_horizontal_scroll():
    decls = _rule(_STYLES, "section.panel{")
    assert decls is not None
    assert "overflow-x:auto" in _compact(decls), (
        "panels must scroll horizontally on narrow screens instead of overflowing"
    )


def test_tables_wrap_long_text():
    decls = _rule(_STYLES, "th,td") or _rule(_STYLES, "td,th")
    assert decls is not None, "missing th,td rule"
    assert "overflow-wrap" in decls or "word-break" in decls, (
        "table cells must wrap long text"
    )


# ---------------------------------------------------------------------------
# Tokens: one palette, extended for the dark hero + soft surfaces
# ---------------------------------------------------------------------------

def test_extended_tokens_present():
    for token in (
        "--surface-inverse", "--ink-inverse", "--ink-inverse-muted",
        "--positive-soft", "--negative-soft", "--warning-soft",
    ):
        assert f"{token}:" in _STYLES, f"missing token {token}"


def test_stylesheet_has_no_remote_urls():
    assert re.search(r"url\(\s*['\"]?(?:https?:)?//", _STYLES) is None

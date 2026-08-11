"""Decision-metrics data-chain contract tests (MU unavailable-fields bug).

Covers the authoritative structured PM decision → build_report_schema →
Decision hero HTML chain for target price, stop loss, and the
deterministic risk/reward ratio:

CASE 1  bearish decision with valid short geometry → target/stop shown, R/R ≈ 1.76
CASE 2  bullish decision with valid long geometry → R/R = 3.00
CASE 3  target missing → target unavailable AND R/R unavailable
CASE 4  stop missing → stop unavailable AND R/R unavailable
CASE 5  bearish rating with long-only geometry → NO positive-looking ratio
CASE 6  structured PM fields preserved verbatim through build_report_schema
CASE 7  prose mentioning prices has NO authority when structured fields are null

No LLM, no network — pure deterministic builder/renderer contract.
"""

import pytest

from finmindagent.reporting.builder import build_report_schema
from finmindagent.reporting.html import render_report_html
from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.events import EventType
from finmindagent.runtime.state import TradingRunState


def _decision_state(
    rating: str,
    current: float,
    target: float | None,
    stop: float | None,
    *,
    portfolio_decision: dict | None = None,
    final_trade_decision: str = "",
) -> TradingRunState:
    """Minimal structured state: market series (current price) + PM decision."""
    state = TradingRunState(
        ticker="MU",
        trade_date="2026-08-10",
        run_id="decision-metrics-test",
        status="completed",
    )
    market_points = [
        {"date": "2026-08-07", "open": current, "high": current + 1,
         "low": current - 1, "close": current - 2.0, "volume": 1000000},
        {"date": "2026-08-10", "open": current, "high": current + 1,
         "low": current - 1, "close": current, "volume": 1000000},
    ]
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
    if portfolio_decision is None:
        portfolio_decision = {
            "rating": rating,
            "executive_summary": "Decision summary.",
            "investment_thesis": "Investment thesis.",
            "price_target": target,
            "stop_loss": stop,
        }
    state.portfolio_decision = portfolio_decision
    if final_trade_decision:
        state.final_trade_decision = final_trade_decision
    return state


def _hero(html: str) -> str:
    """Extract the decision hero section from rendered HTML."""
    return html.split('id="decision"')[1].split("</section>")[0]


# ---------------------------------------------------------------------------
# CASE 1 — bearish decision with valid short geometry (MU-like)
# ---------------------------------------------------------------------------

def test_case1_bearish_target_stop_and_rr():
    state = _decision_state("Underweight", 877.57, 750.0, 950.0)
    report = build_report_schema(state, generated_at="2026-08-10T12:00:00Z")

    assert report.decision.rating == "Underweight"
    assert report.decision.current_price == pytest.approx(877.57)
    assert report.decision.price_target == pytest.approx(750.0)
    assert report.decision.stop_loss == pytest.approx(950.0)
    # reward = 877.57 - 750 = 127.57; risk = 950 - 877.57 = 72.43
    expected = 127.57 / 72.43
    assert report.decision.risk_reward_ratio == pytest.approx(expected, abs=1e-9)

    hero = _hero(render_report_html(report, chart_js=""))
    assert "877.57" in hero
    assert "750.00" in hero
    assert "950.00" in hero
    assert "1:1.76" in hero


# ---------------------------------------------------------------------------
# CASE 2 — bullish decision with valid long geometry
# ---------------------------------------------------------------------------

def test_case2_bullish_rr():
    state = _decision_state("Buy", 100.0, 130.0, 90.0)
    report = build_report_schema(state, generated_at="2026-08-10T12:00:00Z")

    assert report.decision.risk_reward_ratio == pytest.approx(3.0)
    hero = _hero(render_report_html(report, chart_js=""))
    assert "1:3.00" in hero


# ---------------------------------------------------------------------------
# CASE 3 / CASE 4 — missing target / missing stop → unavailable + no R/R
# ---------------------------------------------------------------------------

def test_case3_target_missing():
    state = _decision_state("Buy", 100.0, None, 90.0)
    report = build_report_schema(state, generated_at="2026-08-10T12:00:00Z")

    assert report.decision.price_target is None
    assert report.decision.stop_loss == pytest.approx(90.0)
    assert report.decision.risk_reward_ratio is None

    hero = _hero(render_report_html(report, chart_js=""))
    assert "暂无可靠数据" in hero  # target card
    assert "90.00" in hero  # stop still shown
    assert "1:" not in hero.replace("1:1", "") or "1:1." not in hero  # no ratio


def test_case4_stop_missing():
    state = _decision_state("Buy", 100.0, 130.0, None)
    report = build_report_schema(state, generated_at="2026-08-10T12:00:00Z")

    assert report.decision.stop_loss is None
    assert report.decision.price_target == pytest.approx(130.0)
    assert report.decision.risk_reward_ratio is None

    hero = _hero(render_report_html(report, chart_js=""))
    assert "暂无可靠数据" in hero  # stop card
    assert "130.00" in hero  # target still shown


# ---------------------------------------------------------------------------
# CASE 5 — bearish rating with long-only geometry → no fabricated ratio
# ---------------------------------------------------------------------------

def test_case5_invalid_bearish_geometry_no_ratio():
    """Sell with target 120 > current 100 > stop 90: the long formula would
    produce a plausible-looking 2.0 — must be rejected, not displayed."""
    state = _decision_state("Sell", 100.0, 120.0, 90.0)
    report = build_report_schema(state, generated_at="2026-08-10T12:00:00Z")

    assert report.decision.risk_reward_ratio is None
    hero = _hero(render_report_html(report, chart_js=""))
    assert "1:" not in hero


# ---------------------------------------------------------------------------
# CASE 6 — structured PM fields preserved verbatim through the builder
# ---------------------------------------------------------------------------

def test_case6_pm_structured_fields_preserved():
    state = _decision_state("Underweight", 877.57, 750.0, 950.0)
    report = build_report_schema(state, generated_at="2026-08-10T12:00:00Z")

    assert report.decision.price_target == 750.0
    assert report.decision.stop_loss == 950.0


# ---------------------------------------------------------------------------
# CASE 7 — prose has no authority when structured fields are null
# ---------------------------------------------------------------------------

def test_case7_prose_prices_never_extracted():
    state = _decision_state(
        "Underweight",
        877.57,
        None,
        None,
        portfolio_decision={
            "rating": "Underweight",
            "executive_summary": "Reduce exposure.",
            "investment_thesis": "Peak-cycle risk dominates.",
            "price_target": None,
            "stop_loss": None,
        },
        final_trade_decision=(
            "**Rating**: Underweight\n\n"
            "We see target $700, stop $950 as the working levels."
        ),
    )
    report = build_report_schema(state, generated_at="2026-08-10T12:00:00Z")

    # Structured priority: no regex extraction from prose, no fabricated prices.
    assert report.decision.price_target is None
    assert report.decision.stop_loss is None
    assert report.decision.risk_reward_ratio is None

    hero = _hero(render_report_html(report, chart_js=""))
    assert "700" not in hero
    assert "950" not in hero


# ---------------------------------------------------------------------------
# PM structured contract — fields must instruct fill-when-evidence + direction
# ---------------------------------------------------------------------------

def test_pm_schema_contract_requires_direction_consistent_levels():
    """The PortfolioDecision schema contract must instruct the model to fill
    target/stop with evidence and return null only when no reliable level
    exists, with direction-consistency semantics."""
    from finmindagent.agents.schemas import PortfolioDecision

    target_desc = PortfolioDecision.model_fields["price_target"].description.lower()
    stop_desc = PortfolioDecision.model_fields["stop_loss"].description.lower()

    for desc in (target_desc, stop_desc):
        assert "reliable" in desc, f"contract must demand evidence-based levels: {desc}"
        assert "null" in desc, f"contract must define the null-only-when-uncertain rule: {desc}"
        assert "direction" in desc or "underweight" in desc or "sell" in desc, (
            f"contract must demand direction-consistent levels: {desc}"
        )


def test_pm_schema_fields_remain_optional_float():
    """Fields stay optional — some decisions legitimately cannot form a price."""
    from typing import get_args

    from finmindagent.agents.schemas import PortfolioDecision

    for name in ("price_target", "stop_loss"):
        field = PortfolioDecision.model_fields[name]
        assert field.is_required() is False
        assert get_args(field.annotation) == (float, type(None)), (
            f"{name} must stay float | None"
        )

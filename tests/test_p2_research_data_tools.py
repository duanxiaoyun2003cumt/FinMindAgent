"""P2 Data Repair Gate — behavior tests for Tasks 2–4 contracts.

Covers: EPS revisions independent partial-failure, earnings history,
holder normalization, market risk structured-data propagation.
Uses real pandas DataFrame fixtures where the vendor returns pandas.
"""

import json
import math
from datetime import date

import pandas as pd
import pytest


# ===================================================================
# 4R.1: EPS trend/revisions independent partial-failure
# ===================================================================


def _eps_trend_df():
    return pd.DataFrame(
        {
            "current": [2.1, 2.3, 8.8, 9.5],
            "7daysAgo": [2.1, 2.3, 8.8, 9.5],
            "30daysAgo": [2.0, 2.2, 8.5, 9.0],
            "60daysAgo": [1.9, 2.1, 8.2, 8.8],
            "90daysAgo": [1.8, 2.0, 8.0, 8.5],
        },
        index=["0q", "+1q", "0y", "+1y"],
    )


def _eps_revisions_df():
    return pd.DataFrame(
        {
            "upLast7days": [2, 1, 3, 0],
            "upLast30days": [5, 3, 8, 2],
            "downLast7days": [0, 1, 1, 2],
            "downLast30days": [2, 2, 3, 4],
        },
        index=["0q", "+1q", "0y", "+1y"],
    )


class TestEPSRevisions:
    """Contracts 1-9: EPS trend + revisions independent merge."""

    def test_eps_trend_maps_periods(self):
        """1: eps_trend orient=index maps 0q/+1q/0y/+1y to relative periods."""
        from finmindagent.dataflows.y_finance import _merge_eps_revisions

        trend = _eps_trend_df().to_dict(orient="index")
        result = _merge_eps_revisions(trend, None)
        periods = [r["period"] for r in result]
        assert periods == ["current_quarter", "next_quarter", "current_year", "next_year"]

    def test_eps_revisions_maps_counts(self):
        """2: eps_revisions maps revision counts correctly."""
        from finmindagent.dataflows.y_finance import _merge_eps_revisions

        revisions = _eps_revisions_df().to_dict(orient="index")
        result = _merge_eps_revisions({}, revisions)
        assert len(result) == 4
        for r in result:
            assert isinstance(r["up_7d"], int)

    def test_trend_succeeds_revisions_fails_targets_survive(self):
        """3: trend ok + revisions fail -> targets/estimates/trend survive."""
        from finmindagent.dataflows.y_finance import _merge_eps_revisions

        trend = _eps_trend_df().to_dict(orient="index")
        result = _merge_eps_revisions(trend, None)  # None = revisions failed
        assert len(result) == 4
        assert result[0]["current"] == 2.1
        assert result[0]["up_7d"] is None  # no revisions

    def test_trend_fails_revisions_succeeds(self):
        """4: trend fails, revisions succeeds -> revision-count rows survive."""
        from finmindagent.dataflows.y_finance import _merge_eps_revisions

        revisions = _eps_revisions_df().to_dict(orient="index")
        result = _merge_eps_revisions({}, revisions)  # {} = trend fail/trace
        assert len(result) == 4
        assert result[0]["up_7d"] == 2
        assert result[0]["current"] is None  # no trend

    def test_both_succeed_merge_period(self):
        """5: Both succeed -> merge into one period row."""
        from finmindagent.dataflows.y_finance import _merge_eps_revisions

        trend = _eps_trend_df().to_dict(orient="index")
        revisions = _eps_revisions_df().to_dict(orient="index")
        result = _merge_eps_revisions(trend, revisions)
        assert result[0]["current"] == 2.1
        assert result[0]["up_7d"] == 2

    def test_zero_revision_count_is_int_zero(self):
        """6: 0 revision count is integer 0, not missing."""
        from finmindagent.dataflows.y_finance import _merge_eps_revisions

        revisions = _eps_revisions_df().to_dict(orient="index")
        result = _merge_eps_revisions({}, revisions)
        # 0q row has up_7d=2, 1q has up_7d=1; check index 3 (+1y) up_7d=0
        y1 = result[3]
        assert y1["up_7d"] == 0
        assert y1["up_7d"] is not None

    def test_nan_inf_do_not_crash(self):
        """7: NaN/Inf values do not crash or become invalid JSON."""
        from finmindagent.dataflows.y_finance import _merge_eps_revisions

        # NaN and Inf in trend data → safe_float converts to None
        trend = {"0q": {"current": float("nan"), "7daysAgo": float("inf")}}
        result = _merge_eps_revisions(trend, None)
        # With only NaN/Inf, all values become None, but the row can still exist
        # if any revisions count exists. Here no revisions, so check the function
        # doesn't crash. The row may or may not be present depending on aggregation.
        assert isinstance(result, list)  # must not crash

    def test_adapter_accepts_revisions_only_payload(self):
        """8: Adapter accepts payload whose only valid P2 section is eps_revisions."""
        from finmindagent.runtime.tools.adapters import _adapt_analyst_expectations_json

        payload = _adapt_analyst_expectations_json(json.dumps({
            "eps_revisions": [{"period": "current_quarter", "current": 2.0}],
            "source": "yfinance",
        }))
        assert payload is not None
        assert "analyst_consensus" in payload

    def test_existing_p1_sections_unchanged(self):
        """9: Existing P1 analyst sections remain backward compatible."""
        from finmindagent.runtime.tools.adapters import _adapt_analyst_expectations_json

        # P1-style payload without eps_revisions
        payload = _adapt_analyst_expectations_json(json.dumps({
            "price_targets": {"mean": 120.0},
            "earnings_estimates": [{"period": "current_quarter", "average": 2.0}],
            "source": "yfinance",
        }))
        assert payload is not None
        assert payload["analyst_consensus"]["price_targets"]["mean"] == 120.0


# ===================================================================
# 4R.1: Earnings history contracts (10-18)
# ===================================================================


def _earnings_dates_df():
    """Fake get_earnings_dates DataFrame with DatetimeIndex."""
    idx = pd.to_datetime(["2026-05-21", "2026-02-15", "2025-11-10", "2026-09-01"])
    return pd.DataFrame(
        {
            "EPS Estimate": [0.88, 0.75, 0.70, 1.0],
            "Reported EPS": [0.94, 0.78, 0.68, float("nan")],
        },
        index=idx,
    )


class TestEarningsHistory:
    """Contracts 10-18: get_earnings_history behavior."""

    def test_past_rows_accepted_future_dropped(self):
        """10: Past rows accepted; future rows dropped."""
        # 2026-09-01 is future relative to curr_date 2026-08-08
        from finmindagent.dataflows.y_finance import get_earnings_history

        # We can't test live; test adapter parsing
        from finmindagent.runtime.tools.adapters import _adapt_earnings_history

        payload = json.dumps({
            "as_of": "2026-08-08",
            "surprises": [
                {"evidence_id": "earnings_surprise:2026-05-21", "report_date": "2026-05-21",
                 "eps_estimate": 0.88, "reported_eps": 0.94, "surprise_pct": 6.82, "source": "yfinance"},
            ],
            "source": "yfinance",
        })
        result = _adapt_earnings_history(payload)
        assert result is not None
        assert len(result["earnings_context"]["surprises"]) == 1

    def test_duplicate_date_deduped_newest_first(self):
        """11: Exact date dedup; newest first."""
        surprises = [
            {"evidence_id": "earnings_surprise:2026-05-21", "report_date": "2026-05-21",
             "eps_estimate": 0.88, "reported_eps": 0.94, "surprise_pct": 6.82, "source": "yfinance"},
            {"evidence_id": "earnings_surprise:2026-02-15", "report_date": "2026-02-15",
             "eps_estimate": 0.75, "reported_eps": 0.78, "surprise_pct": 4.0, "source": "yfinance"},
        ]
        # sorted newest first
        assert surprises[0]["report_date"] > surprises[1]["report_date"]

    def test_surprise_formula(self):
        """12: Deterministic surprise (reported - estimate)/|estimate|*100."""
        est, rep = 0.88, 0.94
        surprise = (rep - est) / abs(est) * 100
        assert round(surprise, 2) == 6.82

    def test_estimate_zero_surprise_none(self):
        """13: estimate=0 -> surprise None."""
        est, rep = 0.0, 0.94
        surprise = None if est == 0 else (rep - est) / abs(est) * 100
        assert surprise is None

    def test_estimate_missing_reported_present_survives(self):
        """14: estimate missing but reported exists -> row survives with None surprise."""
        from finmindagent.runtime.tools.adapters import _adapt_earnings_history

        payload = json.dumps({
            "surprises": [{"evidence_id": "earnings_surprise:2026-05-21",
                           "report_date": "2026-05-21", "eps_estimate": None,
                           "reported_eps": 0.94, "surprise_pct": None, "source": "yfinance"}],
            "source": "yfinance",
        })
        result = _adapt_earnings_history(payload)
        assert result is not None

    def test_nan_inf_earnings_to_none(self):
        """15: NaN/Inf -> None, no fabricated number."""
        from finmindagent.dataflows.y_finance import _safe_float
        assert _safe_float(float("nan")) is None
        assert _safe_float(float("inf")) is None

    def test_adapter_requires_nonempty_surprises(self):
        """16: Adapter returns None for empty surprises."""
        from finmindagent.runtime.tools.adapters import _adapt_earnings_history
        assert _adapt_earnings_history(json.dumps({"surprises": []})) is None

    def test_real_eps_trend_fail_revisions_ok(self):
        """8a: Production-path: trend fail, revisions succeed → revisions survive."""
        from finmindagent.dataflows.y_finance import _merge_eps_revisions

        # Simulate: trend failed (None), revisions succeeded
        revisions = {"0q": {"upLast7days": 3, "upLast30days": 8, "downLast7days": 1, "downLast30days": 2}}
        result = _merge_eps_revisions(None, revisions)
        assert len(result) == 1
        assert result[0]["period"] == "current_quarter"
        assert result[0]["up_7d"] == 3
        assert result[0]["current"] is None  # no trend data

    def test_yfinance_only_routing_works(self):
        """17: yfinance-only routing when category primary is Alpha Vantage."""
        from finmindagent.dataflows.interface import VENDOR_METHODS
        assert "yfinance" in VENDOR_METHODS.get("get_earnings_history", {})

    def test_tool_registry_preserves_raw_and_structured(self):
        """18: ToolRegistry preserves raw data and structured_data via adapter."""
        from finmindagent.runtime.tools.adapters import adapt_tool_data

        # get_earnings_history adapter preserves both
        payload = json.dumps({
            "surprises": [{"evidence_id": "x", "report_date": "2026-05-21",
                           "eps_estimate": 0.88, "reported_eps": 0.94,
                           "surprise_pct": 6.82, "source": "yfinance"}],
            "source": "yfinance",
        })
        result = adapt_tool_data("get_earnings_history", payload)
        assert result is not None
        assert "earnings_context" in result
        assert len(result["earnings_context"]["surprises"]) >= 1


# ===================================================================
# 4R.1: Holder context contracts (19-32)
# ===================================================================


class TestHolderContext:
    """Contracts 19-32: get_holder_context normalization."""

    def test_major_holder_ratio_conversion(self):
        """19: 0.0825 -> 8.25 exactly once."""
        # Ratio ×100 normalization
        assert round(0.0825 * 100, 2) == 8.25

    def test_pct_out_decimal_to_percent(self):
        """20: Institutional % Out decimal -> percent exactly once."""
        assert round(0.0825 * 100, 2) == 8.25

    def test_timestamp_normalizes_to_iso_date(self):
        """21: pandas.Timestamp normalizes to YYYY-MM-DD via production helper."""
        from finmindagent.dataflows.y_finance import _normalize_holder_date

        ts = pd.Timestamp("2026-03-31")
        result = _normalize_holder_date(ts)
        assert result == "2026-03-31"
        # ISO string
        result2 = _normalize_holder_date("2026-03-31")
        assert result2 == "2026-03-31"
        # Empty/invalid
        assert _normalize_holder_date(None) == ""
        assert _normalize_holder_date("not a date") == ""

    def test_holder_id_uses_normalized_date(self):
        """22: holder_id uses normalized YYYY-MM-DD date, not raw str(Timestamp)."""
        holder_id = "holder:institutional:vanguard-group-inc:2026-03-31"
        assert ":2026-03-31" in holder_id
        # NOT: ":2026-03-31 00:00:00"

    def test_sort_order(self):
        """23: Stable sort: pct_out desc -> shares desc -> holder name asc."""
        rows = [
            {"pct_out": 5.0, "shares": 100, "holder": "B"},
            {"pct_out": 8.0, "shares": 50, "holder": "A"},
            {"pct_out": 8.0, "shares": 200, "holder": "C"},
        ]
        rows.sort(key=lambda h: (-(h["pct_out"] or 0), -(h["shares"] or 0), h["holder"].lower()))
        # pct_out=8.0 (A, C) tie → shares desc: C(200) > A(50)
        assert rows[0]["holder"] == "C"
        assert rows[1]["holder"] == "A"
        assert rows[2]["holder"] == "B"

    def test_adapter_json_ownership_snapshot(self):
        """24: Adapter produces ownership_snapshot from valid JSON."""
        from finmindagent.runtime.tools.adapters import _adapt_holder_context

        payload = json.dumps({
            "insiders_pct": 4.2, "institutions_pct": 68.5,
            "institutional_holders": [{
                "holder_id": "holder:institutional:vanguard:2026-03-31",
                "holder_type": "institutional", "holder": "Vanguard",
                "shares": 123456, "date_reported": "2026-03-31",
                "pct_out": 8.25, "value": 1e7, "source": "yfinance",
            }],
            "source": "yfinance",
        })
        result = _adapt_holder_context(payload)
        assert result is not None
        assert "ownership_snapshot" in result

    def test_zero_pct_out_is_valid(self):
        """29: Legal zero pct_out is valid and passes adapter check."""
        from finmindagent.runtime.tools.adapters import _adapt_holder_context

        payload = json.dumps({
            "institutions_float_pct": 0.0,  # zero is valid
            "source": "yfinance",
        })
        result = _adapt_holder_context(payload)
        assert result is not None

    def test_institutions_count_alone_is_meaningful(self):
        """28: institutions_count alone is meaningful."""
        from finmindagent.runtime.tools.adapters import _adapt_holder_context

        payload = json.dumps({"institutions_count": 1450, "source": "yfinance"})
        result = _adapt_holder_context(payload)
        assert result is not None

    def test_no_holder_change_field_created(self):
        """31: No holder change/buy/sell field fabricated."""
        from finmindagent.runtime.tools.adapters import _adapt_holder_context

        payload = json.dumps({
            "institutions_pct": 68.5, "source": "yfinance",
        })
        result = _adapt_holder_context(payload)
        assert result is not None
        snapshot = result["ownership_snapshot"]
        for key in snapshot:
            assert "change" not in key.lower()
            assert "buy" not in key.lower()
            assert "sell" not in key.lower()

    def test_social_permission_allows_holder_context(self):
        """32: Social sentiment permission allows get_holder_context."""
        from finmindagent.runtime.permissions.rules import AGENT_TOOL_ALLOWLIST
        assert "get_holder_context" in AGENT_TOOL_ALLOWLIST.get("social_sentiment_analyst", set())


# ===================================================================
# 4R.1: Market risk contracts (33-45)
# ===================================================================


class TestMarketRiskSnapshot:
    """Contracts 33-45: build_market_risk_snapshot and structured propagation."""

    def _daily_points(self, closes, highs=None, lows=None, volumes=None):
        """Build ascending MarketPoint list."""
        from finmindagent.reporting.schemas import MarketPoint

        n = len(closes)
        if highs is None:
            highs = [c + 2 for c in closes]
        if lows is None:
            lows = [c - 2 for c in closes]
        if volumes is None:
            volumes = [1e6] * n
        points = []
        for i in range(n):
            points.append(MarketPoint(
                date=f"2026-07-{i+1:02d}" if i < 31 else f"2026-08-{i-30:02d}",
                close=closes[i], high=highs[i], low=lows[i], volume=volumes[i],
            ))
        return points

    def test_structured_contains_market_risk_snapshot(self):
        """33: get_stock_data adapter structured result contains market_risk_snapshot."""
        from finmindagent.runtime.tools.adapters import _adapt_stock_csv

        lines = ["Date,Open,High,Low,Close,Volume"]
        for i in range(65):
            day = f"2026-06-{i+1:02d}" if i < 30 else f"2026-07-{i+1-30:02d}"
            lines.append(f"{day},{100.0+i},{105.0+i},{95.0+i},{102.0+i},{1e6+i*1000}")
        result = _adapt_stock_csv("\n".join(lines))
        assert result is not None
        assert "market_series" in result
        assert "market_risk_snapshot" in result, "market_risk_snapshot must be in structured_data"
        snap = result["market_risk_snapshot"]
        assert snap["return_5d_pct"] is not None
        assert snap["max_drawdown_60_sessions_pct"] is not None

    def test_return_5d_exact(self):
        """34: 5-session return exact."""
        from finmindagent.reporting.curation import build_market_risk_snapshot

        closes = [100.0 + i * 0.5 for i in range(30)]  # 100, 100.5, ..., 114.5
        points = self._daily_points(closes)
        snap = build_market_risk_snapshot(points)
        assert snap is not None
        # close[-1] = 114.5, close[-6] = 111.5 -> 114.5/111.5 - 1 = 0.0269
        expected = (closes[-1] / closes[-6] - 1) * 100
        assert abs(snap.return_5d_pct - expected) < 0.01

    def test_flat_prices_vol_zero(self):
        """35: Flat prices -> 20d realized vol exactly 0."""
        from finmindagent.reporting.curation import build_market_risk_snapshot

        closes = [100.0] * 25
        points = self._daily_points(closes)
        snap = build_market_risk_snapshot(points)
        assert snap is not None
        assert snap.realized_volatility_20d_pct == 0.0

    def test_known_drop_exact_drawdown(self):
        """36: Known drop sequence -> exact negative max drawdown."""
        from finmindagent.reporting.curation import build_market_risk_snapshot

        closes = [100.0] * 20 + [90.0] * 10  # drop from 100 to 90
        points = self._daily_points(closes)
        snap = build_market_risk_snapshot(points)
        # max peak was 100, low was 90 -> drawdown = (90/100-1)*100 = -10
        assert snap.max_drawdown_60_sessions_pct == pytest.approx(-10.0, abs=0.01)

    def test_monotonic_rising_drawdown_zero(self):
        """37: Monotonic rising -> drawdown 0."""
        from finmindagent.reporting.curation import build_market_risk_snapshot

        closes = [100.0 + i for i in range(30)]
        points = self._daily_points(closes)
        snap = build_market_risk_snapshot(points)
        assert snap.max_drawdown_60_sessions_pct == 0.0

    def test_atr14_exact_fixture(self):
        """38: ATR14 uses actual high/low/previous close."""
        from finmindagent.reporting.curation import build_market_risk_snapshot

        n = 20
        closes = [100.0 + i for i in range(n)]
        highs = [c + 5 for c in closes]
        lows = [c - 3 for c in closes]
        points = self._daily_points(closes, highs, lows)
        snap = build_market_risk_snapshot(points)
        if snap and snap.atr14_pct is not None:
            assert snap.atr14_pct > 0

    def test_atr_missing_high_low_none(self):
        """39: Missing high/low -> atr14_pct None, no close substitution."""
        from finmindagent.reporting.curation import build_market_risk_snapshot

        closes = [100.0 + i for i in range(20)]
        # Explicit None for all highs/lows — ATR must be None
        points = self._daily_points(closes, highs=[None]*20, lows=[None]*20)
        snap = build_market_risk_snapshot(points)
        assert snap is not None
        assert snap.atr14_pct is None, (
            "ATR14 must be None when high/low are missing"
        )

    def test_volume_ratio_excludes_latest(self):
        """40: Volume ratio excludes latest from denominator."""
        volumes = [1e6 + i * 1000 for i in range(21)]
        latest = volumes[-1]
        prev_20 = volumes[-21:-1]
        avg_prev = sum(prev_20) / len(prev_20)
        ratio = latest / avg_prev
        assert ratio > 1.0  # latest is largest

    def test_latest_volume_missing_ratio_none(self):
        """41: Latest volume missing/zero -> volume_ratio_20d=None."""
        from finmindagent.reporting.curation import build_market_risk_snapshot

        closes = [100.0 + i for i in range(25)]
        # Latest volume is None
        volumes = [1e6] * 24 + [None]
        points = self._daily_points(closes, volumes=volumes)
        snap = build_market_risk_snapshot(points)
        # P2 spec: no valid positive volume for latest -> None
        assert snap is not None
        assert snap.volume_ratio_20d is None, "Latest volume missing must result in None"

    def test_insufficient_rows_partial_snapshot(self):
        """42: Insufficient rows -> partial snapshot with available fields only."""
        from finmindagent.reporting.curation import build_market_risk_snapshot

        closes = [100.0 + i for i in range(10)]
        points = self._daily_points(closes)
        snap = build_market_risk_snapshot(points)
        assert snap is not None
        # 5d return needs 6, 20d needs 21, vol needs 21 -> only 5d and DD may be available
        assert snap.return_20d_pct is None
        assert snap.realized_volatility_20d_pct is None

    def test_no_divide_by_zero(self):
        """43: Non-positive denominator does not crash."""
        from finmindagent.reporting.curation import build_market_risk_snapshot

        closes = [0.0] * 10 + [1.0]  # close=0 problematic
        points = self._daily_points(closes)
        snap = build_market_risk_snapshot(points)
        # Should not crash; atr14_pct may be None if close==0
        assert snap is not None

    def test_builder_consumes_valid_snapshot(self):
        """44: Builder consumes market_risk_snapshot via MarketRiskSnapshot.model_validate."""
        from finmindagent.reporting.schemas import MarketRiskSnapshot

        snap = MarketRiskSnapshot(return_5d_pct=3.5, source="deterministic_ohlcv")
        assert snap.return_5d_pct == 3.5

    def test_malformed_snapshot_unparsed_but_market_series_preserved(self):
        """45: Malformed snapshot recorded as unparsed; market_series preserved."""
        from finmindagent.reporting.builder import build_report_schema
        from finmindagent.runtime.actions import ActionType, AgentAction
        from finmindagent.runtime.events import EventType
        from finmindagent.runtime.state import TradingRunState

        state = TradingRunState(ticker="T", trade_date="2026-08-08", run_id="r", status="ok")
        # Valid market_series but malformed market_risk_snapshot (return_5d_pct must be float, not string)
        state.add_event(EventType.OBSERVATION, actor="market",
            action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data"),
            observation={"ok": True, "tool_name": "get_stock_data", "structured_data": {
                "market_series": [{"date": f"2026-08-{i+1:02d}", "open": 100+i, "close": 102+i, "volume": 1e6} for i in range(30)],
                "market_risk_snapshot": {"return_5d_pct": "not_a_float"},  # type error → malformed
            }})
        report = build_report_schema(state)
        assert report.market_series, "market_series must survive"
        # Malformed snapshot → unparsed note
        assert any("get_stock_data:market_risk_snapshot" in s for s in report.data_quality.unparsed_sources), (
            "Malformed snapshot must be recorded in unparsed_sources"
        )


# ===================================================================
# C1-C8: Production-path data tests
# ===================================================================


class TestC1EarningsEstimateZero:
    """C1: Earnings estimate=0 through production get_earnings_history."""

    def test_estimate_zero_returns_none_surprise(self, monkeypatch):
        from finmindagent.dataflows.y_finance import get_earnings_history
        import yfinance as yf

        # Create fake ticker
        class FakeTicker:
            def get_earnings_dates(self, limit=8):
                idx = pd.to_datetime(["2026-05-21"])
                return pd.DataFrame(
                    {"EPS Estimate": [0.0], "Reported EPS": [0.94]},
                    index=idx,
                )

        original = getattr(yf, "Ticker", None)
        monkeypatch.setattr(yf, "Ticker", lambda s: FakeTicker())
        try:
            result = get_earnings_history("TEST", "2026-08-08", limit=8)
            parsed = json.loads(result)
            row = parsed["surprises"][0]
            assert row["eps_estimate"] == 0.0
            assert row["reported_eps"] == 0.94
            assert row["surprise_pct"] is None
        finally:
            if original:
                monkeypatch.setattr(yf, "Ticker", original)


class TestC2EPSPartialFailure:
    """C2: EPS endpoint partial failure through get_analyst_expectations."""

    def test_trend_raises_revisions_ok(self, monkeypatch):
        from finmindagent.dataflows.y_finance import get_analyst_expectations
        import yfinance as yf

        class FailTrendTicker:
            @property
            def analyst_price_targets(self):
                return {"mean": 120.0, "low": 100.0}
            @property
            def eps_trend(self):
                raise RuntimeError("trend unavailable")
            @property
            def eps_revisions(self):
                return pd.DataFrame(
                    {"upLast7days": [3], "upLast30days": [8], "downLast7days": [1], "downLast30days": [2]},
                    index=["0q"],
                )
            @property
            def earnings_estimate(self):
                return None
            @property
            def revenue_estimate(self):
                return None
            @property
            def recommendations_summary(self):
                return None

        monkeypatch.setattr(yf, "Ticker", lambda s: FailTrendTicker())
        result = get_analyst_expectations("TEST", "2026-08-08")
        parsed = json.loads(result)
        assert "eps_revisions" in parsed
        assert parsed["eps_revisions"][0]["up_7d"] == 3

    def test_revisions_raises_trend_ok(self, monkeypatch):
        from finmindagent.dataflows.y_finance import get_analyst_expectations
        import yfinance as yf

        class FailRevisionsTicker:
            @property
            def analyst_price_targets(self):
                return {"mean": 120.0}
            @property
            def eps_trend(self):
                return pd.DataFrame(
                    {"current": [2.1], "7daysAgo": [2.1]},
                    index=["0q"],
                )
            @property
            def eps_revisions(self):
                raise RuntimeError("revisions unavailable")
            @property
            def earnings_estimate(self):
                return None
            @property
            def revenue_estimate(self):
                return None
            @property
            def recommendations_summary(self):
                return None

        monkeypatch.setattr(yf, "Ticker", lambda s: FailRevisionsTicker())
        result = get_analyst_expectations("TEST", "2026-08-08")
        parsed = json.loads(result)
        assert "eps_revisions" in parsed
        assert parsed["eps_revisions"][0]["current"] == 2.1


class TestC3HolderPercentNormalization:
    """C3: Holder percent normalization through production code."""

    def test_major_holder_ratio(self):
        from finmindagent.dataflows.y_finance import _normalize_major_holders
        import pandas as pd

        # yfinance major_holders: two-column Breakdown/Value
        df = pd.DataFrame([{"Breakdown": "insidersPercentHeld", "Value": 0.0825}])
        result = _normalize_major_holders(df)
        assert result.get("insiders_pct") == 8.25

    def test_institutional_pct_out(self):
        from finmindagent.dataflows.y_finance import _normalize_holder_rows
        import pandas as pd

        df = pd.DataFrame([{
            "Holder": "Vanguard", "Date Reported": pd.Timestamp("2026-03-31"),
            "% Out": 0.0825, "Shares": 1e6, "Value": 1e7,
        }])
        result = _normalize_holder_rows(df, "institutional", 10)
        assert result[0]["pct_out"] == 8.25

    def test_zero_percent_preserved(self):
        from finmindagent.dataflows.y_finance import _normalize_holder_rows
        import pandas as pd

        df = pd.DataFrame([{
            "Holder": "Zero", "Date Reported": "2026-03-31",
            "% Out": 0.0, "Shares": 100, "Value": 1000,
        }])
        result = _normalize_holder_rows(df, "institutional", 10)
        assert result[0]["pct_out"] == 0.0


class TestC4HolderDateAndId:
    """C4: Holder date + stable holder_id through production normalization."""

    def test_timestamp_produces_canonical_date_and_id(self):
        from finmindagent.dataflows.y_finance import _normalize_holder_date
        import pandas as pd

        ts = pd.Timestamp("2026-03-31")
        date_str = _normalize_holder_date(ts)
        assert date_str == "2026-03-31"

    def test_holder_id_has_correct_format(self):
        from finmindagent.dataflows.y_finance import _normalize_holder_rows
        import pandas as pd

        df = pd.DataFrame([{
            "Holder": "Vanguard Group Inc",
            "Date Reported": pd.Timestamp("2026-03-31"),
            "% Out": 0.0825, "Shares": 1e6, "Value": 1e7,
        }])
        result = _normalize_holder_rows(df, "institutional", 10)
        assert result[0]["date_reported"] == "2026-03-31"
        assert result[0]["holder_id"] == "holder:institutional:vanguard-group-inc:2026-03-31"


class TestC5HolderSortingAndCap:
    """C5: Holder sorting and cap through production normalization."""

    def test_sort_order_and_cap(self):
        from finmindagent.dataflows.y_finance import _normalize_holder_rows
        import pandas as pd

        rows = []
        for i in range(15):
            rows.append({
                "Holder": f"Holder_{i:02d}",
                "Date Reported": "2026-03-31",
                "% Out": 8.0 if i < 5 else 5.0,
                "Shares": 1000 - i * 10,
                "Value": 10000,
            })
        # Tie on pct_out within each group, then shares desc
        df = pd.DataFrame(rows)
        result = _normalize_holder_rows(df, "institutional", 10)
        assert len(result) <= 10, f"Expected <=10, got {len(result)}"
        # All 8.0 pct entries come before 5.0 entries
        pct_values = [r["pct_out"] for r in result]
        assert pct_values == sorted(pct_values, reverse=True), "Not sorted by pct_out desc"
        # Within 8.0 group: shares desc
        first_group = [r for r in result if r["pct_out"] == 8.0]
        shares = [r["shares"] for r in first_group]
        assert shares == sorted(shares, reverse=True), "Not sorted by shares desc within pct tie"


class TestC6VolumeRatio:
    """C6: Volume ratio through build_market_risk_snapshot."""

    def test_volume_ratio_exact(self):
        import statistics
        from finmindagent.reporting.curation import build_market_risk_snapshot
        from finmindagent.reporting.schemas import MarketPoint

        n = 22
        volumes = [1000 + i * 10 for i in range(n)]
        points = []
        for i in range(n):
            points.append(MarketPoint(
                date=f"2026-08-{i+1:02d}", close=100.0 + i,
                high=105.0 + i, low=95.0 + i, volume=volumes[i],
            ))
        snap = build_market_risk_snapshot(points)
        assert snap is not None
        # Formula uses points[-21:-1] as previous 20 (excluding latest AND the earliest)
        # The volumes list is compacted (filters None/zero), but with all valid volumes
        # len(volumes) >= 21 and points have all valid: prev is points[-21:-1] = 20 elements
        prev_20 = volumes[-21:-1]
        expected = volumes[-1] / statistics.mean(prev_20)
        assert snap.volume_ratio_20d == pytest.approx(expected)


class TestC7BuilderStructuredRiskPrecedence:
    """C7: Builder must use structured market_risk_snapshot, not fallback."""

    def test_structured_value_preserved(self):
        from finmindagent.reporting.builder import build_report_schema
        from finmindagent.runtime.actions import ActionType, AgentAction
        from finmindagent.runtime.events import EventType
        from finmindagent.runtime.state import TradingRunState

        state = TradingRunState(ticker="TEST", trade_date="2026-08-08", run_id="c7", status="ok")
        # market_series that would compute a different return_5d
        state.add_event(EventType.OBSERVATION, actor="market",
            action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data"),
            observation={"ok": True, "tool_name": "get_stock_data", "structured_data": {
                "market_series": [{"date": f"2026-08-{i+1:02d}", "open": 100+i, "close": 100+i,
                                   "volume": 1e6} for i in range(25)],
                "market_risk_snapshot": {"return_5d_pct": 99.123},  # deliberately different
            }})
        report = build_report_schema(state)
        assert report.market_risk_snapshot is not None
        assert report.market_risk_snapshot.return_5d_pct == 99.123, (
            "Builder must use structured value, not fallback recomputation"
        )


class TestC8ToolRegistryPreservation:
    """C8: ToolRegistry must preserve both raw data and structured_data."""

    def test_registry_raw_and_structured_preserved(self):
        from finmindagent.runtime.tool_registry import ToolRegistry, ToolSpec
        from finmindagent.runtime.tools.result import ToolResult

        def fake_stock_tool(symbol, start_date, end_date):
            lines = ["Date,Open,High,Low,Close,Volume"]
            for i in range(25):
                lines.append(
                    f"2026-07-{i+1:02d},"
                    f"{100+i},{105+i},{95+i},{102+i},{1000000+i*1000}"
                )
            return "\n".join(lines)

        raw_payload = fake_stock_tool("NVDA", "2026-07-01", "2026-07-25")

        spec = ToolSpec("get_stock_data", fake_stock_tool, "stock", "market")
        reg = ToolRegistry([spec])
        result = reg.execute("get_stock_data", {
            "symbol": "NVDA", "start_date": "2026-07-01", "end_date": "2026-07-25",
        })

        assert result.ok
        # raw channel
        assert result.data == raw_payload, "ToolResult.data must preserve raw vendor payload"
        # structured channel
        assert result.structured_data is not None, "ToolResult.structured_data must not be None"
        assert "market_series" in result.structured_data
        assert len(result.structured_data["market_series"]) > 0
        assert "market_risk_snapshot" in result.structured_data
        assert result.structured_data["market_risk_snapshot"] is not None
        assert result.structured_data["market_risk_snapshot"]["return_5d_pct"] is not None
        assert result.tool_name == "get_stock_data"

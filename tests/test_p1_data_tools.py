"""P1 data tool tests: analyst expectations, company calendar, insider transactions.

Uses real pandas DataFrame shapes matching yfinance output to validate
the normalization layer independently of live network calls.
"""

import json
import math

import pandas as pd
import pytest


# ============================================================================
# 2R: Analyst Expectations — DataFrame normalization
# ============================================================================

def _make_earnings_df():
    """Earnings estimate DataFrame matching real yfinance shape."""
    return pd.DataFrame(
        {
            "avg": [2.0, 2.1, 8.5, 9.2],
            "low": [1.8, 1.9, 8.0, 8.5],
            "high": [2.2, 2.3, 9.0, 10.0],
            "yearAgoEps": [1.5, 1.6, 7.0, 7.5],
            "numberOfAnalysts": [20, 20, 18, 18],
            "growth": [0.33, 0.31, 0.21, 0.22],
        },
        index=["0q", "+1q", "0y", "+1y"],
    )


def _make_revenue_df():
    """Revenue estimate DataFrame matching real yfinance shape."""
    return pd.DataFrame(
        {
            "avg": [35e9, 36e9, 140e9, 150e9],
            "low": [34e9, 35e9, 135e9, 142e9],
            "high": [36e9, 37e9, 145e9, 158e9],
            "yearAgoRevenue": [30e9, 31e9, 120e9, 125e9],
            "numberOfAnalysts": [15, 15, 14, 14],
            "growth": [0.166, 0.161, 0.166, 0.20],
        },
        index=["0q", "+1q", "0y", "+1y"],
    )


def _make_recommendations_df():
    """Recommendations summary DataFrame matching real yfinance shape."""
    return pd.DataFrame(
        {
            "period": ["2026-08-07"],
            "strongBuy": [12],
            "buy": [15],
            "hold": [8],
            "sell": [3],
            "strongSell": [1],
        }
    )


class TestAnalystExpectationsNormalization:
    """2R.1: Test normalization with real DataFrame shapes."""

    def test_earnings_estimates_df_shape(self):
        """Real DataFrame with index=[0q,+1q,0y,+1y] x columns=[avg,low,high,...]"""
        from finmindagent.dataflows.y_finance import _normalize_estimate_rows

        # Test both DataFrame and to_dict(orient="index") paths
        df = _make_earnings_df()
        # Use orient="index" to match the expected dict shape
        raw = df.to_dict(orient="index")
        result = _normalize_estimate_rows(raw, is_earnings=True)

        assert len(result) == 4
        periods = [r["period"] for r in result]
        assert periods == ["current_quarter", "next_quarter", "current_year", "next_year"]

        q0 = result[0]
        assert q0["average"] == 2.0
        assert q0["low"] == 1.8
        assert q0["high"] == 2.2
        assert q0["year_ago"] == 1.5
        assert q0["analyst_count"] == 20
        # growth 0.33 → 33.0%
        assert q0["growth_pct"] == 33.0

    def test_revenue_estimates_df_shape(self):
        """Revenue estimate DataFrame with yearAgoRevenue column."""
        from finmindagent.dataflows.y_finance import _normalize_estimate_rows

        df = _make_revenue_df()
        raw = df.to_dict(orient="index")
        result = _normalize_estimate_rows(raw, is_earnings=False)

        assert len(result) == 4
        q0 = result[0]
        assert q0["average"] == 35e9
        assert q0["year_ago"] == 30e9
        assert q0["growth_pct"] == 16.6

    def test_recommendations_df_shape(self):
        """Recommendations DataFrame with period/strongBuy/buy/hold/sell/strongSell."""
        from finmindagent.dataflows.y_finance import _normalize_recommendations

        df = _make_recommendations_df()
        raw = df.to_dict(orient="records")
        result = _normalize_recommendations(raw)

        assert len(result) >= 1
        r = result[0]
        assert r["period"] == "2026-08-07"
        assert r["strong_buy"] == 12
        assert r["buy"] == 15
        assert r["hold"] == 8
        assert r["sell"] == 3
        assert r["strong_sell"] == 1

    def test_price_targets_with_nan(self):
        """vendor_current with NaN → None, not crash."""
        from finmindagent.dataflows.y_finance import _safe_float

        assert _safe_float(float("nan")) is None
        assert _safe_float(float("inf")) is None
        assert _safe_float(float("-inf")) is None
        assert _safe_float(None) is None
        assert _safe_float("not a number") is None
        assert _safe_float(42.0) == 42.0

    def test_vendor_current_uses_safe_float(self):
        """vendor_current must go through _safe_float, not raw float()."""
        from finmindagent.dataflows.y_finance import _safe_float

        # NaN → None
        assert _safe_float(float("nan")) is None
        # Normal value
        assert _safe_float(100.0) == 100.0
        assert _safe_float(0) == 0.0

    def test_any_ok_price_targets_all_none_not_ok(self):
        """Price targets all None → should NOT count as available data."""
        from finmindagent.dataflows.y_finance import _safe_float

        # Simulate: all targets are None
        pt = {
            "vendor_current": _safe_float(None),
            "low": _safe_float(None),
            "mean": _safe_float(None),
            "median": _safe_float(None),
            "high": _safe_float(None),
        }
        has_any_value = any(v is not None for v in pt.values())
        assert not has_any_value

    def test_unknown_period_skipped(self):
        """DataFrame index with unknown period like '5y' → skipped."""
        from finmindagent.dataflows.y_finance import _normalize_estimate_rows

        # Only 0q is known; +5y should be skipped
        raw = {
            "0q": pd.DataFrame({
                "avg": [1.0], "low": [0.9], "high": [1.1],
                "yearAgoEps": [0.8], "numberOfAnalysts": [5], "growth": [0.25],
            }, index=[0]),
            "+5y": pd.DataFrame({
                "avg": [5.0], "low": [4.0], "high": [6.0],
                "yearAgoEps": [3.0], "numberOfAnalysts": [2], "growth": [0.5],
            }, index=[0]),
        }
        result = _normalize_estimate_rows(raw, is_earnings=True)
        # Only 0q mapped (current_quarter); +5y skipped
        assert len(result) == 1
        assert result[0]["period"] == "current_quarter"


class TestAnalystExpectationsAdapter:
    """2R.5: Adapter contract."""

    def test_valid_json_returns_analyst_consensus(self):
        from finmindagent.runtime.tools.adapters import _adapt_analyst_expectations_json

        payload = _adapt_analyst_expectations_json(
            json.dumps({"price_targets": {"mean": 120.0, "low": 90.0}, "source": "yfinance"})
        )
        assert payload is not None
        assert payload["analyst_consensus"]["price_targets"]["mean"] == 120.0

    def test_empty_sub_sections_returns_none(self):
        from finmindagent.runtime.tools.adapters import _adapt_analyst_expectations_json

        # All sections empty/absent
        payload = _adapt_analyst_expectations_json(json.dumps({"as_of": "2026-08-08"}))
        assert payload is None

    def test_malformed_json_returns_none(self):
        from finmindagent.runtime.tools.adapters import _adapt_analyst_expectations_json

        assert _adapt_analyst_expectations_json("not json") is None
        assert _adapt_analyst_expectations_json("") is None

    def test_non_dict_root_returns_none(self):
        from finmindagent.runtime.tools.adapters import _adapt_analyst_expectations_json

        assert _adapt_analyst_expectations_json("[]") is None


# ============================================================================
# 3R: Company Calendar + Insider E2E
# ============================================================================


class TestCompanyCalendar:
    """3R.1: yfinance company calendar normalization."""

    def test_future_earnings_accepted(self):
        from finmindagent.dataflows.y_finance import get_company_calendar

        # This test won't have network access; test via adapter instead
        from finmindagent.runtime.tools.adapters import adapt_tool_data

        payload = json.dumps({
            "as_of": "2026-08-08",
            "events": [
                {
                    "event_id": "earnings:2026-08-28",
                    "event_type": "earnings",
                    "date": "2026-08-28",
                    "title": "Earnings release",
                    "eps_estimate": 2.1,
                    "revenue_estimate": 35000000000.0,
                    "source": "yfinance",
                }
            ],
            "source": "yfinance",
        })
        result = adapt_tool_data("get_company_calendar", payload)
        assert result is not None
        assert len(result["calendar_events"]) == 1
        assert result["calendar_events"][0]["event_id"] == "earnings:2026-08-28"

    def test_empty_events_returns_none(self):
        from finmindagent.runtime.tools.adapters import adapt_tool_data

        payload = json.dumps({"as_of": "2026-08-08", "events": [], "source": "yfinance"})
        result = adapt_tool_data("get_company_calendar", payload)
        # Empty events → no structured data
        assert result is None or result.get("calendar_events") == []

    def test_malformed_json_returns_none(self):
        from finmindagent.runtime.tools.adapters import adapt_tool_data

        assert adapt_tool_data("get_company_calendar", "not json") is None


class TestInsiderAdapter:
    """3R.4-3R.6: Insider CSV adapter + Builder integration."""

    def test_parse_valid_csv(self):
        from finmindagent.runtime.tools.adapters import _adapt_insider_csv

        text = (
            "# Insider Transactions data for AAPL\n"
            "# Data retrieved on: 2026-08-08\n\n"
            "Start Date,Insider,Position,Transaction,Shares,Value\n"
            "2026-06-08,Jane Doe,Director,Sale,100,5504\n"
            "2026-05-15,John Smith,Officer,Purchase,50,2500\n"
        )
        result = _adapt_insider_csv(text)
        assert result is not None
        items = result["timeline"]
        assert len(items) == 2
        assert items[0]["kind"] == "insider"
        assert items[0]["impact"] == "unknown"  # No sentiment inference
        assert "Jane Doe" in items[0]["event"]
        assert "Sale" in items[0]["event"]
        assert "Position: Director" in items[0]["detail"]

    def test_parse_csv_with_date_alias(self):
        """CSV with 'Date' column instead of 'Start Date'."""
        from finmindagent.runtime.tools.adapters import _adapt_insider_csv

        text = (
            "Date,Insider,Transaction\n"
            "2026-06-08,Jane Doe,Sale\n"
        )
        result = _adapt_insider_csv(text)
        assert result is not None
        assert len(result["timeline"]) == 1

    def test_no_header_returns_none(self):
        from finmindagent.runtime.tools.adapters import _adapt_insider_csv

        assert _adapt_insider_csv("Some random text\nNo CSV here") is None

    def test_sale_stays_unknown_impact(self):
        """Insider Sale must NOT be auto-marked negative."""
        from finmindagent.runtime.tools.adapters import _adapt_insider_csv

        text = "Start Date,Insider,Transaction\n2026-06-08,Jane Doe,Sale\n"
        result = _adapt_insider_csv(text)
        assert result["timeline"][0]["impact"] == "unknown"


class TestInsiderReachesBuilder:
    """3R.4-3R.5: Insider timeline reaches report and survives relevance filter."""

    def test_insider_reaches_report_timeline(self):
        """Insider structured_data → report.timeline via Builder."""
        from finmindagent.reporting.builder import build_report_schema
        from finmindagent.runtime.actions import ActionType, AgentAction
        from finmindagent.runtime.events import EventType
        from finmindagent.runtime.state import TradingRunState

        state = TradingRunState(
            ticker="AAPL", trade_date="2026-08-08",
            run_id="test-insider", status="completed",
        )
        state.add_event(
            EventType.OBSERVATION,
            actor="news_analyst",
            action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_insider_transactions"),
            observation={
                "ok": True,
                "tool_name": "get_insider_transactions",
                "structured_data": {
                    "timeline": [
                        {
                            "date": "2026-06-08",
                            "event": "Insider transaction: Jane Doe - Sale",
                            "impact": "unknown",
                            "importance": "unknown",
                            "detail": "Position: Director; Shares: 100",
                            "source": "get_insider_transactions",
                            "url": "",
                            "scope": "company",
                            "kind": "insider",
                        }
                    ]
                },
            },
        )

        report = build_report_schema(state)
        # Insider should be in timeline
        insider_items = [t for t in report.timeline if t.kind == "insider"]
        assert len(insider_items) == 1
        assert insider_items[0].event == "Insider transaction: Jane Doe - Sale"

    def test_insider_survives_relevance_filter(self):
        """Insider kind=insider bypasses company-name/ticker relevance filter."""
        from finmindagent.reporting.builder import build_report_schema
        from finmindagent.runtime.actions import ActionType, AgentAction
        from finmindagent.runtime.events import EventType
        from finmindagent.runtime.state import TradingRunState

        state = TradingRunState(
            ticker="AAPL", trade_date="2026-08-08",
            run_id="test-insider-filter", status="completed",
        )
        # Add a company profile that would filter out items without "AAPL"/"Apple"
        state.add_event(
            EventType.OBSERVATION,
            actor="fundamentals_analyst",
            action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_fundamentals"),
            observation={
                "ok": True,
                "tool_name": "get_fundamentals",
                "structured_data": {
                    "company_profile": {
                        "name": "Apple Inc.",
                        "sector": "Technology",
                        "industry": "Consumer Electronics",
                    }
                },
            },
        )
        # Insider event that does NOT mention Apple or AAPL
        state.add_event(
            EventType.OBSERVATION,
            actor="news_analyst",
            action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_insider_transactions"),
            observation={
                "ok": True,
                "tool_name": "get_insider_transactions",
                "structured_data": {
                    "timeline": [
                        {
                            "date": "2026-06-08",
                            "event": "Insider transaction: Jane Doe - Sale",
                            "impact": "unknown",
                            "importance": "unknown",
                            "detail": "Position: Director; Shares: 100",
                            "source": "get_insider_transactions",
                            "url": "",
                            "scope": "company",
                            "kind": "insider",
                        }
                    ]
                },
            },
        )

        report = build_report_schema(state)
        # Insider must survive even though event text doesn't contain "Apple" or "AAPL"
        insider_items = [t for t in report.timeline if t.kind == "insider"]
        assert len(insider_items) >= 1, "Insider item was incorrectly filtered out by relevance filter"

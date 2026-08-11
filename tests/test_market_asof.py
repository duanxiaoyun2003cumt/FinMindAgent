"""Point-in-time / As-Of market data boundary tests (current-plan v10).

Reproduces the PLTR bug: `propagate("PLTR", "2026-08-10")` produced
`1d 2026-08-10 -> 2026-08-22` forward queries for BOTH PLTR and SPY
(trading_graph._fetch_returns: end = start + timedelta(days=holding_days + 7)).

Contract: NO FUTURE BUSINESS DATA — every consumed bar must satisfy
bar.date <= resolved completed-session cutoff; provider exclusive-end
adaptation (cutoff + 1) is allowed and must not be misjudged as leakage.
"""

import datetime as _dt
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import pytest

from finmindagent.dataflows.market_asof import (
    provider_exclusive_end,
    resolve_market_data_cutoff,
    validate_analysis_date,
)
from finmindagent.time_utils import BEIJING_TZ, US_EASTERN_TZ


def _frozen(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, day, hour, minute, tzinfo=US_EASTERN_TZ)


def _frozen_beijing(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, day, hour, minute, tzinfo=BEIJING_TZ)


def _bar_frame(dates, close=100.0):
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Adj Close": close, "Volume": 1000,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests 1-8 — resolve_market_data_cutoff semantics
# ---------------------------------------------------------------------------

def test_validate_analysis_date_rejects_true_future():
    """Test 3/4 — a truly future analysis date (beyond the SYSTEM date) is
    rejected by Guard A; never clamped."""
    with pytest.raises(ValueError):
        validate_analysis_date("2026-08-12", runtime_now=_frozen_beijing(11, 9))
    with pytest.raises(ValueError):
        validate_analysis_date("2026-08-22", runtime_now=_frozen_beijing(11, 9))


def test_validate_analysis_date_accepts_cross_tz_request():
    """Guard A uses the SYSTEM (Beijing) date: 08-11 is legal while New York
    is still 08-10 — the request must NOT be rejected."""
    result = validate_analysis_date("2026-08-11", runtime_now=_frozen_beijing(11, 9, 24))
    assert result == date(2026, 8, 11)


def test_market_resolver_clamps_cross_tz_skew():
    """Guard B never rejects: min(requested, market-local today)."""
    assert resolve_market_data_cutoff("2026-08-11", as_of_now=_frozen(10, 21, 24)) == date(2026, 8, 10)
    # even a true-future request is clamped (Guard A rejects it upstream)
    assert resolve_market_data_cutoff("2026-08-22", as_of_now=_frozen(10, 21, 24)) == date(2026, 8, 10)
    # pre-open: candidate == market today but session incomplete → previous day
    assert resolve_market_data_cutoff("2026-08-22", as_of_now=_frozen(10, 9)) == date(2026, 8, 9)


def test_historical_requested_date_uses_full_day():
    """Test 7 — historical date keeps its own day regardless of runtime."""
    cutoff = resolve_market_data_cutoff("2026-08-10", as_of_now=_frozen(20, 9))
    assert cutoff == date(2026, 8, 10)


def test_preopen_current_day_uses_previous_session():
    """Test 4 — 2026-08-10 08:57 ET pre-open → cutoff 08-09."""
    cutoff = resolve_market_data_cutoff("2026-08-10", as_of_now=_frozen(10, 8, 57))
    assert cutoff == date(2026, 8, 9)


def test_mid_session_current_day_uses_previous_session():
    """Test 5 — 12:00 ET during session → still previous completed session."""
    cutoff = resolve_market_data_cutoff("2026-08-10", as_of_now=_frozen(10, 12, 0))
    assert cutoff == date(2026, 8, 9)


def test_after_close_current_day_uses_today():
    """Test 6 — after the daily close threshold, today's bar is allowed."""
    cutoff = resolve_market_data_cutoff("2026-08-10", as_of_now=_frozen(10, 17, 0))
    assert cutoff == date(2026, 8, 10)


def test_weekend_requested_keeps_date_bars_resolve_backward():
    """Test 3 — weekend request keeps the calendar date; the bars layer picks
    the last available bar at-or-before it (08-07), never 08-10."""
    assert resolve_market_data_cutoff("2026-08-09", as_of_now=_frozen(10, 9)) == date(2026, 8, 9)


def test_provider_exclusive_end_is_cutoff_plus_one():
    """Test 12 — exclusive-end adaptation (08-11) is legal, not leakage."""
    assert provider_exclusive_end(date(2026, 8, 10)) == date(2026, 8, 11)


# ---------------------------------------------------------------------------
# Tests 2/11/12/13 — get_stock_data (y_finance path) query + consumption
# ---------------------------------------------------------------------------

class _FakeTicker:
    def __init__(self, symbol, history_impl):
        self._symbol = symbol
        self._history_impl = history_impl

    def history(self, **kwargs):
        return self._history_impl(self._symbol, kwargs)


def _patch_ticker(monkeypatch, history_impl):
    import yfinance as yf

    monkeypatch.setattr(
        yf, "Ticker", lambda symbol: _FakeTicker(symbol, history_impl)
    )


def test_get_stock_data_historical_no_forward_query(monkeypatch):
    """Test 2 — requested end 08-10 (historical, runtime later) must not
    produce a future query like 08-10 → 08-22; max provider end = cutoff+1."""
    from finmindagent.dataflows.y_finance import get_YFin_data_online

    monkeypatch.setattr("finmindagent.dataflows.market_asof.market_now",
                        lambda tz=US_EASTERN_TZ: _frozen(20, 9))
    calls = []

    def history_impl(symbol, kwargs):
        calls.append((symbol, kwargs.get("start"), kwargs.get("end")))
        return _bar_frame(["2026-08-07", "2026-08-10"])

    _patch_ticker(monkeypatch, history_impl)
    out = get_YFin_data_online("PLTR", "2026-04-12", "2026-08-10")

    assert calls, "provider must be queried"
    for symbol, start, end in calls:
        assert start == "2026-04-12"
        assert end == "2026-08-11", f"exclusive end must be cutoff+1, got {end}"
        assert end != "2026-08-22"
    # the 08-10 bar is consumed (historical completed session)
    assert "2026-08-10" in out


def test_get_stock_data_preopen_filters_today_bar(monkeypatch):
    """Test 4/11 — pre-open 08-10: provider may return an 08-10 row, but the
    pipeline must NOT consume it (defensive filter)."""
    from finmindagent.dataflows.market_asof import market_now
    from finmindagent.dataflows.y_finance import get_YFin_data_online

    monkeypatch.setattr("finmindagent.dataflows.market_asof.market_now",
                        lambda tz=US_EASTERN_TZ: _frozen(10, 8, 57))
    _patch_ticker(monkeypatch, lambda symbol, kwargs: _bar_frame(["2026-08-07", "2026-08-10"]))
    out = get_YFin_data_online("PLTR", "2026-04-12", "2026-08-10")

    data_rows = [line for line in out.splitlines() if line and not line.startswith("#") and not line.startswith(",")]
    assert [row.split(",")[0] for row in data_rows] == ["2026-08-07"], data_rows


def test_get_stock_data_weekend_uses_previous_session(monkeypatch):
    """Test 3 — requested 08-09 (Saturday): 08-10 bar never consumed."""
    from finmindagent.dataflows.y_finance import get_YFin_data_online

    _patch_ticker(monkeypatch, lambda symbol, kwargs: _bar_frame(["2026-08-06", "2026-08-07", "2026-08-10"]))
    out = get_YFin_data_online("PLTR", "2026-04-01", "2026-08-09")

    assert "2026-08-07" in out
    # only the header line mentions 08-10 (request boundary), never a data row
    data_rows = [line for line in out.splitlines() if line and not line.startswith("#")]
    assert not any(line.startswith("2026-08-10") for line in data_rows)


def test_get_stock_data_future_request_rejected(monkeypatch):
    """Test 3/4 — truly future requested date (beyond the SYSTEM date) still
    raises at the dataflow boundary (Guard A); never clamped."""
    from finmindagent.dataflows.y_finance import get_YFin_data_online

    monkeypatch.setattr("finmindagent.dataflows.market_asof.market_now",
                        lambda tz=US_EASTERN_TZ: _frozen(10, 21, 24))
    monkeypatch.setattr("finmindagent.dataflows.market_asof.now_system",
                        lambda: _frozen_beijing(11, 9, 24))
    with pytest.raises(ValueError):
        get_YFin_data_online("PLTR", "2026-04-01", "2026-08-22")


# ---------------------------------------------------------------------------
# Test 1 (KEY) — cross-timezone request accepted: Beijing 08-11 / NY 08-10
# ---------------------------------------------------------------------------

def test_cross_timezone_request_08_11_accepted(monkeypatch):
    """The MU bug: runtime Beijing 2026-08-11 09:24 (= NY 08-10 21:24),
    requested 08-11 → the request must be ACCEPTED with market data as of
    08-10 (post-close, the 08-10 daily bar is available)."""
    from finmindagent.dataflows.y_finance import get_YFin_data_online

    monkeypatch.setattr("finmindagent.dataflows.market_asof.market_now",
                        lambda tz=US_EASTERN_TZ: _frozen(10, 21, 24))
    monkeypatch.setattr("finmindagent.dataflows.market_asof.now_system",
                        lambda: _frozen_beijing(11, 9, 24))
    calls = []
    _patch_ticker(monkeypatch, lambda symbol, kwargs: (calls.append(kwargs) or _bar_frame(["2026-08-07", "2026-08-10"], close=111.11)))
    out = get_YFin_data_online("MU", "2026-04-12", "2026-08-11")

    # no rejection: tool produced data
    assert "# As-Of: requested_end=2026-08-11" in out
    assert "resolved_market_cutoff=2026-08-10" in out
    # 08-10 bar consumed; no future query
    data_rows = [line for line in out.splitlines() if line and not line.startswith("#") and not line.startswith(",")]
    assert [row.split(",")[0] for row in data_rows] == ["2026-08-07", "2026-08-10"]
    assert calls[0]["end"] == "2026-08-11", f"exclusive end, got {calls[0]}"


def test_get_stock_data_single_query_no_next_day_lookup(monkeypatch):
    """Test 13 — target-day missing: exactly ONE query, no target+1 retry."""
    from finmindagent.dataflows.y_finance import get_YFin_data_online

    calls = []
    _patch_ticker(monkeypatch, lambda symbol, kwargs: (calls.append(kwargs) or _bar_frame(["2026-08-07"])))
    get_YFin_data_online("PLTR", "2026-04-01", "2026-08-10")
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Tests 9/10 — benchmark (_fetch_returns in trading_graph)
# ---------------------------------------------------------------------------

def _patch_yf_ticker(monkeypatch, impl):
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", impl)


def test_fetch_returns_defers_when_window_not_elapsed(monkeypatch):
    """Test 9 — PLTR 08-10 pre-open: outcome window (08-22) is in the future;
    NO query is made for either PLTR or SPY — outcome deferred."""
    from finmindagent.graph.trading_graph import FinMindAgentGraph

    monkeypatch.setattr("finmindagent.dataflows.market_asof.market_now",
                        lambda tz=US_EASTERN_TZ: _frozen(10, 8, 57))
    calls = []

    def fake_ticker(symbol):
        @staticmethod
        def history(**kwargs):
            calls.append((symbol, kwargs))
            return _bar_frame(["2026-08-07"])
        return type("T", (), {"history": history})()

    _patch_yf_ticker(monkeypatch, fake_ticker)
    graph = FinMindAgentGraph.__new__(FinMindAgentGraph)
    raw, alpha, days = graph._fetch_returns("PLTR", "2026-08-10")

    assert raw is None and alpha is None
    assert calls == [], f"no forward query allowed, got {calls}"


def test_fetch_returns_queries_both_with_same_boundary(monkeypatch):
    """Test 9/10 — elapsed window: PLTR and SPY queried with the SAME end;
    future SPY rows are filtered; returns use common <= boundary data."""
    from finmindagent.graph.trading_graph import FinMindAgentGraph

    monkeypatch.setattr("finmindagent.dataflows.market_asof.market_now",
                        lambda tz=US_EASTERN_TZ: _frozen(10, 8, 57))
    calls = []

    def fake_ticker(symbol):
        @staticmethod
        def history(**kwargs):
            calls.append((symbol, kwargs))
            if symbol == "SPY":
                return _bar_frame(["2026-07-10", "2026-07-13", "2026-07-14"], close=400.0)
            return _bar_frame(["2026-07-10", "2026-07-13", "2026-07-14"], close=100.0)
        return type("T", (), {"history": history})()

    _patch_yf_ticker(monkeypatch, fake_ticker)
    graph = FinMindAgentGraph.__new__(FinMindAgentGraph)
    raw, alpha, days = graph._fetch_returns("PLTR", "2026-07-01", holding_days=5)

    assert calls, "outcome query must run for an elapsed window"
    assert calls[0][0] == "PLTR" and calls[1][0] == "SPY"
    assert calls[0][1]["end"] == calls[1][1]["end"], "same as-of boundary for both"
    assert raw is not None
    assert alpha is not None


# ---------------------------------------------------------------------------
# Tests 15/16 — current_price as-of and indicator input truncation
# ---------------------------------------------------------------------------

def test_current_price_is_as_of_price():
    """Test 15 — analysis 08-10 pre-open, bars end 08-07 close=172.01:
    current_price = 172.01 with price_as_of = 2026-08-07 (not runtime live)."""
    from finmindagent.reporting.builder import build_report_schema
    from finmindagent.runtime.actions import ActionType, AgentAction
    from finmindagent.runtime.events import EventType
    from finmindagent.runtime.state import TradingRunState

    state = TradingRunState(ticker="PLTR", trade_date="2026-08-10", run_id="r", status="completed")
    state.add_event(
        EventType.OBSERVATION,
        actor="market_analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data"),
        observation={
            "ok": True, "tool_name": "get_stock_data",
            "structured_data": {"market_series": [
                {"date": "2026-08-07", "close": 172.01, "volume": 1},
                {"date": "2026-08-06", "close": 170.0, "volume": 1},
            ]},
        },
    )
    report = build_report_schema(state)
    assert report.decision.current_price == pytest.approx(172.01)
    assert report.decision.price_as_of == "2026-08-07"


def test_report_discloses_analysis_date_vs_market_as_of():
    """Test 16 — the report shows 分析日期=08-11 AND 行情截至=08-10 side by
    side; the cross-timezone lag is a legal state, never a rejection."""
    from finmindagent.reporting.builder import build_report_schema
    from finmindagent.reporting.html import render_report_html
    from finmindagent.runtime.actions import ActionType, AgentAction
    from finmindagent.runtime.events import EventType
    from finmindagent.runtime.state import TradingRunState

    state = TradingRunState(ticker="MU", trade_date="2026-08-11", run_id="r", status="completed")
    state.add_event(
        EventType.OBSERVATION,
        actor="market_analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data"),
        observation={
            "ok": True, "tool_name": "get_stock_data",
            "structured_data": {"market_series": [
                {"date": "2026-08-07", "close": 108.0, "volume": 1},
                {"date": "2026-08-10", "close": 111.11, "volume": 1},
            ]},
        },
    )
    report = build_report_schema(state)
    html = render_report_html(report, chart_js="", locale="zh-CN")

    assert "分析日期" in html and "2026-08-11" in html
    assert "截至" in html and "2026-08-10" in html
    assert "被拒绝" not in html
    # current_price is the as-of price (Test 17): 08-10 close, never future
    assert report.decision.current_price == pytest.approx(111.11)
    assert report.decision.price_as_of == "2026-08-10"


def test_load_ohlcv_filters_future_and_partial_bars(monkeypatch, tmp_path):
    """Test 16 — indicator input must not contain bars after the cutoff,
    and must not bfill from future rows (pre-open 08-10 → cutoff 08-09)."""
    import yfinance as yf

    from finmindagent.dataflows import config as dataflows_config
    from finmindagent.dataflows.stockstats_utils import load_ohlcv

    monkeypatch.setattr("finmindagent.dataflows.market_asof.market_now",
                        lambda tz=US_EASTERN_TZ: _frozen(10, 8, 57))
    real_config = dataflows_config.get_config()
    monkeypatch.setattr(
        "finmindagent.dataflows.stockstats_utils.get_config",
        lambda: {**real_config, "data_cache_dir": str(tmp_path)},
    )
    frame = _bar_frame(["2026-08-06", "2026-08-07", "2026-08-10", "2026-08-11"])
    # mimic download output (index named Date, then reset_index column)
    frame.index.name = "Date"
    frame = frame.reset_index()
    monkeypatch.setattr(yf, "download", lambda *a, **k: frame)

    data = load_ohlcv("PLTR", "2026-08-10")
    dates = sorted(pd.to_datetime(data["Date"]).dt.date.tolist())
    assert dates == [date(2026, 8, 6), date(2026, 8, 7)], f"got {dates}"
    assert date(2026, 8, 10) not in dates
    assert date(2026, 8, 11) not in dates


def test_historical_determinism_same_cutoff():
    """Test 17 — the same historical analysis date resolves to the same
    cutoff regardless of how late the runtime runs."""
    c1 = resolve_market_data_cutoff("2026-08-10", as_of_now=_frozen(20, 9))
    c2 = resolve_market_data_cutoff("2026-08-10", as_of_now=_frozen(31, 9))  # Sept 1
    assert c1 == c2 == date(2026, 8, 10)

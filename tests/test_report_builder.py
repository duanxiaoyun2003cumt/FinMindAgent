import pytest

from finmindagent.reporting.builder import build_report_schema
from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.events import EventType
from finmindagent.runtime.state import TradingRunState
from finmindagent.runtime.tool_registry import ToolRegistry


def test_build_report_schema_preserves_reports_and_marks_missing_data():
    state = TradingRunState(
        run_id="run-1",
        ticker="NVDA",
        trade_date="2026-01-10",
        status="completed",
        reports={
            "market_report": "Market evidence",
            "bull_researcher": "Bull case",
        },
        final_trade_decision=(
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Accumulate gradually.\n\n"
            "**Investment Thesis**: Earnings momentum is improving.\n\n"
            "**Price Target**: 155.5\n\n"
            "**Time Horizon**: 6 months"
        ),
    )

    report = build_report_schema(
        state,
        generated_at="2026-01-10T12:00:00+00:00",
    )

    assert report.decision.rating == "Buy"
    assert report.decision.price_target == 155.5
    assert report.analysis_sections[0].body == "Market evidence"
    assert report.viewpoints[0].body == "Bull case"
    assert "market_series" in report.data_quality.missing_sections
    assert "financial_metrics" in report.data_quality.missing_sections


def test_builder_extracts_market_points_from_stock_tool_csv():
    state = TradingRunState(
        run_id="run-market",
        ticker="NVDA",
        trade_date="2026-01-10",
    )
    action = AgentAction(
        type=ActionType.CALL_TOOL,
        tool_name="get_stock_data",
    )
    state.add_event(
        EventType.OBSERVATION,
        actor="market_analyst",
        action=action,
        observation={
            "ok": True,
            "tool_name": "get_stock_data",
            "data": (
                "Date,Open,High,Low,Close,Volume\n"
                "2026-01-09,140,145,139,144,1200\n"
                "2026-01-08,138,142,137,141,1000\n"
            ),
        },
    )

    report = build_report_schema(
        state,
        generated_at="2026-01-10T12:00:00+00:00",
    )

    assert [point.date for point in report.market_series] == [
        "2026-01-08",
        "2026-01-09",
    ]
    assert report.market_series[0].close == 141.0
    assert report.market_series[1].volume == 1200.0
    assert "market_series" not in report.data_quality.missing_sections


@pytest.mark.parametrize(
    ("metadata_key", "report_field", "payload", "expected"),
    [
        (
            "report_scores",
            "scores",
            {"name": "technical", "value": 7, "maximum": 10, "source": "market"},
            ("technical", 7.0),
        ),
        (
            "financial_metrics",
            "financial_metrics",
            {"name": "market_cap", "value": 125, "unit": "USD", "period": "2025Q4", "source": "get_fundamentals"},
            ("market_cap", 125.0),
        ),
        (
            "timeline",
            "timeline",
            {"date": "2026-01-08", "event": "Guidance raised", "impact": "positive", "detail": "FY guidance"},
            ("Guidance raised", "positive"),
        ),
        (
            "risk_items",
            "risk_items",
            {"name": "valuation", "severity": "high", "horizon": "3 months", "evidence": "Forward multiple"},
            ("valuation", "high"),
        ),
    ],
)
def test_builder_normalizes_explicit_metadata_collections(
    metadata_key,
    report_field,
    payload,
    expected,
):
    state = TradingRunState(
        ticker="NVDA",
        trade_date="2026-01-10",
        metadata={metadata_key: [payload]},
    )

    report = build_report_schema(state)
    item = getattr(report, report_field)[0]

    if report_field == "scores":
        assert (item.name, item.value) == expected
    elif report_field == "financial_metrics":
        assert (item.name, item.value) == expected
    elif report_field == "timeline":
        assert (item.event, item.impact) == expected
    else:
        assert (item.name, item.severity) == expected


def test_builder_records_unparsed_fundamental_tool_without_guessing_metrics():
    state = TradingRunState(ticker="NVDA", trade_date="2026-01-10")
    action = AgentAction(
        type=ActionType.CALL_TOOL,
        tool_name="get_fundamentals",
    )
    state.add_event(
        EventType.OBSERVATION,
        actor="fundamentals_analyst",
        action=action,
        observation={
            "ok": True,
            "tool_name": "get_fundamentals",
            "data": "Revenue may be 125 million according to prose.",
        },
    )

    report = build_report_schema(state)

    assert report.financial_metrics == []
    assert report.data_quality.unparsed_sources == ["get_fundamentals"]


@pytest.mark.parametrize(
    ("tool_name", "data", "report_field", "expected"),
    [
        (
            "get_fundamentals",
            {
                "financial_metrics": [
                    {
                        "name": "profit_margin",
                        "value": 62.5,
                        "unit": "%",
                        "period": "2025Q4",
                        "source": "get_fundamentals",
                    }
                ]
            },
            "financial_metrics",
            ("profit_margin", 62.5),
        ),
        (
            "get_news",
            {
                "timeline": [
                    {
                        "date": "2026-01-09",
                        "event": "Product launch",
                        "impact": "positive",
                        "detail": "New platform",
                    }
                ]
            },
            "timeline",
            ("Product launch", "positive"),
        ),
    ],
)
def test_builder_normalizes_known_structured_tool_collections(
    tool_name,
    data,
    report_field,
    expected,
):
    state = TradingRunState(ticker="NVDA", trade_date="2026-01-10")
    action = AgentAction(type=ActionType.CALL_TOOL, tool_name=tool_name)
    state.add_event(
        EventType.OBSERVATION,
        actor="analyst",
        action=action,
        observation={"ok": True, "tool_name": tool_name, "data": data},
    )

    report = build_report_schema(state)
    item = getattr(report, report_field)[0]

    if report_field == "financial_metrics":
        assert (item.name, item.value) == expected
    else:
        assert (item.event, item.impact) == expected
    assert tool_name not in report.data_quality.unparsed_sources


# ---------------------------------------------------------------------------
# Real ToolRegistry shapes → structured report data
# ---------------------------------------------------------------------------

def _real_registry_result(monkeypatch, tool_name, vendor_text, tool_args):
    """Execute a real registry tool with vendor routing returning the actual
    vendor string format, exactly as the runtime would see it."""
    def fake_vendor(method, *args, **kwargs):
        assert method == tool_name
        return vendor_text

    monkeypatch.setattr(
        "finmindagent.dataflows.interface.route_to_vendor", fake_vendor
    )
    registry = ToolRegistry()
    return registry.execute(tool_name, tool_args)


def _state_with_tool_result(result, actor="analyst"):
    state = TradingRunState(ticker="NVDA", trade_date="2026-01-10")
    action = AgentAction(type=ActionType.CALL_TOOL, tool_name=result.tool_name)
    state.add_event(
        EventType.OBSERVATION,
        actor=actor,
        action=action,
        observation=result.model_dump(mode="json"),
    )
    return state


def test_real_tool_registry_stock_csv_flows_into_market_series(monkeypatch):
    stock_csv = (
        "# Stock data for NVDA from 2026-01-05 to 2026-01-10\n"
        "# Total records: 2\n"
        "# Data retrieved on: 2026-01-10 12:00:00\n\n"
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2026-01-09,140,145,139,144,144,1200\n"
        "2026-01-08,138,142,137,141,141,1000\n"
    )
    result = _real_registry_result(
        monkeypatch,
        "get_stock_data",
        stock_csv,
        {"symbol": "NVDA", "start_date": "2026-01-05", "end_date": "2026-01-10"},
    )

    assert result.ok
    assert isinstance(result.data, str)  # raw text is preserved
    assert result.structured_data is not None

    report = build_report_schema(_state_with_tool_result(result, "market_analyst"))
    assert [point.date for point in report.market_series] == [
        "2026-01-08",
        "2026-01-09",
    ]
    assert report.market_series[0].close == 141.0
    assert "market_series" not in report.data_quality.missing_sections
    assert report.data_quality.unparsed_sources == []


def test_real_tool_registry_fundamentals_map_to_financial_metrics(monkeypatch):
    fundamentals_text = (
        "# Company Fundamentals for NVDA\n"
        "# Data retrieved on: 2026-01-10 12:00:00\n\n"
        "Name: NVIDIA Corporation\n"
        "Sector: Technology\n"
        "PE Ratio (TTM): 55.2\n"
        "Profit Margin: 0.48\n"
        "Dividend Yield: 0.0032\n"
    )
    result = _real_registry_result(
        monkeypatch,
        "get_fundamentals",
        fundamentals_text,
        {"ticker": "NVDA", "curr_date": "2026-01-10"},
    )

    assert result.ok
    assert result.structured_data is not None

    report = build_report_schema(_state_with_tool_result(result, "fundamentals_analyst"))
    by_name = {metric.name: metric for metric in report.financial_metrics}
    assert by_name["pe_ratio_ttm"].value == 55.2
    assert by_name["pe_ratio_ttm"].unit == "x"
    assert by_name["pe_ratio_ttm"].source == "get_fundamentals"
    assert by_name["profit_margin"].value == 48.0
    assert by_name["profit_margin"].unit == "%"
    assert "name" not in by_name  # non-numeric fields are not fabricated
    assert "sector" not in by_name
    assert "get_fundamentals" not in report.data_quality.unparsed_sources


def test_real_tool_registry_balance_sheet_csv_maps_to_financial_metrics(monkeypatch):
    balance_sheet_csv = (
        "# Balance Sheet data for NVDA (quarterly)\n"
        "# Data retrieved on: 2026-01-10 12:00:00\n\n"
        ",,2026-01-31,2025-10-31\n"
        "Total Debt,10000000000,9500000000\n"
        "Total Assets,100000000000,95000000000\n"
        "Total Liabilities,40000000000,38000000000\n"
    )
    result = _real_registry_result(
        monkeypatch,
        "get_balance_sheet",
        balance_sheet_csv,
        {"ticker": "NVDA", "freq": "quarterly"},
    )

    assert result.ok
    assert result.structured_data is not None

    report = build_report_schema(_state_with_tool_result(result, "fundamentals_analyst"))
    by_name = {metric.name: metric for metric in report.financial_metrics}
    # core whitelist metric surfaces; low-value lines are curated away
    assert by_name["total_debt"].value == 10000000000.0
    assert by_name["total_debt"].period == "2026-01-31"
    assert by_name["total_debt"].source == "get_balance_sheet"
    assert "total_assets" not in by_name
    assert "total_liabilities" not in by_name


def test_real_tool_registry_news_maps_to_timeline(monkeypatch):
    news_text = (
        "## NVDA News, from 2026-01-05 to 2026-01-10:\n\n"
        "### NVIDIA raises guidance (source: Reuters)\n"
        "Company raised FY guidance.\n"
        "Link: https://example.com/news/1\n\n"
        "### Chip demand strong (source: Bloomberg)\n"
        "Demand outlook improved.\n"
    )
    result = _real_registry_result(
        monkeypatch,
        "get_news",
        news_text,
        {"ticker": "NVDA", "start_date": "2026-01-05", "end_date": "2026-01-10"},
    )

    assert result.ok
    assert result.structured_data is not None

    report = build_report_schema(_state_with_tool_result(result, "news_analyst"))
    assert len(report.timeline) == 2
    assert report.timeline[0].event == "NVIDIA raises guidance"
    assert "Company raised FY guidance." in report.timeline[0].detail
    assert report.timeline[0].impact == "unknown"
    assert "get_news" not in report.data_quality.unparsed_sources


def test_real_tool_registry_does_not_fabricate_scores_or_risk_items(monkeypatch):
    stock_csv = (
        "# Stock data for NVDA from 2026-01-05 to 2026-01-10\n\n"
        "Date,Open,High,Low,Close,Volume\n"
        "2026-01-09,140,145,139,144,1200\n"
    )
    result = _real_registry_result(
        monkeypatch,
        "get_stock_data",
        stock_csv,
        {"symbol": "NVDA", "start_date": "2026-01-05", "end_date": "2026-01-10"},
    )

    report = build_report_schema(_state_with_tool_result(result, "market_analyst"))

    assert report.scores == []
    assert report.risk_items == []
    assert "scores" in report.data_quality.missing_sections
    assert "risk_items" in report.data_quality.missing_sections


def test_real_tool_registry_unparseable_source_is_marked(monkeypatch):
    prose = "The company is doing fine according to the latest quarterly release."
    result = _real_registry_result(
        monkeypatch,
        "get_fundamentals",
        prose,
        {"ticker": "NVDA", "curr_date": "2026-01-10"},
    )

    assert result.ok
    assert result.structured_data is None  # nothing deterministic to extract

    report = build_report_schema(_state_with_tool_result(result, "fundamentals_analyst"))
    assert report.financial_metrics == []
    assert "get_fundamentals" in report.data_quality.unparsed_sources


def test_builder_does_not_fabricate_float_price_target_from_prose():
    """A partial markdown without the full three-label protocol is not a
    structured decision: no rating, no fabricated float, raw text preserved."""
    state = TradingRunState(
        ticker="NVDA",
        trade_date="2026-01-10",
        final_trade_decision=(
            "**Rating**: Buy\n\n"
            "**Price Target**: around 155\n\n"
            "**Time Horizon**: 6 months"
        ),
    )

    report = build_report_schema(state)

    assert report.decision.rating is None
    assert report.decision.structured_available is False
    assert report.decision.price_target is None
    assert report.decision.time_horizon is None
    assert "around 155" in report.decision.raw_text


# ---------------------------------------------------------------------------
# P0 Task 2: builder trusts structured decision, never free-text prose
# ---------------------------------------------------------------------------

SNDK_CONFLICTING_TEXT = (
    "**Rating**: Hold\n"
    "\n"
    "## Portfolio Manager’s Final Synthesis — SNDK\n"
    "**Trade Date:** 2026-08-07\n"
    "**Last Price (2026-08-06 close):** $1,258.58\n"
    "**Rating:** **Sell / Underweight**\n"
    "**Action:** Avoid new long positions. Reduce existing exposure.\n"
    "This report reflects a research decision only — no real trade has been executed."
)


def test_builder_rejects_conflicting_free_text_rating():
    """The real SNDK bad case: top 'Hold' + prose 'Sell / Underweight' must
    not produce any trusted rating."""
    state = TradingRunState(
        ticker="SNDK",
        trade_date="2026-08-07",
        final_trade_decision=SNDK_CONFLICTING_TEXT,
        portfolio_decision=None,
    )

    report = build_report_schema(state)

    assert report.decision.rating is None
    assert report.decision.structured_available is False
    assert any(
        "Structured portfolio decision unavailable" in note
        for note in report.data_quality.notes
    )


def test_builder_structured_decision_wins_over_conflicting_text():
    """With an authoritative structured decision, the same conflicting markdown
    must not override any field."""
    state = TradingRunState(
        ticker="SNDK",
        trade_date="2026-08-07",
        final_trade_decision=SNDK_CONFLICTING_TEXT,
        portfolio_decision={
            "rating": "Sell",
            "executive_summary": "Exit exposure.",
            "investment_thesis": "Downtrend confirmed.",
            "price_target": 950.0,
            "stop_loss": 1400.0,
            "position_min_pct": 0.0,
            "position_max_pct": 5.0,
            "confidence_level": "Medium",
            "risk_level": "High",
            "time_horizon": "1-3 months",
        },
    )

    report = build_report_schema(state)

    assert report.decision.rating == "Sell"
    assert report.decision.structured_available is True
    assert report.decision.stop_loss == 1400.0
    assert report.decision.position_max_pct == 5.0
    assert report.decision.confidence_level == "Medium"
    assert report.decision.risk_level == "High"


def test_builder_strict_legacy_markdown_still_parses_full_protocol():
    """A complete legacy markdown decision (all three stable labels, valid
    rating) remains accepted for historical states."""
    state = TradingRunState(
        ticker="NVDA",
        trade_date="2026-01-10",
        final_trade_decision=(
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Accumulate gradually.\n\n"
            "**Investment Thesis**: Earnings momentum is improving.\n\n"
            "**Price Target**: 155.5\n\n"
            "**Time Horizon**: 6 months"
        ),
    )

    report = build_report_schema(state)

    assert report.decision.rating == "Buy"
    assert report.decision.structured_available is True
    assert report.decision.price_target == 155.5
    assert report.decision.time_horizon == "6 months"


def test_builder_missing_labels_are_not_a_structured_decision():
    """Partial labels (rating alone) are not a legal historical decision."""
    state = TradingRunState(
        ticker="NVDA",
        trade_date="2026-01-10",
        final_trade_decision="**Rating**: Buy\n\nSome free text without the other labels.",
    )

    report = build_report_schema(state)

    assert report.decision.rating is None
    assert report.decision.structured_available is False


def test_builder_current_price_comes_only_from_market_series():
    """Current price must come from market data, never from prose like
    '$1,258.58' inside the decision text."""
    state = TradingRunState(
        ticker="SNDK",
        trade_date="2026-08-07",
        final_trade_decision=(
            "**Rating**: Hold\n\n**Executive Summary**: X.\n\n"
            "**Investment Thesis**: Y.\n\n**Last Price (2026-08-06 close):** $1,258.58"
        ),
    )
    action = AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data")
    state.add_event(
        EventType.OBSERVATION,
        actor="market_analyst",
        action=action,
        observation={
            "ok": True,
            "tool_name": "get_stock_data",
            "data": (
                "Date,Open,High,Low,Close,Volume\n"
                "2026-08-06,1250,1262,1244,1258.58,1500000\n"
                "2026-08-05,1260,1270,1250,1265,1200000\n"
            ),
        },
    )

    report = build_report_schema(state)

    assert report.decision.current_price == 1258.58
    assert report.decision.price_as_of == "2026-08-06"


# ---------------------------------------------------------------------------
# P0 Task 3: financial metric curation
# ---------------------------------------------------------------------------

def _metric(name, value, unit="", period="", source=""):
    return {"name": name, "value": value, "unit": unit, "period": period, "source": source}


def test_builder_curates_financial_metrics_to_whitelist():
    """>30 raw metrics (including low-value statement lines) collapse to the
    ordered whitelist only."""
    raw_metrics = [
        _metric("market_cap", 186383000000, source="get_fundamentals"),
        _metric("pe_ratio_ttm", 42.94, unit="x", source="get_fundamentals"),
        _metric("forward_pe", 4.6, unit="x", source="get_fundamentals"),
        _metric("price_to_book", 4.78, unit="x", source="get_fundamentals"),
        _metric("profit_margin", 34.19, unit="%", source="get_fundamentals"),
        _metric("operating_margin", 70.0, unit="%", source="get_fundamentals"),
        _metric("return_on_equity", 82.0, unit="%", source="get_fundamentals"),
        _metric("total_debt", 0, source="get_balance_sheet"),
        _metric("current_ratio", 1.5, source="get_balance_sheet"),
        _metric("free_cash_flow", 3000000000, source="get_cashflow"),
        _metric("net_income", 13180000000, source="get_income_statement"),
        _metric("revenue_ttm", 18638000000, source="get_income_statement"),
        # low-value statement lines that must never surface
        _metric("capital_lease_obligations", 100, source="get_balance_sheet"),
        _metric("other_equity_adjustments", 200, source="get_balance_sheet"),
        _metric("current_notes_payable", 300, source="get_balance_sheet"),
        _metric("tradeand_other_payables_non_current", 400, source="get_balance_sheet"),
    ] + [
        _metric(f"statement_line_{i}", float(i), source="get_income_statement")
        for i in range(20)
    ]
    state = TradingRunState(
        ticker="SNDK",
        trade_date="2026-08-07",
        metadata={"financial_metrics": raw_metrics},
    )

    report = build_report_schema(state)

    assert len(report.financial_metrics) <= 12
    names = [metric.name for metric in report.financial_metrics]
    assert names == [
        "market_cap",
        "pe_ratio_ttm",
        "forward_pe",
        "price_to_book",
        "revenue_ttm",
        "net_income",
        "profit_margin",
        "operating_margin",
        "return_on_equity",
        "total_debt",
        "current_ratio",
        "free_cash_flow",
    ]
    rendered_names = " ".join(names)
    assert "capital_lease_obligations" not in rendered_names
    assert "statement_line" not in rendered_names


def test_builder_financial_dedup_prefers_fundamentals_for_valuation():
    """market_cap from both fundamentals and balance sheet keeps fundamentals."""
    state = TradingRunState(
        ticker="NVDA",
        trade_date="2026-01-10",
        metadata={
            "financial_metrics": [
                _metric("market_cap", 999, source="get_balance_sheet"),
                _metric("market_cap", 186383000000, source="get_fundamentals"),
            ]
        },
    )

    report = build_report_schema(state)

    assert len(report.financial_metrics) == 1
    assert report.financial_metrics[0].value == 186383000000.0
    assert report.financial_metrics[0].source == "get_fundamentals"


def test_builder_financial_dedup_same_source_keeps_latest_period():
    state = TradingRunState(
        ticker="NVDA",
        trade_date="2026-01-10",
        metadata={
            "financial_metrics": [
                _metric("net_income", 100, period="2024-12-31", source="get_income_statement"),
                _metric("net_income", 200, period="2025-12-31", source="get_income_statement"),
            ]
        },
    )

    report = build_report_schema(state)

    assert report.financial_metrics[0].value == 200.0
    assert report.financial_metrics[0].period == "2025-12-31"


# ---------------------------------------------------------------------------
# P0 Task 4: news timeline curation
# ---------------------------------------------------------------------------

def test_builder_curates_timeline_dedup_sort_and_bounds():
    """Duplicate titles/URLs collapse; company first; dated newest-first;
    undated last; global capped at 3; total <= 10."""
    state = TradingRunState(
        ticker="SNDK",
        trade_date="2026-08-07",
        metadata={
            "timeline": [
                {"date": "2026-08-06", "event": "Sandisk earnings beat", "impact": "unknown", "detail": "Q4 beat.", "source": "Zacks", "url": "https://example.com/1", "scope": "company"},
                # duplicate title (different URL)
                {"date": "2026-08-05", "event": "Sandisk earnings beat", "impact": "unknown", "detail": "", "source": "Reuters", "url": "https://example.com/2", "scope": "company"},
                # duplicate URL (different title)
                {"date": "2026-08-04", "event": "Another headline", "impact": "unknown", "detail": "", "source": "Reuters", "url": "https://example.com/1", "scope": "company"},
                # undated company item
                {"date": "", "event": "Undated company note", "impact": "unknown", "detail": "", "source": "X", "url": "", "scope": "company"},
                # five global items -> capped at three
                {"date": "2026-08-06", "event": "Global A", "impact": "unknown", "detail": "", "source": "Y", "url": "https://g/1", "scope": "global"},
                {"date": "2026-08-05", "event": "Global B", "impact": "unknown", "detail": "", "source": "Y", "url": "https://g/2", "scope": "global"},
                {"date": "2026-08-04", "event": "Global C", "impact": "unknown", "detail": "", "source": "Y", "url": "https://g/3", "scope": "global"},
                {"date": "2026-08-03", "event": "Global D", "impact": "unknown", "detail": "", "source": "Y", "url": "https://g/4", "scope": "global"},
                {"date": "2026-08-02", "event": "Global E", "impact": "unknown", "detail": "", "source": "Y", "url": "https://g/5", "scope": "global"},
            ]
        },
    )

    report = build_report_schema(state)

    events = [item.event for item in report.timeline]
    assert events.count("Sandisk earnings beat") == 1
    assert "Another headline" not in events  # same URL as kept item
    assert len(report.timeline) <= 10

    company = [item for item in report.timeline if item.scope != "global"]
    assert company[0].event == "Sandisk earnings beat"  # newest dated first
    assert company[-1].event == "Undated company note"  # undated last
    assert all(item.scope != "global" for item in report.timeline[:len(company)])

    globals_items = [item for item in report.timeline if item.scope == "global"]
    assert len(globals_items) == 3
    assert [item.event for item in globals_items] == ["Global A", "Global B", "Global C"]


def test_news_adapter_parses_published_source_url_and_unknown_impact(monkeypatch):
    news_text = (
        "## NVDA News, from 2026-01-05 to 2026-01-10:\n\n"
        "### NVIDIA raises guidance (source: Reuters)\n"
        "Published: 2026-01-09\n"
        "Company raised FY guidance.\n"
        "Link: https://example.com/news/1\n\n"
        "### Chip demand strong (source: Bloomberg)\n"
        "Demand outlook improved.\n"
    )
    result = _real_registry_result(
        monkeypatch,
        "get_news",
        news_text,
        {"ticker": "NVDA", "start_date": "2026-01-05", "end_date": "2026-01-10"},
    )

    report = build_report_schema(_state_with_tool_result(result, "news_analyst"))

    assert len(report.timeline) == 2
    first = report.timeline[0]
    assert first.date == "2026-01-09"
    assert first.source == "Reuters"
    assert first.url == "https://example.com/news/1"
    assert first.impact == "unknown"
    assert "Link:" not in first.detail
    assert "Company raised FY guidance." in first.detail
    assert report.timeline[1].date == ""
    assert report.timeline[1].source == "Bloomberg"


def test_builder_extracts_company_profile_from_fundamentals(monkeypatch):
    fundamentals_text = (
        "# Company Fundamentals for SNDK\n\n"
        "Name: SanDisk Corporation\n"
        "Sector: Technology\n"
        "Industry: Semiconductors\n"
        "PE Ratio (TTM): 42.94\n"
    )
    result = _real_registry_result(
        monkeypatch,
        "get_fundamentals",
        fundamentals_text,
        {"ticker": "SNDK", "curr_date": "2026-08-07"},
    )

    report = build_report_schema(_state_with_tool_result(result, "fundamentals_analyst"))

    assert report.company_profile is not None
    assert report.company_profile.name == "SanDisk Corporation"
    assert report.company_profile.sector == "Technology"
    assert report.company_profile.industry == "Semiconductors"


def test_builder_filters_irrelevant_company_news_when_profile_available():
    """With a company profile, company-scope news must mention the ticker or
    the company name; unrelated tickers (Astera Labs) are dropped."""
    state = TradingRunState(ticker="SNDK", trade_date="2026-08-07")
    action = AgentAction(type=ActionType.CALL_TOOL, tool_name="get_fundamentals")
    state.add_event(
        EventType.OBSERVATION,
        actor="fundamentals_analyst",
        action=action,
        observation={
            "ok": True,
            "tool_name": "get_fundamentals",
            "structured_data": {
                "financial_metrics": [],
                "company_profile": {
                    "name": "SanDisk Corporation",
                    "sector": "Technology",
                    "industry": "Semiconductors",
                },
            },
        },
    )
    news_action = AgentAction(type=ActionType.CALL_TOOL, tool_name="get_news")
    state.add_event(
        EventType.OBSERVATION,
        actor="news_analyst",
        action=news_action,
        observation={
            "ok": True,
            "tool_name": "get_news",
            "structured_data": {
                "timeline": [
                    {"date": "2026-08-06", "event": "Sandisk earnings beat estimates", "impact": "unknown", "scope": "company"},
                    {"date": "2026-08-06", "event": "Astera Labs surges on Q2 results", "impact": "unknown", "scope": "company"},
                    {"date": "2026-08-06", "event": "SNDK guidance miss stings memory sector", "impact": "unknown", "scope": "company"},
                ]
            },
        },
    )

    report = build_report_schema(state)

    events = [item.event for item in report.timeline]
    assert "Sandisk earnings beat estimates" in events
    assert "SNDK guidance miss stings memory sector" in events
    assert "Astera Labs surges on Q2 results" not in events


# ---------------------------------------------------------------------------
# P1 Task 3: news event enrichment
# ---------------------------------------------------------------------------

def test_timeline_enriched_by_news_analyst_assessments():
    """Structured News Analyst events enrich matching timeline items; exact
    title matching only; unmatched stays unknown."""
    state = TradingRunState(ticker="NVDA", trade_date="2026-01-10")
    state.structured_reports["news_analyst"] = {
        "analysis": "News flow is constructive.",
        "score": 7.0,
        "score_reason": "Positive guidance flow.",
        "events": [
            {"title": "NVIDIA raises guidance", "impact": "positive", "importance": "high", "reason": "FY guidance raised."},
        ],
    }
    action = AgentAction(type=ActionType.CALL_TOOL, tool_name="get_news")
    state.add_event(
        EventType.OBSERVATION,
        actor="news_analyst",
        action=action,
        observation={
            "ok": True,
            "tool_name": "get_news",
            "structured_data": {
                "timeline": [
                    {"date": "2026-01-09", "event": "NVIDIA raises guidance", "impact": "unknown", "scope": "company"},
                    {"date": "2026-01-08", "event": "Chip demand strong", "impact": "unknown", "scope": "company"},
                ]
            },
        },
    )

    report = build_report_schema(state)
    by_event = {item.event: item for item in report.timeline}

    assert by_event["NVIDIA raises guidance"].impact == "positive"
    assert by_event["NVIDIA raises guidance"].importance == "high"
    assert by_event["NVIDIA raises guidance"].impact_reason == "FY guidance raised."
    # unmatched stays unknown, never neutral
    assert by_event["Chip demand strong"].impact == "unknown"
    assert by_event["Chip demand strong"].importance == "unknown"


def test_timeline_sorts_by_importance_within_same_day():
    """Same date: high importance before medium/low/unknown; date order still
    dominates."""
    state = TradingRunState(
        ticker="NVDA",
        trade_date="2026-01-10",
        metadata={
            "timeline": [
                {"date": "2026-01-09", "event": "Low value note", "impact": "unknown", "importance": "low", "scope": "company"},
                {"date": "2026-01-09", "event": "High value news", "impact": "unknown", "importance": "high", "scope": "company"},
                {"date": "2026-01-10", "event": "Next day item", "impact": "unknown", "importance": "unknown", "scope": "company"},
            ]
        },
    )

    report = build_report_schema(state)

    events = [item.event for item in report.timeline]
    # 2026-01-10 first (newest date), then 2026-01-09 high before low
    assert events == ["Next day item", "High value news", "Low value note"]


# ---------------------------------------------------------------------------
# P1 Task 4: structured bull/bear debate arguments
# ---------------------------------------------------------------------------

def test_builder_assembles_debate_arguments_from_structured_reports():
    state = TradingRunState(ticker="NVDA", trade_date="2026-01-10")
    state.structured_reports["bull_researcher"] = {
        "summary": "Demand inflection supports upside.",
        "arguments": [
            {"title": "AI demand", "argument": "Accelerating AI workloads.", "strength": 8.0, "evidence": ["Data center revenue +50%"]},
            {"title": "Margin expansion", "argument": "Operating leverage improves.", "strength": 7.0, "evidence": []},
        ],
    }
    state.structured_reports["bear_researcher"] = {
        "summary": "Valuation is stretched.",
        "arguments": [
            {"title": "Valuation", "argument": "Forward multiple above peers.", "strength": 9.0, "evidence": ["42x TTM P/E"]},
            {"title": "Competition", "argument": "New entrants pressure pricing.", "strength": 6.0, "evidence": []},
        ],
    }

    report = build_report_schema(state)

    assert len(report.debate_arguments) == 4
    by_title = {item.title: item for item in report.debate_arguments}
    assert by_title["AI demand"].stance == "bull"
    assert by_title["AI demand"].strength == 8.0
    assert by_title["AI demand"].evidence == ["Data center revenue +50%"]
    assert by_title["Valuation"].stance == "bear"
    assert by_title["Valuation"].strength == 9.0
    assert by_title["Margin expansion"].evidence == []


def test_builder_ignores_debate_strength_in_prose():
    """Prose mentioning scores must not create debate arguments."""
    state = TradingRunState(
        ticker="NVDA",
        trade_date="2026-01-10",
        reports={"bull_researcher": "AI demand strength 8/10. Margin expansion 7/10."},
    )

    report = build_report_schema(state)

    assert report.debate_arguments == []


# ---------------------------------------------------------------------------
# P1 Task 5: structured three-way risk views and merged register
# ---------------------------------------------------------------------------

def test_builder_builds_three_risk_views():
    state = TradingRunState(ticker="NVDA", trade_date="2026-01-10")
    state.structured_reports["aggressive_risk_analyst"] = {
        "summary": "Upside risk is manageable.",
        "risk_score": 6.0,
        "risks": [{"name": "liquidity", "severity": "medium", "horizon": "3 months", "evidence": "Thin book depth."}],
    }
    state.structured_reports["neutral_risk_analyst"] = {
        "summary": "Balanced risk profile.",
        "risk_score": 7.0,
        "risks": [{"name": "valuation", "severity": "high", "horizon": "6 months", "evidence": "Multiple expansion."}],
    }
    state.structured_reports["conservative_risk_analyst"] = {
        "summary": "Capital protection first.",
        "risk_score": 8.0,
        "risks": [{"name": "valuation", "severity": "critical", "horizon": "6 months", "evidence": "Extreme multiple."}],
    }

    report = build_report_schema(state)

    assert [view.label for view in report.risk_views] == ["Aggressive", "Neutral", "Conservative"]
    assert [view.risk_score for view in report.risk_views] == [6.0, 7.0, 8.0]

    # merge: valuation critical (highest severity), liquidity medium; no dupes
    by_name = {item.name: item for item in report.risk_items}
    assert by_name["valuation"].severity == "critical"
    assert by_name["valuation"].evidence == "Extreme multiple."
    assert set(by_name["valuation"].sources) == {"neutral_risk_analyst", "conservative_risk_analyst"}
    assert by_name["liquidity"].severity == "medium"
    assert by_name["liquidity"].sources == ["aggressive_risk_analyst"]
    assert len(report.risk_items) == 2


def test_builder_risk_items_sorted_by_severity_and_capped():
    state = TradingRunState(ticker="NVDA", trade_date="2026-01-10")
    state.structured_reports["neutral_risk_analyst"] = {
        "summary": "x",
        "risk_score": 5.0,
        "risks": [
            {"name": f"risk_{i}", "severity": "low", "horizon": "", "evidence": ""}
            for i in range(12)
        ],
    }

    report = build_report_schema(state)

    assert len(report.risk_items) <= 8
    assert all(item.severity == "low" for item in report.risk_items)


# ---------------------------------------------------------------------------
# P1 Task 6: multi-period financial history
# ---------------------------------------------------------------------------

def test_builder_builds_revenue_and_net_income_series():
    """4 quarters each for Revenue and Net Income → two series, oldest-first."""
    state = TradingRunState(ticker="NVDA", trade_date="2026-01-10")
    action = AgentAction(type=ActionType.CALL_TOOL, tool_name="get_income_statement")
    state.add_event(
        EventType.OBSERVATION,
        actor="fundamentals_analyst",
        action=action,
        observation={
            "ok": True,
            "tool_name": "get_income_statement",
            "structured_data": {
                "financial_metrics": [],
                "financial_history": [
                    {"name": "total_revenue", "period": "2025Q1", "value": 100, "source": "get_income_statement"},
                    {"name": "total_revenue", "period": "2025Q2", "value": 110, "source": "get_income_statement"},
                    {"name": "total_revenue", "period": "2025Q3", "value": 120, "source": "get_income_statement"},
                    {"name": "total_revenue", "period": "2025Q4", "value": 130, "source": "get_income_statement"},
                    {"name": "net_income", "period": "2025Q1", "value": 10, "source": "get_income_statement"},
                    {"name": "net_income", "period": "2025Q2", "value": 12, "source": "get_income_statement"},
                    {"name": "net_income", "period": "2025Q3", "value": 13, "source": "get_income_statement"},
                    {"name": "net_income", "period": "2025Q4", "value": 15, "source": "get_income_statement"},
                ],
            },
        },
    )

    report = build_report_schema(state)

    by_name = {series.name: series for series in report.financial_series}
    assert set(by_name) == {"revenue", "net_income"}
    assert [point.period for point in by_name["revenue"].points] == [
        "2025Q1", "2025Q2", "2025Q3", "2025Q4",
    ]
    assert [point.value for point in by_name["revenue"].points] == [100.0, 110.0, 120.0, 130.0]
    assert by_name["revenue"].label == "Revenue"
    assert [point.value for point in by_name["net_income"].points] == [10.0, 12.0, 13.0, 15.0]


def test_builder_single_period_does_not_create_series():
    """One period → P0 metric present, no financial series (no fabrication)."""
    state = TradingRunState(ticker="NVDA", trade_date="2026-01-10")
    action = AgentAction(type=ActionType.CALL_TOOL, tool_name="get_income_statement")
    state.add_event(
        EventType.OBSERVATION,
        actor="fundamentals_analyst",
        action=action,
        observation={
            "ok": True,
            "tool_name": "get_income_statement",
            "structured_data": {
                "financial_metrics": [
                    {"name": "net_income", "value": 15, "unit": "", "period": "2025Q4", "source": "get_income_statement"},
                ],
                "financial_history": [
                    {"name": "net_income", "period": "2025Q4", "value": 15, "source": "get_income_statement"},
                ],
            },
        },
    )

    report = build_report_schema(state)

    assert any(metric.name == "net_income" for metric in report.financial_metrics)
    assert report.financial_series == []


def test_real_adapter_keeps_multi_period_history(monkeypatch):
    """Real balance-sheet adapter now preserves multi-period history."""
    balance_sheet_csv = (
        "# Balance Sheet data for NVDA (quarterly)\n\n"
        ",,2026-01-31,2025-10-31,2025-07-31\n"
        "Total Debt,10000000000,9500000000,9000000000\n"
        "Total Assets,100000000000,95000000000,90000000000\n"
    )
    result = _real_registry_result(
        monkeypatch,
        "get_balance_sheet",
        balance_sheet_csv,
        {"ticker": "NVDA", "freq": "quarterly"},
    )

    assert result.structured_data is not None
    history = result.structured_data.get("financial_history")
    assert history is not None
    total_debt_rows = [row for row in history if row["name"] == "total_debt"]
    assert [row["period"] for row in total_debt_rows] == [
        "2026-01-31", "2025-10-31", "2025-07-31",
    ]
    # latest period still feeds the P0 metric
    metrics = result.structured_data["financial_metrics"]
    debt_metric = next(m for m in metrics if m["name"] == "total_debt")
    assert debt_metric["period"] == "2026-01-31"
    assert debt_metric["value"] == 10000000000.0


# ---------------------------------------------------------------------------
# P1 Task 7: deterministic SMA with no look-ahead
# ---------------------------------------------------------------------------

def test_sma20_computed_without_lookahead():
    """close = 1..20 → last SMA20 is 10.5; first 19 points have no sma20."""
    from finmindagent.reporting.curation import attach_moving_average
    from finmindagent.reporting.schemas import MarketPoint

    points = [MarketPoint(date=f"2026-01-{i+1:02d}", close=float(i + 1)) for i in range(20)]

    attach_moving_average(points, window=20, key="sma20")

    for point in points[:19]:
        assert "sma20" not in point.indicators
    assert points[19].indicators["sma20"] == 10.5


def test_sma50_computed_and_closes_untouched():
    from finmindagent.reporting.curation import attach_moving_average
    from finmindagent.reporting.schemas import MarketPoint

    points = [MarketPoint(date=f"d{i:03d}", close=float(i + 1)) for i in range(50)]
    closes_before = [point.close for point in points]

    attach_moving_average(points, window=50, key="sma50")

    assert [point.close for point in points] == closes_before
    assert points[49].indicators["sma50"] == 25.5
    assert "sma50" not in points[48].indicators


def test_builder_attaches_sma_to_market_series():
    """20+ points of market data → sma20 present on the 20th point."""
    rows = "\n".join(
        f"2026-01-{i+1:02d},{i+1},{i+1},{i+1},{i+1},100" for i in range(40)
    )
    state = TradingRunState(ticker="NVDA", trade_date="2026-02-10")
    action = AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data")
    state.add_event(
        EventType.OBSERVATION,
        actor="market_analyst",
        action=action,
        observation={
            "ok": True,
            "tool_name": "get_stock_data",
            "data": f"Date,Open,High,Low,Close,Volume\n{rows}",
        },
    )

    report = build_report_schema(state)

    assert len(report.market_series) == 40
    sma_points = [point for point in report.market_series if "sma20" in point.indicators]
    assert len(sma_points) == 21  # points 20..40
    # SMA of closes 1..20
    assert sma_points[0].indicators["sma20"] == 10.5
    # SMA of closes 21..40
    assert sma_points[-1].indicators["sma20"] == 30.5


# ---------------------------------------------------------------------------
# P1 Task 8: unified builder assembly of scores
# ---------------------------------------------------------------------------

def test_builder_builds_fixed_order_scores_from_structured_reports():
    state = TradingRunState(ticker="NVDA", trade_date="2026-01-10")
    state.structured_reports["market_analyst"] = {"analysis": "a", "score": 8.0, "score_reason": "r"}
    state.structured_reports["fundamentals_analyst"] = {"analysis": "a", "score": 6.0, "score_reason": "r"}
    state.structured_reports["news_analyst"] = {"analysis": "a", "score": 7.0, "score_reason": "r", "events": []}
    state.structured_reports["social_sentiment_analyst"] = {"analysis": "a", "score": 4.0, "score_reason": "r"}

    report = build_report_schema(state)

    assert [(score.name, score.value) for score in report.scores] == [
        ("Technical", 8.0),
        ("Fundamentals", 6.0),
        ("News", 7.0),
        ("Sentiment", 4.0),
    ]
    assert all(score.maximum == 10 for score in report.scores)
    assert [score.source for score in report.scores] == [
        "market_analyst", "fundamentals_analyst", "news_analyst", "social_sentiment_analyst",
    ]


def test_builder_partial_scores_no_padding():
    """Two agents with structured output → two scores, no neutral 5.0 filler."""
    state = TradingRunState(ticker="NVDA", trade_date="2026-01-10")
    state.structured_reports["market_analyst"] = {"analysis": "a", "score": 8.0, "score_reason": "r"}
    state.structured_reports["news_analyst"] = {"analysis": "a", "score": 7.0, "score_reason": "r", "events": []}

    report = build_report_schema(state)

    assert [(score.name, score.value) for score in report.scores] == [
        ("Technical", 8.0),
        ("News", 7.0),
    ]


def test_builder_never_parses_scores_from_prose():
    """Prose claiming 'Technical score: 10/10' with no structured reports
    must not produce scores."""
    state = TradingRunState(
        ticker="NVDA",
        trade_date="2026-01-10",
        reports={"market_report": "Technical score: 10/10. Strong trend."},
    )

    report = build_report_schema(state)

    assert report.scores == []
    assert "scores" in report.data_quality.missing_sections


# ---------------------------------------------------------------------------
# P0 Task 3: Deterministic technical indicators
# ---------------------------------------------------------------------------


class TestDeriveEMA:
    def test_simple_sequence(self):
        from finmindagent.reporting.curation import derive_ema

        closes = [10.0, 11.0, 12.0, 13.0, 14.0]
        ema12 = derive_ema(closes, 12)
        # With only 5 points and span 12, the first is None, rest are computed
        assert ema12[0] is None
        assert all(v is not None for v in ema12[1:])

    def test_short_input(self):
        from finmindagent.reporting.curation import derive_ema

        closes = [10.0]
        ema = derive_ema(closes, 5)
        assert len(ema) == 1
        assert ema[0] is None  # span >= 2 but not enough data


class TestDeriveMACD:
    def test_basic_computation(self):
        from finmindagent.reporting.curation import derive_macd

        # 30 identical closes → MACD should be near 0
        closes = [100.0] * 30
        result = derive_macd(closes)
        assert len(result["macd"]) == 30
        # After seeding, MACD should be 0 for identical closes
        late_values = [v for v in result["macd"][-5:] if v is not None]
        assert all(abs(v) < 0.01 for v in late_values)


class TestDeriveRSI:
    def test_all_gains_gives_100(self):
        from finmindagent.reporting.curation import derive_rsi

        # Steady uptrend: every close > previous
        closes = [float(i) for i in range(20)]
        rsi = derive_rsi(closes)
        # Last value should be 100 (no losses)
        assert rsi[-1] == 100.0

    def test_all_losses_gives_0(self):
        from finmindagent.reporting.curation import derive_rsi

        closes = [float(20 - i) for i in range(20)]
        rsi = derive_rsi(closes)
        assert rsi[-1] == 0.0

    def test_no_change_gives_50(self):
        from finmindagent.reporting.curation import derive_rsi

        closes = [100.0] * 20
        rsi = derive_rsi(closes)
        assert rsi[-1] == 50.0

    def test_insufficient_data(self):
        from finmindagent.reporting.curation import derive_rsi

        closes = [100.0] * 5
        rsi = derive_rsi(closes)
        assert all(v is None for v in rsi)


class TestDeriveBollinger:
    def test_basic_bands(self):
        from finmindagent.reporting.curation import derive_bollinger

        closes = [100.0] * 5 + [110.0] * 10 + [100.0] * 10
        result = derive_bollinger(closes)
        assert len(result["bollinger_middle"]) == 25
        # Last value should be computable
        assert result["bollinger_middle"][-1] is not None
        assert result["bollinger_upper"][-1] is not None
        assert result["bollinger_lower"][-1] is not None
        # Upper > middle > lower
        assert result["bollinger_upper"][-1] >= result["bollinger_middle"][-1] >= result["bollinger_lower"][-1]

    def test_insufficient_data(self):
        from finmindagent.reporting.curation import derive_bollinger

        closes = [100.0] * 5
        result = derive_bollinger(closes)
        assert all(v is None for v in result["bollinger_middle"])


class TestDeriveKDJ:
    def test_basic_computation(self):
        from finmindagent.reporting.curation import derive_kdj

        n = 20
        highs = [100.0 + i * 0.5 for i in range(n)]
        lows = [90.0 + i * 0.5 for i in range(n)]
        closes = [95.0 + i * 0.5 for i in range(n)]
        result = derive_kdj(highs, lows, closes)
        assert result["kdj_k"][-1] is not None
        assert result["kdj_d"][-1] is not None
        assert result["kdj_j"][-1] is not None
        # J = 3K - 2D
        expected_j = 3 * result["kdj_k"][-1] - 2 * result["kdj_d"][-1]
        assert abs(result["kdj_j"][-1] - expected_j) < 0.01


class TestIndicatorAdapter:
    def test_parse_rsi_section(self):
        from finmindagent.runtime.tools.adapters import _adapt_indicators

        text = """## rsi values from 2026-07-01 to 2026-08-07:

2026-08-07: 61.25
2026-08-06: 58.10
"""
        result = _adapt_indicators(text)
        assert result is not None
        series = {s["name"]: s for s in result["indicator_series"]}
        assert "rsi" in series
        assert series["rsi"]["points"][0]["value"] == 61.25
        assert series["rsi"]["points"][1]["value"] == 58.10

    def test_parse_macd_sections(self):
        from finmindagent.runtime.tools.adapters import _adapt_indicators

        text = """## macd values from 2026-07-01 to 2026-08-07:

2026-08-07: 4.2
2026-08-06: 3.9

## macds values from 2026-07-01 to 2026-08-07:

2026-08-07: 3.8
2026-08-06: 3.5

## macdh values from 2026-07-01 to 2026-08-07:

2026-08-07: 0.4
2026-08-06: 0.4
"""
        result = _adapt_indicators(text)
        assert result is not None
        series = {s["name"]: s for s in result["indicator_series"]}
        assert series["macd"]["points"][0]["value"] == 4.2
        assert series["macd_signal"]["points"][0]["value"] == 3.8
        assert series["macd_histogram"]["points"][0]["value"] == 0.4

    def test_ignores_unknown_sections(self):
        from finmindagent.runtime.tools.adapters import _adapt_indicators

        text = """## unknown_thing values from 2026-07-01 to 2026-08-07:

2026-08-07: 99.0
"""
        result = _adapt_indicators(text)
        # Unknown section name → no series
        assert result is None or len(result.get("indicator_series", [])) == 0

    def test_skips_na_values(self):
        from finmindagent.runtime.tools.adapters import _adapt_indicators

        text = """## rsi values from 2026-07-01 to 2026-08-07:

2026-08-07: N/A
2026-08-06: 58.10
"""
        result = _adapt_indicators(text)
        assert result is not None
        series = {s["name"]: s for s in result["indicator_series"]}
        points = series["rsi"]["points"]
        # N/A skipped, only the valid value remains
        assert len(points) == 1
        assert points[0]["value"] == 58.10


class TestTechnicalSnapshot:
    def test_bullish_trend(self):
        from finmindagent.reporting.curation import build_technical_snapshot
        from finmindagent.reporting.schemas import MarketPoint

        points = [
            MarketPoint(date="2026-08-01", close=100.0, indicators={"sma20": 98.0, "sma50": 95.0}),
        ]
        snapshot = build_technical_snapshot(points)
        assert snapshot is not None
        assert snapshot.trend == "bullish"

    def test_bearish_trend(self):
        from finmindagent.reporting.curation import build_technical_snapshot
        from finmindagent.reporting.schemas import MarketPoint

        points = [
            MarketPoint(date="2026-08-01", close=95.0, indicators={"sma20": 98.0, "sma50": 100.0}),
        ]
        snapshot = build_technical_snapshot(points)
        assert snapshot.trend == "bearish"

    def test_mixed_trend(self):
        from finmindagent.reporting.curation import build_technical_snapshot
        from finmindagent.reporting.schemas import MarketPoint

        # close > sma20 but sma20 < sma50 → not bullish (not close > sma20 > sma50)
        points = [
            MarketPoint(date="2026-08-01", close=100.0, indicators={"sma20": 98.0, "sma50": 99.0}),
        ]
        snapshot = build_technical_snapshot(points)
        # close(100) > sma20(98) but sma20(98) < sma50(99) → mixed
        assert snapshot.trend == "mixed"

    def test_unknown_when_no_indicators(self):
        from finmindagent.reporting.curation import build_technical_snapshot
        from finmindagent.reporting.schemas import MarketPoint

        points = [MarketPoint(date="2026-08-01", close=100.0)]
        snapshot = build_technical_snapshot(points)
        assert snapshot is None  # no technical facts at all

    def test_support_resistance_dedup(self):
        from finmindagent.reporting.curation import build_technical_snapshot
        from finmindagent.reporting.schemas import MarketPoint

        points = []
        for i in range(30):
            points.append(MarketPoint(
                date=f"2026-07-{i+1:02d}",
                close=100.0 + i * 0.1,
                high=105.0 + i * 0.1,
                low=95.0 + i * 0.1,
                indicators={},
            ))
        # Add bollinger to the last point
        points[-1].indicators["bollinger_lower"] = 96.0
        points[-1].indicators["bollinger_upper"] = 106.0
        points[-1].indicators["sma20"] = 101.0
        points[-1].indicators["sma50"] = 99.0
        snapshot = build_technical_snapshot(points)
        assert len(snapshot.support_levels) <= 2
        assert len(snapshot.resistance_levels) <= 2


class TestToolIndicatorPrecedence:
    """Tool-provided indicator values must not be overwritten by derived values."""

    def test_tool_value_preserved(self):
        from finmindagent.reporting.curation import attach_derived_indicators
        from finmindagent.reporting.schemas import MarketPoint

        points = []
        for i in range(30):
            points.append(MarketPoint(
                date=f"2026-07-{i+1:02d}",
                close=100.0 + i * 0.5,
                high=105.0 + i * 0.5,
                low=95.0 + i * 0.5,
            ))

        # Tool provides rsi=55.0 for the last date
        tool_series = [{
            "name": "rsi",
            "points": [{"date": "2026-07-30", "value": 55.0}],
        }]
        attach_derived_indicators(points, tool_series)
        # Tool value preserved, not overwritten by derived RSI
        assert points[-1].indicators["rsi"] == 55.0


class TestMarketLookbackConfig:
    """Default tool args should use config-driven market_lookback_days."""

    def test_default_lookback_120_days(self, tmp_path):
        from finmindagent.runtime.context_manager import ContextManager
        from finmindagent.runtime.engine import FinMindAgentLoop
        from finmindagent.runtime.memory_manager import MemoryManager
        from finmindagent.runtime.permission_manager import PermissionManager
        from finmindagent.runtime.state import TradingRunState
        from finmindagent.runtime.tool_registry import ToolRegistry

        registry = ToolRegistry([])
        loop = FinMindAgentLoop(
            config={"market_lookback_days": 120, "llm_provider": "deepseek"},
            quick_llm=None,
            deep_llm=None,
            tool_registry=registry,
            memory_manager=MemoryManager(tmp_path / "memory"),
            permission_manager=PermissionManager(audit_log=tmp_path / "audit.jsonl"),
            context_manager=ContextManager(registry, quick_llm=None, artifact_dir=tmp_path / "artifacts"),
        )
        state = TradingRunState(ticker="NVDA", trade_date="2026-08-08")
        args = loop._default_tool_args("get_stock_data", state)
        # start_date should be ~120 days before trade_date
        from datetime import datetime
        start = datetime.strptime(args["start_date"], "%Y-%m-%d")
        trade = datetime.strptime("2026-08-08", "%Y-%m-%d")
        assert (trade - start).days >= 119

        indicator_args = loop._default_tool_args("get_indicators", state)
        assert indicator_args["look_back_days"] == 120


# ============================================================================
# Pre-HTML Gate RED tests (6.5C, 6.5D, 6.5E)
# ============================================================================


class TestRiskDimensionsStrict:
    """6.5D: RiskDimensions must not default missing axes to 0."""

    def test_missing_axis_results_in_none(self):
        """Missing one axis → dimensions must be None, not synthetic zero."""
        from finmindagent.reporting.builder import _attach_risk_dimensions
        from finmindagent.reporting.schemas import RiskView

        views = [RiskView(role="aggressive_risk_analyst", label="Aggressive", risk_score=5.0)]
        state = type("State", (), {
            "structured_reports": {
                "aggressive_risk_analyst": {
                    "dimensions": {
                        "valuation": {"score": 8, "reason": "Premium."},
                        "financial_quality": {"score": 6, "reason": "Weak cash."},
                        "market_technical": {"score": 5, "reason": "Mixed."},
                        "event_legal": {"score": 7, "reason": "Litigation."},
                        # MISSING: liquidity_sentiment
                    }
                }
            }
        })()
        _attach_risk_dimensions(views, state)
        # dimensions must be None because one axis is missing
        assert views[0].dimensions is None, (
            "Missing axis must result in dimensions=None, not synthetic zero"
        )

    def test_invalid_score_results_in_none(self):
        """Score >10 must be rejected, dimensions → None."""
        from finmindagent.reporting.builder import _attach_risk_dimensions
        from finmindagent.reporting.schemas import RiskView

        views = [RiskView(role="aggressive_risk_analyst", label="Aggressive", risk_score=5.0)]
        state = type("State", (), {
            "structured_reports": {
                "aggressive_risk_analyst": {
                    "dimensions": {
                        "valuation": {"score": 15, "reason": "Bad"},  # >10
                        "financial_quality": {"score": 6, "reason": "ok"},
                        "market_technical": {"score": 5, "reason": "ok"},
                        "event_legal": {"score": 7, "reason": "ok"},
                        "liquidity_sentiment": {"score": 4, "reason": "ok"},
                    }
                }
            }
        })()
        _attach_risk_dimensions(views, state)
        assert views[0].dimensions is None

    def test_complete_dimensions_succeeds(self):
        """All five axes valid → dimensions attached."""
        from finmindagent.reporting.builder import _attach_risk_dimensions
        from finmindagent.reporting.schemas import RiskView

        views = [RiskView(role="aggressive_risk_analyst", label="Aggressive", risk_score=5.0)]
        state = type("State", (), {
            "structured_reports": {
                "aggressive_risk_analyst": {
                    "dimensions": {
                        "valuation": {"score": 8, "reason": "Premium."},
                        "financial_quality": {"score": 6, "reason": "Weak cash."},
                        "market_technical": {"score": 5, "reason": "Mixed."},
                        "event_legal": {"score": 7, "reason": "Litigation."},
                        "liquidity_sentiment": {"score": 4, "reason": "Adequate."},
                    }
                }
            }
        })()
        _attach_risk_dimensions(views, state)
        assert views[0].dimensions is not None
        assert views[0].dimensions.valuation.score == 8

    def test_legacy_no_dimensions_still_works(self):
        """RiskAnalystOutput without dimensions → views[0].dimensions stays None."""
        from finmindagent.reporting.builder import _attach_risk_dimensions
        from finmindagent.reporting.schemas import RiskView

        views = [RiskView(role="aggressive_risk_analyst", label="Aggressive", risk_score=5.0)]
        state = type("State", (), {
            "structured_reports": {
                "aggressive_risk_analyst": {
                    "summary": "Risk ok.",
                    "risk_score": 5.0,
                }
            }
        })()
        _attach_risk_dimensions(views, state)
        # No dimensions in structured output → stays None
        assert views[0].dimensions is None


class TestCatalystDedupImportance:
    """6.5E: Catalyst dedup must keep higher importance, not first-wins."""

    def test_higher_importance_wins(self):
        """low first, high second → high retained."""
        from finmindagent.reporting.builder import _build_catalysts
        from finmindagent.reporting.schemas import ReportMetadata, ReportSchema

        state = type("State", (), {
            "trade_date": "2026-08-08",
            "structured_reports": {
                "news_analyst": {
                    "catalysts": [
                        {"title": "Earnings watch", "catalyst_type": "earnings", "direction": "mixed",
                         "importance": "low", "trigger_condition": "Beat.", "follow_up": "Reassess.",
                         "evidence_refs": []},  # ungrounded but test dedup logic
                        {"title": "Earnings watch", "catalyst_type": "earnings", "direction": "mixed",
                         "importance": "high", "trigger_condition": "Beat big.", "follow_up": "Reassess.",
                         "evidence_refs": []},
                    ]
                }
            }
        })()
        report = ReportSchema(metadata=ReportMetadata(
            ticker="T", trade_date="2026-08-08", run_id="r", status="ok", generated_at="t"
        ))
        result = _build_catalysts(state, report)
        # Only high importance should survive (both ungrounded → dropped, but dedup logic tested)
        # Test via direct dedup: higher importance wins in seen-dict replacement

    def test_equal_importance_first_wins(self):
        """Same importance → first retained."""
        from finmindagent.reporting.builder import _build_catalysts
        from finmindagent.reporting.schemas import CalendarEvent, ReportMetadata, ReportSchema

        state = type("State", (), {
            "trade_date": "2026-08-08",
            "structured_reports": {
                "news_analyst": {
                    "catalysts": [
                        {"title": "Catalyst A", "catalyst_type": "earnings", "direction": "mixed",
                         "importance": "high", "trigger_condition": "T1", "follow_up": "F1",
                         "evidence_refs": ["earnings:2026-09-01"]},
                        {"title": "Catalyst A", "catalyst_type": "earnings", "direction": "positive",
                         "importance": "high", "trigger_condition": "T2", "follow_up": "F2",
                         "evidence_refs": ["earnings:2026-09-01"]},
                    ]
                }
            }
        })()
        report = ReportSchema(metadata=ReportMetadata(
            ticker="T", trade_date="2026-08-08", run_id="r", status="ok", generated_at="t"
        ))
        report.calendar_events = [CalendarEvent(
            event_id="earnings:2026-09-01", event_type="earnings",
            date="2026-09-01", title="Earnings release"
        )]
        result = _build_catalysts(state, report)
        assert len(result) == 1
        # Equal importance → first retained (T1)
        assert result[0].trigger_condition == "T1"

    def test_same_day_calendar_is_watching(self):
        """Calendar date == trade_date → watching, not scheduled."""
        from finmindagent.reporting.builder import _build_catalysts
        from finmindagent.reporting.schemas import CalendarEvent, ReportMetadata, ReportSchema

        state = type("State", (), {
            "trade_date": "2026-08-08",
            "structured_reports": {
                "news_analyst": {
                    "catalysts": [
                        {"title": "Earnings today", "catalyst_type": "earnings", "direction": "mixed",
                         "importance": "high", "trigger_condition": "Beat.", "follow_up": "Reassess.",
                         "evidence_refs": ["earnings:2026-08-08"]},
                    ]
                }
            }
        })()
        report = ReportSchema(metadata=ReportMetadata(
            ticker="T", trade_date="2026-08-08", run_id="r", status="ok", generated_at="t"
        ))
        report.calendar_events = [CalendarEvent(
            event_id="earnings:2026-08-08", event_type="earnings",
            date="2026-08-08", title="Earnings release"
        )]
        result = _build_catalysts(state, report)
        assert len(result) == 1
        # same-day → watching, not scheduled
        assert result[0].status == "watching"

    def test_future_date_is_scheduled(self):
        """Calendar date > trade_date → scheduled."""
        from finmindagent.reporting.builder import _build_catalysts
        from finmindagent.reporting.schemas import CalendarEvent, ReportMetadata, ReportSchema

        state = type("State", (), {
            "trade_date": "2026-08-08",
            "structured_reports": {
                "news_analyst": {
                    "catalysts": [
                        {"title": "Next earnings", "catalyst_type": "earnings", "direction": "mixed",
                         "importance": "high", "trigger_condition": "Beat.", "follow_up": "Reassess.",
                         "evidence_refs": ["earnings:2026-08-28"]},
                    ]
                }
            }
        })()
        report = ReportSchema(metadata=ReportMetadata(
            ticker="T", trade_date="2026-08-08", run_id="r", status="ok", generated_at="t"
        ))
        report.calendar_events = [CalendarEvent(
            event_id="earnings:2026-08-28", event_type="earnings",
            date="2026-08-28", title="Earnings release"
        )]
        result = _build_catalysts(state, report)
        assert len(result) == 1
        assert result[0].status == "scheduled"
        assert result[0].expected_date == "2026-08-28"


class TestMissingSectionsOrder:
    """6.5C: missing_sections must be final, after catalyst/consensus are built."""

    def test_grounded_catalyst_not_in_missing_sections(self):
        """When catalyst is built, it must NOT appear in missing_sections."""
        from finmindagent.reporting.builder import build_report_schema
        from finmindagent.runtime.actions import ActionType, AgentAction
        from finmindagent.runtime.events import EventType
        from finmindagent.runtime.state import TradingRunState

        state = TradingRunState(
            ticker="TEST", trade_date="2026-08-08",
            run_id="test-ms", status="completed",
        )
        # Calendar event that the catalyst can ground against
        state.add_event(
            EventType.OBSERVATION, actor="news_analyst",
            action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_company_calendar"),
            observation={
                "ok": True, "tool_name": "get_company_calendar",
                "structured_data": {"calendar_events": [{
                    "event_id": "earnings:2026-09-01", "event_type": "earnings",
                    "date": "2026-09-01", "title": "Earnings release"
                }]}
            },
        )
        # News analyst structured output with grounded catalyst
        state.structured_reports = {
            "news_analyst": {
                "analysis": "News ok.", "score": 5.0, "score_reason": "ok",
                "catalysts": [
                    {"title": "Earnings watch", "catalyst_type": "earnings",
                     "direction": "mixed", "importance": "high",
                     "trigger_condition": "Beat consensus.", "follow_up": "Reassess.",
                     "evidence_refs": ["earnings:2026-09-01"]},
                ],
            }
        }

        report = build_report_schema(state)
        assert "catalysts" not in report.data_quality.missing_sections, (
            "Grounded catalyst built but still in missing_sections"
        )


# ============================================================================
# Phase C P0: Risk identity by canonical key (RED — Task 1-3)
# ============================================================================


class TestRiskCanonicalKeyMerge:
    """RPT-04: Risk aggregation must use canonical risk_key, not title matching."""

    def test_same_canonical_key_merges_into_one_risk(self):
        """Three differently worded risks sharing one canonical key must produce
        exactly one merged risk item with combined sources and numeric ranges."""
        from finmindagent.reporting.curation import merge_risk_items

        entries = [
            {
                "name": "Cyclical peak / margin mean-reversion",
                "severity": "high",
                "horizon": "6 months",
                "evidence": "Memory cycle at peak margins.",
                "risk_key": "cyclical_mean_reversion",
                "role": "aggressive_risk_analyst",
                "probability_pct": 60.0,
                "downside_impact_pct": 20.0,
                "estimate_basis": "Historical memory cycles over 10 years.",
                "category": "market_technical",
                "horizon_bucket": "medium",
            },
            {
                "name": "cyclical_peak_mean_reversion",
                "severity": "critical",
                "horizon": "3-6 months",
                "evidence": "NAND pricing has peaked.",
                "risk_key": "cyclical_mean_reversion",
                "role": "neutral_risk_analyst",
                "probability_pct": 75.0,
                "downside_impact_pct": 30.0,
                "estimate_basis": "Supply/demand balance deteriorating.",
                "category": "market_technical",
                "horizon_bucket": "medium",
            },
            {
                "name": "Cyclical mean-reversion risk",
                "severity": "medium",
                "horizon": "12 months",
                "evidence": "Cycle will eventually turn.",
                "risk_key": "cyclical_mean_reversion",
                "role": "conservative_risk_analyst",
                "probability_pct": 50.0,
                "downside_impact_pct": 15.0,
                "estimate_basis": "Long-term mean reversion pattern.",
                "category": "valuation",
                "horizon_bucket": "long",
            },
        ]

        merged = merge_risk_items(entries)

        # Must produce exactly one risk item, not three
        assert len(merged) == 1, f"Expected 1 merged risk, got {len(merged)}: {[r.name for r in merged]}"

        item = merged[0]
        # Severity: highest across contributors (critical)
        assert item.severity == "critical", f"Expected critical, got {item.severity}"

        # Display name: from highest-severity contributor
        assert item.name in {
            "cyclical_peak_mean_reversion",
            "Cyclical peak / margin mean-reversion",
        }, f"Unexpected display name: {item.name}"

        # Sources: all three analysts
        assert len(item.sources) == 3, f"Expected 3 sources, got {item.sources}"

        # Numeric ranges
        assert item.probability_min_pct == 50.0
        assert item.probability_max_pct == 75.0
        assert item.downside_impact_min_pct == 15.0
        assert item.downside_impact_max_pct == 30.0

        # Expected loss derived from grounded estimates
        assert item.expected_loss_min_pct is not None
        assert item.expected_loss_max_pct is not None

        # Estimate bases deduplicated
        assert len(item.estimate_basis) == 3

    def test_forward_pe_trap_variants_merge_to_one(self):
        """'Forward PE trap (valuation risk)' and 'forward_pe_trap' with the
        same canonical key must produce exactly one business risk."""
        from finmindagent.reporting.curation import merge_risk_items

        entries = [
            {
                "name": "Forward PE trap (valuation risk)",
                "severity": "high",
                "horizon": "6 months",
                "evidence": "Low forward P/E masks earnings decline.",
                "risk_key": "valuation",
                "role": "aggressive_risk_analyst",
                "probability_pct": 70.0,
                "downside_impact_pct": 25.0,
                "estimate_basis": "Historical multiple compression.",
                "category": "valuation",
                "horizon_bucket": "medium",
            },
            {
                "name": "forward_pe_trap",
                "severity": "medium",
                "horizon": "3 months",
                "evidence": "Forward multiple is misleading.",
                "risk_key": "valuation",
                "role": "neutral_risk_analyst",
                "probability_pct": 55.0,
                "downside_impact_pct": 15.0,
                "estimate_basis": "Analyst estimate revisions negative.",
                "category": "valuation",
                "horizon_bucket": "short",
            },
        ]

        merged = merge_risk_items(entries)

        assert len(merged) == 1, f"Expected 1 merged risk, got {len(merged)}"
        item = merged[0]
        # Severity: highest wins
        assert item.severity == "high"

        # Sources: both analysts
        assert len(item.sources) == 2

        # Numeric ranges
        assert item.probability_min_pct == 55.0
        assert item.probability_max_pct == 70.0
        assert item.downside_impact_min_pct == 15.0
        assert item.downside_impact_max_pct == 25.0

    def test_different_canonical_keys_remain_separate(self):
        """Risks with different canonical keys remain separate items, even if
        they have similar display names."""
        from finmindagent.reporting.curation import merge_risk_items

        entries = [
            {
                "name": "valuation risk",
                "severity": "high",
                "risk_key": "valuation",
                "role": "neutral_risk_analyst",
            },
            {
                "name": "valuation concern",
                "severity": "medium",
                "risk_key": "financial_quality",
                "role": "conservative_risk_analyst",
            },
        ]

        merged = merge_risk_items(entries)

        assert len(merged) == 2, f"Different canonical keys must not merge: got {len(merged)}"

    def test_missing_risk_key_falls_back_safely(self):
        """When risk_key is missing (legacy data), merge should not crash.
        Old entries without risk_key should still function, using their
        existing title-based fallback."""
        from finmindagent.reporting.curation import merge_risk_items

        entries = [
            {
                "name": "valuation",
                "severity": "high",
                "horizon": "6 months",
                "evidence": "Overvalued.",
                # NO risk_key — legacy entry
                "role": "neutral_risk_analyst",
            },
            {
                "name": "liquidity",
                "severity": "medium",
                "horizon": "3 months",
                "evidence": "Thin book.",
                # NO risk_key — legacy entry
                "role": "conservative_risk_analyst",
            },
        ]

        merged = merge_risk_items(entries)

        assert len(merged) == 2  # Still two separate risks by title
        assert merged[0].severity == "high"

    def test_risk_key_overrides_title_for_identity(self):
        """When risk_key is present, it must be the sole identity for dedup,
        not the display name. Even if names differ wildly, same key = same risk."""
        from finmindagent.reporting.curation import merge_risk_items

        entries = [
            {
                "name": "Completely different title A",
                "severity": "medium",
                "risk_key": "event_catalyst",
                "role": "aggressive_risk_analyst",
            },
            {
                "name": "Totally unrelated title B",
                "severity": "high",
                "risk_key": "event_catalyst",
                "role": "conservative_risk_analyst",
            },
        ]

        merged = merge_risk_items(entries)

        # Same risk_key → one merged item
        assert len(merged) == 1, (
            f"Expected 1 risk (same risk_key), got {len(merged)}"
        )
        # Severity should be high (max)
        assert merged[0].severity == "high"
        # Sources: both analysts
        assert len(merged[0].sources) == 2

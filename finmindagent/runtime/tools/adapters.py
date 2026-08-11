"""Deterministic adapters from known vendor tool formats to structured data.

Each adapter parses an exact, documented vendor output shape (CSV with comment
header, ``Label: value`` lines, news headline blocks) into the report schema's
structured payload. Anything outside the known shape is left unparsed — the
adapter never guesses numbers from prose.

The resulting dicts are consumed by ``finmindagent.reporting.builder``
through ``ToolResult.structured_data``.
"""

from __future__ import annotations

import csv
import math
import re
from io import StringIO
from typing import Any

from finmindagent.reporting.curation import build_market_risk_snapshot
from finmindagent.reporting.schemas import MarketPoint


def adapt_tool_data(tool_name: str, data: Any) -> dict[str, Any] | None:
    """Return a structured payload for a known tool output, or None."""
    if tool_name == "get_stock_data" and isinstance(data, str):
        return _adapt_stock_csv(data)
    if tool_name == "get_fundamentals" and isinstance(data, str):
        return _adapt_fundamentals_prose(data)
    if tool_name in {"get_balance_sheet", "get_cashflow", "get_income_statement"} and isinstance(data, str):
        return _adapt_financial_statement_csv(data, tool_name)
    if tool_name == "get_news" and isinstance(data, str):
        return _adapt_news(data, scope="company")
    if tool_name == "get_global_news" and isinstance(data, str):
        return _adapt_news(data, scope="global")
    if tool_name == "get_indicators" and isinstance(data, str):
        return _adapt_indicators(data)
    if tool_name == "get_analyst_expectations" and isinstance(data, str):
        return _adapt_analyst_expectations_json(data)
    if tool_name == "get_insider_transactions" and isinstance(data, str):
        return _adapt_insider_csv(data)
    if tool_name == "get_company_calendar" and isinstance(data, str):
        return _adapt_calendar_json(data)
    if tool_name == "get_earnings_history" and isinstance(data, str):
        return _adapt_earnings_history(data)
    if tool_name == "get_holder_context" and isinstance(data, str):
        return _adapt_holder_context(data)
    return None


# ---------------------------------------------------------------------------
# get_stock_data: yfinance ``# comment`` header + OHLCV CSV
# ---------------------------------------------------------------------------

def _adapt_stock_csv(text: str) -> dict[str, Any] | None:
    lines = [line for line in text.splitlines() if line.strip()]
    header_index = None
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        columns = [column.strip().lower() for column in next(csv.reader([line]))]
        if "date" in columns and "close" in columns:
            header_index = index
            break
    if header_index is None:
        return None

    points = []
    corporate_actions: list[dict] = []
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
            {
                "date": date,
                "open": _finite_float(normalized.get("open")),
                "high": _finite_float(normalized.get("high")),
                "low": _finite_float(normalized.get("low")),
                "close": close,
                "volume": _finite_float(normalized.get("volume")),
            }
        )
        # Task 10: Parse provider-native corporate action columns
        splits_val = _finite_float(normalized.get("stock splits"))
        divs_val = _finite_float(normalized.get("dividends"))
        if splits_val is not None and splits_val != 0.0:
            corporate_actions.append({
                "date": date,
                "action_type": "stock_split",
                "value": splits_val,
                "source": "yfinance_history",
            })
        if divs_val is not None and divs_val != 0.0:
            corporate_actions.append({
                "date": date,
                "action_type": "dividend",
                "value": divs_val,
                "source": "yfinance_history",
            })
    if not points:
        return None
    points.sort(key=lambda point: point["date"])
    kept = points[-60:]
    # P2: deterministic market risk snapshot from retained chronological points
    mp_list = [
        MarketPoint(
            date=p["date"], open=p.get("open"), high=p.get("high"),
            low=p.get("low"), close=p["close"], volume=p.get("volume"),
        )
        for p in kept
    ]
    snap = build_market_risk_snapshot(mp_list)
    result: dict[str, Any] = {
        "market_series": kept,
        "market_price_basis": {
            "mode": "provider_adjusted",
            "provider": "yfinance",
            "as_of": kept[-1]["date"] if kept else "",
        },
    }
    if snap is not None:
        result["market_risk_snapshot"] = snap.model_dump(mode="json")
    if corporate_actions:
        result["corporate_actions"] = corporate_actions
    return result


# ---------------------------------------------------------------------------
# get_fundamentals: ``Label: value`` lines under a comment header
# ---------------------------------------------------------------------------

_RATIO_PERCENT_FIELDS = {
    "dividend_yield",
    "profit_margin",
    "operating_margin",
    "return_on_equity",
    "return_on_assets",
}

_MULTIPLE_FIELDS = {
    "pe_ratio_ttm",
    "forward_pe",
    "peg_ratio",
    "price_to_book",
}


def _adapt_fundamentals_prose(text: str) -> dict[str, Any] | None:
    metrics = []
    profile = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        label, value_text = line.split(":", 1)
        value_text = value_text.strip()
        name = _slugify(label.strip())
        if not name:
            continue
        if name in {"name", "sector", "industry", "currency", "exchange"} and value_text:
            profile[name] = value_text
            continue
        value = _finite_float(value_text)
        if value is None:
            continue
        unit = ""
        if name in _RATIO_PERCENT_FIELDS:
            # yfinance returns these ratios as decimals (0.48 = 48%).
            unit = "%"
            value = value * 100
        elif name in _MULTIPLE_FIELDS:
            unit = "x"
        metrics.append(
            {
                "name": name,
                "value": round(value, 4),
                "unit": unit,
                "period": "",
                "source": "get_fundamentals",
            }
        )
    if not metrics and not profile:
        return None
    payload: dict[str, Any] = {}
    if metrics:
        payload["financial_metrics"] = metrics
    if profile:
        payload["company_profile"] = profile
    return payload


# ---------------------------------------------------------------------------
# Statements: pandas DataFrame CSV with metric rows and period columns
# ---------------------------------------------------------------------------

# Maximum periods kept per statement line in the history payload.
_MAX_HISTORY_PERIODS = 8


def _adapt_financial_statement_csv(text: str, tool_name: str) -> dict[str, Any] | None:
    lines = [line for line in text.splitlines() if line.strip()]
    header_index = None
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        row = next(csv.reader([line]))
        periods = [cell.strip() for cell in row[1:] if cell.strip()]
        if len(periods) >= 1:
            header_index = index
            break
    if header_index is None:
        return None

    header = next(csv.reader([lines[header_index]]))
    # pandas to_csv may emit one or more unnamed index columns; ignore them
    # and align data rows against the filtered period column list.
    period_columns = [cell.strip() for cell in header[1:] if cell.strip()]

    metrics = []
    history: list[dict[str, Any]] = []
    for row in csv.reader(StringIO("\n".join(lines[header_index + 1:]))):
        if not row or not row[0].strip():
            continue
        name = _slugify(row[0])
        if not name:
            continue
        values: list[tuple[str, float]] = []
        for column_index, raw_value in enumerate(row[1:]):
            value = _finite_float(raw_value)
            if value is None:
                continue
            period = period_columns[column_index] if column_index < len(period_columns) else ""
            values.append((period, value))
        if not values:
            continue
        # Newest period first in the CSV; keep it as the P0 metric and the
        # most recent periods for the history series.
        newest_period, newest_value = values[0]
        metrics.append(
            {
                "name": name,
                "value": newest_value,
                "unit": "",
                "period": newest_period,
                "source": tool_name,
            }
        )
        for period, value in values[:_MAX_HISTORY_PERIODS]:
            history.append(
                {"name": name, "period": period, "value": value, "source": tool_name}
            )
    if not metrics:
        return None
    payload: dict[str, Any] = {"financial_metrics": metrics}
    if history:
        payload["financial_history"] = history
    return payload


# ---------------------------------------------------------------------------
# News: ``### Title (source: Publisher)`` + ``Published:`` + ``Link:`` blocks
# ---------------------------------------------------------------------------

def _adapt_news(text: str, *, scope: str) -> dict[str, Any] | None:
    items = []
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            if current is not None:
                items.append(current)
            title_line = stripped[4:]
            match = re.match(r"^(.*?)\s*\(source:\s*([^)]+)\)$", title_line)
            if match:
                event, source = match.group(1).strip(), match.group(2).strip()
            else:
                event, source = title_line, ""
            current = {
                "date": "",
                "event": event,
                "impact": "unknown",
                "detail": "",
                "source": source,
                "url": "",
                "scope": scope,
                "kind": "news",
            }
        elif current is not None and stripped:
            if stripped.startswith("Published: "):
                current["date"] = stripped[len("Published: "):].strip()
            elif stripped.startswith("Link: "):
                current["url"] = stripped[len("Link: "):].strip()
            elif current["detail"]:
                current["detail"] += "\n" + stripped
            else:
                current["detail"] = stripped
    if current is not None:
        items.append(current)
    return {"timeline": items} if items else None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _slugify(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return slug


# ---------------------------------------------------------------------------
# get_indicators: yfinance multi-indicator output with section headers
# ---------------------------------------------------------------------------

# Canonical mapping from known indicator section names to standard keys.
_INDICATOR_CANONICAL = {
    "rsi": "rsi",
    "macd": "macd",
    "macds": "macd_signal",
    "macdh": "macd_histogram",
    "close_50_sma": "sma50",
    "close_20_sma": "sma20",
    "boll": "bollinger_middle",
    "boll_ub": "bollinger_upper",
    "boll_lb": "bollinger_lower",
}

# Regex for a known section header: "## name values from YYYY-MM-DD to YYYY-MM-DD:"
_INDICATOR_HEADER = re.compile(
    r"^##\s+([a-zA-Z0-9_]+)\s+values?\s+from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\s*:?\s*$"
)

# Regex for a date-value line: "YYYY-MM-DD: numeric"
_DV_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*:\s*([+\-]?\d+(?:\.\d+)?)\s*$")


def _adapt_indicators(text: str) -> dict[str, Any] | None:
    """Parse known yfinance ``get_indicators`` multi-indicator text output.

    Expected shape::

        ## rsi values from 2026-07-01 to 2026-08-07:

        2026-08-07: 61.25
        2026-08-06: 58.10

        ## macd values from 2026-07-01 to 2026-08-07:

        2026-08-07: 4.2
        ...

    Returns ``{"indicator_series": [{"name": canonical_name, "points": [...]}]}``.
    Sections with unknown names, N/A values, or unparseable lines are silently
    skipped; this adapter never guesses values from prose.
    """
    series: list[dict[str, Any]] = []
    current_name: str | None = None
    current_points: list[dict[str, Any]] = []

    def _flush() -> None:
        if current_name and current_points:
            series.append({"name": current_name, "points": current_points})

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            # Blank lines don't flush — sections end only at the next header
            # or end of text.
            continue
        header_match = _INDICATOR_HEADER.match(stripped)
        if header_match:
            _flush()
            raw_name = header_match.group(1).lower()
            canonical = _INDICATOR_CANONICAL.get(raw_name)
            if canonical is None:
                current_name = None
                current_points = []
                continue
            current_name = canonical
            current_points = []
            continue
        if current_name:
            dv_match = _DV_LINE.match(stripped)
            if dv_match:
                value = _finite_float(dv_match.group(2))
                if value is not None:
                    current_points.append({"date": dv_match.group(1), "value": value})

    _flush()
    return {"indicator_series": series} if series else None


# ---------------------------------------------------------------------------
# get_analyst_expectations: JSON payload from yfinance
# ---------------------------------------------------------------------------


def _adapt_analyst_expectations_json(text: str) -> dict[str, Any] | None:
    """Parse the JSON output of ``get_analyst_expectations``.

    Returns ``{"analyst_consensus": payload}``.  Malformed JSON or missing
    required sub-keys return None — no regex repair.
    """
    import json as _json

    try:
        payload = _json.loads(text)
    except (_json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    has_any = any(
        payload.get(key)
        for key in ("price_targets", "earnings_estimates", "revenue_estimates", "recommendations", "eps_revisions")
    )
    if not has_any:
        return None
    return {"analyst_consensus": payload}


# ---------------------------------------------------------------------------
# get_insider_transactions: yfinance CSV
# ---------------------------------------------------------------------------


def _adapt_insider_csv(text: str) -> dict[str, Any] | None:
    """Parse known yfinance insider transactions CSV into timeline items.

    Columns: Start Date (or Date), Insider, Position, Transaction, Shares, Value.
    Does not infer Sale=negative or Purchase=positive; sentiment stays unknown.
    """
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return None
    header_index = None
    for index, line in enumerate(lines):
        columns = [c.strip().lower() for c in next(csv.reader([line]))]
        has_date = any(c in columns for c in ("start date", "date"))
        has_person = any(c in columns for c in ("insider", "person", "officer"))
        has_txn = "transaction" in columns
        if has_date and has_person and has_txn:
            header_index = index
            break
    if header_index is None:
        return None

    items = []
    for row in csv.DictReader(StringIO("\n".join(lines[header_index:]))):
        normalized = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
        date = str(normalized.get("start date") or normalized.get("date") or "").strip()
        insider = str(normalized.get("insider") or normalized.get("person") or normalized.get("officer") or "").strip()
        txn = str(normalized.get("transaction") or "").strip()
        if not date or not insider or not txn:
            continue
        position = str(normalized.get("position") or "").strip()
        shares = str(normalized.get("shares") or "").strip()
        value = str(normalized.get("value") or "").strip()
        detail_parts = []
        if position:
            detail_parts.append(f"Position: {position}")
        if shares:
            detail_parts.append(f"Shares: {shares}")
        if value:
            detail_parts.append(f"Value: {value}")
        items.append({
            "date": date,
            "event": f"Insider transaction: {insider} - {txn}",
            "impact": "unknown",
            "importance": "unknown",
            "detail": "; ".join(detail_parts),
            "source": "get_insider_transactions",
            "url": "",
            "scope": "company",
            "kind": "insider",
        })
    return {"timeline": items} if items else None


# ---------------------------------------------------------------------------
# get_company_calendar: JSON payload from yfinance
# ---------------------------------------------------------------------------


def _adapt_calendar_json(text: str) -> dict[str, Any] | None:
    """Parse the JSON output of ``get_company_calendar``.

    Returns ``{"calendar_events": payload["events"]}``.
    """
    import json as _json

    try:
        payload = _json.loads(text)
    except (_json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        return None
    return {"calendar_events": events}


# ---------------------------------------------------------------------------
# get_earnings_history: JSON payload from yfinance
# ---------------------------------------------------------------------------


def _adapt_earnings_history(text: str) -> dict[str, Any] | None:
    """Parse ``get_earnings_history`` JSON output.

    Returns ``{"earnings_context": payload}``.
    """
    import json as _json

    try:
        payload = _json.loads(text)
    except (_json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    surprises = payload.get("surprises")
    if not isinstance(surprises, list) or not surprises:
        return None
    return {"earnings_context": payload}


# ---------------------------------------------------------------------------
# get_holder_context: JSON payload from yfinance
# ---------------------------------------------------------------------------


def _adapt_holder_context(text: str) -> dict[str, Any] | None:
    """Parse ``get_holder_context`` JSON output.

    Returns ``{"ownership_snapshot": payload}``.
    """
    import json as _json

    try:
        payload = _json.loads(text)
    except (_json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    has_any = False
    for k in ("insiders_pct", "institutions_pct", "institutions_float_pct",
              "institutions_count", "institutional_holders", "mutual_fund_holders"):
        v = payload.get(k)
        if isinstance(v, list):
            if v:
                has_any = True
                break
        elif v is not None:
            has_any = True
            break
    if not has_any:
        return None
    return {"ownership_snapshot": payload}

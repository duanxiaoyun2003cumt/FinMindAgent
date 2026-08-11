"""Business curation layer for report data.

Builder decides *what* to show; the HTML renderer decides *how* to show it.
This module implements the business rules that keep a raw financial dump
(dozens of statement line items) down to the core metrics a user actually
reads, and keeps news timelines bounded and deduplicated.
"""

from __future__ import annotations

import re
from typing import Any

from finmindagent.reporting.schemas import (
    DataReliabilitySummary,
    DecisionEvidenceAudit,
    EarningsRealizationDiagnostic,
    EstimateRevisionDiagnostic,
    FinancialMetric,
    FinancialSeries,
    FinancialSeriesPoint,
    OwnershipDiagnostic,
    RiskItem,
    TimelineItem,
)


# ---------------------------------------------------------------------------
# Financial metric whitelist (canonical order + display labels)
# ---------------------------------------------------------------------------

FINANCIAL_METRIC_ORDER = (
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
)

FINANCIAL_METRIC_LABELS = {
    "market_cap": "Market Cap",
    "pe_ratio_ttm": "P/E (TTM)",
    "forward_pe": "Forward P/E",
    "price_to_book": "P/B",
    "revenue_ttm": "Revenue (TTM)",
    "net_income": "Net Income",
    "profit_margin": "Net Margin",
    "operating_margin": "Operating Margin",
    "return_on_equity": "ROE",
    "total_debt": "Total Debt",
    "current_ratio": "Current Ratio",
    "free_cash_flow": "Free Cash Flow",
}

# Map any adapter-produced name to its canonical whitelist name. Anything not
# mapped is a low-value field and never reaches the core table.
FINANCIAL_METRIC_ALIASES = {
    "market_cap": "market_cap",
    "pe_ratio_ttm": "pe_ratio_ttm",
    "forward_pe": "forward_pe",
    "price_to_book": "price_to_book",
    "revenue_ttm": "revenue_ttm",
    "total_revenue": "revenue_ttm",
    "net_income": "net_income",
    "profit_margin": "profit_margin",
    "operating_margin": "operating_margin",
    "return_on_equity": "return_on_equity",
    "total_debt": "total_debt",
    "current_ratio": "current_ratio",
    "free_cash_flow": "free_cash_flow",
}

# Fixed source priority: lower number wins when the same canonical metric
# arrives from multiple tools.
_SOURCE_PRIORITY = {
    "get_fundamentals": 1,
    "get_balance_sheet": 2,
    "get_cashflow": 3,
    "get_income_statement": 4,
}


def curate_financial_metrics(metrics: list[FinancialMetric]) -> list[FinancialMetric]:
    """Keep at most one metric per canonical name, ordered by the whitelist.

    Deduplication rules:
    - valuation/overview metrics prefer ``get_fundamentals``;
    - balance-sheet items prefer ``get_balance_sheet``;
    - cash-flow items prefer ``get_cashflow``;
    - income-statement items prefer ``get_income_statement``;
    - same source: latest period wins.
    """
    best: dict[str, FinancialMetric] = {}
    for metric in metrics:
        canonical = FINANCIAL_METRIC_ALIASES.get(metric.name)
        if canonical is None:
            continue
        candidate = metric.model_copy(update={"name": canonical})
        current = best.get(canonical)
        if current is None or _better_metric(candidate, current):
            best[canonical] = candidate
    return [best[name] for name in FINANCIAL_METRIC_ORDER if name in best]


def _better_metric(candidate: FinancialMetric, current: FinancialMetric) -> bool:
    candidate_rank = _SOURCE_PRIORITY.get(candidate.source, 99)
    current_rank = _SOURCE_PRIORITY.get(current.source, 99)
    if candidate_rank != current_rank:
        return candidate_rank < current_rank
    # Same source: newest period wins (ISO dates sort lexicographically).
    return candidate.period > current.period


# ---------------------------------------------------------------------------
# Number formatting for display (schema keeps raw floats)
# ---------------------------------------------------------------------------

def format_financial_value(value: float, unit: str = "") -> str:
    """186.38B / 42.94x / 34.19% / 4.78 — never 1.86383e+11."""
    if unit == "%":
        return f"{value:.2f}%"
    if unit == "x":
        return f"{value:.2f}x"
    return _format_magnitude(value)


def _format_magnitude(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1e12:
        return f"{value / 1e12:.2f}T"
    if absolute >= 1e9:
        return f"{value / 1e9:.2f}B"
    if absolute >= 1e6:
        return f"{value / 1e6:.2f}M"
    if absolute >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:.2f}"


def metric_display_name(canonical_name: str) -> str:
    return FINANCIAL_METRIC_LABELS.get(canonical_name, canonical_name)


# ---------------------------------------------------------------------------
# Historical financial series (Revenue / Net Income)
# ---------------------------------------------------------------------------

# Explicit alias map only — no fuzzy "income" substring matching.
FINANCIAL_SERIES_ALIASES = {
    "revenue": {
        "total_revenue",
        "revenue",
        "operating_revenue",
    },
    "net_income": {
        "net_income",
        "net_income_common_stockholders",
        "net_income_from_continuing_operation_net_minority_interest",
    },
}

FINANCIAL_SERIES_LABELS = {
    "revenue": "Revenue",
    "net_income": "Net Income",
}

MIN_SERIES_POINTS = 2


def build_financial_series(history: list[dict[str, Any]]) -> list[FinancialSeries]:
    """Assemble Revenue / Net Income series from multi-period tool history.

    Points are sorted oldest → newest; only series with >= 2 points are
    created. A single point stays a P0 metric but never becomes a chart.
    """
    series_by_canonical: dict[str, dict[str, Any]] = {}
    for entry in history:
        name = str(entry.get("name") or "")
        canonical = next(
            (key for key, aliases in FINANCIAL_SERIES_ALIASES.items() if name in aliases),
            None,
        )
        if canonical is None:
            continue
        period = str(entry.get("period") or "")
        value = entry.get("value")
        if not period or not isinstance(value, (int, float)):
            continue
        series = series_by_canonical.setdefault(
            canonical,
            {"name": canonical, "source": str(entry.get("source") or ""), "points": {}},
        )
        if series["source"] != str(entry.get("source") or "") and series["source"]:
            continue  # keep the first tool's series only
        series["source"] = str(entry.get("source") or "")
        series["points"][period] = float(value)

    result = []
    for canonical, data in series_by_canonical.items():
        sorted_pairs = sorted(data["points"].items())
        points = []
        prev_value = None
        for period, value in sorted_pairs:
            growth = None
            if prev_value is not None and prev_value != 0:
                growth = (value - prev_value) / abs(prev_value) * 100
            points.append(
                FinancialSeriesPoint(period=period, value=value, growth_pct=growth)
            )
            prev_value = value
        if len(points) < MIN_SERIES_POINTS:
            continue
        result.append(
            FinancialSeries(
                name=canonical,
                label=FINANCIAL_SERIES_LABELS.get(canonical, canonical),
                unit="",
                source=data["source"],
                points=points,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Moving averages (deterministic derivation, never a tool re-call)
# ---------------------------------------------------------------------------


def attach_moving_average(
    points: list[Any],
    window: int,
    key: str,
) -> None:
    """Attach a trailing simple moving average to each MarketPoint.

    Uses only closes at or before the point's own date (no look-ahead). The
    first legal value lands on the ``window``-th point; earlier points keep
    no indicator. Original closes are never modified.
    """
    if window <= 1:
        raise ValueError("window must be >= 2")
    closes = [point.close for point in points]
    running_sum = 0.0
    for index, point in enumerate(points):
        running_sum += closes[index]
        if index >= window:
            running_sum -= closes[index - window]
        if index >= window - 1:
            point.indicators[key] = running_sum / window


# ---------------------------------------------------------------------------
# News timeline curation
# ---------------------------------------------------------------------------

MAX_TIMELINE_ITEMS = 10
MAX_GLOBAL_ITEMS = 3

_IMPORTANCE_RANK = {"high": 0, "medium": 1, "low": 2, "unknown": 3}


def curate_timeline(
    items: list[TimelineItem],
    *,
    max_items: int = MAX_TIMELINE_ITEMS,
    max_global: int = MAX_GLOBAL_ITEMS,
) -> list[TimelineItem]:
    """Deduplicate, sort, and bound the news timeline.

    Rules:
    - same URL or same normalized title → one entry;
    - company news precedes global news;
    - dated items sorted newest-first; undated items go last;
    - total <= max_items, global <= max_global.
    """
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[TimelineItem] = []
    for item in items:
        url_key = item.url.strip().lower()
        title_key = normalize_event_title(item.event)
        if url_key and url_key in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        unique.append(item)

    company = _dated_sort([item for item in unique if item.scope != "global"])
    global_items = _dated_sort([item for item in unique if item.scope == "global"])[:max_global]
    return (company + global_items)[:max_items]


# ---------------------------------------------------------------------------
# P2: Deterministic market risk snapshot
# ---------------------------------------------------------------------------


def build_market_risk_snapshot(points: list[Any]) -> Any | None:
    """Build a ``MarketRiskSnapshot`` from chronologically ascending OHLCV points.

    No look-ahead. Insufficient data → field None, not synthetic zero.
    """
    from finmindagent.reporting.schemas import MarketRiskSnapshot
    import statistics as _stats
    import math as _math

    n = len(points)
    if n < 2:
        return None
    closes = [p.close for p in points]
    highs = [p.high if p.high is not None else p.close for p in points]
    lows = [p.low if p.low is not None else p.close for p in points]
    volumes = [p.volume for p in points if p.volume is not None and p.volume > 0]

    snapshot = MarketRiskSnapshot(as_of=points[-1].date)

    # 5-session return
    if n >= 6 and closes[-6] and closes[-6] != 0:
        snapshot.return_5d_pct = (closes[-1] / closes[-6] - 1) * 100

    # 20-session return
    if n >= 21 and closes[-21] and closes[-21] != 0:
        snapshot.return_20d_pct = (closes[-1] / closes[-21] - 1) * 100

    # 20-session realized volatility
    if n >= 21:
        returns = [closes[i] / closes[i - 1] - 1 for i in range(n - 20, n)]
        if len(returns) >= 2:
            snapshot.realized_volatility_20d_pct = _stats.stdev(returns) * _math.sqrt(252) * 100

    # Trailing max drawdown (up to 60 sessions)
    window = closes[-60:] if n >= 60 else closes
    peak = window[0]
    max_dd = 0.0
    for c in window:
        if c > peak:
            peak = c
        if peak and peak != 0:
            dd = (c / peak - 1) * 100
            if dd < max_dd:
                max_dd = dd
    snapshot.max_drawdown_60_sessions_pct = max_dd

    # ATR14 — requires actual high/low; do not substitute close for missing high/low
    if n >= 15 and points[-1].close and points[-1].close > 0:
        trs = []
        all_valid = True
        for i in range(n - 14, n):
            h = points[i].high
            l = points[i].low
            pc = points[i - 1].close if i > 0 else None
            if h is None or l is None or pc is None or pc == 0:
                all_valid = False
                break
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        if all_valid and trs:
            atr14 = sum(trs) / len(trs)
            snapshot.atr14_pct = atr14 / closes[-1] * 100

    # Volume ratio — requires latest and 20 prior valid positive volumes
    if len(points) >= 21 and points[-1].volume is not None and points[-1].volume > 0:
        latest_vol = points[-1].volume
        prev_20 = [p.volume for p in points[-21:-1] if p.volume is not None and p.volume > 0]
        if len(prev_20) >= 20:
            avg_prev = sum(prev_20[:20]) / 20
            if avg_prev > 0:
                snapshot.volume_ratio_20d = latest_vol / avg_prev

    # Return None only if nothing computable
    if all(v is None for v in (
        snapshot.return_5d_pct, snapshot.return_20d_pct,
        snapshot.realized_volatility_20d_pct, snapshot.max_drawdown_60_sessions_pct,
        snapshot.atr14_pct, snapshot.volume_ratio_20d,
    )):
        return None
    return snapshot


def _dated_sort(items: list[TimelineItem]) -> list[TimelineItem]:
    """Dated items newest-first; undated items go after all dated ones.

    Within the same date, importance priority high > medium > low > unknown.
    """
    dated = sorted(
        (item for item in items if item.date),
        key=lambda item: (
            item.date,
            -_IMPORTANCE_RANK.get(item.importance, 3),
        ),
        reverse=True,
    )
    undated = [item for item in items if not item.date]
    return dated + undated


def normalize_event_title(title: str) -> str:
    """Deterministic title normalization for exact dedup/event matching.

    Lowercase, trim, collapse whitespace, drop punctuation — never fuzzy
    matching across different events.
    """
    lowered = title.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s]", "", lowered)
    return lowered


# ---------------------------------------------------------------------------
# Risk items merge (from the three risk analysts)
# ---------------------------------------------------------------------------

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
MAX_RISK_ITEMS = 8
# Deterministic tie breaker for same-severity duplicates (not a trust ranking).
_RISK_ROLE_PRIORITY = {
    "neutral_risk_analyst": 1,
    "conservative_risk_analyst": 2,
    "aggressive_risk_analyst": 3,
}


def _risk_identity(entry: dict[str, Any]) -> str:
    """Return the canonical identity key for a risk entry.

    Prefer the structured ``risk_key`` field when present (Phase C P0).
    Fall back to the legacy normalized-name identity for entries that lack
    a ``risk_key`` (historical data or tests without canonical keys).
    """
    risk_key = entry.get("risk_key")
    if isinstance(risk_key, str) and risk_key.strip():
        return risk_key.strip()
    # Legacy fallback: normalized display name
    name = str(entry.get("name") or "").strip()
    if name:
        return "name:" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return ""


def merge_risk_items(risk_entries: list[dict[str, Any]]) -> list[RiskItem]:
    """Merge structured risks from all three analysts.

    - Same canonical ``risk_key`` → one item.
    - Legacy entries without ``risk_key`` fall back to normalized-name identity.
    - Severity: highest wins; evidence follows the highest-severity source.
    - Equal severity → deterministic role priority tie-break.
    - Display name: first non-empty human-readable name from the highest-severity
      contributor; deterministic tie-break by analyst order when severities tie.
    - P1: category/horizon_bucket follow winning severity entry.
    - P1: probability/downside ranges from all entries with numeric estimates;
      expected-loss only from entries individually providing both numbers.
      Never average, never fabricate missing values.
    - Evidence deduplicated; sources merged from all contributing analysts.
    - Estimate bases deduplicated across contributors.
    - Final list sorted severity desc → canonical key, capped at MAX_RISK_ITEMS.
    """
    best: dict[str, RiskItem] = {}
    # Collect raw entries per identity for P1 range aggregation
    raw_by_id: dict[str, list[dict[str, Any]]] = {}
    for entry in risk_entries:
        identity = _risk_identity(entry)
        if not identity:
            continue
        severity = str(entry.get("severity") or "").strip().lower()
        role = str(entry.get("role") or "")
        if severity not in SEVERITY_RANK:
            continue
        raw_by_id.setdefault(identity, []).append(entry)
        name = str(entry.get("name") or "").strip()
        category = str(entry.get("category") or "other")
        horizon_bucket = str(entry.get("horizon_bucket") or "unknown")
        # P2 (RPT-26): preserve canonical risk_key through merge when present.
        risk_key = str(entry.get("risk_key") or "").strip()
        candidate = RiskItem(
            name=name,
            severity=severity,
            horizon=str(entry.get("horizon") or ""),
            evidence=str(entry.get("evidence") or ""),
            sources=[role] if role else [],
            category=category,
            horizon_bucket=horizon_bucket,
            risk_key=risk_key,
        )
        current = best.get(identity)
        if current is None:
            best[identity] = candidate
        else:
            # Always merge the new analyst's source into the best entry
            for source in candidate.sources:
                if source and source not in current.sources:
                    current.sources.append(source)
            if SEVERITY_RANK[candidate.severity] > SEVERITY_RANK[current.severity]:
                # Propagate merged sources to the higher-severity candidate
                candidate.sources = current.sources
                best[identity] = candidate
            # When severities are equal, keep the first entry (deterministic
            # by role priority in caller). Lower-severity entries only
            # contribute sources, not other fields.

    # P1: aggregate probability / downside / expected-loss ranges
    for identity, item in best.items():
        entries = raw_by_id.get(identity, [])
        probs = []
        downsides = []
        expected_losses = []
        bases = []
        evidence_set: set[str] = set()
        for e in entries:
            ev = str(e.get("evidence") or "").strip()
            if ev and ev not in evidence_set:
                evidence_set.add(ev)
            p = e.get("probability_pct")
            d = e.get("downside_impact_pct")
            if isinstance(p, (int, float)) and p is not None:
                probs.append(float(p))
            if isinstance(d, (int, float)) and d is not None:
                downsides.append(float(d))
            # expected loss only if BOTH numbers from same entry
            if isinstance(p, (int, float)) and isinstance(d, (int, float)) and p is not None and d is not None:
                expected_losses.append(float(p) / 100.0 * float(d))
            basis = str(e.get("estimate_basis") or "")
            if basis and basis not in bases:
                bases.append(basis)

        if probs:
            item.probability_min_pct = min(probs)
            item.probability_max_pct = max(probs)
        if downsides:
            item.downside_impact_min_pct = min(downsides)
            item.downside_impact_max_pct = max(downsides)
        if expected_losses:
            item.expected_loss_min_pct = min(expected_losses)
            item.expected_loss_max_pct = max(expected_losses)
        if bases:
            item.estimate_basis = bases

    ordered = sorted(
        best.values(),
        key=lambda item: (-SEVERITY_RANK[item.severity], item.name),
    )
    return ordered[:MAX_RISK_ITEMS]


# ---------------------------------------------------------------------------
# P0: Deterministic technical indicator derivation
# ---------------------------------------------------------------------------


def derive_ema(closes: list[float], span: int) -> list[float | None]:
    """Exponential moving average seeded from the first close.

    ``ema_t = alpha * close_t + (1 - alpha) * ema_{t-1}`` where
    ``alpha = 2 / (span + 1)``. The first point is None (seeding point).
    """
    if span < 2 or len(closes) < 2:
        return [None] * len(closes)
    alpha = 2.0 / (span + 1)
    result: list[float | None] = [None]
    ema = closes[0]
    for close in closes[1:]:
        ema = alpha * close + (1 - alpha) * ema
        result.append(ema)
    return result


def derive_macd(closes: list[float]) -> dict[str, list[float | None]]:
    """MACD from EMA12 / EMA26 with signal EMA9, seeded from first close.

    Returns ``{"macd": [...], "macd_signal": [...], "macd_histogram": [...]}``.
    """
    ema12 = derive_ema(closes, 12)
    ema26 = derive_ema(closes, 26)
    n = len(closes)
    macd: list[float | None] = [None] * n
    for i in range(n):
        if ema12[i] is not None and ema26[i] is not None:
            macd[i] = ema12[i] - ema26[i]  # type: ignore[operator]

    # Signal: EMA9 of MACD values (seed from first non-None MACD)
    signal: list[float | None] = [None] * n
    histogram: list[float | None] = [None] * n
    macd_values = [v for v in macd if v is not None]
    if len(macd_values) < 2:
        return {"macd": macd, "macd_signal": signal, "macd_histogram": histogram}

    alpha_s = 2.0 / (9 + 1)
    sig_ema = macd_values[0]
    sig_index = macd.index(macd_values[0])
    signal[sig_index] = sig_ema
    for idx in range(sig_index + 1, n):
        if macd[idx] is not None:
            sig_ema = alpha_s * macd[idx] + (1 - alpha_s) * sig_ema  # type: ignore[operator]
            signal[idx] = sig_ema
            histogram[idx] = macd[idx] - sig_ema  # type: ignore[operator]

    return {"macd": macd, "macd_signal": signal, "macd_histogram": histogram}


def derive_rsi(closes: list[float], period: int = 14) -> list[float | None]:
    """RSI with Wilder smoothing. No value until enough deltas exist.

    - zero-loss window → 100
    - zero-gain window → 0
    - both zero → 50
    """
    n = len(closes)
    if n < period + 1:
        return [None] * n
    result: list[float | None] = [None] * n
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains[i] = delta
        elif delta < 0:
            losses[i] = -delta
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period
    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0 and avg_gain == 0:
            result[i] = 50.0
        elif avg_loss == 0:
            result[i] = 100.0
        elif avg_gain == 0:
            result[i] = 0.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)
    return result


def derive_bollinger(
    closes: list[float], period: int = 20, num_std: float = 2.0
) -> dict[str, list[float | None]]:
    """Bollinger Bands: middle = SMA(period), bands = middle ± num_std * std_dev.

    Standard deviation is population std over the same window.
    """
    n = len(closes)
    middle: list[float | None] = [None] * n
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n
    if n < period:
        return {"bollinger_middle": middle, "bollinger_upper": upper, "bollinger_lower": lower}
    import math as _math
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        sma = sum(window) / period
        variance = sum((x - sma) ** 2 for x in window) / period
        std = _math.sqrt(variance)
        middle[i] = sma
        upper[i] = sma + num_std * std
        lower[i] = sma - num_std * std
    return {"bollinger_middle": middle, "bollinger_upper": upper, "bollinger_lower": lower}


def derive_kdj(
    highs: list[float], lows: list[float], closes: list[float], period: int = 9
) -> dict[str, list[float | None]]:
    """KDJ indicator: stochastic with 9-period rolling high/low.

    - RSV = 100 * (close - low9) / (high9 - low9)
    - initial K = D = 50
    - K = 2/3 * K_prev + 1/3 * RSV
    - D = 2/3 * D_prev + 1/3 * K
    - J = 3 * K - 2 * D
    - high9 == low9 → RSV = 50
    """
    n = len(closes)
    k_vals: list[float | None] = [None] * n
    d_vals: list[float | None] = [None] * n
    j_vals: list[float | None] = [None] * n
    if n < period:
        return {"kdj_k": k_vals, "kdj_d": d_vals, "kdj_j": j_vals}
    k_prev = 50.0
    d_prev = 50.0
    for i in range(period - 1, n):
        high9 = max(highs[i - period + 1 : i + 1])
        low9 = min(lows[i - period + 1 : i + 1])
        if high9 == low9:
            rsv = 50.0
        else:
            rsv = 100.0 * (closes[i] - low9) / (high9 - low9)
        k = 2.0 / 3.0 * k_prev + 1.0 / 3.0 * rsv
        d = 2.0 / 3.0 * d_prev + 1.0 / 3.0 * k
        j = 3.0 * k - 2.0 * d
        k_vals[i] = k
        d_vals[i] = d
        j_vals[i] = j
        k_prev = k
        d_prev = d
    return {"kdj_k": k_vals, "kdj_d": d_vals, "kdj_j": j_vals}


def attach_derived_indicators(
    market_points: list[Any],
    indicator_series: list[dict[str, Any]] | None = None,
) -> None:
    """Merge tool-provided indicators by exact date, then fill remaining slots
    with deterministic derivations from OHLCV. Tool values always win over
    derived values on the same date and key.

    This function modifies ``market_points`` in place.
    """
    if not market_points:
        return

    # Phase 1: merge tool indicator series by exact date
    if indicator_series:
        date_index = {point.date: idx for idx, point in enumerate(market_points)}
        for series in indicator_series:
            key = series.get("name", "")
            if not key:
                continue
            for pt in series.get("points") or []:
                date = pt.get("date", "")
                value = pt.get("value")
                if date not in date_index or value is None:
                    continue
                idx = date_index[date]
                market_points[idx].indicators[key] = float(value)

    closes = [point.close for point in market_points]
    highs = [point.high if point.high is not None else point.close for point in market_points]
    lows = [point.low if point.low is not None else point.close for point in market_points]
    n = len(market_points)

    # Phase 2: fill missing indicators deterministically
    _indicators_keys = {
        "macd", "macd_signal", "macd_histogram",
        "rsi", "bollinger_middle", "bollinger_upper", "bollinger_lower",
        "kdj_k", "kdj_d", "kdj_j",
    }

    def _set_if_missing(idx: int, key: str, value: float | None) -> None:
        if value is not None and key not in market_points[idx].indicators:
            market_points[idx].indicators[key] = value

    # MACD
    macd_result = derive_macd(closes)
    for key in ("macd", "macd_signal", "macd_histogram"):
        values = macd_result[key]
        for i in range(n):
            _set_if_missing(i, key, values[i])

    # RSI
    rsi_result = derive_rsi(closes)
    for i in range(n):
        _set_if_missing(i, "rsi", rsi_result[i])

    # Bollinger
    bb_result = derive_bollinger(closes)
    for key in ("bollinger_middle", "bollinger_upper", "bollinger_lower"):
        values = bb_result[key]
        for i in range(n):
            _set_if_missing(i, key, values[i])

    # KDJ
    kdj_result = derive_kdj(highs, lows, closes)
    for key in ("kdj_k", "kdj_d", "kdj_j"):
        values = kdj_result[key]
        for i in range(n):
            _set_if_missing(i, key, values[i])


def build_technical_snapshot(
    market_points: list[Any],
    source: str = "",
) -> Any | None:
    """Build a ``TechnicalSnapshot`` from the latest market data point.

    Trend rules:
    - bullish when close > sma20 > sma50
    - bearish when close < sma20 < sma50
    - mixed when both SMAs exist but neither ordering holds
    - unknown otherwise

    Support/resistance are simple and reproducible:
    - candidate support = latest Bollinger lower + lowest low in trailing 20
    - candidate resistance = latest Bollinger upper + highest high in trailing 20
    - keep only support < close, resistance > close
    - dedupe after rounding to 4 decimals; max 2 per side
    """
    from finmindagent.reporting.schemas import TechnicalSnapshot

    if not market_points:
        return None

    latest = market_points[-1]
    ind = latest.indicators
    close = latest.close
    sma20 = ind.get("sma20")
    sma50 = ind.get("sma50")

    if close is not None and sma20 is not None and sma50 is not None:
        if close > sma20 > sma50:
            trend = "bullish"
        elif close < sma20 < sma50:
            trend = "bearish"
        else:
            trend = "mixed"
    else:
        trend = "unknown"

    # Support / resistance from trailing 20 points
    trailing = market_points[-20:]
    lows_list = [p.low for p in trailing if p.low is not None]
    highs_list = [p.high for p in trailing if p.high is not None]
    supports: list[float] = []
    resistances: list[float] = []

    bb_lower = ind.get("bollinger_lower")
    if bb_lower is not None:
        supports.append(bb_lower)
    if lows_list:
        supports.append(min(lows_list))

    bb_upper = ind.get("bollinger_upper")
    if bb_upper is not None:
        resistances.append(bb_upper)
    if highs_list:
        resistances.append(max(highs_list))

    # Filter and dedupe
    supports = [s for s in supports if s < close]
    resistances = [r for r in resistances if r > close]
    supports = _dedupe_floats(supports, decimals=4)[:2]
    resistances = _dedupe_floats(resistances, decimals=4)[:2]

    has_any = any(
        ind.get(key) is not None
        for key in (
            "sma20", "sma50", "rsi", "macd", "macd_signal", "macd_histogram",
            "bollinger_middle", "bollinger_upper", "bollinger_lower",
            "kdj_k", "kdj_d", "kdj_j",
        )
    )

    if not has_any and trend == "unknown" and not supports and not resistances:
        return None

    return TechnicalSnapshot(
        as_of=latest.date,
        trend=trend,
        latest_close=close,
        sma20=sma20,
        sma50=sma50,
        rsi=ind.get("rsi"),
        macd=ind.get("macd"),
        macd_signal=ind.get("macd_signal"),
        macd_histogram=ind.get("macd_histogram"),
        bollinger_middle=ind.get("bollinger_middle"),
        bollinger_upper=ind.get("bollinger_upper"),
        bollinger_lower=ind.get("bollinger_lower"),
        kdj_k=ind.get("kdj_k"),
        kdj_d=ind.get("kdj_d"),
        kdj_j=ind.get("kdj_j"),
        support_levels=supports,
        resistance_levels=resistances,
        source=source if has_any else "",
    )


def _dedupe_floats(values: list[float], decimals: int = 4) -> list[float]:
    """Deduplicate float values after rounding to ``decimals`` places."""
    seen: set[int] = set()
    result: list[float] = []
    for v in values:
        key = round(v, decimals)
        # Use the integer representation for dedup
        int_key = int(round(key * (10 ** decimals)))
        if int_key not in seen:
            seen.add(int_key)
            result.append(v)
    return result


# ---------------------------------------------------------------------------
# P2 (RPT-21): Deterministic EPS revision diagnostics
# ---------------------------------------------------------------------------


def build_estimate_revision_diagnostics(
    consensus: Any | None,
) -> list[EstimateRevisionDiagnostic]:
    """Build one diagnostic row per ``AnalystConsensus.eps_revisions`` row.

    Deterministic rules (never parse prose, never call provider/LLM):

    - ``change_30d_pct`` = (current - days_30_ago) / abs(days_30_ago) * 100,
      only when both endpoints exist and days_30_ago != 0; otherwise None.
    - ``direction`` = rising / falling / flat when both endpoints exist
      (numeric comparison); unknown when either endpoint is missing.
      A zero denominator still allows direction via numeric comparison.
    - ``net_revisions_30d`` = up_30d - down_30d only when both counts exist;
      a legal 0 is preserved, never treated as missing.
    - Partial rows survive: trend-only and counts-only rows both produce
      diagnostics with missing fields left None.
    - No ``eps_revisions`` or None consensus → [] (never placeholder rows).
    """
    if consensus is None:
        return []
    revisions = getattr(consensus, "eps_revisions", None)
    if not revisions:
        return []

    diagnostics: list[EstimateRevisionDiagnostic] = []
    for point in revisions:
        current = getattr(point, "current", None)
        days_30_ago = getattr(point, "days_30_ago", None)
        up_30d = getattr(point, "up_30d", None)
        down_30d = getattr(point, "down_30d", None)

        # 30-day percentage change (zero denominator → None, never 0)
        change_30d_pct = None
        if current is not None and days_30_ago is not None and days_30_ago != 0:
            change_30d_pct = (float(current) - float(days_30_ago)) / abs(float(days_30_ago)) * 100

        # Direction via numeric comparison; missing endpoint → unknown
        if current is None or days_30_ago is None:
            direction = "unknown"
        elif current > days_30_ago:
            direction = "rising"
        elif current < days_30_ago:
            direction = "falling"
        else:
            direction = "flat"

        # Net revisions: both counts required; 0 preserved
        net_revisions = None
        if up_30d is not None and down_30d is not None:
            net_revisions = int(up_30d) - int(down_30d)

        diagnostics.append(
            EstimateRevisionDiagnostic(
                period=str(getattr(point, "period", "")),
                current=float(current) if current is not None else None,
                days_30_ago=float(days_30_ago) if days_30_ago is not None else None,
                change_30d_pct=change_30d_pct,
                up_30d=int(up_30d) if up_30d is not None else None,
                down_30d=int(down_30d) if down_30d is not None else None,
                net_revisions_30d=net_revisions,
                direction=direction,
                source=str(getattr(consensus, "source", "") or ""),
            )
        )
    return diagnostics


# ---------------------------------------------------------------------------
# P2 (RPT-22): Deterministic earnings realization diagnostic
# ---------------------------------------------------------------------------


def build_earnings_realization_diagnostic(
    context: Any | None,
) -> EarningsRealizationDiagnostic | None:
    """Build a deterministic earnings realization summary from structured
    ``EarningsContext.surprises``. Never parses prose, never calls LLM/provider.

    Rules:
    - ``surprise_pct is None`` is excluded from sample statistics.
    - surprise_pct > 0 → beat; < 0 → miss; == 0 → inline (no tolerance).
    - beat_rate = beat_count / sample_size * 100 only when sample_size > 0;
      sample_size == 0 → beat_rate None (NOT 0%).
    - mean_surprise = mean of valid values only (raw float, no rounding).
    - latest row = max by ``report_date`` over ALL rows (including None
      surprise); latest_surprise_pct may be None.
    - context None or surprises == [] → None (no placeholder diagnostic).
    """
    if context is None:
        return None
    surprises = getattr(context, "surprises", None)
    if not surprises:
        return None

    valid_values = [
        float(point.surprise_pct)
        for point in surprises
        if point.surprise_pct is not None
    ]
    sample_size = len(valid_values)

    beat_count = sum(1 for v in valid_values if v > 0)
    miss_count = sum(1 for v in valid_values if v < 0)
    inline_count = sum(1 for v in valid_values if v == 0)

    beat_rate_pct = (beat_count / sample_size * 100) if sample_size > 0 else None
    mean_surprise_pct = (sum(valid_values) / len(valid_values)) if valid_values else None

    # Latest row over ALL surprises (including None surprise), by report_date.
    latest = max(surprises, key=lambda point: str(point.report_date or ""))
    latest_report_date = str(latest.report_date or "")
    latest_surprise_pct = (
        float(latest.surprise_pct) if latest.surprise_pct is not None else None
    )

    return EarningsRealizationDiagnostic(
        sample_size=sample_size,
        beat_count=beat_count,
        miss_count=miss_count,
        inline_count=inline_count,
        beat_rate_pct=beat_rate_pct,
        mean_surprise_pct=mean_surprise_pct,
        latest_surprise_pct=latest_surprise_pct,
        latest_report_date=latest_report_date,
        source=str(getattr(context, "source", "") or ""),
    )


# ---------------------------------------------------------------------------
# P2 (RPT-25): Deterministic ownership snapshot diagnostic
# ---------------------------------------------------------------------------


def build_ownership_diagnostic(
    snapshot: Any | None,
) -> OwnershipDiagnostic | None:
    """Build a point-in-time ownership diagnostic from the normalized
    ``OwnershipSnapshot``. Never infers flow, accumulation, or buy/sell.

    Rules:
    - snapshot None → None (no placeholder).
    - Diagnostic exists whenever a snapshot exists, even with zero retained
      holders (concentration then None, not 0).
    - Concentration uses only retained institutional holders with
      ``pct_out is not None``; None is never zero-filled.
    - top_holder = valid_pct[0]; top5 = sum(valid_pct[:5]) — the adapter
      contract already sorts holders by pct_out desc.
    - ``retained_holder_count`` = number of retained institutional holder
      records (NOT ``institutions_count`` provider total).
    """
    if snapshot is None:
        return None

    holders = getattr(snapshot, "institutional_holders", None) or []
    valid_pct = [
        float(h.pct_out)
        for h in holders
        if h.pct_out is not None
    ]

    top_holder_pct = valid_pct[0] if valid_pct else None
    top5_institutional_pct = sum(valid_pct[:5]) if valid_pct else None

    return OwnershipDiagnostic(
        as_of=str(getattr(snapshot, "as_of", "") or ""),
        institutional_pct=(
            float(snapshot.institutions_pct)
            if snapshot.institutions_pct is not None else None
        ),
        insiders_pct=(
            float(snapshot.insiders_pct)
            if snapshot.insiders_pct is not None else None
        ),
        top_holder_pct=top_holder_pct,
        top5_institutional_pct=top5_institutional_pct,
        retained_holder_count=len(holders),
        source=str(getattr(snapshot, "source", "") or ""),
    )


# ---------------------------------------------------------------------------
# P2 (RPT-26): Stable structured evidence registry
# ---------------------------------------------------------------------------

# Deterministic domain priority matching schema order (Task 5).
_EVIDENCE_DOMAIN_ORDER = {
    "market": 0,
    "valuation": 1,
    "earnings": 2,
    "estimates": 3,
    "ownership": 4,
    "news": 5,
    "sentiment": 6,
    "risk": 7,
    "financial_quality": 8,
}


def _sign_direction(value: float | None) -> str:
    """Deterministic direction from numeric sign (None → unknown)."""
    if value is None:
        return "unknown"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def _revision_direction(direction: str) -> str:
    """Map EstimateRevisionDiagnostic direction to evidence direction."""
    return {"rising": "positive", "falling": "negative", "flat": "neutral"}.get(
        direction, "unknown"
    )


def build_evidence_registry(report: Any) -> list[Any]:
    """Build the deterministic evidence registry from structured ReportSchema.

    Only consumes structured fields — no state prose, no LLM. Reuses stable
    structured IDs (earnings evidence_id, holder_id, risk_key). Exact
    evidence_id dedupe only; items without a valid stable identity are
    dropped. Registry order: domain priority → as_of descending →
    evidence_id ascending (never input order).
    """
    from finmindagent.reporting.schemas import EvidenceDirection, ResearchEvidenceItem

    items: list[ResearchEvidenceItem] = []

    # --- MARKET ---
    mrs = getattr(report, "market_risk_snapshot", None)
    if mrs is not None:
        as_of = str(getattr(mrs, "as_of", "") or "")
        if getattr(mrs, "return_20d_pct", None) is not None:
            items.append(ResearchEvidenceItem(
                evidence_id=f"market:return_20d:{as_of}",
                domain="market",
                title="20-day return",
                direction=EvidenceDirection(_sign_direction(mrs.return_20d_pct)),
                value_text=f"{mrs.return_20d_pct:.2f}%",
                as_of=as_of,
                source=str(getattr(mrs, "source", "") or ""),
            ))
        if getattr(mrs, "max_drawdown_60_sessions_pct", None) is not None:
            items.append(ResearchEvidenceItem(
                evidence_id=f"market:drawdown_60:{as_of}",
                domain="market",
                title="60-session max drawdown",
                direction=EvidenceDirection.UNKNOWN,  # magnitude is context
                value_text=f"{mrs.max_drawdown_60_sessions_pct:.2f}%",
                as_of=as_of,
                source=str(getattr(mrs, "source", "") or ""),
            ))

    # --- VALUATION ---
    vc = getattr(report, "valuation_context", None)
    consensus_as_of = ""
    ac = getattr(report, "analyst_consensus", None)
    if ac is not None:
        consensus_as_of = str(getattr(ac, "as_of", "") or "")
    if vc is not None and getattr(vc, "analyst_target_mean", None) is not None:
        items.append(ResearchEvidenceItem(
            evidence_id=f"analyst:target_mean:{consensus_as_of}",
            domain="valuation",
            title="Analyst consensus mean target",
            direction=EvidenceDirection(_sign_direction(getattr(vc, "mean_target_upside_pct", None))),
            value_text=f"{vc.analyst_target_mean:.2f}",
            as_of=consensus_as_of,
            source=str(getattr(ac, "source", "") or "") if ac else "",
        ))

    # --- EARNINGS (reuse structured evidence_id) ---
    ec = getattr(report, "earnings_context", None)
    if ec is not None:
        for point in getattr(ec, "surprises", []) or []:
            eid = str(getattr(point, "evidence_id", "") or "")
            if not eid:
                continue
            items.append(ResearchEvidenceItem(
                evidence_id=eid,
                domain="earnings",
                title="Earnings surprise",
                direction=EvidenceDirection(_sign_direction(point.surprise_pct)),
                value_text=(
                    f"{point.surprise_pct:.2f}%" if point.surprise_pct is not None else ""
                ),
                as_of=str(getattr(point, "report_date", "") or ""),
                source=str(getattr(point, "source", "") or ""),
                source_ref=eid,
            ))

    # --- ESTIMATES ---
    if ac is not None:
        for diag in getattr(report, "estimate_revision_diagnostics", []) or []:
            period = str(getattr(diag, "period", "") or "")
            items.append(ResearchEvidenceItem(
                evidence_id=f"analyst:eps_revision:{period}:{consensus_as_of}",
                domain="estimates",
                title=f"EPS revision ({period})",
                direction=EvidenceDirection(_revision_direction(getattr(diag, "direction", "unknown"))),
                value_text=(
                    f"{diag.change_30d_pct:.2f}%" if diag.change_30d_pct is not None else ""
                ),
                as_of=consensus_as_of,
                source=str(getattr(ac, "source", "") or ""),
            ))

    # --- OWNERSHIP ---
    od = getattr(report, "ownership_diagnostic", None)
    osnap = getattr(report, "ownership_snapshot", None)
    if od is not None:
        items.append(ResearchEvidenceItem(
            evidence_id=f"ownership:summary:{od.as_of}",
            domain="ownership",
            title="Ownership summary",
            direction=EvidenceDirection.UNKNOWN,  # snapshot only, never flow
            value_text=f"Institutional {od.institutional_pct}% / Insiders {od.insiders_pct}%"
                       if od.institutional_pct is not None or od.insiders_pct is not None else "",
            as_of=od.as_of,
            source=od.source,
        ))
    if osnap is not None:
        for holder in getattr(osnap, "institutional_holders", []) or []:
            hid = str(getattr(holder, "holder_id", "") or "")
            if not hid:
                continue
            items.append(ResearchEvidenceItem(
                evidence_id=hid,
                domain="ownership",
                title=f"Holder: {getattr(holder, 'holder', '') or ''}",
                direction=EvidenceDirection.UNKNOWN,  # point-in-time snapshot
                value_text=f"{holder.pct_out}%" if holder.pct_out is not None else "",
                as_of=str(getattr(holder, "date_reported", "") or ""),
                source=str(getattr(holder, "source", "") or ""),
                source_ref=hid,
            ))

    # --- SENTIMENT (grounded structured signals only) ---
    for signal in getattr(report, "sentiment_signals", []) or []:
        title = str(getattr(signal, "title", "") or "")
        if not title:
            continue
        # Exact deterministic identity: normalized title, spaces → "_"
        norm = normalize_event_title(title).replace(" ", "_")
        if not norm:
            continue
        direction = str(getattr(signal, "direction", "unknown") or "unknown")
        if direction not in ("positive", "negative", "neutral", "mixed", "unknown"):
            direction = "unknown"
        # source_ref: only exact grounded refs (builder filters them)
        refs = [str(r) for r in (getattr(signal, "evidence_refs", []) or [])]
        source_ref = refs[0] if refs else ""
        items.append(ResearchEvidenceItem(
            evidence_id=f"sentiment:{norm}",
            domain="sentiment",
            title=title,
            direction=EvidenceDirection(direction),
            as_of=str(getattr(signal, "as_of", "") or ""),
            source=str(getattr(signal, "source", "") or ""),
            source_ref=source_ref,
        ))

    # --- NEWS / CATALYSTS (grounded stable ref preferred, exact normalized title fallback) ---
    calendar_ids: set[str] = {
        ce.event_id for ce in (getattr(report, "calendar_events", None) or [])
        if getattr(ce, "event_id", "")
    }
    for catalyst in getattr(report, "catalysts", []) or []:
        title = str(getattr(catalyst, "title", "") or "")
        if not title:
            continue
        # Stable identity: prefer grounded evidence_ref matching calendar event_id
        stable_ref = ""
        for ref in (getattr(catalyst, "evidence_refs", []) or []):
            if str(ref) in calendar_ids:
                stable_ref = str(ref)
                break
        if stable_ref:
            evidence_id = f"catalyst:{stable_ref}"
            source_ref = stable_ref
        else:
            # Fallback: exact deterministic normalized title identity only
            norm = normalize_event_title(title).replace(" ", "_")
            if not norm:
                continue
            evidence_id = f"catalyst:{norm}"
            source_ref = ""
        direction = str(getattr(catalyst, "direction", "unknown") or "unknown")
        if direction not in ("positive", "negative", "neutral", "mixed", "unknown"):
            direction = "unknown"
        items.append(ResearchEvidenceItem(
            evidence_id=evidence_id,
            domain="news",
            title=title,
            direction=EvidenceDirection(direction),
            as_of=str(getattr(catalyst, "expected_date", "") or ""),
            source="news_analyst",
            source_ref=source_ref,
        ))

    # --- RISK (only with canonical risk_key; never guessed from display name) ---
    for risk in getattr(report, "risk_items", []) or []:
        risk_key = str(getattr(risk, "risk_key", "") or "").strip()
        if not risk_key:
            continue  # no canonical identity → omit, never fabricate
        items.append(ResearchEvidenceItem(
            evidence_id=f"risk:{risk_key}",
            domain="risk",
            title=risk.name,
            direction=EvidenceDirection.UNKNOWN,  # severity is not decision direction
            value_text=str(getattr(risk, "evidence", "") or ""),
            source=",".join(risk.sources) if risk.sources else "",
        ))

    # --- Dedupe by exact evidence_id (one ID, one fact) ---
    seen: dict[str, ResearchEvidenceItem] = {}
    for item in items:
        seen.setdefault(item.evidence_id, item)

    # --- Deterministic order: domain priority → as_of desc → evidence_id asc ---
    # ISO dates sort descending via negated codepoints; empty as_of sorts last
    # ("￿" = U+FFFF, the largest codepoint → most negative → first in desc).
    def _as_of_desc_key(value: str) -> tuple:
        return tuple(-ord(c) for c in (value or "￿"))

    ordered = sorted(
        seen.values(),
        key=lambda item: (
            _EVIDENCE_DOMAIN_ORDER.get(item.domain, 99),
            _as_of_desc_key(item.as_of),
            item.evidence_id,
        ),
    )
    return ordered


# ---------------------------------------------------------------------------
# P2 (RPT-27): Deterministic decision evidence audit
# ---------------------------------------------------------------------------


def build_decision_evidence_audit(
    report: Any,
    decision_direction: str,
) -> DecisionEvidenceAudit | None:
    """Classify registry evidence against the authoritative PM direction.

    ``decision_direction`` must come from a validated structured
    ``portfolio_decision`` (never from prose). Rules:
    - neutral/unknown decision → every evidence item is context.
    - matching direction → support.
    - opposite positive/negative direction → conflict.
    - neutral/mixed/unknown evidence → context.
    No scores, no weighting, no thresholds. Preserves registry order
    within each bucket.
    """
    if decision_direction not in ("positive", "negative", "neutral", "unknown"):
        return None

    support_refs: list[str] = []
    conflict_refs: list[str] = []
    context_refs: list[str] = []

    for item in getattr(report, "evidence_registry", []) or []:
        direction = str(getattr(item.direction, "value", "") or "")
        if decision_direction in ("neutral", "unknown"):
            context_refs.append(item.evidence_id)
        elif direction == decision_direction:
            support_refs.append(item.evidence_id)
        elif direction in ("positive", "negative"):
            conflict_refs.append(item.evidence_id)
        else:
            context_refs.append(item.evidence_id)

    return DecisionEvidenceAudit(
        decision_direction=decision_direction,
        support_refs=support_refs,
        conflict_refs=conflict_refs,
        context_refs=context_refs,
    )


# ---------------------------------------------------------------------------
# P2 (RPT-28): Deterministic data reliability summary
# ---------------------------------------------------------------------------


def build_data_reliability_summary(
    data_quality: Any | None,
) -> DataReliabilitySummary:
    """Summarize DataQuality with plain counts — no score, no weighting.

    Mirrors the final data_quality state exactly:
    source_count = len(sources); failed = ok is False; truncated = True;
    conflict_count = len(conflicts); missing = len(missing_sections);
    stale_market_data = any 'stale' in notes.
    """
    dq = data_quality
    if dq is None:
        return DataReliabilitySummary()

    return DataReliabilitySummary(
        source_count=len(getattr(dq, "sources", []) or []),
        failed_source_count=sum(
            1 for s in (getattr(dq, "sources", []) or []) if not s.ok
        ),
        truncated_source_count=sum(
            1 for s in (getattr(dq, "sources", []) or []) if s.truncated
        ),
        conflict_count=len(getattr(dq, "conflicts", []) or []),
        missing_section_count=len(getattr(dq, "missing_sections", []) or []),
        stale_market_data=any(
            "stale" in note.lower()
            for note in (getattr(dq, "notes", []) or [])
        ),
    )

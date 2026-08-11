from typing import Annotated
from datetime import date, datetime
from finmindagent.time_utils import now_system
import math
from dateutil.relativedelta import relativedelta
import pandas as pd
import yfinance as yf
import os
from .market_asof import (
    provider_exclusive_end,
    resolve_market_data_cutoff,
    validate_analysis_date,
)
from .stockstats_utils import StockstatsUtils, _clean_dataframe, yf_retry, load_ohlcv, filter_financials_by_date

def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Two-layer as-of governance (current-plan v11):
    # Guard A — analysis request vs RUNTIME (system) date: only a true
    # future request (beyond the system's current date) is rejected.
    # Cross-timezone skew (Beijing 08-11 while New York is still 08-10) is
    # a legitimate request, never a rejection.
    validate_analysis_date(end_date)
    # Guard B — market data boundary: clamps to min(requested, market-local
    # today) and to the previous completed session when today's daily bar is
    # not complete yet. Provider end is the exclusive adaptation cutoff+1 so
    # the cutoff's own completed daily bar is included. Consumption is
    # always filtered to rows <= cutoff (provider is never the authority).
    cutoff = resolve_market_data_cutoff(end_date)
    provider_end = provider_exclusive_end(cutoff).strftime("%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range.
    # Explicit auto_adjust=True: use provider-adjusted close so OHLCV,
    # technical indicators (stockstats also uses auto_adjust=True), and
    # current price share the same price basis.
    # actions=True returns Stock Splits/Dividends columns for corporate-action
    # integrity checks (Task 10).
    data = yf_retry(lambda: ticker.history(
        start=start_date, end=provider_end, auto_adjust=True, actions=True,
    ))

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output (keeps market-local
    # wall clock), then defensively drop any bar after the business cutoff —
    # provider may return future/partial rows; they must never be consumed.
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data = data[data.index.date <= cutoff]

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information (as-of diagnostics: requested vs resolved cutoff)
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# As-Of: requested_end={end_date}; resolved_market_cutoff={cutoff}; provider_query_end={provider_end}; market_timezone=America/New_York; interval=1d\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {now_system().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string

def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Optimized: Get stock data once and calculate indicators for all dates
    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)
        
        # Generate the date range we need
        current_dt = curr_date_dt
        date_values = []
        
        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # Look up the indicator value for this date
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Not a trading day (weekend or holiday)"
            
            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)
        
        # Build the result string
        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"
        
    except Exception as e:
        print(f"Error getting bulk stockstats data: {e}")
        # Fallback to original implementation if bulk method fails
        ind_string = ""
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_date_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date_dt.strftime("%Y-%m-%d")
            )
            ind_string += f"{curr_date_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            curr_date_dt = curr_date_dt - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference"]
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from stockstats import wrap

    data = load_ohlcv(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    
    # Calculate the indicator for all rows at once
    df[indicator]  # This triggers stockstats to calculate the indicator
    
    # Create a dictionary mapping date strings to indicator values
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]
        
        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)
    
    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
) -> str:

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
        )
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get company fundamentals overview from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = yf_retry(lambda: ticker_obj.info)

        if not info:
            return f"No fundamentals data found for symbol '{ticker}'"

        fields = [
            ("Name", info.get("longName")),
            ("Sector", info.get("sector")),
            ("Industry", info.get("industry")),
            ("Currency", info.get("currency")),
            ("Exchange", info.get("exchange")),
            ("Market Cap", info.get("marketCap")),
            ("PE Ratio (TTM)", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")),
            ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")),
            ("EPS (TTM)", info.get("trailingEps")),
            ("Forward EPS", info.get("forwardEps")),
            ("Dividend Yield", info.get("dividendYield")),
            ("Beta", info.get("beta")),
            ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")),
            ("50 Day Average", info.get("fiftyDayAverage")),
            ("200 Day Average", info.get("twoHundredDayAverage")),
            ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit", info.get("grossProfits")),
            ("EBITDA", info.get("ebitda")),
            ("Net Income", info.get("netIncomeToCommon")),
            ("Profit Margin", info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("Return on Equity", info.get("returnOnEquity")),
            ("Return on Assets", info.get("returnOnAssets")),
            ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")),
            ("Book Value", info.get("bookValue")),
            ("Free Cash Flow", info.get("freeCashflow")),
        ]

        lines = []
        for label, value in fields:
            if value is not None:
                lines.append(f"{label}: {value}")

        header = f"# Company Fundamentals for {ticker.upper()}\n"
        header += f"# Data retrieved on: {now_system().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + "\n".join(lines)

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get balance sheet data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_balance_sheet)
        else:
            data = yf_retry(lambda: ticker_obj.balance_sheet)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No balance sheet data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Balance Sheet data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {now_system().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get cash flow data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_cashflow)
        else:
            data = yf_retry(lambda: ticker_obj.cashflow)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No cash flow data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Cash Flow data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {now_system().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get income statement data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_income_stmt)
        else:
            data = yf_retry(lambda: ticker_obj.income_stmt)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No income statement data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Income Statement data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {now_system().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get insider transactions data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: ticker_obj.insider_transactions)
        
        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Insider Transactions data for {ticker.upper()}\n"
        header += f"# Data retrieved on: {now_system().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string

    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"


# ---------------------------------------------------------------------------
# P1: Analyst expectations (price targets, EPS/revenue estimates, recommendations)
# ---------------------------------------------------------------------------

_ESTIMATE_PERIODS = {
    "0q": "current_quarter",
    "+1q": "next_quarter",
    "0y": "current_year",
    "+1y": "next_year",
}


def get_analyst_expectations(ticker: str, curr_date: str | None = None) -> str:
    """Fetch analyst price targets, forward EPS/revenue estimates, and
    recommendation summary from yfinance.

    Returns a JSON string with keys price_targets, earnings_estimates,
    revenue_estimates, recommendations, source.  Failure of one sub-source
    does not erase the others.  If all are unavailable, returns a clear
    error string.
    """
    import json as _json

    try:
        t = yf.Ticker(ticker.upper())
    except Exception as e:
        return f"Error creating yfinance Ticker for {ticker}: {str(e)}"

    result: dict = {"as_of": curr_date or now_system().strftime("%Y-%m-%d"), "source": "yfinance"}
    any_ok = False

    # --- price targets ---
    try:
        pt_info = yf_retry(lambda: t.analyst_price_targets)
        if pt_info is not None and isinstance(pt_info, dict):
            current_raw = pt_info.get("current")
            pt = {
                "vendor_current": _safe_float(current_raw),
                "low": _safe_float(pt_info.get("low")),
                "mean": _safe_float(pt_info.get("mean")),
                "median": _safe_float(pt_info.get("median")),
                "high": _safe_float(pt_info.get("high")),
            }
            if any(v is not None for v in pt.values()):
                result["price_targets"] = pt
                any_ok = True
    except Exception:
        pass

    # --- earnings estimates ---
    try:
        earnings = yf_retry(lambda: t.earnings_estimate)
        if earnings is not None:
            ee_dict = earnings.to_dict(orient="index") if hasattr(earnings, "to_dict") else None
            if ee_dict is None:
                ee_dict = dict(earnings) if isinstance(earnings, dict) else None
            if ee_dict is not None:
                result["earnings_estimates"] = _normalize_estimate_rows(ee_dict, is_earnings=True)
            if result["earnings_estimates"]:
                any_ok = True
    except Exception:
        pass

    # --- revenue estimates ---
    try:
        revenue = yf_retry(lambda: t.revenue_estimate)
        if revenue is not None:
            rev_dict = revenue.to_dict(orient="index") if hasattr(revenue, "to_dict") else None
            if rev_dict is None:
                rev_dict = dict(revenue) if isinstance(revenue, dict) else None
            if rev_dict is not None:
                result["revenue_estimates"] = _normalize_estimate_rows(rev_dict, is_earnings=False)
            if result["revenue_estimates"]:
                any_ok = True
    except Exception:
        pass

    # --- recommendations ---
    try:
        recs = yf_retry(lambda: t.recommendations_summary)
        if recs is not None:
            if hasattr(recs, "to_dict"):
                recs_dict = recs.to_dict(orient="records")
            elif isinstance(recs, dict):
                recs_dict = list(recs.values()) if all(isinstance(v, dict) for v in recs.values()) else recs
            else:
                recs_dict = None
            if recs_dict is not None:
                result["recommendations"] = _normalize_recommendations(recs_dict)
            if result["recommendations"]:
                any_ok = True
    except Exception:
        pass

    # --- P2: EPS trend + revisions (independent guarded calls) ---
    trend_dict = None
    revisions_dict = None
    try:
        eps_trend = yf_retry(lambda: t.eps_trend)
        if eps_trend is not None:
            trend_dict = eps_trend.to_dict(orient="index") if hasattr(eps_trend, "to_dict") else None
            if trend_dict is None and isinstance(eps_trend, dict):
                trend_dict = eps_trend
    except Exception:
        pass
    try:
        eps_rev = yf_retry(lambda: t.eps_revisions)
        if eps_rev is not None:
            revisions_dict = eps_rev.to_dict(orient="index") if hasattr(eps_rev, "to_dict") else None
            if revisions_dict is None and isinstance(eps_rev, dict):
                revisions_dict = eps_rev
    except Exception:
        pass
    merged = _merge_eps_revisions(trend_dict, revisions_dict)
    if merged:
        result["eps_revisions"] = merged
        any_ok = True

    if not any_ok:
        return f"No analyst expectations data found for {ticker}."

    return _json.dumps(result, default=str)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(result):
        return result
    return None


def _normalize_estimate_rows(raw: dict, *, is_earnings: bool) -> list[dict]:
    """Normalize yfinance estimate dict-of-DataFrames to stable JSON rows.

    Keys like '0q', '+1q', '0y', '+1y' → ForwardEstimatePoint periods.
    DataFrames have columns: avg, low, high, yearAgoEps/yearAgoRevenue,
    numberOfAnalysts, growth.
    """
    import pandas as _pd
    rows = []
    for key, period_label in _ESTIMATE_PERIODS.items():
        df = raw.get(key)
        if df is None or not isinstance(df, (_pd.DataFrame, dict)):
            continue
        if isinstance(df, _pd.DataFrame) and df.empty:
            continue
        try:
            if isinstance(df, _pd.DataFrame):
                first = df.iloc[0]
            else:
                first = df
            avg = _safe_float(first.get("avg"))
            low = _safe_float(first.get("low"))
            high = _safe_float(first.get("high"))
            if is_earnings:
                yago = _safe_float(first.get("yearAgoEps"))
            else:
                yago = _safe_float(first.get("yearAgoRevenue"))
            num = first.get("numberOfAnalysts")
            analyst_count = int(num) if num is not None and str(num).replace(".", "").isdigit() else None
            growth_raw = first.get("growth")
            growth_pct = None
            if growth_raw is not None:
                g = _safe_float(growth_raw)
                if g is not None:
                    # yfinance growth is ratio decimal; convert to percent
                    growth_pct = round(g * 100, 2)
            row = {
                "period": period_label,
                "average": avg,
                "low": low,
                "high": high,
                "year_ago": yago,
                "analyst_count": analyst_count,
                "growth_pct": growth_pct,
            }
            rows.append(row)
        except Exception:
            continue
    return rows


# ---------------------------------------------------------------------------
# P1: Company calendar (earnings dates)
# ---------------------------------------------------------------------------


def get_company_calendar(ticker: str, curr_date: str | None = None) -> str:
    """Fetch future earnings dates from yfinance calendar.

    Returns a JSON string with events array. Only future dates (>= curr_date)
    are included. Duplicate dates are deduped. Returns an error string if
    no calendar data is available.
    """
    import json as _json
    from datetime import date as _date

    try:
        t = yf.Ticker(ticker.upper())
    except Exception as e:
        return f"Error creating yfinance Ticker for {ticker}: {str(e)}"

    ref_date = _date.today()
    if curr_date:
        try:
            ref_date = _date.fromisoformat(curr_date)
        except (ValueError, TypeError):
            pass

    try:
        cal = yf_retry(lambda: t.calendar)
        if cal is None:
            return f"No company calendar data found for {ticker}."
    except Exception:
        return f"No company calendar data found for {ticker}."

    events = []
    seen_dates: set[str] = set()

    def _add_earnings(date_str: str, eps_est=None, rev_est=None):
        if not date_str or date_str in seen_dates:
            return
        try:
            event_date = _date.fromisoformat(date_str)
        except (ValueError, TypeError):
            return
        if event_date < ref_date:
            return
        seen_dates.add(date_str)
        events.append({
            "event_id": f"earnings:{date_str}",
            "event_type": "earnings",
            "date": date_str,
            "title": "Earnings release",
            "eps_estimate": _safe_float(eps_est),
            "revenue_estimate": _safe_float(rev_est),
            "source": "yfinance",
        })

    # yfinance calendar may be a dict with 'Earnings Date' etc.
    if isinstance(cal, dict):
        earnings_dates = cal.get("Earnings Date")
        if isinstance(earnings_dates, list):
            for d in earnings_dates:
                _add_earnings(str(d) if not isinstance(d, _date) else d.isoformat())
        # Also try 'Earnings High', 'Earnings Low' etc.
        eps_avg = _safe_float(cal.get("Earnings Average"))
        rev_avg = _safe_float(cal.get("Revenue Average"))
        if earnings_dates and isinstance(earnings_dates, list):
            for d in earnings_dates:
                date_str = str(d) if not isinstance(d, _date) else d.isoformat()
                _add_earnings(date_str, eps_avg, rev_avg)
    elif hasattr(cal, "to_dict"):
        cal_dict = cal.to_dict() if callable(cal.to_dict) else dict(cal)
        if isinstance(cal_dict, dict):
            earnings_dates = cal_dict.get("Earnings Date")
            if isinstance(earnings_dates, list):
                eps_avg = _safe_float(cal_dict.get("Earnings Average"))
                rev_avg = _safe_float(cal_dict.get("Revenue Average"))
                for d in earnings_dates:
                    date_str = str(d) if not isinstance(d, _date) else d.isoformat()
                    _add_earnings(date_str, eps_avg, rev_avg)

    if not events:
        return f"No future earnings dates found for {ticker}."

    return _json.dumps({
        "as_of": curr_date or _date.today().isoformat(),
        "events": events,
        "source": "yfinance",
    }, default=str)


# ---------------------------------------------------------------------------
# P2: EPS trend + revisions merge
# ---------------------------------------------------------------------------


def _safe_int(v):
    """Convert to int, rejecting NaN/Inf/overflow."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(f):
        return None
    return int(f)


def _merge_eps_revisions(trend: dict | None, revisions: dict | None) -> list[dict]:
    """Merge yfinance eps_trend + eps_revisions into one eps_revisions list.

    Supports trend-only, revisions-only, both, or neither.
    Rows keyed by period (0q/+1q/0y/+1y). NaN/Inf → None.
    """
    if not trend and not revisions:
        return []
    result = []
    for raw_period, label in _ESTIMATE_PERIODS.items():
        trend_row = trend.get(raw_period) if isinstance(trend, dict) else {}
        rev_row = revisions.get(raw_period) if isinstance(revisions, dict) else {}
        if not isinstance(trend_row, dict):
            trend_row = {}
        if not isinstance(rev_row, dict):
            rev_row = {}
        current = _safe_float(trend_row.get("current"))
        days_7 = _safe_float(trend_row.get("7daysAgo"))
        days_30 = _safe_float(trend_row.get("30daysAgo"))
        days_60 = _safe_float(trend_row.get("60daysAgo"))
        days_90 = _safe_float(trend_row.get("90daysAgo"))
        up7 = _safe_int(rev_row.get("upLast7days"))
        up30 = _safe_int(rev_row.get("upLast30days"))
        down7 = _safe_int(rev_row.get("downLast7days"))
        down30 = _safe_int(rev_row.get("downLast30days"))
        if current is not None or any(v is not None for v in (days_7, days_30, days_60, days_90, up7, up30, down7, down30)):
            result.append({
                "period": label,
                "current": current,
                "days_7_ago": days_7,
                "days_30_ago": days_30,
                "days_60_ago": days_60,
                "days_90_ago": days_90,
                "up_7d": up7,
                "up_30d": up30,
                "down_7d": down7,
                "down_30d": down30,
            })
    return result


# ---------------------------------------------------------------------------
# P2: get_earnings_history
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# P2: get_holder_context
# ---------------------------------------------------------------------------


def get_holder_context(ticker: str, curr_date: str | None = None, limit: int = 10) -> str:
    """Fetch major holders, institutional holders, and mutual fund holders.

    Returns JSON with ownership_snapshot fields. Partial endpoint failure
    does not erase other sections.
    """
    import json as _json

    try:
        t = yf.Ticker(ticker.upper())
    except Exception as e:
        return f"Error creating yfinance Ticker for {ticker}: {str(e)}"

    result: dict = {"source": "yfinance"}
    any_ok = False

    # --- major holders ---
    try:
        mh = yf_retry(lambda: t.major_holders)
        if mh is not None:
            mh_dict = _normalize_major_holders(mh)
            if mh_dict:
                result.update(mh_dict)
                any_ok = True
    except Exception:
        pass

    # --- institutional holders ---
    try:
        inst = yf_retry(lambda: t.institutional_holders)
        if inst is not None and hasattr(inst, "to_dict"):
            inst_list = _normalize_holder_rows(inst, "institutional", limit)
            if inst_list:
                result["institutional_holders"] = inst_list
                any_ok = True
    except Exception:
        pass

    # --- mutual fund holders ---
    try:
        mf = yf_retry(lambda: t.mutualfund_holders)
        if mf is not None and hasattr(mf, "to_dict"):
            mf_list = _normalize_holder_rows(mf, "mutual_fund", limit)
            if mf_list:
                result["mutual_fund_holders"] = mf_list
                any_ok = True
    except Exception:
        pass

    # as_of: latest date_reported from retained rows
    latest_date = ""
    for lst in (result.get("institutional_holders") or [], result.get("mutual_fund_holders") or []):
        for h in lst:
            d = h.get("date_reported", "")
            if d and d > latest_date:
                latest_date = d
    result["as_of"] = latest_date
    result.setdefault("insiders_pct", None)
    result.setdefault("institutions_pct", None)
    result.setdefault("institutions_float_pct", None)
    result.setdefault("institutions_count", None)
    result.setdefault("institutional_holders", [])
    result.setdefault("mutual_fund_holders", [])

    if not any_ok:
        return f"No holder data found for {ticker}."

    return _json.dumps(result, default=str)


def _normalize_major_holders(mh) -> dict:
    """Normalize yfinance major_holders to flat percent dict."""
    result: dict = {}
    if hasattr(mh, "to_dict"):
        mh_dict = mh.to_dict(orient="records") if hasattr(mh, "to_dict") else {}
    elif isinstance(mh, dict):
        mh_dict = mh
    else:
        return result

    # Known semantic keys -> canonical names; yfinance percent ratios ×100
    _KEY_MAP = {
        "insidersPercentHeld": "insiders_pct",
        "institutionsPercentHeld": "institutions_pct",
        "institutionsFloatPercentHeld": "institutions_float_pct",
        "institutionsCount": "institutions_count",
    }
    if isinstance(mh_dict, list):
        for row in mh_dict:
            if isinstance(row, dict):
                breakdown = str(row.get("Breakdown") or row.get("") or "").strip().lower()
                val = row.get("Value")
                for ykey, ckey in _KEY_MAP.items():
                    if ckey == "institutions_count":
                        if ykey.lower() in breakdown:
                            try:
                                result[ckey] = int(float(str(val)))
                            except (ValueError, TypeError):
                                pass
                    elif ykey.lower() in breakdown:
                        v = _safe_float(val)
                        if v is not None:
                            result[ckey] = round(v * 100, 2)
    elif isinstance(mh_dict, dict):
        for ykey, ckey in _KEY_MAP.items():
            v = _safe_float(mh_dict.get(ykey))
            if v is not None:
                if ckey == "institutions_count":
                    result[ckey] = int(v)
                else:
                    result[ckey] = round(v * 100, 2)
    return result


def _normalize_holder_date(value) -> str:
    """Convert pandas Timestamp/datetime/date/ISO to YYYY-MM-DD."""
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else ""
    s = str(value).strip()
    if not s:
        return ""
    # Try ISO parse
    try:
        d = date.fromisoformat(s[:10])
        return d.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass
    return ""


def _slugify_holder(name: str) -> str:
    import re as _re
    return _re.sub(r"-+", "-", _re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-")


def _normalize_holder_rows(df, holder_type: str, limit: int) -> list[dict]:
    """Normalize yfinance institutional/mutualfund_holders DataFrame."""
    rows_raw = df.to_dict(orient="records") if hasattr(df, "to_dict") else list(df)
    result = []
    for row in rows_raw:
        if not isinstance(row, dict):
            continue
        holder_name = str(row.get("Holder") or "")
        if not holder_name:
            continue
        date_reported = _normalize_holder_date(row.get("Date Reported"))
        pct_out_raw = row.get("% Out")
        pct_out = _safe_float(pct_out_raw)
        if pct_out is not None:
            pct_out = round(pct_out * 100, 2)
        result.append({
            "holder_id": f"holder:{holder_type}:{_slugify_holder(holder_name)}:{date_reported or 'unknown'}",
            "holder_type": holder_type,
            "holder": holder_name,
            "shares": _safe_float(row.get("Shares")),
            "date_reported": date_reported,
            "pct_out": pct_out,
            "value": _safe_float(row.get("Value")),
            "source": "yfinance",
        })
    result.sort(key=lambda h: (
        -(h["pct_out"] or 0),
        -(h["shares"] or 0),
        h["holder"].lower(),
    ))
    return result[:limit]


def get_earnings_history(ticker: str, curr_date: str | None = None, limit: int = 8) -> str:
    """Fetch historical earnings dates with EPS estimate/reported pairs.

    Deterministically calculates surprise_pct from estimate and reported EPS.
    Only past dates (<= curr_date) are kept. Future rows dropped.
    """
    import json as _json
    from datetime import date as _date

    try:
        t = yf.Ticker(ticker.upper())
    except Exception as e:
        return f"Error creating yfinance Ticker for {ticker}: {str(e)}"

    ref_date = _date.today()
    if curr_date:
        try:
            ref_date = _date.fromisoformat(curr_date)
        except (ValueError, TypeError):
            pass

    try:
        raw = yf_retry(lambda: t.get_earnings_dates(limit=max(limit, 8)))
    except Exception:
        return f"No earnings history data found for {ticker}."

    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return f"No earnings history data found for {ticker}."

    surprises = []
    seen_reports: set[str] = set()
    try:
        if hasattr(raw, "iterrows"):
            for idx, row in raw.iterrows():
                report_date = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
                if report_date in seen_reports:
                    continue
                try:
                    rd = _date.fromisoformat(report_date)
                except (ValueError, TypeError):
                    continue
                if rd > ref_date:
                    continue
                seen_reports.add(report_date)
                estimate = _safe_float(row.get("EPS Estimate"))
                reported = _safe_float(row.get("Reported EPS"))
                surprise = None
                if estimate is not None and estimate != 0 and reported is not None:
                    surprise = (reported - estimate) / abs(estimate) * 100
                if reported is not None or estimate is not None:
                    surprises.append({
                        "evidence_id": f"earnings_surprise:{report_date}",
                        "report_date": report_date,
                        "eps_estimate": estimate,
                        "reported_eps": reported,
                        "surprise_pct": surprise,
                        "source": "yfinance",
                    })
    except Exception:
        pass

    if not surprises:
        return f"No earnings history data found for {ticker}."

    surprises.sort(key=lambda s: s["report_date"], reverse=True)
    return _json.dumps({
        "as_of": curr_date or _date.today().isoformat(),
        "surprises": surprises[:limit],
        "source": "yfinance",
    }, default=str)


def _normalize_recommendations(raw: list | dict) -> list[dict]:
    """Normalize yfinance recommendations_summary to stable list.

    Input is already ``to_dict(orient="records")`` → list of dicts with keys
    period, strongBuy, buy, hold, sell, strongSell.
    """
    import pandas as _pd
    items = raw
    if isinstance(raw, _pd.DataFrame):
        items = raw.to_dict(orient="records")
    elif isinstance(raw, dict):
        items = list(raw.values()) if all(isinstance(v, dict) for v in raw.values()) else [raw]
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            result.append({
                "period": str(item.get("period", "")),
                "strong_buy": int(item.get("strongBuy", 0) or 0),
                "buy": int(item.get("buy", 0) or 0),
                "hold": int(item.get("hold", 0) or 0),
                "sell": int(item.get("sell", 0) or 0),
                "strong_sell": int(item.get("strongSell", 0) or 0),
            })
        except Exception:
            continue
    return result
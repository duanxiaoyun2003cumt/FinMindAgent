"""Presentation numeric formatting.

Formats values for display only; business precision lives in the schema.
No investment judgment is made here.
"""

from __future__ import annotations

from typing import Any

from finmindagent.reporting.localization import text


def format_number(value: float | int) -> str:
    """1,258.58 — locale-neutral thousands separator, no trailing junk."""
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def format_compact_number(value: float | int) -> str:
    """186383000000 → 186.38B; 1250000000 → 1.25B; 42500000 → 42.50M.

    Never 1.86383e+11.
    """
    absolute = abs(float(value))
    if absolute >= 1e12:
        return f"{value / 1e12:.2f}T"
    if absolute >= 1e9:
        return f"{value / 1e9:.2f}B"
    if absolute >= 1e6:
        return f"{value / 1e6:.2f}M"
    if absolute >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def format_percentage(value: float | int) -> str:
    """34.19 → 34.19% — no padded zeroes."""
    return f"{value:.2f}%".replace(".00%", "%")


def format_multiple(value: float | int) -> str:
    """42.94 → 42.94x."""
    return f"{value:.2f}x".replace(".00x", "x")


_CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",
    "CNY": "¥",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "HKD": "HK$",
    "KRW": "₩",
    "TWD": "NT$",
}


def _currency_prefix(currency: str | None, locale: str = "zh-CN") -> str:
    """Return a deterministic currency prefix for display.

    USD → $, CNY → ¥, unknown → ISO code + space.
    Locale affects CNY display: zh-CN prefers 元, en prefers ¥.
    """
    if currency is None:
        return ""
    upper = currency.upper().strip()
    if upper == "CNY":
        if locale == "zh-CN":
            return "¥"
        return "¥"
    if upper in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[upper]
    # Unknown currency: display ISO code
    return f"{upper} "


def format_price(value: float | int | None, currency: str | None = None, locale: str = "zh-CN") -> str:
    """Format a price with optional currency prefix. None → ''."""
    if value is None:
        return ""
    prefix = _currency_prefix(currency, locale)
    return f"{prefix}{value:,.2f}"


def format_position(
    minimum: float | int | None,
    maximum: float | int | None,
    locale: str = "zh-CN",
) -> str:
    """min=5,max=8 → 5%–8%; max only → ≤8% (zh) / Up to 8% (en); none → ''."""
    if minimum is not None and maximum is not None:
        return f"{minimum:g}%–{maximum:g}%"
    if maximum is not None:
        if locale == "en":
            return f"{text('up_to', 'en')} {maximum:g}%"
        return f"≤{maximum:g}%"
    if minimum is not None:
        if locale == "en":
            return f"{text('at_least', 'en')} {minimum:g}%"
        return f"≥{minimum:g}%"
    return ""


def format_metric_value(value: float | int, unit: str = "") -> str:
    """Unit-aware display: % / x / magnitude for money-like values."""
    if unit == "%":
        return format_percentage(value)
    if unit == "x":
        return format_multiple(value)
    return format_compact_number(value)


def format_optional(value: Any, locale: str = "zh-CN") -> str:
    """Unified missing display; never None/null/-/n/a mixed."""
    if value is None or value == "":
        return text("unavailable", locale)
    return str(value)

"""Time semantics utilities: system time vs market time.

核心原则：系统时间 ≠ 市场时间

- System / Runtime / Audit / CLI / artifact-generated 时间 → Asia/Shanghai
  (UTC+08:00)。所有系统时间都是 timezone-aware，绝不允许 naive datetime。
- Market / Quote / News / Earnings 时间 → exchange / source timezone
  (America/New_York for US with real DST, Asia/Hong_Kong for HK)。绝不默认
  转换成北京时间 —— ``format_market_timestamp`` 没有 tz 参数时保留数据源
  原始 offset。
- date-only 业务日期（trade_date / price_as_of / as_of）保持原始日期，不参与
  时区转换。

Standard library only (``zoneinfo``); no third-party dependency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

BEIJING_TZ = ZoneInfo("Asia/Shanghai")
US_EASTERN_TZ = ZoneInfo("America/New_York")
HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")

# Exchange → market timezone mapping (source-of-truth for market timestamps).
EXCHANGE_TIMEZONES: dict[str, ZoneInfo] = {
    "NASDAQ": US_EASTERN_TZ,
    "NYSE": US_EASTERN_TZ,
    "AMEX": US_EASTERN_TZ,
    "HKEX": HONG_KONG_TZ,
    "SEHK": HONG_KONG_TZ,
}

_SYSTEM_FMT = "%Y-%m-%d %H:%M:%S"


def now_system() -> datetime:
    """Current system/runtime time in Asia/Shanghai (always aware)."""
    return datetime.now(BEIJING_TZ)


def system_timestamp() -> str:
    """ISO-8601 system timestamp with +08:00 offset, e.g.
    ``2026-08-10T11:30:00+08:00``. Used by runtime events / audit / tool
    results — never a naive string."""
    return now_system().isoformat()


def exchange_timezone(exchange: str | None) -> ZoneInfo | None:
    """Market timezone for an exchange name, or None when unknown."""
    if not exchange:
        return None
    return EXCHANGE_TIMEZONES.get(str(exchange).upper())


def _coerce_datetime(value: Any) -> datetime | None:
    """Accept an aware/naive datetime or an ISO-8601 string."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def format_system_timestamp(value: Any, fmt: str = _SYSTEM_FMT) -> str:
    """Format a system timestamp in Asia/Shanghai with explicit +08:00.

    ``value`` may be an aware datetime (any zone), an ISO string with offset,
    or a legacy naive string — naive values are interpreted as UTC (the
    historical canonical storage) and converted to Beijing. The output always
    carries ``+08:00`` so it can never be mistaken for an ambiguous bare time.
    """
    dt = _coerce_datetime(value)
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # legacy canonical storage was UTC
    return dt.astimezone(BEIJING_TZ).strftime(f"{fmt} %z").replace("+0800", "+08:00")


def format_market_timestamp(value: Any, tz: ZoneInfo | None = None) -> str:
    """Format a market timestamp in its market timezone — never Beijing.

    - With ``tz`` (e.g. ``US_EASTERN_TZ``): converts to that market timezone
      and renders wall clock + abbreviation (``2026-08-10 09:30 EDT``).
    - Without ``tz``: preserves the source representation/offset verbatim
      (``2026-08-10T09:30:00-04:00``) — no conversion, no Beijing default.
    """
    dt = _coerce_datetime(value)
    if dt is None:
        return ""
    if tz is None:
        return dt.isoformat()
    dt = dt.astimezone(tz)
    abbr = dt.strftime("%Z") or f"{dt.utcoffset()}"
    return dt.strftime(f"{_SYSTEM_FMT} {abbr}")

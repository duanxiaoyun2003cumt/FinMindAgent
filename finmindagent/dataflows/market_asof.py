"""Point-in-time / As-Of market data boundary helpers (current-plan v10).

核心原则（NO FUTURE BUSINESS DATA）：
对于 analysis_date = D，任何被分析消费的行情数据都必须满足
``business_data_time <= as_of_boundary(D)``。为寻找"最近交易日 / 有效价格 /
benchmark"而向 D 之后扫描是禁止的 —— 找不到目标日 bar 时只向过去选择
last available bar。

两个概念严格区分（§55/§57）：
- ``resolve_market_data_cutoff``：业务上最大允许消费的 market date
  （completed daily session 语义）。
- ``provider_exclusive_end``：yfinance ``end`` 参数是 exclusive，允许为
  cutoff + 1 天以包含 cutoff 当天的完整 daily bar —— 这不构成未来泄漏，
  最终消费仍以 ``<= cutoff`` 的防御性过滤为准。

市场时间基于目标市场时区（美股 America/New_York，含 DST，绝不写死 UTC-5），
系统日志时区（Asia/Shanghai）与市场语义不混用。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from finmindagent.time_utils import BEIJING_TZ, US_EASTERN_TZ, now_system

# 美股常规交易时段收盘（market-local）。daily bar 在该时刻之后才视为完成。
DAILY_SESSION_CLOSE = time(16, 0)


def market_now(market_timezone: Any = US_EASTERN_TZ) -> datetime:
    """Current time in the target market's own timezone."""
    return datetime.now(market_timezone)


def validate_analysis_date(
    requested_date: str | date,
    *,
    runtime_now: datetime | None = None,
    runtime_timezone: Any = BEIJING_TZ,
) -> date:
    """Guard A — Analysis Request Guard（analysis request 是否真正未来）。

    比较对象是 **runtime / 系统时区**（Asia/Shanghai）的当前自然日，而不是
    市场时区。北京已进入 08-11 而纽约仍为 08-10 时，requested=08-11 是
    ``requested <= runtime_local_date`` → 合法；只有用户请求真正尚未发生的
    日期（requested > runtime-local today）才拒绝。绝不 silent clamp。
    """
    if isinstance(requested_date, str):
        requested = date.fromisoformat(requested_date)
    else:
        requested = requested_date
    # Runtime date comes from the project's canonical system clock
    # (Asia/Shanghai); never OS-local or market-local.
    now = runtime_now or now_system()
    runtime_today = now.date()
    if requested > runtime_today:
        raise ValueError(
            f"Requested analysis date {requested} is in the future "
            f"(system current analysis date is {runtime_today}); "
            f"the request is rejected."
        )
    return requested


def resolve_market_data_cutoff(
    requested_date: str | date,
    *,
    as_of_now: datetime | None = None,
    market_timezone: Any = US_EASTERN_TZ,
) -> date:
    """Guard B — Market Data Boundary（行情业务允许截止日期）。

    只做市场侧解析，**永不拒绝**（拒绝由 Guard A 负责）：
    ``market_candidate = min(requested, market-local today)`` —— 跨时区 skew
    （北京 08-11 / 纽约 08-10）自然收窄到市场当前自然日，不构成未来泄漏。

    - candidate == market-local 今天 且 日线 session 尚未完成（now < 收盘）
      → 退回 previous calendar day（实际最后可用 bar 由数据层
      ``max(bar.date <= cutoff)`` 决定，周末/节假日自然处理）。
    - 其余 → candidate（历史日期允许使用当天完整 daily bar）。
    """
    if isinstance(requested_date, str):
        requested = date.fromisoformat(requested_date)
    else:
        requested = requested_date
    now = as_of_now or market_now(market_timezone)
    market_today = now.date()
    candidate = min(requested, market_today)
    if candidate == market_today and now.time() < DAILY_SESSION_CLOSE:
        return market_today - timedelta(days=1)
    return candidate


def provider_exclusive_end(cutoff: str | date) -> date:
    """yfinance ``end`` is exclusive: cutoff + 1 day includes cutoff's bar.

    This is an adapter detail, NOT a data boundary — consumption is still
    governed by the ``<= cutoff`` defensive filter.
    """
    if isinstance(cutoff, str):
        cutoff = date.fromisoformat(cutoff)
    return cutoff + timedelta(days=1)

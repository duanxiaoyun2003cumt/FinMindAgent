"""System vs Market time semantics contract tests (current-plan v7).

核心原则：系统时间 ≠ 市场时间
- System / Runtime / Audit / CLI / artifact-generated 时间 → Asia/Shanghai (+08:00)
- Market / Quote / News / Earnings 时间 → exchange / source timezone
  (America/New_York for US, Asia/Hong_Kong for HK — 绝不统一转成北京时间)
- date-only 业务日期（trade_date / price_as_of / as_of）保持原始日期

No new third-party deps — standard library zoneinfo only.
"""

import datetime as _dt
from datetime import datetime, timedelta, timezone

import pytest

from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.events import EventType, RuntimeEvent

FROZEN_UTC = "2026-08-10T03:30:00Z"  # == 2026-08-10 11:30:00 +08:00


def _beijing(frozen: datetime) -> datetime:
    from finmindagent.time_utils import BEIJING_TZ

    if frozen.tzinfo is None:
        frozen = frozen.replace(tzinfo=timezone.utc)
    return frozen.astimezone(BEIJING_TZ)


# ===========================================================================
# Test 1/2 — System timestamp + formatter
# ===========================================================================

def test_system_now_is_beijing_aware():
    from finmindagent.time_utils import BEIJING_TZ, now_system

    now = now_system()
    assert now.tzinfo is not None, "system timestamp must be timezone-aware"
    assert now.utcoffset() == timedelta(hours=8), "Beijing is always UTC+08:00"
    assert str(now.tzinfo) == "Asia/Shanghai" or now.tzinfo == BEIJING_TZ


def test_system_timestamp_iso_keeps_offset():
    from finmindagent.time_utils import system_timestamp

    ts = system_timestamp()
    assert ts.endswith("+08:00"), f"system ISO timestamp must carry +08:00, got {ts!r}"
    assert "+08:00" in ts and "Z" not in ts


def test_format_system_timestamp_converts_utc_to_beijing():
    from finmindagent.time_utils import format_system_timestamp

    out = format_system_timestamp(FROZEN_UTC)
    assert out == "2026-08-10 11:30:00 +08:00", f"got {out!r}"
    # offset must never be dropped
    assert "+08:00" in out


def test_format_system_timestamp_aware_input_any_zone():
    from finmindagent.time_utils import format_system_timestamp

    dt = datetime(2026, 8, 10, 9, 30, tzinfo=timezone(timedelta(hours=-4)))
    out = format_system_timestamp(dt)
    assert out == "2026-08-10 21:30:00 +08:00"


# ===========================================================================
# Test 3/4/5 — Market timestamps: US tz / DST / HK
# ===========================================================================

def test_us_market_timestamp_keeps_new_york():
    from finmindagent.time_utils import US_EASTERN_TZ, format_market_timestamp

    out = format_market_timestamp("2026-08-10T09:30:00-04:00", US_EASTERN_TZ)
    assert "09:30" in out and "EDT" in out, f"got {out!r}"
    assert "Shanghai" not in out and "+08" not in out


def test_us_dst_handled_by_zoneinfo():
    from finmindagent.time_utils import US_EASTERN_TZ

    summer = datetime(2026, 8, 10, 9, 30, tzinfo=US_EASTERN_TZ)
    winter = datetime(2026, 1, 12, 9, 30, tzinfo=US_EASTERN_TZ)
    assert summer.utcoffset() == timedelta(hours=-4), "August = EDT (UTC-4)"
    assert winter.utcoffset() == timedelta(hours=-5), "January = EST (UTC-5)"


def test_hk_market_timestamp_keeps_hong_kong():
    from finmindagent.time_utils import HONG_KONG_TZ, format_market_timestamp

    out = format_market_timestamp("2026-08-10T09:30:00+08:00", HONG_KONG_TZ)
    assert "09:30" in out and "HKT" in out, f"got {out!r}"
    # same wall-clock offset as Beijing but the market semantic is HK, not CN
    assert "Shanghai" not in out


def test_market_timestamp_without_tz_preserves_source():
    """No tz given → keep the source offset verbatim; NEVER default to Beijing."""
    from finmindagent.time_utils import format_market_timestamp

    out = format_market_timestamp("2026-08-10T09:30:00-04:00")
    assert "-04:00" in out and "+08" not in out, f"got {out!r}"


# ===========================================================================
# Test 6/7/8 — System + market coexist on the same instant
# ===========================================================================

def test_system_and_market_coexist():
    from finmindagent.time_utils import format_market_timestamp, format_system_timestamp

    event_ts = format_system_timestamp("2026-08-10T03:30:00Z")
    market_ts = format_market_timestamp("2026-08-10T09:30:00-04:00")
    assert event_ts == "2026-08-10 11:30:00 +08:00"
    assert "-04:00" in market_ts
    assert event_ts != market_ts  # 同一 instant，两个语义，互不污染


def test_runtime_log_and_quote_same_instant():
    from finmindagent.time_utils import format_market_timestamp, format_system_timestamp

    runtime = format_system_timestamp("2026-08-10T13:30:00Z")
    quote = format_market_timestamp("2026-08-10T09:30:00-04:00")
    assert runtime == "2026-08-10 21:30:00 +08:00"
    assert quote == "2026-08-10T09:30:00-04:00"  # source preserved verbatim
    # both describe 13:30Z; representations differ by semantics
    assert runtime != quote


def test_news_published_time_stays_edt():
    from finmindagent.time_utils import US_EASTERN_TZ, format_market_timestamp

    out = format_market_timestamp("2026-08-10T08:15:00-04:00", US_EASTERN_TZ)
    assert "08:15" in out and "EDT" in out
    assert "20:15" not in out, "news time must not be converted to Beijing"


# ===========================================================================
# Runtime event / audit timestamps → Beijing
# ===========================================================================

def test_runtime_event_timestamp_is_beijing(monkeypatch):
    from finmindagent.runtime import events as events_mod

    frozen = datetime(2026, 8, 10, 11, 30, 0, tzinfo=timezone(timedelta(hours=8)))
    monkeypatch.setattr(events_mod, "system_now", lambda: frozen.isoformat())
    ev = RuntimeEvent(run_id="r", step=0, type=EventType.ACTION)
    assert ev.timestamp == "2026-08-10T11:30:00+08:00"


def test_permission_audit_timestamp_is_beijing(monkeypatch, tmp_path):
    from finmindagent.runtime.permissions import audit as audit_mod
    from finmindagent.runtime.permissions.audit import PermissionAuditLog

    frozen = datetime(2026, 8, 10, 11, 30, 0, tzinfo=timezone(timedelta(hours=8)))
    monkeypatch.setattr(audit_mod, "system_now", lambda: frozen.isoformat())
    log = PermissionAuditLog(tmp_path / "audit.jsonl")
    log.write({"run_id": "r", "decision": "allow"})
    import json

    entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert entry["timestamp"] == "2026-08-10T11:30:00+08:00"


def test_builder_generated_at_default_is_beijing(monkeypatch):
    from finmindagent.reporting import builder as builder_mod
    from finmindagent.runtime.state import TradingRunState

    frozen = datetime(2026, 8, 10, 11, 30, 0, tzinfo=timezone(timedelta(hours=8)))
    monkeypatch.setattr(builder_mod, "now_system", lambda: frozen)
    report = builder_mod.build_report_schema(
        TradingRunState(ticker="X", trade_date="2026-08-10", run_id="r", status="completed")
    )
    assert report.metadata.generated_at == "2026-08-10T11:30:00+08:00"


def test_cli_activity_timestamp_is_beijing(monkeypatch):
    """CLI MessageBuffer shows Asia/Shanghai wall clock regardless of OS tz."""
    from cli.main import MessageBuffer

    frozen = datetime(2026, 8, 10, 11, 30, 5, tzinfo=timezone(timedelta(hours=8)))
    monkeypatch.setattr("cli.main.now_system", lambda: frozen)
    buf = MessageBuffer()
    buf.add_message("System", "Runtime loop started")
    assert buf.messages[-1][0] == "11:30:05"


# ===========================================================================
# Test 10 — HTML: system fields Beijing, market fields untouched
# ===========================================================================

def test_html_provenance_retrieved_at_shows_beijing_with_offset(tmp_path):
    """retrieved_at is a system timestamp → Beijing + offset, never truncated."""
    from finmindagent.reporting.builder import build_report_schema
    from finmindagent.reporting.html import render_report_html
    from finmindagent.runtime.actions import AgentAction
    from finmindagent.runtime.state import TradingRunState

    state = TradingRunState(ticker="MU", trade_date="2026-08-10", run_id="r", status="completed")
    state.add_event(
        EventType.OBSERVATION,
        actor="market_analyst",
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name="get_stock_data"),
        observation={
            "ok": True,
            "tool_name": "get_stock_data",
            "timestamp": "2026-08-10T03:30:00Z",
            "structured_data": {
                "market_series": [{"date": "2026-08-10", "close": 100.0, "volume": 1}],
            },
        },
    )
    report = build_report_schema(state)
    html = render_report_html(report, chart_js="")

    assert "11:30:00 +08:00" in html, "provenance retrieved_at must show Beijing with offset"
    # market date-only fields untouched
    assert "2026-08-10" in html
    # no naive bare timestamps
    assert "T03:30:00" not in html


# ===========================================================================
# Invariants (§24)
# ===========================================================================

def test_invariant_system_always_aware():
    from finmindagent.time_utils import now_system

    for _ in range(3):
        assert now_system().tzinfo is not None


def test_invariant_system_formatter_always_shanghai():
    from finmindagent.time_utils import format_system_timestamp

    for value in (FROZEN_UTC, "2026-01-01T00:00:00Z", datetime(2026, 8, 10, 9, 30, tzinfo=timezone.utc)):
        assert "+08:00" in format_system_timestamp(value)


def test_invariant_market_formatter_never_defaults_to_beijing():
    from finmindagent.time_utils import format_market_timestamp

    out = format_market_timestamp("2026-08-10T09:30:00-04:00")
    assert "+08" not in out and "Shanghai" not in out


def test_invariant_us_timezone_from_zoneinfo():
    import zoneinfo

    from finmindagent.time_utils import US_EASTERN_TZ

    assert isinstance(US_EASTERN_TZ, zoneinfo.ZoneInfo)
    assert US_EASTERN_TZ.key == "America/New_York"


def test_invariant_hk_timezone_from_zoneinfo():
    import zoneinfo

    from finmindagent.time_utils import HONG_KONG_TZ

    assert isinstance(HONG_KONG_TZ, zoneinfo.ZoneInfo)
    assert HONG_KONG_TZ.key == "Asia/Hong_Kong"


def test_invariant_no_browser_local_timezone():
    """HTML/JS must not rely on toLocaleString/browser local tz."""
    from pathlib import Path

    html_src = Path("finmindagent/reporting/html.py").read_text(encoding="utf-8")
    assert "toLocaleString" not in html_src
    assert "getTimezoneOffset" not in html_src


def test_invariant_no_naive_datetime_production():
    """No datetime.now() without tz remains as system time."""
    import re

    from pathlib import Path

    import finmindagent.time_utils

    module_dir = Path(finmindagent.time_utils.__file__).parent
    offenders = []
    for path in module_dir.rglob("*.py"):
        if "test" in path.name:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"datetime\.now\(\)", line):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"naive datetime.now() still used as system time: {offenders}"

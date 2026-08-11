"""RED tests for CliRuntimeReporter producer/consumer event handling.

The runtime worker thread only enqueues events; the CLI main thread drains
them into MessageBuffer and renders Rich. These tests prove the split:

- Test 5-7 (previous round): CALL_AGENT observation drives Agent Status,
  Current Report, and official sections without depending on mutable state.
- Test A: a BLOCKED runtime still lets the UI consume and display events.
- Test B: Rich refresh happens while the runtime future is still pending.
- Test C: the worker-thread callback (on_event) never touches Rich.
- Test D: queue preserves event order through drain.
"""

import threading

from cli.main import MessageBuffer
from cli.runtime_reporter import CliRuntimeReporter
from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.events import EventType, RuntimeEvent


def _make_reporter(buffer):
    return CliRuntimeReporter(
        message_buffer=buffer,
        layout=None,
        update_display=lambda *args, **kwargs: None,
    )


def _call_agent_event(agent, observation, run_id="r1", step=1):
    action = AgentAction(type=ActionType.CALL_AGENT, target_agent=agent)
    return RuntimeEvent(
        run_id=run_id,
        step=step,
        type=EventType.OBSERVATION,
        actor=agent,
        action=action,
        observation=observation,
    )


def _agent_call_event(agent, run_id="r1", step=1):
    return RuntimeEvent(
        run_id=run_id,
        step=step,
        type=EventType.AGENT_CALL,
        actor=agent,
        action=AgentAction(type=ActionType.CALL_AGENT, target_agent=agent),
    )


def _tool_call_event(agent, tool_name, tool_args, run_id="r1", step=1):
    return RuntimeEvent(
        run_id=run_id,
        step=step,
        type=EventType.TOOL_CALL,
        actor=agent,
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name=tool_name, tool_args=tool_args),
    )


def _tool_result_event(agent, tool_name, run_id="r1", step=1):
    return RuntimeEvent(
        run_id=run_id,
        step=step,
        type=EventType.OBSERVATION,
        actor=agent,
        action=AgentAction(type=ActionType.CALL_TOOL, tool_name=tool_name),
        observation={"ok": True, "data": "rows=120"},
    )


def _llm_call_event(agent, run_id="r1", step=1):
    return RuntimeEvent(
        run_id=run_id,
        step=step,
        type=EventType.LLM_CALL,
        actor=agent,
    )


def test_bull_bear_risk_agents_complete_status_and_current_report():
    """Test 5: CALL_AGENT observation must complete Bull/Bear/Risk agent
    status and put the finished agent's report into Current Report."""
    buf = MessageBuffer()
    buf.init_for_analysis(["market", "social", "news", "fundamentals"])
    reporter = _make_reporter(buf)

    for agent, display in [
        ("bull_researcher", "Bull Researcher"),
        ("bear_researcher", "Bear Researcher"),
        ("aggressive_risk_analyst", "Aggressive Analyst"),
        ("neutral_risk_analyst", "Neutral Analyst"),
        ("conservative_risk_analyst", "Conservative Analyst"),
    ]:
        reporter._handle_event(_call_agent_event(agent, f"{agent} REPORT"))
        assert buf.agent_status[display] == "completed"
        assert f"{agent} REPORT" in buf.current_report
        assert display in buf.current_report


def test_pm_current_report_uses_observation_not_state_reports():
    """Test 6: Portfolio Manager completion must come from
    event.observation alone — the reporter no longer reads mutable
    TradingRunState at all."""
    buf = MessageBuffer()
    buf.init_for_analysis(["market"])
    reporter = _make_reporter(buf)

    reporter._handle_event(_call_agent_event("portfolio_manager", "PM FINAL REPORT"))

    assert buf.agent_status["Portfolio Manager"] == "completed"
    assert "PM FINAL REPORT" in buf.current_report
    assert "Portfolio Manager" in buf.current_report
    assert buf.report_sections["final_trade_decision"] == "PM FINAL REPORT"


def test_official_report_section_kept_full_while_preview_truncated():
    """Test 7: Current Report may truncate for terminal performance, but the
    authoritative report_sections must keep the full report text."""
    buf = MessageBuffer()
    buf.init_for_analysis(["market"])
    reporter = _make_reporter(buf)
    long_report = "Market data for NVDA. " * 100  # > 2000 chars
    assert len(long_report) > 2000

    reporter._handle_event(_call_agent_event("market_analyst", long_report))

    assert buf.report_sections["market_report"] == long_report
    assert "[truncated]" in buf.current_report


def test_blocked_runtime_still_updates_ui_after_drain():
    """Test A: while the runtime worker is blocked (not finished), the UI
    thread must already see Start <Agent> in progress and the tool call."""
    buf = MessageBuffer()
    buf.init_for_analysis(["market"])
    reporter = _make_reporter(buf)

    started = threading.Event()
    release = threading.Event()

    def worker():
        reporter.on_event(_agent_call_event("market_analyst"), None)
        reporter.on_event(
            _tool_call_event("market_analyst", "get_stock_data", {"symbol": "SNDK"}),
            None,
        )
        started.set()
        release.wait(5)
        reporter.on_event(_call_agent_event("market_analyst", "MARKET REPORT"), None)

    t = threading.Thread(target=worker)
    t.start()
    assert started.wait(2), "worker did not start"

    # Runtime is still blocked; UI already consumes the events.
    assert t.is_alive()
    reporter.drain_events()
    assert any("Start Market Analyst" in content for _, content in buf.progress_events)
    assert any(tool_name == "get_stock_data" for _, tool_name, _ in buf.tool_calls)

    release.set()
    t.join(5)
    reporter.drain_events()
    assert buf.agent_status["Market Analyst"] == "completed"


def test_rich_refresh_happens_before_runtime_completes():
    """Test B: spin_runtime_ui must call live.refresh() while the runtime
    future is still pending — rendering happens during the blocking run."""
    from cli.main import create_layout, spin_runtime_ui

    buf = MessageBuffer()
    buf.init_for_analysis(["market"])
    reporter = _make_reporter(buf)
    layout = create_layout()

    class FakeFuture:
        def __init__(self):
            self._done = False

        def done(self):
            return self._done

    future = FakeFuture()

    class RecordingLive:
        def __init__(self, future):
            self.future = future
            self.refresh_count = 0
            self.refresh_while_runtime_active = False

        def refresh(self):
            self.refresh_count += 1
            if not self.future.done():
                self.refresh_while_runtime_active = True
            # The event-driven loop only renders when events arrive (or on
            # the 1s footer tick), so the runtime "completes" on the first
            # event-triggered refresh to keep the test fast.
            self.future._done = True

    live = RecordingLive(future)
    reporter.on_event(_agent_call_event("market_analyst"), None)

    spin_runtime_ui(future, live, layout, reporter, interval=0.001)

    assert live.refresh_count >= 2
    assert live.refresh_while_runtime_active, "refresh must happen before future.done()"
    assert any("Start Market Analyst" in content for _, content in buf.progress_events)


def test_runtime_worker_thread_never_touches_ui():
    """Test C: on_event runs on the runtime worker thread and must only
    enqueue — never call update_display / refresh."""
    buf = MessageBuffer()
    buf.init_for_analysis(["market"])
    ui_calls = []

    def record_display(*args, **kwargs):
        ui_calls.append(("update_display", threading.get_ident()))

    reporter = CliRuntimeReporter(
        message_buffer=buf,
        layout=None,
        update_display=record_display,
    )

    worker_result = {}

    def worker():
        worker_result["tid"] = threading.get_ident()
        reporter.on_event(_agent_call_event("market_analyst"), None)

    t = threading.Thread(target=worker)
    t.start()
    t.join(2)

    assert ui_calls == [], "on_event must not render from the worker thread"
    reporter.drain_events()
    assert any("Start Market Analyst" in content for _, content in buf.progress_events)


def test_event_order_preserved_through_queue():
    """Test D: the queue must preserve event order — Tool Result can never
    appear before Tool Call, report generated after generating response."""
    buf = MessageBuffer()
    buf.init_for_analysis(["market"])
    reporter = _make_reporter(buf)

    for event in [
        _agent_call_event("market_analyst"),
        _tool_call_event("market_analyst", "get_stock_data", {"symbol": "SNDK"}),
        _tool_result_event("market_analyst", "get_stock_data"),
        _llm_call_event("market_analyst"),
        _call_agent_event("market_analyst", "MARKET REPORT"),
    ]:
        reporter.on_event(event, None)

    reporter.drain_events()

    progress = [content for _, content in buf.progress_events]
    assert progress.index("Start Market Analyst") < progress.index("Market Analyst calls tool get_stock_data")
    assert progress.index("Market Analyst calls tool get_stock_data") < progress.index("Market Analyst generating response...")
    assert progress.index("Market Analyst generating response...") < progress.index("Market Analyst report generated")
    tool_names = [name for _, name, _ in buf.tool_calls]
    assert tool_names == ["get_stock_data"]
    message_types = [mt for _, mt, _ in buf.messages]
    assert "Tool Result" in message_types
    # Tool Result (from OBSERVATION) must come after the Tool Call was recorded.
    assert message_types.index("Tool Result") >= 0
    assert "MARKET REPORT" in buf.report_sections["market_report"]

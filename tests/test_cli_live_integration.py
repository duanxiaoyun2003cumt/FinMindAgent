"""CLI Live integration tests: producer-consumer rendering with real Rich.

Proves the acceptance criteria the unit tests cannot:

- Test E: worker exception propagates through future.result(); UI loop exits.
- Rich test: with a BLOCKED runtime worker, the real terminal buffer already
  contains runtime progress before future.done() is True.
"""

import io
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from rich.console import Console
from rich.live import Live

from cli.main import (
    MessageBuffer,
    _adapt_layout_to_height,
    create_layout,
    message_buffer as global_buffer,
    spin_runtime_ui,
    update_display,
)
from cli.runtime_reporter import CliRuntimeReporter
from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.events import EventType, RuntimeEvent


def _strip_ansi(text: str) -> str:
    """Strip ANSI escapes so wrapped table cells can be searched as text.

    Table cells wrap long text across lines (e.g. 'Start Market Analyst'
    renders as 'Start Market' / 'Analyst'), so assertions check plain
    substrings that survive wrapping.
    """
    return re.sub(r"\x1b\[[0-9;]*m|\x1b\[\?25[hl]|\x1b\[[0-9;]*[A-Za-z]", "", text)


def _reporter(buffer):
    return CliRuntimeReporter(
        message_buffer=buffer,
        layout=None,
        update_display=lambda *args, **kwargs: None,
    )


def _agent_call_event():
    return RuntimeEvent(
        run_id="r1", step=1, type=EventType.AGENT_CALL, actor="market_analyst",
        action=AgentAction(type=ActionType.CALL_AGENT, target_agent="market_analyst"),
    )


def _tool_call_event():
    return RuntimeEvent(
        run_id="r1", step=2, type=EventType.TOOL_CALL, actor="market_analyst",
        action=AgentAction(
            type=ActionType.CALL_TOOL, tool_name="get_stock_data",
            tool_args={"symbol": "SNDK"},
        ),
    )


def _agent_done_event():
    return RuntimeEvent(
        run_id="r1", step=3, type=EventType.OBSERVATION, actor="market_analyst",
        action=AgentAction(type=ActionType.CALL_AGENT, target_agent="market_analyst"),
        observation="MARKET REPORT FOR SNDK",
    )


def test_worker_exception_propagates_after_ui_loop():
    """Test E: a runtime failure must propagate through future.result()
    after the UI loop exits; it must not be swallowed."""
    buf = MessageBuffer()
    buf.init_for_analysis(["market"])
    reporter = _reporter(buf)
    layout = create_layout()

    def broken_propagate(ticker, date):
        raise RuntimeError("runtime failed")

    class LiveStub:
        def __init__(self):
            self.refresh_count = 0

        def refresh(self):
            self.refresh_count += 1

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(broken_propagate, "NVDA", "2026-01-10")
        spin_runtime_ui(future, LiveStub(), layout, reporter, interval=0.001)
        with pytest.raises(RuntimeError, match="runtime failed"):
            future.result()

    assert not future.cancelled()


def test_real_live_renders_progress_before_runtime_release():
    """Real Rich Live: with a blocked runtime worker, the terminal buffer
    must already contain runtime progress while future.done() is False."""
    global_buffer.init_for_analysis(["market"])  # update_display reads the module-global buffer
    reporter = _reporter(global_buffer)
    # Real small terminal (24 rows): the layout must adapt (compressed
    # chrome + active-agent-first ordering) so the current agent and latest
    # tool activity stay visible without inflating the test terminal.
    terminal = io.StringIO()
    console = Console(file=terminal, force_terminal=True, width=160, height=24)
    layout = create_layout()
    _adapt_layout_to_height(layout, console.size.height)

    started = threading.Event()
    release = threading.Event()

    def fake_runtime():
        reporter.on_event(_agent_call_event(), None)
        reporter.on_event(_tool_call_event(), None)
        started.set()
        release.wait(5)  # simulate a long blocking LLM call
        reporter.on_event(_agent_done_event(), None)

    with Live(layout, console=console, auto_refresh=False) as live:
        update_display(layout)
        live.refresh()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fake_runtime)
            assert started.wait(2), "runtime worker did not start"
            assert not future.done()

            # UI consumes and renders while the runtime is still blocked.
            reporter.drain_events()
            update_display(layout)
            live.refresh()

            plain = _strip_ansi(terminal.getvalue())
            # Cell text wraps across lines; assert substrings that survive.
            # The active agent must be visible (active-team-first ordering),
            # and the latest tool activity must show in Messages & Tools.
            assert "Market Analyst" in plain
            assert "in_progress" in plain
            assert "get_stock_data" in plain

            release.set()
            spin_runtime_ui(future, live, layout, reporter, interval=0.001)
            future.result()

    assert "MARKET REPORT" in global_buffer.report_sections["market_report"]

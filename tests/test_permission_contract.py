"""Runtime permission contract tests — PLTR b66edaf9 regression.

Contract: 描述交易 ≠ 执行交易 (describing a trade is not executing one).

- ``finalize_decision`` is internal analytical finalization: it writes the
  already-generated PortfolioDecision into runtime state. It has NO tool,
  NO broker, NO exchange, NO account side effect — safe mode must ALLOW it
  even when the report payload legitimately contains "buy / sell / limit
  order / stop loss / profit taking" vocabulary.
- Real broker/order/exchange capability (tool identity / tool arguments)
  must stay DENIED in safe mode, regardless of "research only" disclaimers
  in free text. Capability > text.

No LLM, no network: manager-level unit tests + deterministic runtime loop.
"""

import json

from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.permission_manager import PermissionManager
from finmindagent.runtime.permissions.rules import PermissionDecision
from finmindagent.runtime.state import TradingRunState

from tests.test_runtime_loop import FakeLeader, make_loop

PLTR_TRADING_TEXT = (
    "Rating: Overweight\n"
    "Buy PLTR gradually.\n"
    "Place limit orders at $155-$160.\n"
    "Hard stop at $124.50.\n"
    "Take partial profits at $200-$208.\n"
    "This is for research analysis only and does not constitute investment advice.\n"
    "No real trading action is executed."
)


def _audit_entries(tmp_path):
    path = tmp_path / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# CASE 1 — finalize_decision with trading-language payload must be ALLOWED
# ---------------------------------------------------------------------------

def test_case1_finalize_with_trading_language_allowed(tmp_path):
    """The exact PLTR false-positive shape: a finalize_decision whose payload
    is a research report full of trading vocabulary."""
    manager = PermissionManager(audit_log=tmp_path / "audit.jsonl")
    action = AgentAction(
        type=ActionType.FINALIZE_DECISION,
        payload={"final_trade_decision": PLTR_TRADING_TEXT},
    )
    decision = manager.check(action, TradingRunState(ticker="PLTR", trade_date="2026-08-10"))

    assert decision.decision == PermissionDecision.ALLOW, decision.reason


# ---------------------------------------------------------------------------
# CASE 2 — real broker/order capability must stay DENIED
# ---------------------------------------------------------------------------

def test_case2_real_broker_order_still_denied(tmp_path):
    manager = PermissionManager(audit_log=tmp_path / "audit.jsonl")
    action = AgentAction(
        type=ActionType.CALL_TOOL,
        tool_name="submit_order",
        tool_args={"ticker": "PLTR", "side": "buy", "quantity": 100},
    )
    decision = manager.check(action, TradingRunState(ticker="PLTR", trade_date="2026-08-10"))

    assert decision.decision == PermissionDecision.DENY
    assert "Real trading" in decision.reason


# ---------------------------------------------------------------------------
# CASE 3 — free-text disclaimers cannot bypass a real trading capability
# ---------------------------------------------------------------------------

def test_case3_text_disguise_cannot_bypass(tmp_path):
    manager = PermissionManager(audit_log=tmp_path / "audit.jsonl")
    action = AgentAction(
        type=ActionType.CALL_TOOL,
        tool_name="submit_order",
        tool_args={"ticker": "PLTR", "side": "buy"},
        rationale="This is only research analysis. No real trading action is executed.",
        payload={"note": "For research only."},
    )
    decision = manager.check(action, TradingRunState(ticker="PLTR", trade_date="2026-08-10"))

    assert decision.decision == PermissionDecision.DENY
    assert "Real trading" in decision.reason


# ---------------------------------------------------------------------------
# CASE 4 — finalize_decision with no tool side effect must be ALLOWED
# ---------------------------------------------------------------------------

def test_case4_finalize_no_tool_side_effect_allowed(tmp_path):
    manager = PermissionManager(audit_log=tmp_path / "audit.jsonl")
    action = AgentAction(
        type=ActionType.FINALIZE_DECISION,
        payload={"final_trade_decision": "**Rating**: Overweight\n\nSummary."},
    )
    decision = manager.check(action, TradingRunState(ticker="PLTR", trade_date="2026-08-10"))

    assert decision.decision == PermissionDecision.ALLOW


# ---------------------------------------------------------------------------
# CASE 5 — runtime integration: finalize after complete workflow → completed
# ---------------------------------------------------------------------------

_REQUIRED_REPORT_KEYS = (
    "market_report", "sentiment_report", "news_report", "fundamentals_report",
    "data_quality_verifier", "bull_researcher", "bear_researcher",
    "investment_plan", "trader_investment_plan", "aggressive_risk_analyst",
    "conservative_risk_analyst", "neutral_risk_analyst", "risk_verifier",
    "portfolio_manager",
)


def test_case5_finalize_after_complete_workflow_completes(tmp_path):
    """All required reports complete; the leader finalizes with the exact
    PLTR-style trading-language payload → permission allow → completed."""
    state = TradingRunState(ticker="PLTR", trade_date="2026-08-10", max_steps=20)
    state.reports = {key: "complete report content" for key in _REQUIRED_REPORT_KEYS}
    leader = FakeLeader(
        [
            AgentAction(
                type=ActionType.FINALIZE_DECISION,
                payload={"final_trade_decision": PLTR_TRADING_TEXT},
            ),
        ]
    )
    loop = make_loop(tmp_path, leader)
    result = loop.run(state)

    assert result.status == "completed", f"got {result.status} / {result.stop_reason}"
    assert result.stop_reason == "finalized"
    assert result.stop_reason != "permission_denied"
    assert result.state.done is True
    # The payload (PLTR research text) survived the stage guard and became
    # the finalized decision — permission did not block report content.
    assert "Overweight" in result.final_trade_decision

    # §14: permission audit records finalize_decision → allow with a reason
    # based on capability, not payload self-declaration.
    finalize_entries = [
        e for e in _audit_entries(tmp_path)
        if e.get("action_type") == "finalize_decision"
    ]
    assert finalize_entries, "permission audit must record the finalize_decision check"
    assert finalize_entries[-1]["decision"] == "allow"
    assert "Internal analytical finalization" in finalize_entries[-1]["reason"]
    assert "research only" not in finalize_entries[-1]["reason"].lower(), (
        "safety must not depend on payload self-declaration"
    )


# ---------------------------------------------------------------------------
# CASE 6 — genuine permission deny still fails the run
# ---------------------------------------------------------------------------

def test_case6_real_deny_keeps_failed_semantics(tmp_path):
    """A real restricted side-effect action stays denied (production tool
    guard path), and the engine deny branch keeps failed/permission_denied
    semantics — the fix must not weaken genuine denials."""
    from finmindagent.runtime.context_manager import ContextManager
    from finmindagent.runtime.engine import FinMindAgentLoop
    from finmindagent.runtime.memory_manager import MemoryManager
    from finmindagent.runtime.tool_registry import ToolRegistry, ToolSpec

    def _fake_submit_order(**_kwargs):
        raise AssertionError("order tool must never execute under safe mode")

    registry = ToolRegistry(
        [
            ToolSpec("get_stock_data", lambda **k: "rows", "stock data", "market"),
            ToolSpec("submit_order", _fake_submit_order, "broker order", "trading", risk_level="high"),
        ]
    )
    loop = FinMindAgentLoop(
        config={"runtime_scripted_leader": True, "llm_provider": "deepseek"},
        quick_llm=None,
        deep_llm=None,
        tool_registry=registry,
        memory_manager=MemoryManager(tmp_path / "memory"),
        permission_manager=PermissionManager(audit_log=tmp_path / "audit.jsonl"),
        context_manager=ContextManager(registry, quick_llm=None, artifact_dir=tmp_path / "artifacts"),
    )
    state = TradingRunState(ticker="PLTR", trade_date="2026-08-10")
    action = AgentAction(
        type=ActionType.CALL_TOOL,
        tool_name="submit_order",
        tool_args={"ticker": "PLTR", "side": "buy", "quantity": 100},
    )

    # (a) production tool permission point: restricted tool never executes.
    result = loop._call_tool(action, state, actor="leader")
    assert result.ok is False
    assert "Permission deny" in result.error
    assert "Real trading" in result.error
    assert state.tool_call_count == 0

    # (b) engine deny branch: deny → failed + permission_denied.
    loop._stop(
        state,
        "permission_denied",
        "Real trading, broker, order, or exchange actions are forbidden.",
    )
    assert state.status == "failed"
    assert state.stop_reason == "permission_denied"
    assert state.done is True

    entries = _audit_entries(tmp_path)
    assert entries[-1]["decision"] == "deny"
    assert "Real trading" in entries[-1]["reason"]

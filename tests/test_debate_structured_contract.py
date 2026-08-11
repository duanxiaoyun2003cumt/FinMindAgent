"""Bull/Bear researcher structured-output contract tests (SPCX 9dd1f8d7 regression).

SPCX failure chain:
  bull_researcher called → DebateResearchOutput structured invoke →
  arguments[*].evidence = 4-5 items > schema max 3 → ValidationError
  (invoke stage) → flash fail, pro fail, route=unavailable →
  free-text fallback (non-authoritative) → HTML 多头观点 = 暂无可靠数据

Contract requirements (current-plan v8):
1. DebateResearchOutput.evidence stays max 3 (strict schema unchanged).
2. Bull/bear prompt explicitly demands 1-3 evidence, strongest first.
3. JSON schema carries maxItems=3 for evidence.
4. Deterministic bounded normalization handles the provider's occasional
   collection overflow (4-5 legal strings → first 3), with diagnostics.
5. Nothing else is ever repaired: invalid types / missing fields / wrong
   semantics still fail validation.
6. Free-text fallback never becomes report authority.
7. Structured bull success → HTML 多头观点 visible; bull+bear → debate complete.
"""

import pytest

from finmindagent.agents.schemas import DebateResearchOutput
from finmindagent.runtime.actions import ActionType, AgentAction
from finmindagent.runtime.events import EventType
from finmindagent.runtime.state import TradingRunState

EVIDENCE_OVERFLOW = ["e1", "e2", "e3", "e4", "e5"]


def _argument(title="Fundamental inflection", evidence=None, strength=8.0):
    return {
        "title": title,
        "argument": "Detailed reasoning.",
        "strength": strength,
        # Preserve the exact value: a string must NOT be converted to a
        # character list (invalid-type cases depend on it staying a string).
        "evidence": list(evidence) if isinstance(evidence, list) else (evidence or []),
    }


# ---------------------------------------------------------------------------
# CASE 1 — canonical schema stays strict (max 3 evidence)
# ---------------------------------------------------------------------------

def test_schema_rejects_more_than_3_evidence():
    """The strict schema contract must keep protecting max 3 — never relax."""
    with pytest.raises(Exception) as exc_info:
        DebateResearchOutput.model_validate({
            "summary": "Bull case.",
            "arguments": [_argument(evidence=["e1", "e2", "e3", "e4"])],
        })
    assert "evidence" in str(exc_info.value)


def test_schema_accepts_exactly_3_evidence():
    out = DebateResearchOutput.model_validate({
        "summary": "Bull case.",
        "arguments": [
            _argument(evidence=["e1", "e2", "e3"]),
            _argument(title="Second", evidence=["x"]),
        ],
    })
    assert len(out.arguments[0].evidence) == 3


# ---------------------------------------------------------------------------
# CASE 3 — JSON schema propagates maxItems
# ---------------------------------------------------------------------------

def test_json_schema_evidence_max_items_is_3():
    schema = DebateResearchOutput.model_json_schema()
    assert schema["properties"]["arguments"]["maxItems"] == 4
    assert schema["properties"]["arguments"]["minItems"] == 2
    evidence_def = schema["$defs"]["ScoredArgument"]["properties"]["evidence"]
    assert evidence_def["maxItems"] == 3
    assert evidence_def["items"]["type"] == "string"


# ---------------------------------------------------------------------------
# CASE 2 — bull researcher prompt explicitly limits evidence
# ---------------------------------------------------------------------------

def _debate_prompt(engine) -> str:
    spec = engine.subagents["bull_researcher"]
    return engine._debate_researcher_prompt(spec, TradingRunState(ticker="SPCX", trade_date="2026-08-10"), [])


def test_debate_prompt_contract_limits_evidence(tmp_path):
    from tests.test_runtime_loop import make_loop

    loop = make_loop(tmp_path, None)
    prompt = _debate_prompt(loop)
    assert "evidence" in prompt.lower()
    assert "3" in prompt
    assert "strongest" in prompt.lower()
    assert "never" in prompt.lower() or "more than 3" in prompt.lower()


# ---------------------------------------------------------------------------
# CASE 4/5/6/7 — deterministic bounded normalization
# ---------------------------------------------------------------------------

def test_normalize_truncates_overflow_to_schema_max():
    from finmindagent.runtime.structured_normalize import normalize_to_schema_max

    schema = DebateResearchOutput.model_json_schema()
    payload = {
        "summary": "Bull case.",
        "arguments": [
            _argument(evidence=EVIDENCE_OVERFLOW),
            _argument(title="Second", evidence=["a", "b"]),
        ],
    }
    normalized, records = normalize_to_schema_max(payload, schema)

    assert normalized["arguments"][0]["evidence"] == ["e1", "e2", "e3"]
    assert normalized["arguments"][0]["title"] == "Fundamental inflection"
    assert normalized["arguments"][0]["strength"] == 8.0
    assert normalized["arguments"][0]["argument"] == "Detailed reasoning."
    # within-limit argument untouched
    assert normalized["arguments"][1]["evidence"] == ["a", "b"]
    # truncations recorded
    assert records, "normalization must be observable"
    assert records[0]["action"] == "truncate"
    assert records[0]["from"] == 5 and records[0]["to"] == 3


def test_normalize_untouched_when_within_limit():
    from finmindagent.runtime.structured_normalize import normalize_to_schema_max

    schema = DebateResearchOutput.model_json_schema()
    payload = {
        "summary": "Bull case.",
        "arguments": [
            _argument(evidence=["e1", "e2", "e3"]),
            _argument(title="Second", evidence=["x"]),
        ],
    }
    normalized, records = normalize_to_schema_max(payload, schema)

    assert normalized == payload
    assert records == []


def test_normalize_does_not_repair_invalid_types():
    from finmindagent.runtime.structured_normalize import normalize_to_schema_max

    schema = DebateResearchOutput.model_json_schema()
    payload = {
        "summary": "Bull case.",
        "arguments": [_argument(evidence="not-a-list")],
    }
    normalized, records = normalize_to_schema_max(payload, schema)

    assert normalized["arguments"][0]["evidence"] == "not-a-list"  # untouched
    assert records == []
    with pytest.raises(Exception):
        DebateResearchOutput.model_validate(normalized)


def test_normalize_does_not_fabricate_missing_fields():
    from finmindagent.runtime.structured_normalize import normalize_to_schema_max

    schema = DebateResearchOutput.model_json_schema()
    payload = {
        "summary": "Bull case.",
        "arguments": [{"argument": "no title here", "strength": 8.0, "evidence": ["e1"]}],
    }
    normalized, records = normalize_to_schema_max(payload, schema)
    assert records == []
    with pytest.raises(Exception):
        DebateResearchOutput.model_validate(normalized)


# ---------------------------------------------------------------------------
# CASE 8 — free-text fallback is NOT report authority
# ---------------------------------------------------------------------------

def test_free_text_fallback_not_authoritative():
    from finmindagent.reporting.builder import build_report_schema
    from finmindagent.reporting.html import render_report_html

    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="r", status="completed")
    state.reports["bull_researcher"] = (
        "# SPCX Bullish Case — 2026-08-10\n\n"
        "- Technical Momentum Is Turning\n- Catalysts Are Strongly Bullish\n"
    )
    report = build_report_schema(state)
    assert report.debate_arguments == []

    html = render_report_html(report, chart_js="")
    debate_zone = html.split('id="debate"')[1].split("</section>")[0]
    assert "暂无可靠数据" in debate_zone
    assert "Technical Momentum" not in debate_zone


# ---------------------------------------------------------------------------
# CASE 9 — structured bull success populates HTML
# ---------------------------------------------------------------------------

def test_structured_bull_populates_html():
    from finmindagent.reporting.builder import build_report_schema
    from finmindagent.reporting.html import render_report_html

    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="r", status="completed")
    state.structured_reports["bull_researcher"] = {
        "summary": "Bull case.",
        "arguments": [
            {"title": "Fundamental inflection", "argument": "Earnings accelerating.", "strength": 8.0,
             "evidence": ["Q2 beat", "Guidance raised", "Capex cycle"]},
            {"title": "Institutional sponsorship", "argument": "Top holders adding.", "strength": 6.5,
             "evidence": ["13F filings"]},
        ],
    }
    report = build_report_schema(state)
    assert any(a.stance == "bull" for a in report.debate_arguments)

    html = render_report_html(report, chart_js="")
    bull_zone = html.split("多头观点")[1].split("空头观点")[0]
    assert "Fundamental inflection" in bull_zone
    assert "暂无可靠数据" not in bull_zone


# ---------------------------------------------------------------------------
# CASE 10/11 — completeness contract
# ---------------------------------------------------------------------------

def _completeness_for(tmp_path, bull_payload, bear_payload):
    from tests.test_runtime_loop import make_loop

    loop = make_loop(tmp_path, None)
    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="r", status="completed")
    if bull_payload:
        state.structured_reports["bull_researcher"] = bull_payload
    if bear_payload:
        state.structured_reports["bear_researcher"] = bear_payload
    return loop._compute_report_completeness(state)


_BULL_VALID = {
    "summary": "Bull case.",
    "arguments": [
        {"title": "Growth", "argument": "a", "strength": 8.0, "evidence": ["e1"]},
        {"title": "Moats", "argument": "b", "strength": 6.0, "evidence": ["e2"]},
    ],
}
_BEAR_VALID = {
    "summary": "Bear case.",
    "arguments": [
        {"title": "Valuation", "argument": "c", "strength": 7.0, "evidence": ["e1"]},
        {"title": "Cyclicality", "argument": "d", "strength": 5.0, "evidence": ["e2"]},
    ],
}


def test_bull_and_bear_complete_debate_complete(tmp_path):
    completeness = _completeness_for(tmp_path, _BULL_VALID, _BEAR_VALID)
    assert completeness["debate"] == "complete"


def test_bull_missing_debate_stays_missing(tmp_path):
    completeness = _completeness_for(tmp_path, None, _BEAR_VALID)
    assert completeness["debate"] != "complete"
    assert completeness["debate"] == "missing"


# ---------------------------------------------------------------------------
# SPCX regression fixture — 4-5 evidence through the real pipeline
# ---------------------------------------------------------------------------

class _FakeBoundLLM:
    def __init__(self, schema, payload_builder):
        self._schema = schema
        self._payload_builder = payload_builder

    def invoke(self, prompt):
        # Simulates the provider returning a parsed object for the bound schema.
        return self._schema.model_validate(self._payload_builder(self._schema))


class _FakeStructuredLLM:
    model_name = "deepseek-v4-flash"

    def __init__(self, payload_builder):
        self._payload_builder = payload_builder

    def with_structured_output(self, schema, **kwargs):
        return _FakeBoundLLM(schema, self._payload_builder)


def _spcx_loop(tmp_path, payload_builder):
    from finmindagent.runtime.context_manager import ContextManager
    from finmindagent.runtime.engine import FinMindAgentLoop
    from finmindagent.runtime.memory_manager import MemoryManager
    from finmindagent.runtime.permission_manager import PermissionManager
    from finmindagent.runtime.tool_registry import ToolRegistry

    registry = ToolRegistry([])
    fake = _FakeStructuredLLM(payload_builder)
    return FinMindAgentLoop(
        config={"runtime_scripted_leader": True, "llm_provider": "deepseek"},
        quick_llm=fake,
        deep_llm=None,
        tool_registry=registry,
        memory_manager=MemoryManager(tmp_path / "memory"),
        permission_manager=PermissionManager(audit_log=tmp_path / "audit.jsonl"),
        context_manager=ContextManager(registry, quick_llm=None, artifact_dir=tmp_path / "artifacts"),
    )


def _run_bull(loop, state=None):
    spec = loop.subagents["bull_researcher"]
    return loop._run_debate_researcher(spec, state or TradingRunState(ticker="SPCX", trade_date="2026-08-10"), [])


def test_spcx_fixture_overflow_normalized_through_pipeline(tmp_path):
    """SPCX shape: model returns 4-5 evidence → pipeline must produce a valid
    canonical DebateResearchOutput with evidence <= 3."""
    def overflow_payload(schema):
        return {
            "summary": "SPCX Bullish Case",
            "arguments": [
                _argument("Technical Momentum Is Turning", evidence=["e1", "e2", "e3", "e4"]),
                _argument("Catalysts Are Strongly Bullish", evidence=["c1", "c2", "c3", "c4", "c5"]),
            ],
        }

    loop = _spcx_loop(tmp_path, overflow_payload)
    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="spcx-fixture", status="running")
    text = _run_bull(loop, state)

    assert "bull_researcher" in state.structured_reports
    canonical = DebateResearchOutput.model_validate(state.structured_reports["bull_researcher"])
    assert all(len(arg.evidence) <= 3 for arg in canonical.arguments)
    assert text  # rendered markdown exists

    diag = state.metadata.get("structured_diagnostics", {}).get("bull_researcher", {})
    assert diag.get("success") is True
    normalizations = diag.get("normalizations") or []
    assert any(n.get("from", 0) > n.get("to", 0) for n in normalizations), (
        "normalization must be observable in structured diagnostics"
    )


def test_spcx_fixture_invalid_type_still_fails(tmp_path):
    """Invalid field types must still fail — bounded normalization never
    repairs non-length issues."""
    def bad_payload(schema):
        return {
            "summary": "SPCX Bullish Case",
            "arguments": [_argument("X", evidence="not-a-list")],
        }

    loop = _spcx_loop(tmp_path, bad_payload)
    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="spcx-bad", status="running")
    text = _run_bull(loop, state)

    assert "bull_researcher" not in state.structured_reports
    assert text  # free-text fallback still produced for workflow continuity
    diag = state.metadata.get("structured_diagnostics", {}).get("bull_researcher", {})
    assert diag.get("success") is False

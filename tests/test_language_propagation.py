"""Language propagation contract tests (current-plan v9).

Bug: config `output_language=Chinese` only reached the HTML UI (via a
hard-coded zh-CN default); agent prompts had no language contract, so
agent-generated natural language stayed English.

Contract:
- Config language is normalized to ONE canonical value at the boundary
  (Chinese → zh-CN, English → en); invalid values fail loudly.
- A single language instruction helper produces the strong contract
  (Chinese: Simplified Chinese for ALL natural-language content, with
  schema keys / enums / tickers / indicators / IDs preserved; English:
  unchanged English contract).
- The runtime carries output_language; EVERY content agent prompt
  (engine `_subagent_prompt` path — all 14 agents) receives it at the
  top of the prompt (trusted layer), for structured AND free-text paths.
- HTML locale comes from the same source of truth (state.output_language).
- Schema keys / enums / evidence IDs are never translated.
"""

import pytest

from finmindagent.agents.schemas import DebateResearchOutput, PortfolioDecision
from finmindagent.runtime.state import TradingRunState


# ---------------------------------------------------------------------------
# Test 1/2 — Config normalization
# ---------------------------------------------------------------------------

def test_normalize_chinese_to_canonical():
    from finmindagent.language import LANGUAGE_CHINESE, normalize_language

    for raw in ("Chinese", "chinese", "zh-CN", "zh_CN", "zh", "中文"):
        assert normalize_language(raw) == LANGUAGE_CHINESE, raw


def test_normalize_english_to_canonical():
    from finmindagent.language import LANGUAGE_ENGLISH, normalize_language

    for raw in ("English", "english", "en", "en-US", "en_US"):
        assert normalize_language(raw) == LANGUAGE_ENGLISH, raw


def test_invalid_language_raises():
    from finmindagent.language import normalize_language

    with pytest.raises(ValueError):
        normalize_language("abc")
    with pytest.raises(ValueError):
        normalize_language("日本語")


# ---------------------------------------------------------------------------
# Test 3 — Common language instruction
# ---------------------------------------------------------------------------

def test_chinese_instruction_contract():
    from finmindagent.language import LANGUAGE_CHINESE, language_instruction

    text = language_instruction(LANGUAGE_CHINESE)
    assert "Simplified Chinese" in text
    assert "zh-CN" in text
    assert "all natural-language" in text.lower()
    # preserves machine-readable contract
    assert "schema keys" in text.lower() or "field names" in text.lower()
    assert "enum" in text.lower()


def test_english_instruction_contract():
    from finmindagent.language import LANGUAGE_ENGLISH, language_instruction

    text = language_instruction(LANGUAGE_ENGLISH)
    assert "English" in text
    assert "Simplified Chinese" not in text


# ---------------------------------------------------------------------------
# Test 4-10 — every content agent prompt carries the contract
# ---------------------------------------------------------------------------

def _engine_prompt(loop, agent_name, language):
    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="r")
    state.output_language = language
    spec = loop.subagents[agent_name]
    if agent_name in {"bull_researcher", "bear_researcher"}:
        return loop._debate_researcher_prompt(spec, state, [])
    return loop._subagent_prompt(spec, state, [])


@pytest.mark.parametrize("agent_name", [
    "market_analyst",
    "social_sentiment_analyst",
    "news_analyst",
    "fundamentals_analyst",
    "bull_researcher",
    "bear_researcher",
    "research_manager",
    "trader",
    "aggressive_risk_analyst",
    "conservative_risk_analyst",
    "neutral_risk_analyst",
    "portfolio_manager",
])
def test_chinese_contract_in_every_agent_prompt(tmp_path, agent_name):
    from tests.test_runtime_loop import make_loop

    loop = make_loop(tmp_path, None)
    prompt = _engine_prompt(loop, agent_name, "zh-CN")
    assert "Simplified Chinese" in prompt, f"{agent_name} prompt lacks Chinese contract"
    assert "zh-CN" in prompt, f"{agent_name} prompt lacks zh-CN"
    assert "all natural-language" in prompt.lower(), (
        f"{agent_name} prompt lacks the all-content language rule"
    )


@pytest.mark.parametrize("agent_name", [
    "market_analyst",
    "bull_researcher",
    "research_manager",
    "trader",
    "aggressive_risk_analyst",
    "portfolio_manager",
])
def test_english_contract_in_agent_prompts(tmp_path, agent_name):
    from tests.test_runtime_loop import make_loop

    loop = make_loop(tmp_path, None)
    prompt = _engine_prompt(loop, agent_name, "en")
    assert "English" in prompt
    assert "Simplified Chinese" not in prompt


# ---------------------------------------------------------------------------
# Test 12 — free-text fallback path carries the same contract
# ---------------------------------------------------------------------------

def test_free_text_fallback_prompt_carries_contract(tmp_path):
    """The fallback path reuses the same composed prompt — it must inherit
    the language contract, never silently switch to English."""
    from tests.test_runtime_loop import make_loop

    loop = make_loop(tmp_path, None)
    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="r")
    state.output_language = "zh-CN"
    spec = loop.subagents["market_analyst"]
    prompt = loop._subagent_prompt(spec, state, [])
    # the same prompt object is what the fallback LLM.invoke receives
    assert "Simplified Chinese" in prompt and "zh-CN" in prompt


# ---------------------------------------------------------------------------
# Test 11 — structured pipeline preserves Chinese values
# ---------------------------------------------------------------------------

def test_structured_chinese_values_not_rewritten():
    """Chinese structured values pass through build_report_schema unchanged."""
    from finmindagent.reporting.builder import build_report_schema

    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="r", status="completed")
    state.output_language = "zh-CN"
    state.structured_reports["bull_researcher"] = {
        "summary": "公司当前处于高增长阶段。",
        "arguments": [
            {"title": "技术动能回升", "argument": "RSI 已回升至 52.7，MACD 出现金叉。",
             "strength": 8.0, "evidence": ["成交量放大", "站上 50 日均线"]},
            {"title": "机构增持", "argument": "头部机构加仓。", "strength": 6.0, "evidence": ["13F 披露"]},
        ],
    }
    report = build_report_schema(state)
    bull = next(a for a in report.debate_arguments if a.stance == "bull")
    assert bull.title == "技术动能回升"  # Chinese value preserved verbatim
    assert "RSI 已回升至 52.7" in bull.body  # Chinese + proper nouns preserved
    assert bull.evidence == ["成交量放大", "站上 50 日均线"]


# ---------------------------------------------------------------------------
# Test 13/14 — schema keys and enums stay machine-readable
# ---------------------------------------------------------------------------

def test_schema_keys_remain_english():
    schema = DebateResearchOutput.model_json_schema()
    props = set(schema["properties"].keys()) | set(schema["$defs"]["ScoredArgument"]["properties"].keys())
    assert {"summary", "arguments", "title", "argument", "strength", "evidence"} <= props


def test_enums_stay_stable():
    decision = PortfolioDecision.model_validate({
        "rating": "Overweight",
        "executive_summary": "中文摘要",
        "investment_thesis": "中文逻辑",
    })
    assert decision.rating.value == "Overweight"
    assert decision.rating.value != "增持"


# ---------------------------------------------------------------------------
# Test 15 — instruction preserves proper nouns
# ---------------------------------------------------------------------------

def test_chinese_instruction_preserves_proper_nouns():
    from finmindagent.language import LANGUAGE_CHINESE, language_instruction

    text = language_instruction(LANGUAGE_CHINESE)
    low = text.lower()
    assert "ticker" in low
    assert "indicator" in low or "rsi" in low or "macd" in low
    assert "evidence" in low and "id" in low


# ---------------------------------------------------------------------------
# Test 16 — Chinese HTML end-to-end
# ---------------------------------------------------------------------------

def test_chinese_html_end_to_end():
    from finmindagent.reporting.builder import build_report_schema
    from finmindagent.reporting.html import render_report_html

    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="r", status="completed")
    state.output_language = "zh-CN"
    state.reports["market_report"] = "SPCX 当前价格仍位于 50 日均线下方，RSI 回升至 52.7。"
    state.structured_reports["bull_researcher"] = {
        "summary": "多头逻辑主要来自技术动能与机构增持。",
        "arguments": [
            {"title": "技术动能回升", "argument": "RSI 与 MACD 出现改善。", "strength": 8.0,
             "evidence": ["成交量放大"]},
            {"title": "机构增持", "argument": "头部机构持续加仓。", "strength": 6.0, "evidence": ["13F"]},
        ],
    }
    report = build_report_schema(state)
    html = render_report_html(report, chart_js="", locale="zh-CN")

    assert "技术动能回升" in html  # agent content in Chinese
    assert "RSI 与 MACD 出现改善" in html
    assert "多头观点" in html  # UI label Chinese
    # proper nouns retained
    assert "SPCX" in html
    assert "RSI" in html


# ---------------------------------------------------------------------------
# Test 17 — English regression
# ---------------------------------------------------------------------------

def test_english_html_uses_english_locale():
    from finmindagent.reporting.builder import build_report_schema
    from finmindagent.reporting.html import render_report_html

    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="r", status="completed")
    state.output_language = "en"
    report = build_report_schema(state)
    html = render_report_html(report, chart_js="", locale="en")
    assert "Investment Analysis Report" in html


# ---------------------------------------------------------------------------
# Test 18 — no OS/browser locale dependency
# ---------------------------------------------------------------------------

def test_language_deterministic_regardless_of_env(monkeypatch):
    from finmindagent.language import LANGUAGE_CHINESE, LANGUAGE_ENGLISH, language_instruction

    zh = language_instruction(LANGUAGE_CHINESE)
    en = language_instruction(LANGUAGE_ENGLISH)
    assert zh == language_instruction(LANGUAGE_CHINESE)  # deterministic
    assert "Simplified Chinese" in zh and "Simplified Chinese" not in en
    assert not zh.lower().startswith(("ja", "ko", "de"))  # never OS-derived


# ---------------------------------------------------------------------------
# Test 19 — replay metadata round-trips; old snapshots stay compatible
# ---------------------------------------------------------------------------

def test_output_language_round_trips_through_snapshot():
    import json

    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="r", status="completed")
    state.output_language = "zh-CN"
    dumped = json.loads(state.model_dump_json())
    assert dumped["output_language"] == "zh-CN"

    reloaded = TradingRunState.model_validate(dumped)
    assert reloaded.output_language == "zh-CN"


def test_legacy_snapshot_without_language_loads_with_default():
    """Old snapshots (no output_language) must load with the legacy default."""
    state = TradingRunState.model_validate({
        "ticker": "SPCX",
        "trade_date": "2026-08-10",
        "run_id": "old",
        "status": "completed",
    })
    assert state.output_language == "zh-CN"  # legacy default, never crash


# ---------------------------------------------------------------------------
# SPCX regression — full prompt chain under language=Chinese
# ---------------------------------------------------------------------------

def test_spcx_regression_all_agent_prompts_chinese(tmp_path):
    from finmindagent.language import normalize_language
    from tests.test_runtime_loop import make_loop

    loop = make_loop(tmp_path, None)
    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="spcx-lang")
    # simulate config output_language=Chinese flowing into the runtime
    state.output_language = normalize_language("Chinese")

    for agent_name in (
        "market_analyst", "news_analyst", "fundamentals_analyst",
        "bull_researcher", "bear_researcher", "research_manager", "trader",
        "aggressive_risk_analyst", "neutral_risk_analyst",
        "conservative_risk_analyst", "portfolio_manager",
    ):
        prompt = _engine_prompt(loop, agent_name, state.output_language)
        assert "Simplified Chinese" in prompt, agent_name
        assert "zh-CN" in prompt, agent_name


def test_generate_report_artifact_uses_state_language(tmp_path):
    """HTML artifact locale must come from the same source of truth as the
    agent generation language — not a hard-coded default."""
    from finmindagent.reporting.html import generate_report_artifact

    state = TradingRunState(ticker="SPCX", trade_date="2026-08-10", run_id="r", status="completed")
    state.output_language = "en"
    path = generate_report_artifact(state, tmp_path / "artifacts")
    html = path.read_text(encoding="utf-8")
    assert '<html lang="en">' in html

    state.output_language = "zh-CN"
    path = generate_report_artifact(state, tmp_path / "artifacts2")
    html = path.read_text(encoding="utf-8")
    assert '<html lang="zh-CN">' in html

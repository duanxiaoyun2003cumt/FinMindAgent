# Runtime Loop Architecture

FinMindAgent now runs through a Claude Code style runtime in
`finmindagent/runtime/`. The old fixed LangGraph DAG has been removed.
`FinMindAgentGraph.propagate()` remains only as a compatibility facade over
the runtime loop.

## Main Flow

`FinMindAgentLoop.run()` executes a bounded loop:

1. Build static + dynamic context.
2. Inject active recall memories.
3. Check token budget and compact when needed.
4. Ask the leader for one structured `AgentAction`.
5. Validate and repair malformed action JSON once.
6. Run code-level permission checks.
7. Execute a tool, sub-agent, report write, compact, or finalization action.
8. Wrap observations in event log entries.
9. Apply tool output budget and artifact persistence.
10. Stop on `finalize_decision`, max steps, max tool calls, parse failure, or permission stop.

## Core Modules

- `engine.py`: `FinMindAgentLoop`, sub-agent specs, deterministic fallback leader.
- `state.py`: `TradingRunState` and legacy state projection.
- `actions.py`: Pydantic action schema.
- `events.py`: event stream schema.
- `tool_registry.py`: allowlisted data tools and metadata.
- `context_manager.py`: prompt assembly, output budgeting, compact handoff.
- `memory_manager.py` and `runtime/memory/*`: active recall and structured memory writes.
- `permission_manager.py` and `runtime/permissions/*`: hard deny, scope rules, audit logging.
- `runtime/tools/*`: `ToolResult`, budgets, wrappers.

## Sub-Agents

The runtime exposes the original FinMindAgent roles as callable specialists:

- `market_analyst`
- `social_sentiment_analyst`
- `news_analyst`
- `fundamentals_analyst`
- `bull_researcher`
- `bear_researcher`
- `research_manager`
- `trader`
- `aggressive_risk_analyst`
- `conservative_risk_analyst`
- `neutral_risk_analyst`
- `portfolio_manager`
- `risk_verifier`
- `data_quality_verifier`

Analyst agents receive only their local tool observations. Debate, trader,
risk, and portfolio agents receive report summaries rather than unbounded raw
tool output.

## Compatibility

`FinMindAgentGraph.propagate(ticker, date)` still returns:

```python
(final_state, processed_signal)
```

The `final_state` keeps the existing keys such as `market_report`,
`investment_plan`, `trader_investment_plan`, `risk_debate_state`, and
`final_trade_decision`.

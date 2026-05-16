# FinMindAgent

FinMindAgent is a research-only financial analysis agent harness. It runs a controlled agent runtime that gathers market evidence, builds analyst reports, stages research and risk debates, and produces a final `Buy / Overweight / Hold / Underweight / Sell` decision.

It does not place orders, execute trades, connect to brokers, or provide personalized financial advice.

## Highlights

- While-loop agent runtime in `finmindagent/runtime/`, with structured actions, event logs, tool budgets, permission checks, and stage guards.
- Specialist subagents for market, sentiment, news, fundamentals, research, trader, risk, portfolio, data-quality, and guardrail checks.
- Context management with active recall memory, compacting, external-data isolation, and prompt-injection safeguards.
- Markdown decision memory at `.finmindagent/memory/trading_memory.md`, plus structured runtime memory for recall.
- CLI progress display for Progress, Messages & Tools, and Current Report.

## Safety Boundary

FinMindAgent is designed for analysis and education. The runtime hard-denies broker/order/exchange execution, shell actions, credential writes, and misleading language such as "order executed" or "trade filled." Final reports must include evidence and research disclaimers.

Always verify market data independently before making any financial decision.

## Installation

```bash
git clone https://github.com/duanxiaoyun2003cumt/FinMindAgent.git
cd FinMindAgent
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy the example environment file and fill only the providers you plan to use:

```bash
copy .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

## Environment Variables

Common provider keys:

```text
OPENAI_API_KEY=
GOOGLE_API_KEY=
ANTHROPIC_API_KEY=
XAI_API_KEY=
DEEPSEEK_API_KEY=
DASHSCOPE_API_KEY=
ZHIPU_API_KEY=
OPENROUTER_API_KEY=
ALPHA_VANTAGE_API_KEY=
```

Optional runtime path overrides:

```text
FINMINDAGENT_RESULTS_DIR=
FINMINDAGENT_CACHE_DIR=
FINMINDAGENT_MEMORY_LOG_PATH=
```

Enterprise provider examples live in `.env.enterprise.example`.

## CLI Usage

Run the interactive CLI:

```bash
finmindagent
```

The CLI uses the same runtime path as the Python API, including context management, memory, permission audit, event logging, and final decision persistence.

## Python Usage

```python
from finmindagent.default_config import DEFAULT_CONFIG
from finmindagent.graph.trading_graph import FinMindAgentGraph

config = DEFAULT_CONFIG.copy()
graph = FinMindAgentGraph(debug=True, config=config)

final_state, decision = graph.propagate("AAPL", "2026-05-15")
print(decision)
```

## Project Layout

```text
finmindagent/
  agents/          Specialist agent prompts, schemas, and memory helpers
  dataflows/       Market, news, fundamentals, and indicator data adapters
  graph/           Compatibility facade for FinMindAgentGraph.propagate()
  llm_clients/     Provider client factory and adapters
  runtime/         Agent loop, actions, context, tools, memory, permissions
cli/               Interactive command-line interface
docs/              Runtime architecture and migration notes
tests/             Unit and regression tests
```

## Runtime Flow

The runtime advances through required stages:

```text
market/social/news/fundamentals
-> data_quality_verifier
-> bull/bear researchers
-> research_manager
-> trader
-> aggressive/conservative/neutral risk analysts
-> risk_verifier
-> portfolio_manager
-> finalize_decision
```

The leader model can propose actions, but dependency guards prevent skipping required reports. Failures are recorded as structured observations rather than silently leaving legacy report fields empty.

## Local State

Runtime outputs are intentionally ignored by git:

```text
.finmindagent/logs/
.finmindagent/cache/
.finmindagent/memory/
reports/
```

Do not commit `.env`, generated memory, logs, cached market data, or local IDE settings.

## Tests

Run the core runtime tests:

```bash
python -m pytest tests/test_runtime_loop.py -q
```

Run the full suite:

```bash
python -m pytest
```

Some integration paths may require optional provider credentials or network access.

## Documentation

- `docs/runtime_loop_architecture.md`
- `docs/context_memory_permission_design.md`


## Contributing

Pull requests are welcome. Please keep changes scoped, add tests for runtime behavior, avoid committing generated artifacts, and never include real credentials in examples, logs, or fixtures.

## License

This project is released under the license in `LICENSE`.

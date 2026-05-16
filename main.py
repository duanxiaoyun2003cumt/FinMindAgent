from dotenv import load_dotenv

from finmindagent.default_config import DEFAULT_CONFIG
from finmindagent.graph.trading_graph import FinMindAgentGraph


# Load local environment variables. Keep real credentials in .env, never in source.
load_dotenv()

config = DEFAULT_CONFIG.copy()
ticker = "MSFT"
trade_date = "2026-05-18"

fma = FinMindAgentGraph(debug=True, config=config)

_, decision = fma.propagate(ticker, trade_date)
print(decision)

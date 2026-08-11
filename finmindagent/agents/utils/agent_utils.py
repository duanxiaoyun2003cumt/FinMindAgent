from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from finmindagent.agents.utils.core_stock_tools import (
    get_stock_data
)
from finmindagent.agents.utils.technical_indicators_tools import (
    get_indicators
)
from finmindagent.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from finmindagent.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return the shared language contract for the configured output language.

    Single source of truth: ``finmindagent.language`` — the same contract
    the runtime engine injects into every content-agent prompt. Covers the
    legacy agent-graph path (analysts + portfolio manager); the runtime
    engine path uses ``language_instruction()`` directly.
    """
    from finmindagent.dataflows.config import get_config
    from finmindagent.language import language_instruction, normalize_language

    raw = get_config().get("output_language", "Chinese")
    try:
        canonical = normalize_language(raw)
    except ValueError:
        canonical = "zh-CN"  # legacy config values keep the project default
    return language_instruction(canonical)


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        

import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FINMINDAGENT_HOME = os.path.join(_PROJECT_ROOT, ".finmindagent")

DEFAULT_CONFIG = {
    "project_dir": _PROJECT_ROOT,
    "results_dir": os.getenv("FINMINDAGENT_RESULTS_DIR", os.path.join(_FINMINDAGENT_HOME, "logs")),
    "data_cache_dir": os.getenv("FINMINDAGENT_CACHE_DIR", os.path.join(_FINMINDAGENT_HOME, "cache")),
    "memory_log_path": os.getenv("FINMINDAGENT_MEMORY_LOG_PATH", os.path.join(_FINMINDAGENT_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Runtime loop settings
    "runtime_scripted_leader": True,
    "runtime_max_steps": 30,
    "runtime_max_tool_calls": 30,
    "runtime_context_tokens": 80_000,
    "runtime_max_tokens": 120_000,
    "permission_mode": "safe",
    "max_position_pct": 10,
    # Market data lookback in calendar days — used for get_stock_data and
    # get_indicators default lookback so we reliably have ~60 trading days.
    "market_lookback_days": 120,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "Chinese",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
}

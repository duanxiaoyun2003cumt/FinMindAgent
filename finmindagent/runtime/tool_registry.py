"""Runtime registry for data tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from finmindagent.runtime.tools.result import ToolResult
from finmindagent.runtime.tools.wrappers import invoke_tool, tool_display_name


@dataclass
class ToolSpec:
    name: str
    func: Callable[..., Any]
    description: str
    category: str
    risk_level: str = "low"
    search_hint: str = ""
    schema: dict[str, Any] | None = None


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec] | None = None):
        self._tools: dict[str, ToolSpec] = {}
        for spec in tools or default_tool_specs():
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def tool_index(self) -> list[dict[str, str]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "search_hint": spec.search_hint,
                "category": spec.category,
            }
            for spec in self.list_tools()
        ]

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        spec = self.get(name)
        return invoke_tool(spec.func, spec.name, args)


def _spec(func: Callable[..., Any], description: str, category: str, hint: str = "") -> ToolSpec:
    return ToolSpec(
        name=tool_display_name(func),
        func=func,
        description=description,
        category=category,
        search_hint=hint,
        risk_level="low",
    )


def default_tool_specs() -> list[ToolSpec]:
    return [
        _spec(get_stock_data, "Fetch historical OHLCV market data.", "market", "price volume history"),
        _spec(get_indicators, "Calculate technical indicators from market data.", "market", "RSI MACD SMA ATR"),
        _spec(get_news, "Fetch ticker-specific news.", "news", "company headlines events"),
        _spec(get_global_news, "Fetch macro and global market news.", "news", "macro rates geopolitics"),
        _spec(get_insider_transactions, "Fetch insider transaction information.", "news", "insider buying selling"),
        _spec(get_fundamentals, "Fetch company fundamental metrics.", "fundamentals", "valuation ratios"),
        _spec(get_balance_sheet, "Fetch balance sheet data.", "fundamentals", "assets liabilities equity"),
        _spec(get_cashflow, "Fetch cashflow statement data.", "fundamentals", "operating free cash flow"),
        _spec(get_income_statement, "Fetch income statement data.", "fundamentals", "revenue margin earnings"),
    ]


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    return _route_to_vendor("get_stock_data", symbol, start_date, end_date)


def get_indicators(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
    results = []
    for ind in indicators:
        try:
            results.append(_route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days))
        except ValueError as exc:
            results.append(str(exc))
    return "\n\n".join(results)


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    return _route_to_vendor("get_news", ticker, start_date, end_date)


def get_global_news(curr_date: str, look_back_days: int = 7, limit: int = 5) -> str:
    return _route_to_vendor("get_global_news", curr_date, look_back_days, limit)


def get_insider_transactions(ticker: str) -> str:
    return _route_to_vendor("get_insider_transactions", ticker)


def get_fundamentals(ticker: str, curr_date: str) -> str:
    return _route_to_vendor("get_fundamentals", ticker, curr_date)


def get_balance_sheet(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    return _route_to_vendor("get_balance_sheet", ticker, freq, curr_date)


def get_cashflow(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    return _route_to_vendor("get_cashflow", ticker, freq, curr_date)


def get_income_statement(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    return _route_to_vendor("get_income_statement", ticker, freq, curr_date)


def _route_to_vendor(tool_name: str, *args: Any) -> str:
    from finmindagent.dataflows.interface import route_to_vendor

    return route_to_vendor(tool_name, *args)

"""Facade for active recall and memory extraction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from finmindagent.runtime.memory.active_recall import active_recall as select_memories
from finmindagent.time_utils import system_timestamp
from finmindagent.runtime.memory.extract import MemoryExtractor
from finmindagent.runtime.memory.schemas import MemoryItem, MemoryType
from finmindagent.runtime.memory.store import MemoryStore
from finmindagent.runtime.state import TradingRunState


def _pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:+.1f}%"
    return str(value) if value is not None else "n/a"


class MemoryManager:
    def __init__(self, memory_dir: str | Path, selector_llm: Any | None = None):
        self.store = MemoryStore(memory_dir)
        self.selector_llm = selector_llm
        self.extractor = MemoryExtractor(self.store)

    def active_recall(self, query: str, state: TradingRunState, max_memories: int = 5):
        return select_memories(query, state, self.store, self.selector_llm, max_memories)

    def write_outcome_reflection(
        self,
        *,
        ticker: str,
        trade_date: str,
        decision: str = "",
        raw_return: Any = None,
        alpha_return: Any = None,
        holding_days: Any = None,
        reflection: str = "",
    ) -> Path:
        """Write/update the structured outcome memory for a completed trade.

        Idempotent: the memory id is ``outcome_{ticker}_{trade_date}``, so
        resolving the same trade again overwrites the same file instead of
        duplicating it. Content follows the single outcome-memory contract
        consumed by active recall (see MemoryItem.content format).
        """
        memory_id = f"outcome_{ticker}_{trade_date}"
        now = system_timestamp()
        summary_parts = [f"{ticker} trade outcome on {trade_date}"]
        if raw_return is not None:
            summary_parts.append(f"raw return {_pct(raw_return)}")
        if alpha_return is not None:
            summary_parts.append(f"alpha {_pct(alpha_return)}")
        item = MemoryItem(
            id=memory_id,
            type=MemoryType.DECISION_REFLECTION,
            scope="ticker",
            ticker=ticker,
            tags=["outcome", "reflection", ticker],
            created_at=now,
            updated_at=now,
            summary=", ".join(summary_parts) + ".",
            stale_after_days=30,
            content="\n".join(
                [
                    f"Decision date: {trade_date}",
                    f"Ticker: {ticker}",
                    f"Original decision: {decision}",
                    f"Raw return: {_pct(raw_return)}",
                    f"Alpha return: {_pct(alpha_return)}",
                    f"Holding days: {holding_days if holding_days is not None else 'n/a'}",
                    f"Reflection: {reflection}",
                ]
            ),
        )
        return self.store.write(item)

    def write_after_run(self, state: TradingRunState):
        return self.extractor.extract(state)


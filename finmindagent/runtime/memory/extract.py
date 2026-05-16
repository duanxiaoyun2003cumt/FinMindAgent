"""Memory extraction at the end of a runtime run."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from finmindagent.runtime.memory.schemas import MemoryItem
from finmindagent.runtime.memory.store import MemoryStore
from finmindagent.runtime.state import FinMindRunState


class MemoryExtractor:
    def __init__(self, store: MemoryStore):
        self.store = store

    def extract(self, state: FinMindRunState) -> list[Path]:
        if state.metadata.get("memory_written"):
            return []
        if not state.final_trade_decision:
            return []
        now = datetime.now(timezone.utc).isoformat()
        item = MemoryItem(
            id=f"decision_{state.ticker}_{state.trade_date}_{state.run_id[:8]}",
            type="decision_reflection",
            scope="ticker",
            ticker=state.ticker,
            tags=["runtime", "decision"],
            created_at=now,
            updated_at=now,
            summary=f"{state.ticker} decision on {state.trade_date}; useful for future same-ticker context.",
            stale_after_days=30,
            content="\n\n".join(
                [
                    f"# {state.ticker} {state.trade_date} Decision",
                    "## Final Trade Decision",
                    state.final_trade_decision,
                    "## Data Summary",
                    "\n".join(sorted(state.artifacts.keys())) or "No large artifacts recorded.",
                    "## Guardrail",
                    state.reports.get("risk_verifier", "No risk verifier report recorded."),
                ]
            ),
        )
        return [self.store.write(item)]


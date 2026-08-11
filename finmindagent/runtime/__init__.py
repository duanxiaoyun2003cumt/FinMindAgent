"""Claude Code style runtime loop for FinMindAgent."""

from finmindagent.runtime.engine import FinMindAgentLoop
from finmindagent.runtime.state import TradingRunResult, TradingRunState

__all__ = ["FinMindAgentLoop", "TradingRunResult", "TradingRunState", "create_runtime", "run_analysis"]


def __getattr__(name):
    if name in {"create_runtime", "run_analysis"}:
        from finmindagent.runtime.runner import create_runtime, run_analysis

        return {"create_runtime": create_runtime, "run_analysis": run_analysis}[name]
    raise AttributeError(name)

"""Claude Code style runtime loop for FinMindAgent."""

from finmindagent.runtime.engine import FinMindAgentLoop
from finmindagent.runtime.state import (
    FinMindRunResult,
    FinMindRunState,
)

__all__ = [
    "FinMindAgentLoop",
    "FinMindRunResult",
    "FinMindRunState",

]


def __getattr__(name):
    if name in {"create_runtime", "run_analysis"}:
        from finmindagent.runtime.runner import create_runtime, run_analysis

        return {"create_runtime": create_runtime, "run_analysis": run_analysis}[name]
    raise AttributeError(name)

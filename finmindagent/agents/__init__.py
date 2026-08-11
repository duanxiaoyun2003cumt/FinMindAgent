"""Runtime-era agent package.

The old LangGraph node factories are no longer imported at package import time.
Structured schemas and memory helpers remain available from their direct module
paths, while execution is owned by ``finmindagent.runtime``.
"""

__all__: list[str] = []

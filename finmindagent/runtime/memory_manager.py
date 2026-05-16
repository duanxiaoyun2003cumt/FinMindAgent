"""Facade for active recall and memory extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from finmindagent.runtime.memory.active_recall import active_recall as select_memories
from finmindagent.runtime.memory.extract import MemoryExtractor
from finmindagent.runtime.memory.store import MemoryStore
from finmindagent.runtime.state import FinMindRunState


class MemoryManager:
    def __init__(self, memory_dir: str | Path, selector_llm: Any | None = None):
        self.store = MemoryStore(memory_dir)
        self.selector_llm = selector_llm
        self.extractor = MemoryExtractor(self.store)

    def active_recall(self, query: str, state: FinMindRunState, max_memories: int = 5):
        return select_memories(query, state, self.store, self.selector_llm, max_memories)

    def write_after_run(self, state: FinMindRunState):
        return self.extractor.extract(state)


"""Active recall selection for structured memories."""

from __future__ import annotations

import json
import re
from typing import Any

from finmindagent.runtime.memory.schemas import MemoryItem
from finmindagent.runtime.memory.store import MemoryStore


def active_recall(
    query: str,
    state: Any,
    store: MemoryStore | None = None,
    selector_llm: Any | None = None,
    max_memories: int = 5,
) -> list[MemoryItem]:
    store = store or MemoryStore(getattr(state, "memory_dir", "memory"))
    headers = store.scan_headers()
    if not headers:
        return []

    selected_ids = _select_ids(query, headers, selector_llm, max_memories + 2)
    valid = {item.id for item in headers}
    selected: list[MemoryItem] = []
    already = getattr(state, "injected_memory_ids", set())
    for memory_id in selected_ids:
        if memory_id not in valid or memory_id in already:
            continue
        full = store.read_full(memory_id)
        if not full:
            continue
        full.content = full.content[:4096]
        selected.append(full)
        already.add(memory_id)
        if len(selected) >= max_memories:
            break
    if hasattr(state, "injected_memory_ids"):
        state.injected_memory_ids = already
    return selected


def _select_ids(query: str, headers: list[MemoryItem], selector_llm: Any | None, limit: int) -> list[str]:
    if selector_llm is not None:
        index = "\n".join(
            f"{m.id} | {m.type} | {m.ticker or ''} | {','.join(m.tags)} | {m.updated_at or ''} | {m.summary}"
            for m in headers
        )
        prompt = (
            "Select only clearly relevant memory IDs. Return JSON array of IDs, no prose.\n"
            f"Query: {query}\nMemory index:\n{index}\nMax IDs: {limit}"
        )
        try:
            raw = selector_llm.invoke(prompt)
            content = getattr(raw, "content", raw)
            ids = json.loads(content)
            if isinstance(ids, list):
                return [str(x) for x in ids][:limit]
        except Exception:
            pass

    q = query.lower()
    scored: list[tuple[int, str]] = []
    for item in headers:
        haystack = " ".join([item.id, item.type, item.ticker or "", " ".join(item.tags), item.summary]).lower()
        score = 0
        for token in set(re.findall(r"[A-Za-z0-9_.-]+", q)):
            if len(token) > 2 and token in haystack:
                score += 1
        if item.ticker and item.ticker.lower() in q:
            score += 3
        if score >= 2:
            scored.append((score, item.id))
    return [memory_id for _, memory_id in sorted(scored, reverse=True)[:limit]]


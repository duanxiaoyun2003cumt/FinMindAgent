"""Context compaction helpers."""

from __future__ import annotations

from typing import Any

from finmindagent.runtime.events import EventType, RuntimeEvent


def micro_compact_events(events: list[RuntimeEvent], keep_recent: int = 12) -> list[RuntimeEvent]:
    """Compact old events while preserving tool call/result adjacency."""
    if len(events) <= keep_recent:
        return events

    cutoff = len(events) - keep_recent
    compacted: list[RuntimeEvent] = []
    i = 0
    while i < cutoff:
        event = events[i]
        next_event = events[i + 1] if i + 1 < len(events) else None
        if event.type == EventType.TOOL_CALL and next_event and next_event.type == EventType.OBSERVATION:
            compacted.append(_placeholder(event))
            compacted.append(_placeholder(next_event))
            i += 2
            continue
        if event.type == EventType.OBSERVATION and i > 0 and events[i - 1].type == EventType.TOOL_CALL:
            compacted.append(_placeholder(event))
        elif event.type in {EventType.FINAL, EventType.ERROR, EventType.PERMISSION, EventType.MEMORY}:
            compacted.append(event)
        i += 1

    return compacted + events[cutoff:]


def _placeholder(event: RuntimeEvent) -> RuntimeEvent:
    clone = event.model_copy(deep=True)
    clone.observation = "[compacted: see artifact/event log for full content]"
    clone.metadata = {**clone.metadata, "compacted": True}
    return clone


def extract_summary_text(response: Any) -> str:
    text = getattr(response, "content", response)
    if not isinstance(text, str):
        text = str(text)
    start = text.find("<summary>")
    end = text.find("</summary>")
    if start != -1 and end != -1 and end > start:
        return text[start + len("<summary>") : end].strip()
    return text.strip()


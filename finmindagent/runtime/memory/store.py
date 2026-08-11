"""Markdown frontmatter memory store."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from finmindagent.runtime.memory.schemas import MemoryItem
from finmindagent.time_utils import now_system, system_timestamp

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


class MemoryStore:
    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def scan_headers(self) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        for path in sorted(self.memory_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            item = self._parse_memory(text, path, body_lines=30)
            if item:
                items.append(item)
        return items

    def read_full(self, memory_id: str) -> MemoryItem | None:
        for path in self.memory_dir.glob("*.md"):
            text = path.read_text(encoding="utf-8")
            item = self._parse_memory(text, path)
            if item and item.id == memory_id:
                return item
        return None

    def write(self, item: MemoryItem) -> Path:
        path = self.memory_dir / f"{safe_memory_filename(item.id)}.md"
        now = system_timestamp()
        item.updated_at = item.updated_at or now
        item.created_at = item.created_at or now
        frontmatter = {
            "id": item.id,
            "type": item.type,
            "scope": item.scope,
            "ticker": item.ticker or "",
            "market": item.market or "",
            "tags": item.tags,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "last_verified_at": item.last_verified_at or "",
            "summary": item.summary,
            "stale_after_days": item.stale_after_days or 30,
        }
        text = "---\n" + "\n".join(f"{k}: {format_frontmatter_value(v)}" for k, v in frontmatter.items()) + "\n---\n\n" + item.content
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        return path

    def _parse_memory(self, text: str, path: Path, body_lines: int | None = None) -> MemoryItem | None:
        match = FRONTMATTER_RE.match(text)
        if not match:
            return None
        meta = parse_frontmatter(match.group(1))
        body = match.group(2)
        if body_lines is not None:
            body = "\n".join(body.splitlines()[:body_lines])
        if not meta.get("id"):
            return None
        item = MemoryItem(
            id=str(meta.get("id")),
            type=str(meta.get("type", "")),
            scope=str(meta.get("scope", "global")),
            ticker=str(meta["ticker"]) if meta.get("ticker") else None,
            market=str(meta["market"]) if meta.get("market") else None,
            tags=meta.get("tags") if isinstance(meta.get("tags"), list) else [],
            created_at=str(meta.get("created_at", "")) or None,
            updated_at=str(meta.get("updated_at", "")) or None,
            last_verified_at=str(meta.get("last_verified_at", "")) or None,
            summary=str(meta.get("summary", "")),
            stale_after_days=int(meta.get("stale_after_days") or 30),
            path=str(path),
            content=body.strip(),
        )
        item.is_stale = is_stale(item)
        if item.is_stale:
            item.stale_warning = "STALE WARNING: this memory is a historical snapshot; verify current data before use."
        return item


def parse_frontmatter(text: str) -> dict:
    meta = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            meta[key.strip()] = [x.strip().strip("\"'") for x in inner.split(",") if x.strip()]
        else:
            meta[key.strip()] = value.strip("\"'")
    return meta


def format_frontmatter_value(value):
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]"
    text = str(value)
    if any(ch in text for ch in [":", "#", "\n"]):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def is_stale(item: MemoryItem) -> bool:
    if not item.updated_at or not item.stale_after_days:
        return False
    try:
        updated = datetime.fromisoformat(item.updated_at.replace("Z", "+00:00"))
        age = now_system() - updated
        return age.days > item.stale_after_days
    except Exception:
        return False


def safe_memory_filename(memory_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", memory_id)[:120]


"""JSONL permission audit log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finmindagent.runtime.events import system_now


class PermissionAuditLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        if not self.path:
            return
        payload = {"timestamp": system_now(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


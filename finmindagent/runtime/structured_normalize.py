"""Schema-aware bounded collection normalization for structured LLM output.

Purpose (current-plan v8 §15-18): when a provider occasionally returns more
items than a collection's declared schema maximum (e.g. 4-5 evidence items
for ``DebateResearchOutput.arguments[*].evidence`` whose max is 3), the
overflow is *deterministically* truncated to the schema maximum instead of
discarding the whole structured output.

Boundaries — this module ONLY ever:
- truncates a list to a ``maxItems`` declared in the schema;
- truncates from the front (``[:max]``), preserving model order (the prompt
  contract requires strongest-first output, so front-truncation is the
  semantically correct choice);
- records every truncation as a diagnostic entry.

It NEVER:
- repairs invalid types, missing fields, or wrong semantics;
- reorders, rephrases, merges, or fabricates content;
- touches fields that are not declared lists with ``maxItems``.
"""

from __future__ import annotations

from typing import Any


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any] | None:
    name = ref.rsplit("/", 1)[-1]
    return (root.get("$defs") or {}).get(name) or (root.get("definitions") or {}).get(name)


def normalize_to_schema_max(
    value: Any,
    json_schema: dict[str, Any],
    path: str = "$",
    records: list[dict[str, Any]] | None = None,
    root: dict[str, Any] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Recursively truncate collections to the schema's declared ``maxItems``.

    ``value`` is mutated in place (dict/list); the (value, records) pair is
    returned for convenience. Only list-length overflow is ever changed.
    ``root`` is the top-level JSON schema carrying ``$defs`` (auto-seeded).
    """
    if records is None:
        records = []
    if root is None:
        root = json_schema

    if isinstance(json_schema, dict) and "$ref" in json_schema:
        resolved = _resolve_ref(json_schema["$ref"], root)
        json_schema = resolved or json_schema

    kind = json_schema.get("type") if isinstance(json_schema, dict) else None

    if kind == "object" and isinstance(value, dict):
        for key, prop_schema in (json_schema.get("properties") or {}).items():
            if key in value and isinstance(value[key], (list, dict)):
                normalize_to_schema_max(value[key], prop_schema, f"{path}.{key}", records, root)

    elif kind == "array" and isinstance(value, list):
        max_items = json_schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            records.append({
                "path": path,
                "action": "truncate",
                "from": len(value),
                "to": max_items,
            })
            del value[max_items:]
        items_schema = json_schema.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(value):
                if isinstance(item, (list, dict)):
                    normalize_to_schema_max(item, items_schema, f"{path}[{index}]", records, root)

    return value, records

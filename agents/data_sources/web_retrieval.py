"""Interface placeholder for future search/scrape retrieval."""

from __future__ import annotations

from typing import Any


def build_search_queries(event: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Build conservative public search queries for future retrieval integrations."""
    title = str(spec.get("title") or event.get("title") or "").strip()
    rules = str(spec.get("rules") or event.get("rules") or "").strip()
    if not title and not rules:
        return []
    base = title or rules
    return [
        f"{base} consensus forecast",
        f"{base} official nowcast",
        f"{base} latest public data",
    ]


"""JSONL logging helpers for forecast agents."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def serialize_for_json(value: Any) -> Any:
    """Best-effort serializer for dataclasses and nested values."""
    if is_dataclass(value):
        return serialize_for_json(asdict(value))
    if isinstance(value, dict):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_for_json(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    file_path = Path(path)
    encoded = json.dumps(serialize_for_json(payload), ensure_ascii=True, default=str)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")


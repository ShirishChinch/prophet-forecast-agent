"""FRED public CSV helpers.

FRED graph CSV endpoints do not require an API key and are suitable for
lightweight public market/macro features. Failures return empty data; callers
should never depend on these features being present.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import csv
from datetime import datetime, timedelta
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@dataclass(frozen=True)
class SeriesSnapshot:
    """Latest value and simple trailing changes for one series."""

    series_id: str
    latest: float | None
    change_7d: float | None
    change_30d: float | None
    change_90d: float | None
    latest_date: str | None
    status: str


def get_series_snapshot(series_id: str, as_of: datetime | None = None) -> SeriesSnapshot:
    """Fetch one FRED series and compute recent changes."""
    rows = _download_series(series_id)
    if not rows:
        return SeriesSnapshot(series_id, None, None, None, None, None, "missing")

    cutoff = as_of.date() if as_of else None
    if cutoff is not None:
        rows = [(date, value) for date, value in rows if date.date() <= cutoff]
    if not rows:
        return SeriesSnapshot(series_id, None, None, None, None, None, "no_rows_before_as_of")

    latest_date, latest = rows[-1]
    return SeriesSnapshot(
        series_id=series_id,
        latest=latest,
        change_7d=_change_since(rows, latest, latest_date - timedelta(days=7)),
        change_30d=_change_since(rows, latest, latest_date - timedelta(days=30)),
        change_90d=_change_since(rows, latest, latest_date - timedelta(days=90)),
        latest_date=latest_date.date().isoformat(),
        status="ok",
    )


@lru_cache(maxsize=128)
def _download_series(series_id: str) -> tuple[tuple[datetime, float], ...]:
    query = urlencode({"id": series_id})
    request = Request(
        f"{FRED_CSV_URL}?{query}",
        headers={"User-Agent": "ai-prophet-public-data/0.1"},
    )
    try:
        with urlopen(request, timeout=12.0) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError, ValueError):
        return ()

    rows: list[tuple[datetime, float]] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        date_raw = row.get("observation_date")
        value_raw = row.get(series_id)
        if not date_raw or value_raw in {None, "", "."}:
            continue
        try:
            date = datetime.fromisoformat(date_raw)
            value = float(value_raw)
        except ValueError:
            continue
        if math.isfinite(value):
            rows.append((date, value))
    rows.sort(key=lambda item: item[0])
    return tuple(rows)


def _change_since(
    rows: list[tuple[datetime, float]] | tuple[tuple[datetime, float], ...],
    latest: float,
    target_date: datetime,
) -> float | None:
    prior: float | None = None
    for date, value in rows:
        if date <= target_date:
            prior = value
        else:
            break
    if prior is None:
        return None
    return latest - prior


def flatten_snapshot(prefix: str, snapshot: SeriesSnapshot) -> dict[str, float]:
    """Convert a snapshot into numeric feature names."""
    out: dict[str, float] = {}
    for key in ("latest", "change_7d", "change_30d", "change_90d"):
        value = getattr(snapshot, key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            out[f"{prefix}_{key}"] = float(value)
    return out


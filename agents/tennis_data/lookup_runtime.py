"""Runtime tennis empirical bucket lookup."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_ARTIFACT = Path("agents/tennis_data/artifacts/tennis_lookup.json")
MIN_SUPPORTED_SAMPLES = 20
SAMPLE_CAPS = (
    (20, 50, 0.03),
    (50, 100, 0.07),
    (100, None, 0.12),
)
FALLBACK_MULTIPLIERS = {
    0: 1.00,
    1: 0.75,
    2: 0.50,
}

_CACHE: dict[Path, dict[str, Any] | None] = {}


@dataclass(frozen=True)
class TennisLookupResult:
    probability: float
    applied: bool
    source: str
    details: dict[str, Any]


def maybe_apply_tennis_lookup(
    probability: float,
    event: dict[str, Any],
    *,
    artifact_path: str | Path | None = None,
) -> TennisLookupResult:
    """Return tennis bucket residual calibration when support is strong enough."""
    raw = max(0.01, min(0.99, float(probability)))
    if not _looks_like_tennis(event):
        return TennisLookupResult(raw, False, "not_tennis", {})

    artifact = _load(Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT)
    if not artifact or artifact.get("table_type") != "tennis_bucket_lookup":
        return TennisLookupResult(raw, False, "missing_tennis_lookup", {})

    tables = artifact.get("tables")
    if not isinstance(tables, dict):
        return TennisLookupResult(raw, False, "invalid_tennis_lookup", {})

    bucket_low, bucket_high = _bucket_bounds(raw)
    series = _series_key(event)
    day = _day_before_close(event)
    day_key = str(day) if day is not None else "ALL"
    fallback_keys = [
        (f"{series}|{day_key}", 0),
        (f"{series}|ALL", 1),
        (f"ALL|{day_key}", 1),
        ("ALL|ALL", 2),
    ]
    for key, fallback_steps in fallback_keys:
        row = _find_bucket(tables.get(key), bucket_low, bucket_high)
        if row is None:
            continue
        n = int(row.get("n") or 0)
        if n < MIN_SUPPORTED_SAMPLES:
            continue
        empirical = float(row["empirical_p_yes"])
        mean_market_odds = float(row.get("mean_market_odds") or raw)
        uncapped_adjustment = empirical - mean_market_odds
        base_cap = _sample_cap(n)
        if base_cap <= 0.0:
            continue
        fallback_multiplier = FALLBACK_MULTIPLIERS.get(fallback_steps, 0.50)
        effective_cap = base_cap * fallback_multiplier
        capped_adjustment = _clip(uncapped_adjustment, -effective_cap, effective_cap)
        calibrated = _clip(raw + capped_adjustment, 0.01, 0.99)
        return TennisLookupResult(
            probability=calibrated,
            applied=True,
            source=f"tennis_bucket_lookup:{key}",
            details={
                "lookup_key": key,
                "series": series,
                "day_before_close": day,
                "bucket_low": bucket_low,
                "bucket_high": bucket_high,
                "n": n,
                "empirical_p_yes": empirical,
                "mean_market_odds": mean_market_odds,
                "raw_market_probability": raw,
                "uncapped_adjustment": uncapped_adjustment,
                "base_cap": base_cap,
                "fallback_steps": fallback_steps,
                "fallback_multiplier": fallback_multiplier,
                "effective_cap": effective_cap,
                "capped_adjustment": capped_adjustment,
                "calibrated_probability": calibrated,
            },
        )
    return TennisLookupResult(
        raw,
        False,
        "no_supported_tennis_bucket",
        {
            "series": series,
            "day_before_close": day,
            "bucket_low": bucket_low,
            "bucket_high": bucket_high,
            "raw_market_probability": raw,
        },
    )


def _load(path: Path) -> dict[str, Any] | None:
    resolved = path.resolve()
    if resolved in _CACHE:
        return _CACHE[resolved]
    if not resolved.exists():
        _CACHE[resolved] = None
        return None
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        data = None
    _CACHE[resolved] = data
    return data


def _find_bucket(rows: Any, low: int, high: int) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if int(row.get("bucket_low") or -1) == low and int(row.get("bucket_high") or -1) == high:
            return row
    return None


def _bucket_bounds(probability: float) -> tuple[int, int]:
    cent = max(1, min(99, int(round(probability * 100.0))))
    low = ((cent - 1) // 5) * 5 + 1
    high = min(99, low + 4)
    return low, high


def _sample_cap(n: int) -> float:
    for low, high, cap in SAMPLE_CAPS:
        if n >= low and (high is None or n < high):
            return cap
    return 0.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _looks_like_tennis(event: dict[str, Any]) -> bool:
    text = " ".join(
        str(event.get(key) or "")
        for key in ("title", "market_ticker", "event_ticker", "category", "description")
    ).lower()
    return any(token in text for token in ("tennis", "itf", "atp", "wta"))


def _series_key(event: dict[str, Any]) -> str:
    text = " ".join(str(event.get(key) or "") for key in ("market_ticker", "event_ticker", "title")).lower()
    if "wta" in text:
        return "wta"
    if "atp" in text:
        return "atp"
    if "itfw" in text or "women" in text:
        return "wta"
    if "itf" in text:
        return "itf"
    return "unknown"


def _day_before_close(event: dict[str, Any]) -> int | None:
    value = (
        event.get("expected_expiration_time")
        or event.get("close_time")
        or event.get("expiration_time")
        or event.get("latest_expiration_time")
    )
    if not value:
        return None
    try:
        close_time = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=UTC)
    days = round((close_time.astimezone(UTC) - datetime.now(UTC)).total_seconds() / 86400.0)
    if days < 0:
        return 0
    return max(0, min(15, int(days)))

"""Runtime sector empirical bucket residual lookup."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any

from agents.sector_data.route_verifier import verify_sector_route


DEFAULT_ARTIFACT = Path("agents/sector_data/artifacts/sector_bucket_lookup.json")
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
    3: 0.35,
}

_CACHE: dict[Path, dict[str, Any] | None] = {}


@dataclass(frozen=True)
class SectorLookupResult:
    """Sector bucket residual calibration result."""

    probability: float
    applied: bool
    source: str
    details: dict[str, Any]


def maybe_apply_sector_lookup(
    probability: float,
    event: dict[str, Any],
    *,
    template_family: str | None = None,
    artifact_path: str | Path | None = None,
) -> SectorLookupResult:
    """Apply empirical residual calibration from the sector lookup artifact.

    The lookup keeps the current market probability as the anchor:

        calibrated = market_prob + capped(empirical_yes_rate - mean_market_odds)

    Cells with fewer than 20 rows are ignored. Broader fallback cells get
    smaller caps.
    """
    raw = _clip(float(probability), 0.01, 0.99)
    if os.environ.get("SECTOR_BUCKET_LOOKUP") == "0":
        return SectorLookupResult(raw, False, "disabled", {})

    artifact = _load(Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT)
    if not artifact or artifact.get("table_type") != "sector_bucket_lookup":
        return SectorLookupResult(raw, False, "missing_sector_bucket_lookup", {})

    tables = artifact.get("tables")
    if not isinstance(tables, dict):
        return SectorLookupResult(raw, False, "invalid_sector_bucket_lookup", {})

    rule_sector = _sector_key(event, template_family)
    rule_subtype = _subtype_key(event, rule_sector)
    verification = verify_sector_route(
        event,
        proposed_sector=rule_sector,
        proposed_subtype=rule_subtype,
        template_family=template_family,
    )
    sector = verification.sector
    subtype = verification.subtype
    day = _day_before_close(event)
    day_key = str(day) if day is not None else "ALL"
    bucket_low, bucket_high = _bucket_bounds(raw)

    candidates = [
        (f"{sector}|{subtype}|{day_key}", 0),
        (f"{sector}|{subtype}|ALL", 1),
        (f"{sector}|ALL|{day_key}", 1),
        (f"{sector}|ALL|ALL", 2),
        (f"ALL|ALL|{day_key}", 2),
        ("ALL|ALL|ALL", 3),
    ]
    seen: set[str] = set()
    for key, fallback_steps in candidates:
        if key in seen:
            continue
        seen.add(key)
        row = _find_bucket(tables.get(key), bucket_low, bucket_high)
        if row is None:
            continue
        n = int(row.get("n") or 0)
        if n < MIN_SUPPORTED_SAMPLES:
            continue
        empirical = _to_float(row.get("empirical_p_yes"))
        mean_market = _to_float(row.get("mean_market_odds"))
        if empirical is None or mean_market is None:
            continue
        base_cap = _sample_cap(n)
        if base_cap <= 0.0:
            continue
        fallback_multiplier = FALLBACK_MULTIPLIERS.get(min(fallback_steps, 3), 0.35)
        effective_cap = base_cap * fallback_multiplier
        uncapped_adjustment = empirical - mean_market
        capped_adjustment = _clip(uncapped_adjustment, -effective_cap, effective_cap)
        calibrated = _clip(raw + capped_adjustment, 0.01, 0.99)
        return SectorLookupResult(
            probability=calibrated,
            applied=True,
            source=f"sector_bucket_lookup:{key}",
            details={
                "lookup_key": key,
                "sector": sector,
                "subtype": subtype,
                "rule_sector": rule_sector,
                "rule_subtype": rule_subtype,
                "llm_route_verification": {
                    "used": verification.used,
                    "agree": verification.agree,
                    "confidence": verification.confidence,
                    "reason": verification.reason,
                    "raw": verification.raw,
                },
                "day_before_close": day,
                "bucket_low": bucket_low,
                "bucket_high": bucket_high,
                "n": n,
                "empirical_p_yes": empirical,
                "mean_market_odds": mean_market,
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

    return SectorLookupResult(
        raw,
        False,
        "no_supported_sector_bucket",
        {
            "sector": sector,
            "subtype": subtype,
            "rule_sector": rule_sector,
            "rule_subtype": rule_subtype,
            "llm_route_verification": {
                "used": verification.used,
                "agree": verification.agree,
                "confidence": verification.confidence,
                "reason": verification.reason,
                "raw": verification.raw,
            },
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


def _sample_cap(n: int) -> float:
    for low, high, cap in SAMPLE_CAPS:
        if n >= low and (high is None or n < high):
            return cap
    return 0.0


def _bucket_bounds(probability: float) -> tuple[int, int]:
    cent = max(1, min(99, int(round(probability * 100.0))))
    low = ((cent - 1) // 5) * 5 + 1
    high = min(99, low + 4)
    return low, high


def _sector_key(event: dict[str, Any], template_family: str | None) -> str:
    text = _event_text(event)
    route = str(template_family or "").upper()
    if _has_any(text, ("tennis", "itf", "atp", "wta")):
        return "sports_tennis"
    if _has_any(text, ("mlb", "baseball", "kxmlb")):
        return "sports_baseball"
    if _has_any(text, ("nba", "wnba", "basketball", "ncaamb", "ncaawb")):
        return "sports_basketball"
    if _has_any(text, ("nhl", "hockey", "kxnhl")):
        return "sports_hockey"
    if _has_any(text, ("nfl", "football", "ncaafb")):
        return "sports_football"
    if _has_any(text, ("soccer", "epl", "uefa", "premier league", "champions league")):
        return "sports_soccer"
    if _has_any(text, ("golf", "pga", "masters", "pgatour")):
        return "sports_golf"
    if _has_any(text, ("ufc", "mma", "boxing", "fight")):
        return "sports_combat"
    if _has_any(text, ("btc", "bitcoin", "eth", "ethereum", "crypto", "xrp", "solana")):
        return "crypto_price"
    if _has_any(text, ("weather", "temperature", "hurricane", "rain", "snow", "storm")):
        return "weather"
    if _has_any(text, ("cpi", "inflation", "gdp", "fed", "fomc", "unemployment", "jobs report", "payroll", "rate cut", "rate hike")):
        return "macro"
    if _has_any(text, ("election", "president", "senate", "congress", "trump", "biden", "democrat", "republican", "government")):
        return "politics"
    if _has_any(text, ("eurovision", "survivor", "oscar", "emmy", "grammy", "box office", "album", "song", "movie", "music")):
        return "culture"
    if _has_any(text, ("nasdaq", "s&p", "spx", "stock", "yield", "treasury", "oil", "gas price", "dollar", "above $", "below $")):
        return "financials"
    if route == "SPORTS":
        return "generic"
    if route == "WEATHER":
        return "weather"
    if route == "MACRO_RELEASE":
        return "macro"
    if route == "POLITICS_ELECTIONS_POLICY":
        return "politics"
    if route == "CULTURE_AWARDS_ENTERTAINMENT":
        return "culture"
    if route == "PRICE_THRESHOLD":
        return "financials"
    return "generic"


def _subtype_key(event: dict[str, Any], sector: str) -> str:
    text = _event_text(event)
    if sector == "sports_tennis":
        if "wta" in text or "kxwtamatch" in text or "itfw" in text or "women" in text:
            return "wta"
        if "atp" in text or "kxatpmatch" in text:
            return "atp"
        if "itf" in text:
            return "itf"
    if sector.startswith("sports_"):
        return sector.replace("sports_", "")
    if sector == "crypto_price":
        if "btc" in text or "bitcoin" in text:
            return "btc"
        if "eth" in text or "ethereum" in text:
            return "eth"
        if "xrp" in text:
            return "xrp"
        if "solana" in text or " sol " in f" {text} ":
            return "solana"
    return "all"


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


def _event_text(event: dict[str, Any]) -> str:
    return " ".join(
        str(event.get(key) or "")
        for key in (
            "title",
            "market_ticker",
            "event_ticker",
            "category",
            "description",
            "rules",
            "yes_sub_title",
            "subtitle",
        )
    ).lower()


def _has_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))

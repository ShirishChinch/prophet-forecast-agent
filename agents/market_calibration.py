"""Empirical calibration for market-implied probabilities.

This layer answers one narrow question: when Kalshi says a contract is priced
at p, how often did similar historical contracts actually resolve YES?
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_PATH = Path("agents/order_flow/artifacts/market_prior_calibration.json")
DEFAULT_MIN_SAMPLES = 50
DEFAULT_MIN_CALIBRATED_PROBABILITY = 0.0001

_ARTIFACT_CACHE: dict[Path, dict[str, Any] | None] = {}


@dataclass(frozen=True)
class CalibrationResult:
    """Calibrated probability plus diagnostics."""

    probability: float
    applied: bool
    source: str
    details: dict[str, Any]


def calibrate_market_probability(
    probability: float,
    *,
    template_family: str | None = None,
    event: dict[str, Any] | None = None,
    artifact_path: str | Path | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> CalibrationResult:
    """Return empirical lookup-table calibration for a market probability.

    Calibration is disabled by setting `MARKET_PRIOR_CALIBRATION=0`.
    Missing or weak artifacts fail closed and return the input probability.
    """
    raw = _valid_probability(probability)
    if raw is None:
        return CalibrationResult(0.50, False, "invalid_probability", {})
    if os.environ.get("MARKET_PRIOR_CALIBRATION") == "0":
        return CalibrationResult(raw, False, "disabled", {})

    artifact = _load_artifact(Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT_PATH)
    if not artifact:
        return CalibrationResult(raw, False, "missing_artifact", {})
    if not _artifact_passes_backtest(artifact):
        return CalibrationResult(raw, False, "artifact_failed_backtest", {})
    if artifact.get("table_type") == "symmetric_threshold_cubic_residual":
        return _apply_symmetric_threshold_cubic(raw, artifact)
    if artifact.get("table_type") == "symmetric_signed_parabola_residual":
        return _apply_symmetric_signed_parabola(raw, artifact)
    if artifact.get("table_type") == "sector_time_odds":
        return _lookup_sector_time_odds(
            raw,
            template_family=template_family,
            event=event,
            artifact=artifact,
            min_samples=min_samples,
        )

    template_key = _clean_template(template_family)
    lookup_sources: list[tuple[str, list[dict[str, Any]]]] = []
    by_template = artifact.get("by_template")
    if template_key and isinstance(by_template, dict):
        rows = by_template.get(template_key)
        if isinstance(rows, list):
            lookup_sources.append((f"market_prior_calibration:{template_key}", rows))
    global_rows = artifact.get("global")
    if isinstance(global_rows, list):
        lookup_sources.append(("market_prior_calibration:global", global_rows))

    for source, rows in lookup_sources:
        row = _find_bin(rows, raw)
        if row is None:
            continue
        n = int(row.get("n") or 0)
        if n < min_samples:
            continue
        calibrated = _valid_probability(row.get("calibrated_probability"))
        if calibrated is None:
            continue
        return CalibrationResult(
            probability=_clamp(calibrated),
            applied=True,
            source=source,
            details={
                "raw_market_probability": raw,
                "bin_low": row.get("bin_low"),
                "bin_high": row.get("bin_high"),
                "n": n,
                "empirical_yes_rate": row.get("empirical_yes_rate"),
                "mean_prior": row.get("mean_prior"),
                "calibrated_probability": calibrated,
            },
        )

    return CalibrationResult(raw, False, "no_supported_bin", {"raw_market_probability": raw})


def _apply_symmetric_signed_parabola(
    probability: float,
    artifact: dict[str, Any],
) -> CalibrationResult:
    """Apply a symmetric one-parameter residual curve.

    The fitted residual is:

        forecast_probability - market_probability
            = coefficient * (p - 0.5) * abs(p - 0.5)

    This keeps YES/NO complements exactly symmetric:
    f(1 - p) == 1 - f(p).
    """
    coefficient = _to_float(artifact.get("coefficient"))
    if coefficient is None:
        return CalibrationResult(probability, False, "invalid_symmetric_curve", {})
    min_probability = _to_float(artifact.get("min_probability"))
    if min_probability is None:
        min_probability = DEFAULT_MIN_CALIBRATED_PROBABILITY
    min_probability = max(0.0, min(0.01, min_probability))

    centered = probability - 0.5
    residual = coefficient * centered * abs(centered)
    calibrated = max(min_probability, min(1.0 - min_probability, probability + residual))
    return CalibrationResult(
        probability=calibrated,
        applied=True,
        source="market_prior_calibration:symmetric_signed_parabola_residual",
        details={
            "raw_market_probability": probability,
            "coefficient": coefficient,
            "centered_probability": centered,
            "residual": residual,
            "calibrated_probability": calibrated,
            "min_probability": min_probability,
            "curve": "p + coefficient * (p - 0.5) * abs(p - 0.5)",
        },
    )


def _apply_symmetric_threshold_cubic(
    probability: float,
    artifact: dict[str, Any],
) -> CalibrationResult:
    """Apply a symmetric curve with 50 as a local sink.

    The residual is:

        forecast_probability - market_probability
            = coefficient * x * (x^2 - sink_radius^2), where x = p - 0.5

    With a positive coefficient, probabilities inside the sink radius are
    pulled toward 50%, and probabilities outside it are pushed farther away.
    """
    coefficient = _to_float(artifact.get("coefficient"))
    sink_radius = _to_float(artifact.get("sink_radius"))
    if coefficient is None or sink_radius is None:
        return CalibrationResult(probability, False, "invalid_symmetric_threshold_cubic", {})
    min_probability = _to_float(artifact.get("min_probability"))
    if min_probability is None:
        min_probability = DEFAULT_MIN_CALIBRATED_PROBABILITY
    min_probability = max(0.0, min(0.01, min_probability))
    sink_radius = max(0.0, min(0.49, sink_radius))

    centered = probability - 0.5
    residual = coefficient * centered * ((centered * centered) - (sink_radius * sink_radius))
    calibrated = max(min_probability, min(1.0 - min_probability, probability + residual))
    return CalibrationResult(
        probability=calibrated,
        applied=True,
        source="market_prior_calibration:symmetric_threshold_cubic_residual",
        details={
            "raw_market_probability": probability,
            "coefficient": coefficient,
            "sink_radius": sink_radius,
            "centered_probability": centered,
            "residual": residual,
            "calibrated_probability": calibrated,
            "min_probability": min_probability,
            "curve": "p + coefficient * x * (x^2 - sink_radius^2), x = p - 0.5",
        },
    )


def _lookup_sector_time_odds(
    probability: float,
    *,
    template_family: str | None,
    event: dict[str, Any] | None,
    artifact: dict[str, Any],
    min_samples: int,
) -> CalibrationResult:
    """Lookup by sector, time-to-close bucket, and integer odds cent."""
    sector = _runtime_sector(template_family, event or {})
    time_bucket = _event_time_bucket(event or {}) or "unknown"
    odds_cent = _odds_cent(probability)
    tables = artifact.get("tables")
    if not isinstance(tables, dict):
        return CalibrationResult(probability, False, "invalid_sector_time_artifact", {})

    fallback_keys = [
        f"{sector}|{time_bucket}",
        f"{sector}|ALL",
        f"ALL|{time_bucket}",
        "ALL|ALL",
    ]
    for key in fallback_keys:
        rows = tables.get(key)
        if not isinstance(rows, list):
            continue
        row = _find_odds_cent(rows, odds_cent)
        if row is None:
            continue
        n = int(row.get("n") or 0)
        if n < min_samples:
            continue
        calibrated = _valid_probability(row.get("calibrated_probability"))
        if calibrated is None:
            continue
        return CalibrationResult(
            probability=_clamp(calibrated),
            applied=True,
            source=f"market_prior_calibration:sector_time_odds:{key}",
            details={
                "raw_market_probability": probability,
                "sector": sector,
                "time_bucket": time_bucket,
                "lookup_key": key,
                "odds_cent": odds_cent,
                "n": n,
                "empirical_yes_rate": row.get("empirical_yes_rate"),
                "mean_prior": row.get("mean_prior"),
                "calibrated_probability": calibrated,
            },
        )
    return CalibrationResult(
        probability,
        False,
        "no_supported_sector_time_odds_cell",
        {
            "raw_market_probability": probability,
            "sector": sector,
            "time_bucket": time_bucket,
            "odds_cent": odds_cent,
        },
    )


def _load_artifact(path: Path) -> dict[str, Any] | None:
    resolved = path.resolve()
    if resolved in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[resolved]
    if not resolved.exists():
        _ARTIFACT_CACHE[resolved] = None
        return None
    try:
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        artifact = None
    _ARTIFACT_CACHE[resolved] = artifact
    return artifact


def _artifact_passes_backtest(artifact: dict[str, Any]) -> bool:
    """Require a positive market-disjoint backtest before changing live priors."""
    if os.environ.get("MARKET_PRIOR_CALIBRATION_FORCE") == "1":
        return True
    if artifact.get("allow_runtime") is True:
        return True
    backtest = artifact.get("backtest")
    if not isinstance(backtest, dict):
        return True
    improvement = _to_float(backtest.get("brier_improvement"))
    if improvement is None:
        return True
    return improvement > 0.0


def _find_bin(rows: list[dict[str, Any]], probability: float) -> dict[str, Any] | None:
    for row in rows:
        low = _to_float(row.get("bin_low"))
        high = _to_float(row.get("bin_high"))
        if low is None or high is None:
            continue
        if low <= probability < high or (probability == 1.0 and high == 1.0):
            return row
    return None


def _find_odds_cent(rows: list[dict[str, Any]], odds_cent: int) -> dict[str, Any] | None:
    for row in rows:
        if int(row.get("odds_cent") or -1) == odds_cent:
            return row
    return None


def _event_time_bucket(event: dict[str, Any]) -> str | None:
    from datetime import UTC, datetime

    deadline_value = (
        event.get("close_time")
        or event.get("expected_expiration_time")
        or event.get("expiration_time")
        or event.get("latest_expiration_time")
    )
    if not deadline_value:
        return None
    try:
        deadline = datetime.fromisoformat(str(deadline_value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    hours = (deadline - datetime.now(UTC)).total_seconds() / 3600.0
    return _time_bucket(hours)


def _time_bucket(hours_to_close: float | None) -> str:
    if hours_to_close is None or not math.isfinite(hours_to_close):
        return "unknown"
    minutes = hours_to_close * 60.0
    if minutes <= 1:
        return "<=1m"
    if minutes <= 5:
        return "<=5m"
    if minutes <= 15:
        return "<=15m"
    if hours_to_close <= 1:
        return "<=1h"
    if hours_to_close <= 4:
        return "<=4h"
    if hours_to_close <= 24:
        return "<=24h"
    if hours_to_close <= 168:
        return "<=7d"
    return ">7d"


def _odds_cent(probability: float) -> int:
    return max(1, min(99, int(round(probability * 100.0))))


def _clean_template(value: str | None) -> str | None:
    if not value:
        return None
    return str(value).strip().lower().replace(" ", "_")


def _runtime_sector(template_family: str | None, event: dict[str, Any]) -> str:
    """Map router template names into the sector keys used by training."""
    title = str(event.get("title") or event.get("yes_sub_title") or "")
    ticker = str(event.get("market_ticker") or event.get("event_ticker") or event.get("ticker") or "")
    if title or ticker:
        try:
            from agents.order_flow.features import classify_market_template

            sector = classify_market_template(title, ticker)
            if sector:
                return sector
        except Exception:
            pass

    template = _clean_template(template_family) or "generic"
    template_map = {
        "sports": "sports",
        "macro_release": "macro",
        "weather": "weather",
        "politics_elections_policy": "politics",
        "culture_awards_entertainment": "culture",
        "price_threshold": "financials",
        "company_tech_announcement": "generic",
        "generic_news_unique": "generic",
        "market_prior_baseline": "generic",
        "informed_flow_public_signal": "generic",
    }
    return template_map.get(template, template)


def _valid_probability(value: Any) -> float | None:
    number = _to_float(value)
    if number is None:
        return None
    if number > 1.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        return None
    return float(number)


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _clamp(value: float) -> float:
    return max(0.01, min(0.99, float(value)))

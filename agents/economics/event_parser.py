"""Rule-based parser for Kalshi-style economics events."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


COUNTRY_ALIASES = {
    "us": "USD",
    "u.s.": "USD",
    "united states": "USD",
    "america": "USD",
    "germany": "DEM",
    "spain": "ESP",
    "japan": "JPY",
    "australia": "AUD",
    "canada": "CAD",
    "uk": "GBP",
    "united kingdom": "GBP",
    "eurozone": "EUR",
    "euro area": "EUR",
}


@dataclass(frozen=True)
class EconomicEventSpec:
    """Structured economic event fields used by JPMaQS models."""

    model_type: str
    country_code: str | None
    variable: str
    condition: str
    threshold: float | None
    bucket_width: float | None
    target_date_text: str | None
    yes_outcome: str | None
    confidence: float


def parse_economic_event(event: dict[str, Any], spec: dict[str, Any]) -> EconomicEventSpec:
    """Parse a Prophet/Kalshi economics event into model-routing fields."""
    title = str(spec.get("title") or event.get("title") or "")
    rules = str(spec.get("rules") or event.get("rules") or event.get("description") or "")
    outcomes = spec.get("outcomes") or event.get("outcomes") or []
    yes_outcome = str(outcomes[0]).strip() if isinstance(outcomes, list) and outcomes else None
    text = f"{title} {rules} {yes_outcome or ''}".lower()

    variable, model_type = _detect_variable_and_model(text)
    country_code = _detect_country_code(text)
    condition = _detect_condition(text)
    threshold = _detect_threshold(text)
    bucket_width = _infer_bucket_width(yes_outcome, variable, condition)
    target_date_text = _extract_date_text(text)
    confidence = _confidence(variable, country_code, threshold, condition)

    return EconomicEventSpec(
        model_type=model_type,
        country_code=country_code,
        variable=variable,
        condition=condition,
        threshold=threshold,
        bucket_width=bucket_width,
        target_date_text=target_date_text,
        yes_outcome=yes_outcome,
        confidence=confidence,
    )


def _detect_variable_and_model(text: str) -> tuple[str, str]:
    if "core cpi" in text or "cpi core" in text:
        return "core_cpi", "inflation"
    if "cpi" in text or "inflation" in text:
        return "headline_cpi", "inflation"
    if "gdp" in text:
        return "gdp", "growth"
    if "housing start" in text:
        return "housing_starts", "growth"
    if "10-year" in text or "10 yr" in text or "10y" in text or "treasury" in text or "yield" in text:
        return "ten_year_yield", "yield"
    if "fed funds" in text or "target rate" in text:
        return "policy_rate_level", "policy"
    if any(token in text for token in ("rate action", "cut", "hike", "maintain", "central bank", "bank of japan", "bank of canada", "reserve bank")):
        return "policy_decision", "policy"
    if "gas price" in text or "gas prices" in text:
        return "gas_price", "commodity_price"
    return "unknown_macro", "market_prior"


def _detect_country_code(text: str) -> str | None:
    for alias, code in sorted(COUNTRY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return code
    return None


def _detect_condition(text: str) -> str:
    if "exactly" in text:
        return "exactly"
    if any(token in text for token in ("above", "greater than", "strictly greater", "over")):
        return "above"
    if any(token in text for token in ("below", "less than", "under")):
        return "below"
    if any(token in text for token in ("cut", "hike", "maintain")):
        return "class"
    return "unknown"


def _detect_threshold(text: str) -> float | None:
    patterns = (
        r"(?:above|below|over|under|greater than|less than|strictly greater than|exactly)\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"\$(-?\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _infer_bucket_width(yes_outcome: str | None, variable: str, condition: str) -> float | None:
    if condition != "exactly":
        return None
    outcome = (yes_outcome or "").lower()
    if "%" in outcome:
        return 0.1
    if variable in {"headline_cpi", "core_cpi"}:
        return 0.1
    return None


def _extract_date_text(text: str) -> str | None:
    month = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
    match = re.search(rf"\b({month}\s+\d{{1,2}},?\s+\d{{4}}|{month}\s+\d{{4}}|q[1-4]\s+\d{{4}})\b", text)
    return match.group(0) if match else None


def _confidence(variable: str, country_code: str | None, threshold: float | None, condition: str) -> float:
    score = 0.35
    if variable != "unknown_macro":
        score += 0.25
    if country_code:
        score += 0.15
    if threshold is not None or condition == "class":
        score += 0.15
    if condition != "unknown":
        score += 0.10
    return min(0.95, score)

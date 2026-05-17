"""Stubbed LLM-style parsing and evidence extraction helpers.

These are intentionally standard-library-only placeholders. They mimic the
structure of LLM tool outputs without requiring any API calls in v1.
"""

from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Any

from agents.templates import (
    COMPANY_TECH_ANNOUNCEMENT,
    CULTURE_AWARDS_ENTERTAINMENT,
    GENERIC_NEWS_UNIQUE,
    MACRO_RELEASE,
    POLITICS_ELECTIONS_POLICY,
    PRICE_THRESHOLD,
    SPORTS,
    WEATHER,
)

_THRESHOLD_PATTERN = re.compile(
    r"(?:above|over|exceed|exceeds|below|under|at least|at most|greater than|less than)\s*\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_event_with_stub_llm_or_rules(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize the event payload and extract simple structured hints."""
    title = str(event.get("title") or event.get("question") or "").strip()
    description = str(event.get("description") or "").strip()
    rules = str(event.get("rules") or "").strip()
    category = str(event.get("category") or "").strip()
    close_time = event.get("close_time")
    outcomes = list(event.get("outcomes") or []) if isinstance(event.get("outcomes"), list) else []
    threshold_value = _extract_threshold_value(title) or _extract_threshold_value(rules)
    threshold_direction = _extract_threshold_direction(title) or _extract_threshold_direction(rules)
    location_hint = _extract_location_hint(title, description)
    asset_hint = _extract_asset_hint(title, description)

    return {
        "title": title,
        "description": description,
        "rules": rules,
        "close_time": close_time,
        "category": category,
        "outcomes": outcomes,
        "event_ticker": str(event.get("event_ticker") or "").strip(),
        "market_ticker": str(event.get("market_ticker") or "").strip(),
        "threshold_value": threshold_value,
        "threshold_direction": threshold_direction,
        "location_hint": location_hint,
        "asset_hint": asset_hint,
        "event_text": " ".join(part for part in (title, description, rules) if part).strip(),
        "parser_mode": "rules_stub",
    }


def classify_event_with_stub_llm(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the closest template when keyword routing is ambiguous."""
    text = str(spec.get("event_text") or "").lower()
    category = str(spec.get("category") or "").lower()

    if any(word in text for word in ("award", "emmy", "grammy", "oscar", "eurovision", "anime")):
        return {
            "closest_template": CULTURE_AWARDS_ENTERTAINMENT,
            "template_confidence": 0.65,
            "reason": "culture/awards keywords in event text",
        }
    if any(word in text for word in ("launch", "announce", "product", "earnings", "openai", "apple", "google", "meta")):
        return {
            "closest_template": COMPANY_TECH_ANNOUNCEMENT,
            "template_confidence": 0.62,
            "reason": "company/product announcement keywords in event text",
        }
    if any(word in text for word in ("temperature", "weather", "hurricane", "storm", "rainfall", "snow")):
        return {
            "closest_template": WEATHER,
            "template_confidence": 0.70,
            "reason": "weather keywords in event text",
        }
    if any(word in text for word in ("election", "president", "senate", "bill", "policy", "government")):
        return {
            "closest_template": POLITICS_ELECTIONS_POLICY,
            "template_confidence": 0.68,
            "reason": "politics/policy keywords in event text",
        }
    if _category_has_sports(category) or _has_any_word_or_phrase(text, ("match", "game", "tournament", "playoff", "final")):
        return {
            "closest_template": SPORTS,
            "template_confidence": 0.63,
            "reason": "sports-like keywords or category",
        }
    if any(word in text for word in ("cpi", "inflation", "gdp", "jobs", "unemployment", "fed", "fomc", "rate")):
        return {
            "closest_template": MACRO_RELEASE,
            "template_confidence": 0.67,
            "reason": "macro keywords in event text",
        }
    if spec.get("threshold_value") is not None or any(word in text for word in ("btc", "bitcoin", "eth", "ethereum", "stock", "shares")):
        return {
            "closest_template": PRICE_THRESHOLD,
            "template_confidence": 0.58,
            "reason": "threshold or asset-like language in event text",
        }
    return {
        "closest_template": GENERIC_NEWS_UNIQUE,
        "template_confidence": 0.40,
        "reason": "no strong reusable template match found",
    }


def build_feature_plan_hint(template_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Return a structured feature/data plan as if proposed by an LLM."""
    title = str(spec.get("title") or "")
    if template_name == PRICE_THRESHOLD:
        return {
            "needed_sources": ["public_market_data", "spot_price_history", "volatility_proxy"],
            "structured_features": [
                "current_spot_price",
                "threshold_value",
                "distance_to_threshold",
                "time_to_close",
                "recent_return_1d",
                "realized_vol_7d",
            ],
            "unstructured_evidence_questions": [
                f"Is there recent public news likely to move the underlying for '{title}'?",
            ],
        }
    if template_name == MACRO_RELEASE:
        return {
            "needed_sources": ["public_market_data", "consensus_macro_estimates", "macro_reference_series"],
            "structured_features": [
                "market_prior",
                "consensus_forecast",
                "time_to_release",
                "rates_move",
                "commodity_move_proxy",
            ],
            "unstructured_evidence_questions": [
                f"What public macro evidence is relevant to '{title}' and is it already priced?",
            ],
        }
    if template_name == SPORTS:
        return {
            "needed_sources": ["public_market_data", "sports_odds", "ratings_or_elo", "injury_news"],
            "structured_features": [
                "market_prior",
                "sportsbook_implied_prob",
                "rating_diff",
                "home_away_flag",
            ],
            "unstructured_evidence_questions": [
                f"Are there credible lineup or injury updates for '{title}'?",
            ],
        }
    if template_name == WEATHER:
        return {
            "needed_sources": ["public_market_data", "official_forecast", "historical_weather_normals"],
            "structured_features": [
                "official_forecast_probability",
                "time_to_event",
                "threshold_distance",
                "forecast_confidence",
            ],
            "unstructured_evidence_questions": [
                f"What official public forecast best addresses '{title}'?",
            ],
        }
    if template_name == POLITICS_ELECTIONS_POLICY:
        return {
            "needed_sources": ["public_market_data", "polls_or_public_signal", "recent_public_news"],
            "structured_features": [
                "market_prior",
                "time_to_deadline",
                "institutional_constraint_score",
                "polling_signal",
            ],
            "unstructured_evidence_questions": [
                f"What credible recent public evidence changes the odds for '{title}'?",
            ],
        }
    if template_name == COMPANY_TECH_ANNOUNCEMENT:
        return {
            "needed_sources": ["public_market_data", "official_company_sources", "major_news_coverage"],
            "structured_features": [
                "market_prior",
                "time_to_deadline",
                "official_signal_strength",
                "source_credibility",
            ],
            "unstructured_evidence_questions": [
                f"What official public signal exists for '{title}'?",
            ],
        }
    if template_name == CULTURE_AWARDS_ENTERTAINMENT:
        return {
            "needed_sources": ["public_market_data", "bookmaker_odds", "expert_predictions", "press_coverage"],
            "structured_features": [
                "market_prior",
                "bookmaker_implied_prob",
                "public_odds_consensus",
                "momentum_signal",
            ],
            "unstructured_evidence_questions": [
                f"What credible public prediction sources exist for '{title}'?",
            ],
        }
    return {
        "needed_sources": ["public_market_data", "recent_public_news"],
        "structured_features": [
            "market_prior",
            "time_to_deadline",
            "reasoning_adjustment",
        ],
        "unstructured_evidence_questions": [
            f"What public evidence is most relevant to '{title}'?",
            "How likely is that evidence to already be reflected in market prices?",
        ],
    }


def extract_structured_evidence_stub(template_name: str, spec: dict[str, Any], raw_data: dict[str, Any]) -> dict[str, Any]:
    """Return a neutral structured evidence summary.

    This is intentionally conservative: no external retrieval is performed in v1.
    """
    _ = raw_data
    text = str(spec.get("event_text") or "").lower()
    if template_name in {COMPANY_TECH_ANNOUNCEMENT, POLITICS_ELECTIONS_POLICY, CULTURE_AWARDS_ENTERTAINMENT, GENERIC_NEWS_UNIQUE}:
        if "official" in text:
            credibility = "medium"
        else:
            credibility = "low"
        return {
            "direction": "neutral",
            "strength": "weak",
            "freshness": "old",
            "already_priced_likelihood": "high",
            "credibility": credibility,
            "reasoning_adjustment": 0.0,
            "summary": "No external evidence fetched in v1; leave judgment anchored to the prior.",
        }
    return {
        "direction": "neutral",
        "strength": "weak",
        "freshness": "old",
        "already_priced_likelihood": "high",
        "credibility": "low",
        "reasoning_adjustment": 0.0,
        "summary": "No external evidence fetched in v1.",
    }


def parse_close_time(value: Any) -> datetime | None:
    """Parse an event close_time into a timezone-aware datetime."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _extract_threshold_value(text: str) -> float | None:
    match = _THRESHOLD_PATTERN.search(text or "")
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_threshold_direction(text: str) -> str | None:
    lowered = (text or "").lower()
    if any(token in lowered for token in ("above", "over", "exceed", "exceeds", "greater than", "at least")):
        return "above"
    if any(token in lowered for token in ("below", "under", "less than", "at most")):
        return "below"
    return None


def _extract_location_hint(title: str, description: str) -> str | None:
    combined = " ".join(part for part in (title, description) if part)
    match = re.search(r"\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})", combined)
    if not match:
        return None
    return match.group(1).strip()


def _extract_asset_hint(title: str, description: str) -> str | None:
    combined = f"{title} {description}".lower()
    if "bitcoin" in combined or "btc" in combined:
        return "BTC"
    if "ethereum" in combined or "eth" in combined:
        return "ETH"
    if "nasdaq" in combined:
        return "NASDAQ"
    if "s&p" in combined or "spx" in combined:
        return "SPX"
    if "oil" in combined:
        return "OIL"
    if "gold" in combined:
        return "GOLD"
    return None


def _category_has_sports(category: str) -> bool:
    return bool(re.search(r"(?<![a-z0-9])sports?(?![a-z0-9])", category.lower()))


def _has_any_word_or_phrase(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    for pattern in patterns:
        if " " in pattern:
            if pattern in lowered:
                return True
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", lowered):
            return True
    return False

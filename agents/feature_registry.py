"""Feature plan and structured feature helpers for forecast templates."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import agents.llm_tools as llm_tools
from agents.prior import PriorEstimate, clamp_probability, normalize_probability
from agents.templates import (
    COMPANY_TECH_ANNOUNCEMENT,
    CULTURE_AWARDS_ENTERTAINMENT,
    GENERIC_NEWS_UNIQUE,
    INFORMED_FLOW_PUBLIC_SIGNAL,
    MACRO_RELEASE,
    POLITICS_ELECTIONS_POLICY,
    PRICE_THRESHOLD,
    SPORTS,
    TemplateRoute,
    WEATHER,
)


def build_feature_plan(
    event: dict[str, Any],
    spec: dict[str, Any],
    route: TemplateRoute,
    prior_estimate: PriorEstimate,
) -> dict[str, Any]:
    """Build a template-specific feature plan."""
    hint = llm_tools.build_feature_plan_hint(route.template_name, spec)
    return {
        "template_name": route.template_name,
        "template_confidence": route.template_confidence,
        "prior_source": prior_estimate.prior_source,
        "needed_sources": hint["needed_sources"],
        "structured_features": hint["structured_features"],
        "unstructured_evidence_questions": hint["unstructured_evidence_questions"],
        "needs_external_fetch": False,
        "v1_mode": "stub_only_no_api_calls",
    }


def collect_data(
    event: dict[str, Any],
    spec: dict[str, Any],
    route: TemplateRoute,
    feature_plan: dict[str, Any],
    prior_estimate: PriorEstimate,
) -> dict[str, Any]:
    """Collect raw data.

    v1 performs no real API access. It only surfaces event-native public fields
    and declares source status explicitly.
    """
    market_snapshot = {
        key: event.get(key)
        for key in (
            "best_bid",
            "best_ask",
            "yes_price",
            "price",
            "midpoint",
            "market_prob",
            "implied_prob",
            "recent_price_change_5m",
            "recent_price_change_1h",
            "recent_price_change_24h",
            "volume_zscore",
            "large_trade_flag",
            "order_book_imbalance",
            "spread_change",
            "liquidity_depth",
            "cross_market_confirmation",
            "news_lag_flag",
            "current_spot_price",
            "consensus_forecast",
            "sportsbook_implied_prob",
            "official_forecast_probability",
            "forecast_confidence",
            "polling_signal",
            "bookmaker_implied_prob",
        )
        if event.get(key) is not None
    }
    evidence_summary = llm_tools.extract_structured_evidence_stub(route.template_name, spec, {})
    source_status = {source: "not_fetched_v1" for source in feature_plan.get("needed_sources", [])}

    return {
        "fetched": False,
        "market_snapshot": market_snapshot,
        "evidence_summary": evidence_summary,
        "source_status": source_status,
        "notes": [
            "No external API calls performed in v1.",
            f"Prior source: {prior_estimate.prior_source}",
        ],
    }


def compute_structured_features(
    event: dict[str, Any],
    spec: dict[str, Any],
    route: TemplateRoute,
    prior_estimate: PriorEstimate,
    feature_plan: dict[str, Any],
    raw_data: dict[str, Any],
) -> dict[str, float]:
    """Compute flat numeric features for template models."""
    _ = (event, feature_plan)
    evidence = raw_data.get("evidence_summary", {})
    market_snapshot = raw_data.get("market_snapshot", {})
    close_time = llm_tools.parse_close_time(spec.get("close_time"))
    days_to_close = _days_to_close(close_time)
    hours_to_close = days_to_close * 24.0 if days_to_close >= 0 else -1.0
    current_spot = _first_probability_like_value(
        market_snapshot.get("current_spot_price"),
        market_snapshot.get("price"),
        market_snapshot.get("yes_price"),
    )
    threshold_value = _as_float(spec.get("threshold_value"))
    distance_to_threshold = 0.0
    if current_spot is not None and threshold_value is not None:
        distance_to_threshold = current_spot - threshold_value

    features: dict[str, float] = {
        "prior": clamp_probability(prior_estimate.probability),
        "template_confidence": route.template_confidence,
        "title_length": float(len(str(spec.get("title") or ""))),
        "description_length": float(len(str(spec.get("description") or ""))),
        "rules_length": float(len(str(spec.get("rules") or ""))),
        "outcomes_count": float(len(spec.get("outcomes") or [])),
        "days_to_close": days_to_close,
        "hours_to_close": hours_to_close,
        "has_market_prior": 1.0 if (
            prior_estimate.prior_source.startswith("market")
            or prior_estimate.prior_source.startswith("kalshi_public")
            or "event_field" in prior_estimate.prior_source
        ) else 0.0,
        "market_prior_distance_from_even": prior_estimate.probability - 0.50,
        "threshold_value": threshold_value if threshold_value is not None else 0.0,
        "current_spot_price": current_spot if current_spot is not None else 0.0,
        "distance_to_threshold": distance_to_threshold,
        "abs_distance_to_threshold": abs(distance_to_threshold),
        "time_pressure": _time_pressure(days_to_close),
        "reasoning_adjustment": _as_float(evidence.get("reasoning_adjustment")) or 0.0,
        "evidence_strength_score": _label_to_score(str(evidence.get("strength") or "")),
        "evidence_freshness_score": _freshness_to_score(str(evidence.get("freshness") or "")),
        "already_priced_score": _already_priced_to_score(str(evidence.get("already_priced_likelihood") or "")),
        "credibility_score": _label_to_score(str(evidence.get("credibility") or "")),
        "market_microstructure_available": _market_microstructure_available(market_snapshot),
        "recent_price_change_5m": _as_float(market_snapshot.get("recent_price_change_5m")) or 0.0,
        "recent_price_change_1h": _as_float(market_snapshot.get("recent_price_change_1h")) or 0.0,
        "recent_price_change_24h": _as_float(market_snapshot.get("recent_price_change_24h")) or 0.0,
        "volume_zscore": _as_float(market_snapshot.get("volume_zscore")) or 0.0,
        "order_book_imbalance": _as_float(market_snapshot.get("order_book_imbalance")) or 0.0,
        "spread_change": _as_float(market_snapshot.get("spread_change")) or 0.0,
        "liquidity_depth": _as_float(market_snapshot.get("liquidity_depth")) or 0.0,
        "large_trade_flag": 1.0 if bool(market_snapshot.get("large_trade_flag")) else 0.0,
        "cross_market_confirmation": _as_float(market_snapshot.get("cross_market_confirmation")) or 0.0,
        "news_lag_flag": 1.0 if bool(market_snapshot.get("news_lag_flag")) else 0.0,
        "consensus_forecast": _first_probability_like_value(market_snapshot.get("consensus_forecast")),
        "sportsbook_implied_prob": _first_probability_like_value(market_snapshot.get("sportsbook_implied_prob")),
        "official_forecast_probability": _first_probability_like_value(market_snapshot.get("official_forecast_probability")),
        "forecast_confidence_score": _label_to_score(str(market_snapshot.get("forecast_confidence") or "")),
        "polling_signal": _first_probability_like_value(market_snapshot.get("polling_signal")),
        "bookmaker_implied_prob": _first_probability_like_value(market_snapshot.get("bookmaker_implied_prob")),
        "is_price_threshold": 1.0 if route.template_name == PRICE_THRESHOLD else 0.0,
        "is_macro_release": 1.0 if route.template_name == MACRO_RELEASE else 0.0,
        "is_sports": 1.0 if route.template_name == SPORTS else 0.0,
        "is_weather": 1.0 if route.template_name == WEATHER else 0.0,
        "is_politics_policy": 1.0 if route.template_name == POLITICS_ELECTIONS_POLICY else 0.0,
        "is_company_tech": 1.0 if route.template_name == COMPANY_TECH_ANNOUNCEMENT else 0.0,
        "is_culture_awards": 1.0 if route.template_name == CULTURE_AWARDS_ENTERTAINMENT else 0.0,
        "is_generic_news": 1.0 if route.template_name == GENERIC_NEWS_UNIQUE else 0.0,
        "is_market_prior_baseline": 1.0 if route.template_name == "MARKET_PRIOR_BASELINE" else 0.0,
    }

    # Fill missing probability-like fields with neutral defaults.
    for key in ("consensus_forecast", "sportsbook_implied_prob", "official_forecast_probability", "polling_signal", "bookmaker_implied_prob"):
        if features[key] is None:
            features[key] = 0.0

    return features


def _days_to_close(close_time: datetime | None) -> float:
    if close_time is None:
        return -1.0
    now = datetime.now(UTC)
    return (close_time - now).total_seconds() / 86400.0


def _time_pressure(days_to_close: float) -> float:
    if days_to_close < 0:
        return 0.0
    if days_to_close <= 1.0:
        return 1.0
    if days_to_close <= 3.0:
        return 0.7
    if days_to_close <= 7.0:
        return 0.4
    return 0.1


def _label_to_score(label: str) -> float:
    normalized = label.strip().lower()
    if normalized == "strong":
        return 0.9
    if normalized == "medium":
        return 0.6
    if normalized == "weak":
        return 0.3
    if normalized == "high":
        return 0.9
    if normalized == "low":
        return 0.3
    return 0.0


def _freshness_to_score(label: str) -> float:
    normalized = label.strip().lower()
    if normalized == "breaking":
        return 1.0
    if normalized == "recent":
        return 0.7
    if normalized == "old":
        return 0.2
    return 0.0


def _already_priced_to_score(label: str) -> float:
    normalized = label.strip().lower()
    if normalized == "low":
        return 0.9
    if normalized == "medium":
        return 0.5
    if normalized == "high":
        return 0.1
    return 0.0


def _market_microstructure_available(snapshot: dict[str, Any]) -> float:
    keys = (
        "recent_price_change_5m",
        "recent_price_change_1h",
        "recent_price_change_24h",
        "volume_zscore",
        "large_trade_flag",
        "order_book_imbalance",
        "spread_change",
        "liquidity_depth",
    )
    hits = sum(1 for key in keys if snapshot.get(key) is not None)
    return hits / float(len(keys)) if hits else 0.0


def _first_probability_like_value(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        normalized = normalize_probability(value)
        if normalized is not None:
            return normalized
        numeric = _as_float(value)
        if numeric is not None:
            return numeric
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

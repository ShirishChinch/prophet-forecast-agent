"""Curated edge features for the LLM-assisted residual layer.

These are features the LLM is allowed to query and quantify. They are not
probability forecasts. Each feature is signed so positive values support YES,
and the residual model learns how much to trust it versus the market prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
import re
from typing import Any


@dataclass(frozen=True)
class EdgeFeatureSpec:
    """Operational definition for one queryable edge feature."""

    name: str
    template: str
    description: str
    preferred_sources: tuple[str, ...]
    measurement: str
    range_min: float = -1.0
    range_max: float = 1.0


CURATED_EDGE_FEATURES: tuple[EdgeFeatureSpec, ...] = (
    EdgeFeatureSpec(
        "sports__sportsbook_gap_edge",
        "sports",
        "Consensus sportsbook/odds price minus Kalshi prior, signed for the YES side.",
        ("sportsbook odds pages", "odds aggregators", "exchange odds"),
        "Normalize to [-1, 1] from public odds gap; 0 if unavailable.",
    ),
    EdgeFeatureSpec(
        "sports__injury_lineup_edge",
        "sports",
        "Key injury, scratch, lineup, goalie, pitcher, or minutes-news impact.",
        ("official injury reports", "team lineups", "beat reporters", "league game notes"),
        "Positive if news helps YES side; magnitude reflects player/team importance.",
    ),
    EdgeFeatureSpec(
        "sports__starting_pitcher_goalie_edge",
        "sports",
        "Starting pitcher, goalie, or equivalent high-leverage participant mismatch.",
        ("official probable starters", "team announcements", "league preview pages"),
        "Positive if starter matchup helps YES side; 0 if not relevant.",
    ),
    EdgeFeatureSpec(
        "sports__player_recent_form_edge",
        "sports",
        "Player prop recent-form gap versus the market threshold.",
        ("official box scores", "stat pages", "player logs"),
        "Recent rate minus threshold, scaled to [-1, 1].",
    ),
    EdgeFeatureSpec(
        "sports__team_recent_form_edge",
        "sports",
        "Team recent-form, rest, travel, or head-to-head edge.",
        ("league standings", "recent results", "schedule pages"),
        "Positive if recent form/rest setup helps YES side.",
    ),
    EdgeFeatureSpec(
        "sports__weather_venue_edge",
        "sports",
        "Weather/venue effect for totals, baseball, football, golf, or outdoor events.",
        ("weather forecast", "ballpark weather", "venue conditions"),
        "Positive if conditions help YES side.",
    ),
    EdgeFeatureSpec(
        "macro__consensus_gap_edge",
        "macro",
        "Latest public economist consensus relative to the event threshold/bucket.",
        ("Reuters polls", "Trading Economics", "Investing.com", "Econoday", "official calendars"),
        "Positive if consensus points to YES.",
    ),
    EdgeFeatureSpec(
        "macro__nowcast_gap_edge",
        "macro",
        "Nowcast gap from official/public nowcast models versus threshold.",
        ("Atlanta Fed GDPNow", "Cleveland Fed nowcast", "regional Fed nowcasts"),
        "Positive if nowcast points to YES.",
    ),
    EdgeFeatureSpec(
        "macro__rates_market_gap_edge",
        "macro",
        "Rates/OIS/FedWatch-implied path versus policy/rate market condition.",
        ("CME FedWatch", "public OIS commentary", "Treasury futures/rates pages"),
        "Positive if rates market supports YES.",
    ),
    EdgeFeatureSpec(
        "macro__expert_revision_momentum",
        "macro",
        "Direction of recent expert/consensus revisions.",
        ("economic calendar revisions", "poll update articles", "analyst notes summaries"),
        "Positive if revisions moved toward YES.",
    ),
    EdgeFeatureSpec(
        "crypto__spot_threshold_gap",
        "crypto_price",
        "Current spot distance to threshold, signed for YES.",
        ("Coinbase", "Binance", "CoinMarketCap", "CoinGecko"),
        "Spot minus threshold divided by threshold, clipped to [-1, 1].",
    ),
    EdgeFeatureSpec(
        "crypto__momentum_edge",
        "crypto_price",
        "Short-horizon crypto return/momentum toward the threshold.",
        ("price charts", "exchange data", "market data pages"),
        "Positive if recent move supports YES.",
    ),
    EdgeFeatureSpec(
        "crypto__flow_liquidation_edge",
        "crypto_price",
        "Funding, open-interest, liquidation, or ETF-flow signal.",
        ("Coinglass", "ETF flow trackers", "exchange market summaries"),
        "Positive if flows support YES.",
    ),
    EdgeFeatureSpec(
        "weather__official_forecast_gap_edge",
        "weather",
        "Official forecast relative to event threshold.",
        ("NWS", "NOAA", "National Weather Service", "official meteorological agency"),
        "Positive if official forecast supports YES.",
    ),
    EdgeFeatureSpec(
        "weather__model_confidence_edge",
        "weather",
        "Ensemble/model agreement and forecast confidence.",
        ("ensemble forecasts", "forecast discussions", "weather model pages"),
        "Positive if confidence supports YES, negative if uncertainty favors NO.",
    ),
    EdgeFeatureSpec(
        "politics__poll_consensus_edge",
        "politics",
        "Polling, approval, or public political forecast consensus gap.",
        ("poll aggregators", "official election pages", "reputable election forecasters"),
        "Positive if public polling/forecasting supports YES.",
    ),
    EdgeFeatureSpec(
        "politics__institutional_constraint_edge",
        "politics",
        "Procedural, legal, vote-count, or institutional constraint.",
        ("official calendars", "court dockets", "legislative pages", "reputable reporting"),
        "Positive if constraints make YES more likely.",
    ),
    EdgeFeatureSpec(
        "company__official_signal_edge",
        "company",
        "Official company/filing/blog/product-cycle signal.",
        ("company blogs", "SEC filings", "official social accounts", "event pages"),
        "Positive if official-source evidence supports YES.",
    ),
    EdgeFeatureSpec(
        "culture__expert_consensus_edge",
        "culture",
        "Expert/bookmaker/critic consensus for awards, entertainment, or culture events.",
        ("bookmaker odds", "expert prediction pages", "critic aggregators"),
        "Positive if public consensus supports YES.",
    ),
    EdgeFeatureSpec(
        "generic__official_confirmation_edge",
        "generic",
        "Official confirmation or denial signal.",
        ("official pages", "press releases", "filings", "verified public statements"),
        "Positive if official evidence supports YES.",
    ),
    EdgeFeatureSpec(
        "generic__credible_news_consensus_edge",
        "generic",
        "Reputable news/source consensus for one-off events.",
        ("wire services", "major outlets", "domain-specific reputable sources"),
        "Positive if credible news consensus supports YES.",
    ),
)


MARKET_RULE_FEATURES = (
    "edge__prior",
    "edge__prior_extreme",
    "edge__prior_low",
    "edge__prior_high",
    "edge__market_spread",
    "edge__market_volume_log",
    "edge__hours_to_finish",
    "edge__short_horizon",
    "edge__wide_spread_short_horizon",
    "edge__many_legs",
    "edge__yes_leg_share",
    "edge__no_leg_share",
    "edge__weird_low_prior_active",
    "edge__weird_high_prior_active",
    "edge__recent_price_change_1h",
    "edge__recent_price_change_24h",
    "edge__volume_zscore",
    "edge__large_trade_flag",
    "edge__order_book_imbalance",
)


def curated_feature_names(template: str | None = None) -> list[str]:
    """Return curated LLM feature names, optionally filtered by template."""
    normalized = _normalize_template(template)
    return [
        spec.name
        for spec in CURATED_EDGE_FEATURES
        if normalized is None or spec.template == normalized
    ]


def all_runtime_feature_names() -> list[str]:
    """Return deterministic market-rule plus curated LLM feature names."""
    return list(MARKET_RULE_FEATURES) + curated_feature_names(None)


def default_llm_feature_row() -> dict[str, float]:
    """Return all curated LLM features at neutral values."""
    return {name: 0.0 for name in curated_feature_names(None)}


def build_market_rule_features(
    event: dict[str, Any],
    prior: float,
    *,
    now: datetime | None = None,
) -> dict[str, float]:
    """Build deterministic public market/title features used by the edge model."""
    title = str(event.get("title") or event.get("yes_sub_title") or "")
    spread = _spread(event)
    volume = _first_float(event.get("volume_fp"), event.get("volume"), event.get("volume_24h_fp"), 0.0)
    hours_to_finish = _hours_to_finish(event, now=now)
    leg_count = _leg_count(title)
    yes_count = len(re.findall(r"\byes\b", title.lower()))
    no_count = len(re.findall(r"\bno\b", title.lower()))
    side_count = max(1, yes_count + no_count)
    prior_value = _clip(float(prior), 0.01, 0.99)

    return {
        "edge__prior": prior_value,
        "edge__prior_extreme": abs(prior_value - 0.5) * 2.0,
        "edge__prior_low": 1.0 if prior_value <= 0.05 else 0.0,
        "edge__prior_high": 1.0 if prior_value >= 0.95 else 0.0,
        "edge__market_spread": spread,
        "edge__market_volume_log": math.log1p(max(0.0, volume)),
        "edge__hours_to_finish": hours_to_finish,
        "edge__short_horizon": 1.0 if 0.0 <= hours_to_finish <= 4.0 else 0.0,
        "edge__wide_spread_short_horizon": spread * (1.0 if 0.0 <= hours_to_finish <= 4.0 else 0.0),
        "edge__many_legs": float(min(1.0, max(0, leg_count - 1) / 10.0)),
        "edge__yes_leg_share": yes_count / side_count,
        "edge__no_leg_share": no_count / side_count,
        "edge__weird_low_prior_active": 1.0 if prior_value <= 0.05 and spread <= 0.05 else 0.0,
        "edge__weird_high_prior_active": 1.0 if prior_value >= 0.95 and spread <= 0.05 else 0.0,
        "edge__recent_price_change_1h": _clip(_first_float(event.get("recent_price_change_1h"), 0.0), -1.0, 1.0),
        "edge__recent_price_change_24h": _clip(_first_float(event.get("recent_price_change_24h"), 0.0), -1.0, 1.0),
        "edge__volume_zscore": _clip(_first_float(event.get("volume_zscore"), 0.0) / 10.0, -1.0, 1.0),
        "edge__large_trade_flag": 1.0 if bool(event.get("large_trade_flag")) else 0.0,
        "edge__order_book_imbalance": _clip(_first_float(event.get("order_book_imbalance"), 0.0), -1.0, 1.0),
    }


def normalize_llm_feature_payload(payload: dict[str, Any], template: str | None = None) -> dict[str, float]:
    """Map an LLM JSON payload into the curated numeric schema."""
    features = default_llm_feature_row()
    allowed = set(curated_feature_names(template)) if template else set(features)
    raw_features = payload.get("features")
    if isinstance(raw_features, dict):
        items = raw_features.items()
    elif isinstance(raw_features, list):
        items = ((item.get("name"), item.get("value")) for item in raw_features if isinstance(item, dict))
    else:
        items = ()

    for name, value in items:
        key = str(name or "").strip()
        if key not in features or (allowed and key not in allowed):
            continue
        features[key] = _clip(_first_float(value, 0.0), -1.0, 1.0)
    return features


def _normalize_template(template: str | None) -> str | None:
    if template is None:
        return None
    value = template.lower()
    mapping = {
        "price_threshold": "crypto_price",
        "macro_release": "macro",
        "sports": "sports",
        "weather": "weather",
        "politics_elections_policy": "politics",
        "company_tech_announcement": "company",
        "culture_awards_entertainment": "culture",
        "generic_news_unique": "generic",
        "market_prior_baseline": "generic",
    }
    return mapping.get(value, value)


def _spread(event: dict[str, Any]) -> float:
    direct = _maybe_float(event.get("spread"))
    if direct is not None:
        return _clip(direct, 0.0, 1.0)
    bid = _maybe_float(event.get("best_bid") or event.get("yes_bid_dollars") or event.get("yes_bid"))
    ask = _maybe_float(event.get("best_ask") or event.get("yes_ask_dollars") or event.get("yes_ask"))
    if bid is None or ask is None:
        return 1.0
    if bid > 1.0:
        bid /= 100.0
    if ask > 1.0:
        ask /= 100.0
    return _clip(abs(ask - bid), 0.0, 1.0)


def _hours_to_finish(event: dict[str, Any], *, now: datetime | None) -> float:
    direct = _maybe_float(event.get("hours_to_finish"))
    if direct is not None:
        return direct
    time_value = event.get("finish_time") or event.get("close_time") or event.get("expiration_time") or event.get("expected_expiration_time")
    if not time_value:
        return -1.0
    try:
        text = str(time_value).replace("Z", "+00:00")
        finish = datetime.fromisoformat(text)
        if finish.tzinfo is None:
            finish = finish.replace(tzinfo=UTC)
    except ValueError:
        return -1.0
    base = now or datetime.now(UTC)
    return (finish - base).total_seconds() / 3600.0


def _leg_count(title: str) -> int:
    return max(1, str(title).count(",") + 1)


def _first_float(*values: Any) -> float:
    for value in values:
        number = _maybe_float(value)
        if number is not None:
            return number
    return 0.0


def _maybe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

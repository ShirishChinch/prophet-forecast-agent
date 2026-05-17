"""Prior probability extraction for forecast events."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

from agents.kalshi_public import extract_bid_ask, get_public_market_for_event
from agents.market_calibration import calibrate_market_probability
from agents.sector_data.lookup_runtime import maybe_apply_sector_lookup
from agents.tennis_data.lookup_runtime import maybe_apply_tennis_lookup
from agents.templates import TemplateRoute

MARKET_PRIOR_FIELDS = (
    "market_prob",
    "implied_prob",
    "midpoint",
    "price",
    "yes_price",
    "market_price",
)


@dataclass(frozen=True)
class PriorEstimate:
    """Probability prior plus provenance."""

    probability: float
    prior_source: str
    raw_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


def clamp_probability(value: Any) -> float:
    """Clamp a probability to the Prophet Arena valid range."""
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return 0.50
    if not math.isfinite(prob):
        return 0.50
    return max(0.01, min(0.99, prob))


def normalize_probability(value: Any) -> float | None:
    """Normalize a raw probability or market price into [0, 1]."""
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(prob):
        return None
    if prob > 1.0:
        prob = prob / 100.0
    if prob < 0.0 or prob > 1.0:
        return None
    return prob


def estimate_prior(
    event: dict[str, Any],
    spec: dict[str, Any],
    route: TemplateRoute,
) -> PriorEstimate:
    """Estimate the event prior, preferring the market when available."""
    best_bid = normalize_probability(event.get("best_bid"))
    best_ask = normalize_probability(event.get("best_ask"))
    if best_bid is not None and best_ask is not None:
        wide_spread_prior = _wide_spread_fallback(event, best_bid, best_ask)
        if wide_spread_prior is not None:
            return wide_spread_prior
        midpoint = (best_bid + best_ask) / 2.0
        return _with_market_calibration(
            PriorEstimate(
                probability=clamp_probability(midpoint),
                prior_source="market_bid_ask_midpoint",
                raw_value=midpoint,
                details={"best_bid": best_bid, "best_ask": best_ask},
            ),
            route,
            event,
        )

    for field_name in MARKET_PRIOR_FIELDS:
        prob = normalize_probability(event.get(field_name))
        if prob is not None:
            return _with_market_calibration(
                PriorEstimate(
                    probability=clamp_probability(prob),
                    prior_source=f"event_field:{field_name}",
                    raw_value=prob,
                    details={"field_name": field_name},
                ),
                route,
                event,
            )

    kalshi_market = get_public_market_for_event(event)
    if kalshi_market is not None:
        public_bid, public_ask = extract_bid_ask(kalshi_market)
        if public_bid is not None and public_ask is not None:
            midpoint = (public_bid + public_ask) / 2.0
            return _with_market_calibration(
                PriorEstimate(
                    probability=clamp_probability(midpoint),
                    prior_source="kalshi_public_bid_ask_midpoint",
                    raw_value=midpoint,
                    details={
                        "ticker": kalshi_market.get("ticker"),
                        "event_ticker": kalshi_market.get("event_ticker"),
                        "yes_bid": public_bid,
                        "yes_ask": public_ask,
                        "status": kalshi_market.get("status"),
                        "title": kalshi_market.get("title"),
                        "rules_primary": kalshi_market.get("rules_primary"),
                    },
                ),
                route,
                event,
            )

    category = str(spec.get("category") or event.get("category") or "").strip().lower()
    if "sport" in category:
        base_rate = 0.50
        source = "category_base_rate:sports"
    elif category in {"economics", "financials", "politics"}:
        base_rate = 0.50
        source = f"category_base_rate:{category or 'generic'}"
    else:
        base_rate = 0.50
        source = "default_0.50"

    return PriorEstimate(
        probability=clamp_probability(base_rate),
        prior_source=source,
        raw_value=base_rate,
        details={},
    )


def _wide_spread_fallback(event: dict[str, Any], bid: float, ask: float) -> PriorEstimate | None:
    """Avoid treating a 0/100 market as a real 50% prior.

    Kalshi multileg combo markets often show `yes_bid=0`, `yes_ask=1`.
    The midpoint is mechanically 0.50 but carries almost no information.
    Prefer last traded price if present; otherwise use a conservative
    parlay-style prior based on the number of visible legs.
    """
    if ask - bid < 0.95:
        return None

    for field_name in ("last_price_dollars", "last_price", "last_trade_price", "previous_price"):
        last = normalize_probability(event.get(field_name))
        if last is not None and last > 0.0:
            return PriorEstimate(
                probability=clamp_probability(last),
                prior_source=f"wide_spread_last_price:{field_name}",
                raw_value=last,
                details={"best_bid": bid, "best_ask": ask, "spread": ask - bid},
            )

    title = str(event.get("title") or event.get("yes_sub_title") or "")
    leg_count = _visible_leg_count(title)
    if leg_count >= 3:
        parlay_prior = 0.65 ** leg_count
        return PriorEstimate(
            probability=clamp_probability(parlay_prior),
            prior_source="wide_spread_multileg_parlay_heuristic",
            raw_value=parlay_prior,
            details={
                "best_bid": bid,
                "best_ask": ask,
                "spread": ask - bid,
                "visible_leg_count": leg_count,
                "per_leg_assumption": 0.65,
            },
        )
    return None


def _visible_leg_count(title: str) -> int:
    text = str(title)
    yes_no_count = len(re.findall(r"\b(?:yes|no)\b", text.lower()))
    comma_count = text.count(",") + 1 if text.strip() else 0
    return max(yes_no_count, comma_count)


def _with_market_calibration(
    estimate: PriorEstimate,
    route: TemplateRoute,
    event: dict[str, Any],
) -> PriorEstimate:
    """Apply learned empirical odds calibration when an artifact is available."""
    tennis_lookup = maybe_apply_tennis_lookup(estimate.probability, event)
    if tennis_lookup.applied:
        details = dict(estimate.details)
        details["tennis_lookup"] = tennis_lookup.details
        details["uncalibrated_probability"] = estimate.probability
        details["uncalibrated_prior_source"] = estimate.prior_source
        return PriorEstimate(
            probability=clamp_probability(tennis_lookup.probability),
            prior_source=f"{estimate.prior_source}|{tennis_lookup.source}",
            raw_value=estimate.raw_value,
            details=details,
        )

    sector_lookup = maybe_apply_sector_lookup(
        estimate.probability,
        event,
        template_family=route.template_name,
    )
    if sector_lookup.applied:
        details = dict(estimate.details)
        details["sector_bucket_lookup"] = sector_lookup.details
        details["uncalibrated_probability"] = estimate.probability
        details["uncalibrated_prior_source"] = estimate.prior_source
        return PriorEstimate(
            probability=clamp_probability(sector_lookup.probability),
            prior_source=f"{estimate.prior_source}|{sector_lookup.source}",
            raw_value=estimate.raw_value,
            details=details,
        )

    calibration = calibrate_market_probability(
        estimate.probability,
        template_family=route.template_name,
        event=event,
    )
    if not calibration.applied:
        return estimate
    details = dict(estimate.details)
    details["market_prior_calibration"] = calibration.details
    details["uncalibrated_probability"] = estimate.probability
    details["uncalibrated_prior_source"] = estimate.prior_source
    return PriorEstimate(
        probability=clamp_probability(calibration.probability),
        prior_source=f"{estimate.prior_source}|{calibration.source}",
        raw_value=estimate.raw_value,
        details=details,
    )

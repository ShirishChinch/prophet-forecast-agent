"""Collect structured public features for economics events."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agents.data_sources.market_prices import collect_fast_market_features
from agents.economics.event_parser import EconomicEventSpec


def collect_economic_public_features(
    *,
    event: dict[str, Any],
    spec: dict[str, Any],
    event_spec: EconomicEventSpec,
    as_of: datetime | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Collect only fast public numeric features relevant to the event family."""
    _ = (event, spec)
    fast_features, fast_status = collect_fast_market_features(as_of=as_of)
    selected = _select_features_for_event(event_spec, fast_features)
    diagnostics = {
        "fast_market_status": fast_status,
        "selected_feature_count": len(selected),
    }
    return selected, diagnostics


def public_signal_adjustment(event_spec: EconomicEventSpec, features: dict[str, float]) -> tuple[float, float, str]:
    """Convert public fast features into a very small residual signal.

    This is not a trained model. It only gives directionally sensible tiny
    adjustments while we collect enough historical feature/label data.
    """
    if not features:
        return 0.0, 0.0, "No public fast features available."

    if event_spec.model_type == "yield":
        signal = (
            0.55 * _scaled(features.get("us_10y_yield_change_30d"), 0.50)
            + 0.30 * _scaled(features.get("us_5y_yield_change_30d"), 0.50)
            + 0.15 * _scaled(features.get("us_10y_5y_slope_change_30d"), 0.25)
        )
        return _bounded(signal, 0.015), 0.12, "Used public 10Y/5Y yield momentum and curve-slope changes."

    if event_spec.model_type == "inflation":
        signal = (
            0.35 * _scaled(features.get("wti_oil_change_30d"), 15.0)
            + 0.25 * _scaled(features.get("gasoline_change_30d"), 0.30)
            + 0.15 * _scaled(features.get("us_10y_yield_change_30d"), 0.50)
            - 0.25 * _scaled(features.get("dollar_index_change_30d"), 3.0)
        )
        return _bounded(signal, 0.012), 0.10, "Used public oil, gasoline, rates, and dollar-index momentum."

    if event_spec.model_type == "growth":
        signal = (
            0.35 * _scaled(features.get("sp500_change_30d"), 250.0)
            - 0.25 * _scaled(features.get("credit_spread_hy_change_30d"), 0.75)
            + 0.20 * _scaled(features.get("us_10y_yield_change_30d"), 0.50)
        )
        return _bounded(signal, 0.012), 0.09, "Used public equity, credit-spread, and rates momentum."

    if event_spec.model_type == "policy":
        signal = (
            0.45 * _scaled(features.get("us_5y_yield_change_30d"), 0.50)
            + 0.30 * _scaled(features.get("us_10y_yield_change_30d"), 0.50)
            + 0.15 * _scaled(features.get("us_10y_5y_slope_change_30d"), 0.25)
        )
        return _bounded(signal, 0.010), 0.08, "Used public rates and yield-curve momentum."

    return 0.0, 0.0, "No public signal rule for event type."


def _select_features_for_event(event_spec: EconomicEventSpec, features: dict[str, float]) -> dict[str, float]:
    common = {
        key: value
        for key, value in features.items()
        if key.startswith(("us_10y", "us_5y", "dollar_index"))
    }
    if event_spec.model_type == "inflation":
        return {
            **common,
            **{key: value for key, value in features.items() if key.startswith(("wti_oil", "gasoline"))},
        }
    if event_spec.model_type == "growth":
        return {
            **common,
            **{key: value for key, value in features.items() if key.startswith(("sp500", "credit_spread_hy"))},
        }
    if event_spec.model_type in {"yield", "policy"}:
        return common
    return common


def _scaled(value: float | None, scale: float) -> float:
    if value is None or scale <= 0:
        return 0.0
    return _bounded(float(value) / scale, 1.0)


def _bounded(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))

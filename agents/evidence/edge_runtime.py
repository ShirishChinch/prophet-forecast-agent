"""Runtime adapter for the trained LLM/market-rule edge residual model."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from agents.evidence.edge_feature_schema import build_market_rule_features, default_llm_feature_row
from agents.evidence.llm_edge_extractor import extract_llm_edge_features
from agents.kalshi_public import get_public_market_for_event
from agents.prior import clamp_probability


DEFAULT_ARTIFACT = Path("agents/evidence/artifacts/live_edge_residual.joblib")


def predict_edge_probability(
    *,
    event: dict[str, Any],
    template_name: str,
    prior: float,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
) -> tuple[float, float, str, dict[str, float]]:
    """Return `(p_model, confidence, explanation, features)`.

    The model predicts a capped residual around the market prior. If no
    deployable artifact exists, it returns the prior with zero confidence.
    """
    prior_value = clamp_probability(prior)
    artifact = _load_artifact(Path(artifact_path))
    if artifact is None:
        return prior_value, 0.0, "No trained LLM edge artifact; stayed at prior.", {}

    report = artifact.get("report") or {}
    if not bool(report.get("deployable")):
        return prior_value, 0.0, "LLM edge artifact is not deployable on held-out Brier.", {}

    enriched_event = _with_public_market_fields(event)
    features = build_market_rule_features(enriched_event, prior_value)
    llm_features, llm_payload = extract_llm_edge_features(enriched_event, template_name)
    features.update(llm_features)

    feature_columns = list(artifact.get("feature_columns") or [])
    if not feature_columns:
        return prior_value, 0.0, "LLM edge artifact has no selected features.", features

    row = {column: _finite(features.get(column, 0.0)) for column in feature_columns}
    model = artifact.get("model")
    if model is None:
        return prior_value, 0.0, "LLM edge artifact has no fitted model.", features

    max_delta = float(artifact.get("max_delta") or 0.04)
    raw_delta = float(model.predict(pd.DataFrame([row], columns=feature_columns))[0])
    delta = max(-max_delta, min(max_delta, raw_delta))
    p_model = clamp_probability(prior_value + delta)
    improvement = float(report.get("brier_improvement") or 0.0)
    confidence = 0.35 if improvement > 0 else 0.0
    if llm_payload.get("error"):
        confidence *= 0.65
    elif llm_payload.get("skipped"):
        confidence *= 0.75
    explanation = (
        f"Trained LLM/market-rule residual edge applied with capped delta {delta:+.3f}; "
        f"selected_features={len(feature_columns)}."
    )
    return p_model, confidence, explanation, features


@lru_cache(maxsize=4)
def _load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
    except Exception:
        return None
    return artifact if isinstance(artifact, dict) else None


def _with_public_market_fields(event: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(event)
    if any(key in enriched for key in ("spread", "volume_fp", "yes_bid_dollars", "yes_ask_dollars")):
        return enriched
    market = get_public_market_for_event(event)
    if not market:
        return enriched
    for key in (
        "ticker",
        "event_ticker",
        "title",
        "yes_sub_title",
        "no_sub_title",
        "yes_bid_dollars",
        "yes_ask_dollars",
        "last_price_dollars",
        "volume",
        "volume_24h",
        "open_interest",
        "liquidity",
        "close_time",
        "expiration_time",
        "expected_expiration_time",
    ):
        if key in market and enriched.get(key) is None:
            enriched[key] = market.get(key)
    bid = _probability(market.get("yes_bid_dollars") or market.get("yes_bid"))
    ask = _probability(market.get("yes_ask_dollars") or market.get("yes_ask"))
    if bid is not None and ask is not None:
        enriched["spread"] = abs(ask - bid)
    if enriched.get("volume_fp") is None:
        enriched["volume_fp"] = market.get("volume") or market.get("volume_24h") or 0.0
    return enriched


def _probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number > 1.0:
        number /= 100.0
    if number < 0.0 or number > 1.0:
        return None
    return number


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0

"""Load and apply trained economics ML artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from agents.economics.base_model import probability_from_distribution
from agents.economics.event_parser import EconomicEventSpec
from agents.prior import clamp_probability


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class ArtifactPrediction:
    """Prediction from a trained artifact."""

    p_model: float
    confidence: float
    data_quality: float
    explanation: str
    diagnostics: dict[str, Any]


def load_artifact(model_type: str) -> dict[str, Any] | None:
    """Load a trained artifact if present."""
    try:
        import joblib  # type: ignore
    except Exception:
        return None

    path = ARTIFACT_DIR / f"{model_type}_model.joblib"
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
    except Exception:
        return None
    return artifact if isinstance(artifact, dict) else None


def predict_with_artifact(
    *,
    artifact: dict[str, Any],
    event_spec: EconomicEventSpec,
    live_features: dict[str, float],
    prior: float,
) -> ArtifactPrediction | None:
    """Apply a trained artifact to live JPMaQS features."""
    feature_names = list(artifact.get("feature_names") or [])
    model = artifact.get("model")
    residual_sigma = _safe_float(artifact.get("residual_sigma")) or 0.25
    target_mode = str(artifact.get("target_mode") or "level")
    if not feature_names or model is None:
        return None

    row = []
    missing = 0
    for name in feature_names:
        value = live_features.get(name)
        if value is None or not _is_finite(value):
            missing += 1
            row.append(math.nan)
        else:
            row.append(float(value))

    coverage = 1.0 - (missing / max(1, len(feature_names)))
    if coverage < 0.20:
        return None

    try:
        import pandas as pd  # type: ignore

        frame = pd.DataFrame([row], columns=feature_names)
        pred_value = float(model.predict(frame)[0])
    except Exception:
        return None

    current_value = _current_value_for_event(
        artifact=artifact,
        event_spec=event_spec,
        live_features=live_features,
    )
    p_distribution = _probability_from_prediction(
        pred_value=pred_value,
        current_value=current_value,
        residual_sigma=residual_sigma,
        target_mode=target_mode,
        event_spec=event_spec,
        prior=prior,
    )
    if p_distribution is None:
        return None

    metrics = artifact.get("metrics") or {}
    train_score = _safe_float(metrics.get("test_r2")) or 0.0
    baseline_improvement = _safe_float(metrics.get("rmse_improvement_vs_baseline")) or 0.0
    if baseline_improvement <= 0.0:
        p_model = (0.98 * clamp_probability(prior)) + (0.02 * clamp_probability(p_distribution))
        confidence = min(0.04, 0.02 + 0.02 * coverage)
    else:
        artifact_weight = min(0.20, 0.04 + 0.60 * baseline_improvement)
        p_model = ((1.0 - artifact_weight) * clamp_probability(prior)) + (
            artifact_weight * clamp_probability(p_distribution)
        )
        confidence = min(0.30, max(0.05, 0.06 + 0.16 * coverage + 0.10 * max(0.0, train_score)))
    data_quality = min(0.80, coverage)

    return ArtifactPrediction(
        p_model=clamp_probability(p_model),
        confidence=confidence,
        data_quality=data_quality,
        explanation=(
            f"Used trained {artifact.get('model_type', 'economic')} JPMaQS artifact "
            f"with {coverage:.0%} live feature coverage."
        ),
        diagnostics={
            "artifact_version": artifact.get("version"),
            "model_type": artifact.get("model_type"),
            "target_mode": target_mode,
            "predicted_value": pred_value,
            "current_value": current_value,
            "residual_sigma": residual_sigma,
            "feature_coverage": coverage,
            "missing_features": missing,
            "n_features": len(feature_names),
            "metrics": metrics,
        },
    )


def _probability_from_prediction(
    *,
    pred_value: float,
    current_value: float | None,
    residual_sigma: float,
    target_mode: str,
    event_spec: EconomicEventSpec,
    prior: float,
) -> float | None:
    if event_spec.condition == "class":
        outcome = (event_spec.yes_outcome or "").lower()
        if target_mode == "policy_delta":
            if "hike" in outcome:
                return clamp_probability(0.50 + pred_value)
            if "cut" in outcome:
                return clamp_probability(0.50 - pred_value)
            if "maintain" in outcome or "hold" in outcome:
                return clamp_probability(0.60 - abs(pred_value))
        return clamp_probability(prior)

    mean = pred_value
    if target_mode == "delta_to_current":
        if current_value is None:
            return None
        mean = current_value + pred_value

    return probability_from_distribution(
        mean=mean,
        sigma=max(0.01, residual_sigma),
        condition=event_spec.condition,
        threshold=event_spec.threshold,
        bucket_width=event_spec.bucket_width,
    )


def _current_value_for_event(
    *,
    artifact: dict[str, Any],
    event_spec: EconomicEventSpec,
    live_features: dict[str, float],
) -> float | None:
    """Find the current target value matching this event family/country."""
    country = event_spec.country_code
    if not country:
        return None
    candidates = _target_candidates_for_event(country, event_spec)
    target_tickers = set(str(ticker) for ticker in artifact.get("target_tickers") or [])
    for ticker in candidates:
        if ticker not in target_tickers:
            continue
        value = live_features.get(f"jpmaqs__{ticker}__latest")
        if _is_finite(value):
            return float(value)
    for ticker in target_tickers:
        if not ticker.startswith(f"{country}_"):
            continue
        value = live_features.get(f"jpmaqs__{ticker}__latest")
        if _is_finite(value):
            return float(value)
    return None


def _target_candidates_for_event(country: str, event_spec: EconomicEventSpec) -> list[str]:
    if event_spec.model_type == "yield":
        return [f"{country}_GB10YYLD_NSA"]
    if event_spec.model_type == "inflation":
        if event_spec.variable == "core_cpi":
            return [f"{country}_CPIC_SA_P1M1ML12", f"{country}_CPIH_SA_P1M1ML12"]
        return [f"{country}_CPIH_SA_P1M1ML12", f"{country}_CPIC_SA_P1M1ML12"]
    if event_spec.model_type == "growth":
        return [f"{country}_RGDP_SA_P1Q1QL4", f"{country}_RGDP_SA_P1Q1QL1AR"]
    return []


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

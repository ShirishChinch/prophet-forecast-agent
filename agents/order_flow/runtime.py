"""Runtime adapter for a trained order-flow residual artifact."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from agents.order_flow.features import FEATURE_COLUMNS
from agents.prior import clamp_probability


DEFAULT_ARTIFACT = Path("agents/order_flow/artifacts/order_flow_residual.joblib")
DEFAULT_TEMPLATE_ARTIFACT_DIR = Path("agents/order_flow/artifacts/templates")


def predict_order_flow_probability(
    features: dict[str, float],
    prior: float,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    template_family: str | None = None,
    allow_global_fallback: bool = True,
) -> tuple[float, float, str]:
    """Return `(p_model, confidence, explanation)` from a trained residual model.

    If no trained artifact exists, confidence is zero and p_model is the prior.
    """
    artifact = _load_template_artifact(template_family)
    artifact_name = f"template:{template_family}" if artifact is not None and template_family else "global"
    if template_family and artifact is None and not allow_global_fallback:
        prior_value = clamp_probability(prior)
        return prior_value, 0.0, f"No positive order-flow artifact for template:{template_family}; stayed at prior."
    if artifact is None:
        artifact = _load_artifact(Path(artifact_path))
        artifact_name = "global"
    prior_value = clamp_probability(prior)
    if artifact is None:
        return prior_value, 0.0, "No trained order-flow residual artifact; stayed at prior."

    feature_columns = artifact.get("feature_columns") or FEATURE_COLUMNS
    row = {column: _finite(features.get(column, 0.0)) for column in feature_columns}
    model = artifact["model"]
    max_delta = float(artifact.get("max_delta") or 0.05)
    raw_delta = float(model.predict(pd.DataFrame([row], columns=feature_columns))[0])
    delta = max(-max_delta, min(max_delta, raw_delta))
    report = artifact.get("report") or {}
    improvement = float(report.get("brier_improvement") or 0.0)
    confidence = 0.40 if improvement > 0.0 else 0.05
    return (
        clamp_probability(prior_value + delta),
        confidence,
        f"Trained public order-flow residual ({artifact_name}) applied with capped delta {delta:+.3f}.",
    )


@lru_cache(maxsize=4)
def _load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
    except Exception:
        return None
    return artifact if isinstance(artifact, dict) and "model" in artifact else None


def _load_template_artifact(template_family: str | None) -> dict[str, Any] | None:
    if not template_family:
        return None
    safe_family = "".join(char for char in template_family if char.isalnum() or char in {"_", "-"})
    if not safe_family:
        return None
    artifact = _load_artifact(DEFAULT_TEMPLATE_ARTIFACT_DIR / f"{safe_family}.joblib")
    if artifact is None:
        return None
    report = artifact.get("report") or {}
    if float(report.get("brier_improvement") or 0.0) <= 0.0:
        return None
    return artifact


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0

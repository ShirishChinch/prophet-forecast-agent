"""Small feature selection utilities for high-dimensional JPMaQS panels."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SelectionResult:
    """Selected features plus diagnostics."""

    features: dict[str, float]
    selected_names: list[str]
    diagnostics: dict[str, float]


def select_runtime_features(
    features: dict[str, float],
    *,
    max_features: int,
    prefer_fast_keywords: tuple[str, ...] = (),
) -> SelectionResult:
    """Select a compact runtime feature set without seeing labels.

    Training-time selection should be rolling-fold and label-aware. This helper
    is for live inference, when we only have the latest JPMaQS snapshot and need
    to cap dimensionality before passing features to a small model/artifact.
    """
    scored: list[tuple[float, str, float]] = []
    for name, value in features.items():
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            continue
        score = 1.0
        lowered = name.lower()
        if any(keyword in lowered for keyword in prefer_fast_keywords):
            score += 0.25
        score += min(0.50, abs(float(value)) / 10.0)
        scored.append((score, name, float(value)))

    scored.sort(reverse=True)
    selected = scored[:max_features]
    compact = {name: value for _, name, value in selected}
    return SelectionResult(
        features=compact,
        selected_names=[name for _, name, _ in selected],
        diagnostics={
            "input_feature_count": float(len(features)),
            "selected_feature_count": float(len(compact)),
        },
    )


def stable_feature_policy() -> dict[str, object]:
    """Document the intended train-time selection policy."""
    return {
        "manual_filter": "model_type + country/region + feature family whitelist",
        "availability_filter": "point-in-time only; drop stale or future-released values",
        "sparsity_filter": "drop features with poor coverage in each training fold",
        "ranking": "train-fold only: IC/correlation, elastic net, or permutation importance",
        "stability": "keep features repeatedly selected with stable sign across folds",
        "max_features": {
            "inflation": 120,
            "growth": 120,
            "yield": 80,
            "policy": 80,
        },
    }

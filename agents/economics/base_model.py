"""Base classes and probability helpers for economics models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any

from agents.economics.event_parser import EconomicEventSpec
from agents.prior import clamp_probability


@dataclass(frozen=True)
class EconomicModelOutput:
    """Output from an economics-specific model."""

    p_model: float
    confidence: float
    explanation: str
    data_quality: float
    model_name: str
    diagnostics: dict[str, Any]


class BaseEconomicModel(ABC):
    """Common interface for all economic models."""

    model_name = "base_economic_model"
    max_features = 100

    @abstractmethod
    def predict(
        self,
        *,
        event_spec: EconomicEventSpec,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> EconomicModelOutput:
        """Return a probability estimate for the event's YES condition."""

    def no_edge(self, prior: float, reason: str, diagnostics: dict[str, Any] | None = None) -> EconomicModelOutput:
        return EconomicModelOutput(
            p_model=clamp_probability(prior),
            confidence=0.05,
            explanation=reason,
            data_quality=0.0,
            model_name=self.model_name,
            diagnostics=diagnostics or {},
        )


def normal_cdf(value: float) -> float:
    """Standard normal CDF using only the standard library."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def probability_from_distribution(
    *,
    mean: float,
    sigma: float,
    condition: str,
    threshold: float | None,
    bucket_width: float | None,
) -> float | None:
    """Convert a point/distribution forecast into a binary probability."""
    if threshold is None or sigma <= 0 or not math.isfinite(mean) or not math.isfinite(sigma):
        return None
    if condition == "above":
        return clamp_probability(1.0 - normal_cdf((threshold - mean) / sigma))
    if condition == "below":
        return clamp_probability(normal_cdf((threshold - mean) / sigma))
    if condition == "exactly":
        width = bucket_width or 0.1
        lower = threshold - (width / 2.0)
        upper = threshold + (width / 2.0)
        prob = normal_cdf((upper - mean) / sigma) - normal_cdf((lower - mean) / sigma)
        return clamp_probability(prob)
    return None


def weak_signal_from_features(features: dict[str, float], max_abs_move: float) -> tuple[float, float]:
    """Produce a tiny standardized signal from selected features.

    This is intentionally weak until trained artifacts exist.
    """
    if not features:
        return 0.0, 0.0
    values = [value for value in features.values() if isinstance(value, int | float) and math.isfinite(float(value))]
    if not values:
        return 0.0, 0.0
    clipped = [max(-3.0, min(3.0, float(value))) for value in values[:50]]
    signal = sum(clipped) / max(1.0, len(clipped) * 3.0)
    return max(-max_abs_move, min(max_abs_move, signal * max_abs_move)), min(0.45, len(clipped) / 100.0)

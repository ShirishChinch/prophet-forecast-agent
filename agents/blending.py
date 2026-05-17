"""Conservative blending of model outputs with the market prior."""

from __future__ import annotations

from dataclasses import dataclass
import math

from agents.prior import clamp_probability
from agents.template_models import ModelOutput


@dataclass(frozen=True)
class BlendResult:
    """Final blend result plus diagnostics."""

    probability: float
    weight_on_models: float
    data_quality: float
    diagnostics: dict[str, float]


def blend(
    prior: float,
    model_outputs: list[ModelOutput],
    template_confidence: float,
    data_quality_hint: float = 0.0,
) -> BlendResult:
    """Blend model outputs with the prior, defaulting to heavy prior weight."""
    prior_value = clamp_probability(prior)
    candidate_outputs = [
        output for output in model_outputs
        if output.model_name != "baseline_market_prior"
        and math.isfinite(output.p_model)
    ]
    conviction_outputs = [
        output for output in candidate_outputs
        if output.model_name == "llm_conviction_nudge"
        and output.confidence > 0.0
        and output.data_quality > 0.0
    ]
    ordinary_outputs = [
        output for output in candidate_outputs
        if output.model_name != "llm_conviction_nudge"
    ]
    if not candidate_outputs:
        return BlendResult(
            probability=prior_value,
            weight_on_models=0.0,
            data_quality=0.0,
            diagnostics={"model_mean": prior_value, "model_support": 0.0},
        )

    weights: list[float] = []
    for output in ordinary_outputs:
        weight = max(0.0, output.confidence) * max(0.0, output.data_quality)
        weights.append(weight)

    total_weight = sum(weights)
    if total_weight <= 0.0:
        conviction_delta = _conviction_delta(prior_value, conviction_outputs)
        if conviction_delta != 0.0:
            return BlendResult(
                probability=clamp_probability(prior_value + conviction_delta),
                weight_on_models=1.0,
                data_quality=max(output.data_quality for output in conviction_outputs),
                diagnostics={
                    "model_mean": prior_value,
                    "model_support": 0.0,
                    "llm_conviction_delta": conviction_delta,
                },
            )
        return BlendResult(
            probability=prior_value,
            weight_on_models=0.0,
            data_quality=0.0,
            diagnostics={"model_mean": prior_value, "model_support": 0.0},
        )

    model_mean = sum(output.p_model * weight for output, weight in zip(ordinary_outputs, weights, strict=True)) / total_weight
    avg_confidence = sum(output.confidence for output in ordinary_outputs) / len(ordinary_outputs)
    avg_data_quality = sum(output.data_quality for output in ordinary_outputs) / len(ordinary_outputs)
    model_support = min(1.0, total_weight / max(1.0, len(ordinary_outputs) * 0.45))
    effective_quality = max(data_quality_hint, avg_data_quality)

    base_weight = 0.05 + (0.30 * template_confidence * avg_confidence * effective_quality * model_support)
    weight_on_models = min(0.35, max(0.0, base_weight))

    if abs(model_mean - prior_value) > 0.20:
        weight_on_models *= 0.55
    elif abs(model_mean - prior_value) > 0.10:
        weight_on_models *= 0.80

    blended = ((1.0 - weight_on_models) * prior_value) + (weight_on_models * model_mean)
    conviction_delta = _conviction_delta(prior_value, conviction_outputs)
    if conviction_delta != 0.0:
        blended = clamp_probability(blended + conviction_delta)
        effective_quality = max(effective_quality, max(output.data_quality for output in conviction_outputs))

    return BlendResult(
        probability=clamp_probability(blended),
        weight_on_models=weight_on_models,
        data_quality=effective_quality,
        diagnostics={
            "model_mean": model_mean,
            "avg_confidence": avg_confidence,
            "avg_data_quality": avg_data_quality,
            "model_support": model_support,
            "llm_conviction_delta": conviction_delta,
        },
    )


def _conviction_delta(prior: float, outputs: list[ModelOutput]) -> float:
    """Apply high-conviction LLM checks as explicit residuals."""
    delta = 0.0
    for output in outputs:
        delta += output.p_model - prior
    return max(-0.15, min(0.15, delta))

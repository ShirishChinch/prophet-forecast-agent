"""Template-specific heuristic models with a common interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agents.prior import clamp_probability
from agents.templates import (
    COMPANY_TECH_ANNOUNCEMENT,
    CULTURE_AWARDS_ENTERTAINMENT,
    GENERIC_NEWS_UNIQUE,
    INFORMED_FLOW_PUBLIC_SIGNAL,
    MACRO_RELEASE,
    MARKET_PRIOR_BASELINE,
    POLITICS_ELECTIONS_POLICY,
    PRICE_THRESHOLD,
    SPORTS,
    WEATHER,
)

try:
    from agents.economics.model_router import EconomicModelRouter
except Exception:  # pragma: no cover - economics layer must never break base agent import
    EconomicModelRouter = None  # type: ignore

try:
    from agents.order_flow.features import classify_market_template
    from agents.order_flow.runtime import predict_order_flow_probability
except Exception:  # pragma: no cover - order-flow layer must never break base agent import
    classify_market_template = None  # type: ignore
    predict_order_flow_probability = None  # type: ignore

try:
    from agents.order_flow.llm_context import SuspiciousMove, extract_context_features, should_review_with_llm
except Exception:  # pragma: no cover - LLM context layer is optional
    SuspiciousMove = None  # type: ignore
    extract_context_features = None  # type: ignore
    should_review_with_llm = None  # type: ignore

try:
    from agents.evidence.edge_runtime import predict_edge_probability
except Exception:  # pragma: no cover - evidence edge layer is optional
    predict_edge_probability = None  # type: ignore

try:
    from agents.llm_conviction_nudge import evaluate_factual_resolution_override, evaluate_llm_conviction_nudge
except Exception:  # pragma: no cover - LLM nudge layer is optional
    evaluate_factual_resolution_override = None  # type: ignore
    evaluate_llm_conviction_nudge = None  # type: ignore


@dataclass(frozen=True)
class ModelOutput:
    """Common output for all template models."""

    p_model: float
    confidence: float
    explanation: str
    data_quality: float
    model_name: str


class BaseTemplateModel(ABC):
    """Common interface for all placeholder models."""

    model_name: str = "base_model"

    @abstractmethod
    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        """Return a conservative probability estimate."""


class BaselineMarketPriorModel(BaseTemplateModel):
    model_name = "baseline_market_prior"

    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        prior_source = str(context.get("prior_source", ""))
        has_market_prior = 1.0 if (
            prior_source.startswith("market")
            or prior_source.startswith("kalshi_public")
            or "event_field" in prior_source
        ) else 0.0
        confidence = 0.95 if has_market_prior else 0.40
        data_quality = 0.95 if has_market_prior else 0.35
        return ModelOutput(
            p_model=clamp_probability(prior),
            confidence=confidence,
            explanation="Use the market-implied prior as the baseline anchor.",
            data_quality=data_quality,
            model_name=self.model_name,
        )


class PriceThresholdModel(BaseTemplateModel):
    model_name = "price_threshold_heuristic"

    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        _ = context
        p_model = prior
        confidence = 0.28
        data_quality = 0.20
        threshold = features.get("threshold_value", 0.0)
        current_spot = features.get("current_spot_price", 0.0)
        days_to_close = features.get("days_to_close", -1.0)
        if threshold > 0 and current_spot > 0:
            normalized_gap = (current_spot - threshold) / max(abs(threshold), 1.0)
            move = max(-0.05, min(0.05, normalized_gap * 2.0))
            if 0.0 <= days_to_close < 2.0:
                move *= 0.5
            p_model = clamp_probability(prior + move)
            confidence = 0.48
            data_quality = 0.45
            explanation = "Used threshold gap and time-to-close for a small adjustment."
        else:
            explanation = "No spot/threshold inputs available; stay close to prior."
        return ModelOutput(
            p_model=p_model,
            confidence=confidence,
            explanation=explanation,
            data_quality=data_quality,
            model_name=self.model_name,
        )


class MacroReleaseModel(BaseTemplateModel):
    model_name = "macro_release_heuristic"

    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        _ = context
        p_model = prior
        confidence = 0.25
        data_quality = 0.20
        consensus = features.get("consensus_forecast", 0.0)
        days_to_close = features.get("days_to_close", -1.0)
        if consensus > 0:
            delta = max(-0.03, min(0.03, (consensus - prior) * 0.35))
            if 0.0 <= days_to_close < 2.0:
                delta *= 0.5
            p_model = clamp_probability(prior + delta)
            confidence = 0.42
            data_quality = 0.38
            explanation = "Consensus-like structured signal available; nudged prior slightly."
        else:
            explanation = "No macro structured inputs beyond prior; remain near market baseline."
        return ModelOutput(
            p_model=p_model,
            confidence=confidence,
            explanation=explanation,
            data_quality=data_quality,
            model_name=self.model_name,
        )


class SportsModel(BaseTemplateModel):
    model_name = "sports_heuristic"

    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        _ = context
        sportsbook = features.get("sportsbook_implied_prob", 0.0)
        if sportsbook > 0:
            p_model = clamp_probability((0.65 * prior) + (0.35 * sportsbook))
            confidence = 0.50
            data_quality = 0.45
            explanation = "Used sportsbook-like implied probability conservatively."
        else:
            p_model = clamp_probability(prior)
            confidence = 0.30
            data_quality = 0.22
            explanation = "No sports-specific structured edge in v1; stay near prior."
        return ModelOutput(
            p_model=p_model,
            confidence=confidence,
            explanation=explanation,
            data_quality=data_quality,
            model_name=self.model_name,
        )


class WeatherModel(BaseTemplateModel):
    model_name = "weather_heuristic"

    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        _ = context
        official = features.get("official_forecast_probability", 0.0)
        forecast_confidence = features.get("forecast_confidence_score", 0.0)
        if official > 0:
            weight = 0.20 + (0.20 * forecast_confidence)
            p_model = clamp_probability(((1.0 - weight) * prior) + (weight * official))
            confidence = 0.50
            data_quality = 0.50
            explanation = "Official forecast proxy available; blended lightly with prior."
        else:
            p_model = clamp_probability(prior)
            confidence = 0.28
            data_quality = 0.20
            explanation = "No official weather forecast input available in v1."
        return ModelOutput(
            p_model=p_model,
            confidence=confidence,
            explanation=explanation,
            data_quality=data_quality,
            model_name=self.model_name,
        )


class AdjustmentOnlyModel(BaseTemplateModel):
    """Generic retrieval-heavy model that uses only structured evidence stubs."""

    model_name = "adjustment_only_model"
    adjustment_scale: float = 0.35
    explanation_text: str = "Used neutral structured evidence summary conservatively."

    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        _ = context
        adjustment = features.get("reasoning_adjustment", 0.0) * self.adjustment_scale
        adjustment *= (1.0 - features.get("already_priced_score", 0.0))
        p_model = clamp_probability(prior + adjustment)
        confidence = 0.25 + (0.20 * features.get("credibility_score", 0.0))
        data_quality = 0.20 + (0.20 * features.get("evidence_strength_score", 0.0))
        return ModelOutput(
            p_model=p_model,
            confidence=confidence,
            explanation=self.explanation_text,
            data_quality=data_quality,
            model_name=self.model_name,
        )


class PoliticsPolicyModel(AdjustmentOnlyModel):
    model_name = "politics_policy_reasoner"
    adjustment_scale = 0.40
    explanation_text = "Politics/policy template used only conservative reasoning adjustments in v1."


class CompanyTechAnnouncementModel(AdjustmentOnlyModel):
    model_name = "company_tech_reasoner"
    adjustment_scale = 0.35
    explanation_text = "Company/tech template used only conservative reasoning adjustments in v1."


class CultureAwardsEntertainmentModel(AdjustmentOnlyModel):
    model_name = "culture_awards_reasoner"
    adjustment_scale = 0.35
    explanation_text = "Culture/awards template used only conservative reasoning adjustments in v1."


class GenericNewsUniqueModel(AdjustmentOnlyModel):
    model_name = "generic_news_reasoner"
    adjustment_scale = 0.25
    explanation_text = "Generic template stayed close to the prior with minimal adjustment."


class InformedFlowPublicSignalModel(BaseTemplateModel):
    model_name = "public_flow_signal"

    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        _ = context
        availability = features.get("market_microstructure_available", 0.0)

        if predict_order_flow_probability is not None:
            runtime_features = _order_flow_runtime_features(features)
            runtime_features.update(_maybe_extract_llm_context_features(features, prior, context))
            template_family = _context_template_family(context, runtime_features)
            p_trained, trained_confidence, trained_explanation = predict_order_flow_probability(
                runtime_features,
                prior,
                template_family=template_family,
                allow_global_fallback=False,
            )
            if trained_confidence > 0.0:
                return ModelOutput(
                    p_model=p_trained,
                    confidence=trained_confidence,
                    explanation=trained_explanation,
                    data_quality=max(0.25, min(0.55, 0.25 + availability)),
                    model_name="trained_order_flow_residual",
                )

        if availability <= 0.0:
            return ModelOutput(
                p_model=clamp_probability(prior),
                confidence=0.0,
                explanation="No positive trained order-flow artifact or live microstructure inputs; stayed at prior.",
                data_quality=0.0,
                model_name=self.model_name,
            )

        flow_score = (
            0.30 * features.get("recent_price_change_1h", 0.0)
            + 0.20 * features.get("recent_price_change_24h", 0.0)
            + 0.20 * features.get("order_book_imbalance", 0.0)
            + 0.15 * features.get("volume_zscore", 0.0)
            + 0.10 * features.get("cross_market_confirmation", 0.0)
            - 0.10 * features.get("spread_change", 0.0)
        )
        if features.get("large_trade_flag", 0.0) > 0:
            flow_score += 0.01
        if features.get("news_lag_flag", 0.0) > 0:
            flow_score += 0.01
        move = max(-0.03, min(0.03, flow_score))
        p_model = clamp_probability(prior + move)
        return ModelOutput(
            p_model=p_model,
            confidence=0.35,
            explanation="Used public market-flow signals only for a very small adjustment.",
            data_quality=min(0.50, availability),
            model_name=self.model_name,
        )


def _order_flow_runtime_features(features: dict[str, float]) -> dict[str, float]:
    """Map live microstructure features into the residual artifact schema."""
    runtime_features = {
        "prior": features.get("prior", 0.50),
        "spread": abs(features.get("spread_change", 0.0)),
        "price_change_1": features.get("recent_price_change_1h", 0.0),
        "price_change_3": features.get("recent_price_change_1h", 0.0),
        "price_change_12": 0.5 * (
            features.get("recent_price_change_1h", 0.0)
            + features.get("recent_price_change_24h", 0.0)
        ),
        "price_change_24": features.get("recent_price_change_24h", 0.0),
        "realized_vol_24": abs(features.get("recent_price_change_24h", 0.0)),
        "volume": features.get("liquidity_depth", 0.0),
        "volume_zscore_24": features.get("volume_zscore", 0.0),
        "open_interest": 0.0,
        "open_interest_change_24": 0.0,
        "trade_count_24": features.get("market_microstructure_available", 0.0) * 24.0,
    }
    for key in (
        "llm_context_available",
        "llm_direction_score",
        "llm_evidence_strength",
        "llm_source_quality",
        "llm_freshness_score",
        "llm_cross_market_confirmation",
        "llm_stale_market_correction",
        "llm_near_resolution",
        "llm_already_priced_likelihood",
        "llm_temporal_leakage_risk",
    ):
        if key in features:
            runtime_features[key] = features[key]
    return runtime_features


def _maybe_extract_llm_context_features(
    features: dict[str, float],
    prior: float,
    context: dict[str, Any],
) -> dict[str, float]:
    """Optionally spend OpenAI credits to encode public context for weird flow."""
    import os

    if os.environ.get("ORDER_FLOW_LLM_ENABLED") != "1":
        return {}
    if SuspiciousMove is None or extract_context_features is None or should_review_with_llm is None:
        return {}

    runtime_features = _order_flow_runtime_features(features)
    if not should_review_with_llm(runtime_features):
        return {}

    event = context.get("event") or {}
    spec = context.get("spec") or {}
    title = str(spec.get("title") or event.get("title") or "")
    move = SuspiciousMove(
        market_ticker=str(event.get("market_ticker") or event.get("event_ticker") or ""),
        title=title,
        timestamp=datetime.now(UTC).isoformat(),
        prior=prior,
        price_change_1=runtime_features.get("price_change_1", 0.0),
        price_change_24=runtime_features.get("price_change_24", 0.0),
        volume_zscore_24=runtime_features.get("volume_zscore_24", 0.0),
        trade_size=runtime_features.get("volume", 0.0),
        trade_size_percentile=0.0,
        category=str(spec.get("category") or event.get("category") or ""),
    )
    context_features, _payload = extract_context_features(move)
    return context_features


def _context_template_family(context: dict[str, Any], features: dict[str, float]) -> str | None:
    if classify_market_template is None:
        return None
    event = context.get("event") or {}
    spec = context.get("spec") or {}
    title = str(spec.get("title") or event.get("title") or "")
    ticker = str(event.get("market_ticker") or event.get("event_ticker") or "")
    family = classify_market_template(title, ticker)
    prior = float(features.get("prior", 0.0) or 0.0)
    if family == "sports":
        if prior >= 0.95 or (0.0 < prior <= 0.05):
            return "sports"
        if features.get("trade_count_24", 0.0) >= 3.0:
            return "liquid_sports"
        return "sports"
    if family in {"crypto_price", "generic"}:
        return family
    return None


class JPMaQSEconomicModel(BaseTemplateModel):
    """Economics model adapter that uses compact JPMaQS feature policies."""

    model_name = "jpmaqs_economic_router"

    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        _ = features
        if EconomicModelRouter is None:
            return ModelOutput(
                p_model=clamp_probability(prior),
                confidence=0.05,
                explanation="JPMaQS economics layer unavailable; stayed at market prior.",
                data_quality=0.0,
                model_name=self.model_name,
            )

        try:
            router = EconomicModelRouter()
            output = router.predict(
                event=context.get("event") or {},
                spec=context.get("spec") or {},
                prior=prior,
                context=context,
            )
        except Exception as exc:
            return ModelOutput(
                p_model=clamp_probability(prior),
                confidence=0.05,
                explanation=f"JPMaQS economics model failed closed ({type(exc).__name__}); stayed at market prior.",
                data_quality=0.0,
                model_name=self.model_name,
            )

        return ModelOutput(
            p_model=output.p_model,
            confidence=output.confidence,
            explanation=output.explanation,
            data_quality=output.data_quality,
            model_name=output.model_name,
        )


class LLMEdgeResidualModel(BaseTemplateModel):
    """Trained residual model from curated LLM/queryable evidence features."""

    model_name = "llm_edge_residual"

    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        _ = features
        if predict_edge_probability is None:
            return ModelOutput(
                p_model=clamp_probability(prior),
                confidence=0.0,
                explanation="LLM edge residual layer unavailable; stayed at market prior.",
                data_quality=0.0,
                model_name=self.model_name,
            )
        try:
            p_model, confidence, explanation, edge_features = predict_edge_probability(
                event=context.get("event") or {},
                template_name=str((context.get("template_name") or "")),
                prior=prior,
            )
        except Exception as exc:
            return ModelOutput(
                p_model=clamp_probability(prior),
                confidence=0.0,
                explanation=f"LLM edge residual failed closed ({type(exc).__name__}); stayed at prior.",
                data_quality=0.0,
                model_name=self.model_name,
            )
        if confidence <= 0.0:
            data_quality = 0.0
        else:
            nonzero = sum(1 for value in edge_features.values() if abs(float(value or 0.0)) > 1e-12)
            data_quality = min(0.55, 0.25 + 0.03 * nonzero)
        return ModelOutput(
            p_model=p_model,
            confidence=confidence,
            explanation=explanation,
            data_quality=data_quality,
            model_name=self.model_name,
        )


class LLMConvictionNudgeModel(BaseTemplateModel):
    """Fixed +/-2 point LLM evidence nudge."""

    model_name = "llm_conviction_nudge"

    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        _ = features
        if evaluate_llm_conviction_nudge is None:
            return ModelOutput(
                p_model=clamp_probability(prior),
                confidence=0.0,
                explanation="LLM conviction nudge unavailable; stayed at prior.",
                data_quality=0.0,
                model_name=self.model_name,
            )
        try:
            result = evaluate_llm_conviction_nudge(
                event=context.get("event") or {},
                template_name=str(context.get("template_name") or ""),
                prior=prior,
            )
        except Exception as exc:
            return ModelOutput(
                p_model=clamp_probability(prior),
                confidence=0.0,
                explanation=f"LLM conviction nudge failed closed ({type(exc).__name__}); stayed at prior.",
                data_quality=0.0,
                model_name=self.model_name,
            )
        return ModelOutput(
            p_model=result.p_model,
            confidence=result.confidence if result.applied else 0.0,
            explanation=result.explanation,
            data_quality=result.data_quality if result.applied else 0.0,
            model_name=self.model_name,
        )


class FactualResolutionOverrideModel(BaseTemplateModel):
    """Near-certain override when a public fact already settles the event."""

    model_name = "llm_factual_resolution_override"

    def predict_probability(
        self,
        features: dict[str, float],
        prior: float,
        context: dict[str, Any],
    ) -> ModelOutput:
        _ = features
        if evaluate_factual_resolution_override is None:
            return ModelOutput(
                p_model=clamp_probability(prior),
                confidence=0.0,
                explanation="Factual override unavailable; stayed at prior.",
                data_quality=0.0,
                model_name=self.model_name,
            )
        try:
            result = evaluate_factual_resolution_override(
                event=context.get("event") or {},
                template_name=str(context.get("template_name") or ""),
                prior=prior,
            )
        except Exception as exc:
            return ModelOutput(
                p_model=clamp_probability(prior),
                confidence=0.0,
                explanation=f"Factual override failed closed ({type(exc).__name__}); stayed at prior.",
                data_quality=0.0,
                model_name=self.model_name,
            )
        return ModelOutput(
            p_model=result.p_model,
            confidence=result.confidence if result.applied else 0.0,
            explanation=result.explanation,
            data_quality=result.data_quality if result.applied else 0.0,
            model_name=self.model_name,
        )


def evaluate_template_models(
    template_name: str,
    features: dict[str, float],
    prior: float,
    context: dict[str, Any],
) -> list[ModelOutput]:
    """Build and run the model stack for the routed template."""
    models: list[BaseTemplateModel] = [BaselineMarketPriorModel()]
    context = dict(context)
    context["template_name"] = template_name

    template_model: BaseTemplateModel | None = None
    if template_name == PRICE_THRESHOLD:
        template_model = PriceThresholdModel()
    elif template_name == MACRO_RELEASE:
        template_model = None
    elif template_name == SPORTS:
        template_model = SportsModel()
    elif template_name == WEATHER:
        template_model = WeatherModel()
    elif template_name == POLITICS_ELECTIONS_POLICY:
        template_model = PoliticsPolicyModel()
    elif template_name == COMPANY_TECH_ANNOUNCEMENT:
        template_model = CompanyTechAnnouncementModel()
    elif template_name == CULTURE_AWARDS_ENTERTAINMENT:
        template_model = CultureAwardsEntertainmentModel()
    elif template_name == GENERIC_NEWS_UNIQUE:
        template_model = GenericNewsUniqueModel()
    elif template_name == MARKET_PRIOR_BASELINE:
        template_model = None

    if template_model is not None:
        models.append(template_model)

    if _should_run_trained_order_flow(template_name, context, features):
        models.append(InformedFlowPublicSignalModel())

    category = str((context.get("spec") or {}).get("category") or (context.get("event") or {}).get("category") or "").lower()
    if "economic" in category or template_name == MACRO_RELEASE:
        models.append(JPMaQSEconomicModel())

    models.append(LLMEdgeResidualModel())
    if _should_run_factual_resolution_override(template_name, context, prior):
        models.append(FactualResolutionOverrideModel())
    if _should_run_llm_conviction_nudge(template_name, context, prior):
        models.append(LLMConvictionNudgeModel())

    return [model.predict_probability(features, prior, context) for model in models]


def _should_run_trained_order_flow(
    template_name: str,
    context: dict[str, Any],
    features: dict[str, float],
) -> bool:
    if template_name == INFORMED_FLOW_PUBLIC_SIGNAL:
        return True
    family = _context_template_family(context, features)
    return family in {"crypto_price", "sports", "liquid_sports", "generic"}


def _should_run_llm_conviction_nudge(
    template_name: str,
    context: dict[str, Any],
    prior: float,
) -> bool:
    import os

    if os.environ.get("LLM_CONVICTION_NUDGE_ENABLED") == "0":
        return False
    if os.environ.get("LLM_CONVICTION_FORCE") == "1":
        return True
    if not os.environ.get("OPENAI_API_KEY"):
        return False

    event = context.get("event") or {}
    spec = context.get("spec") or {}
    text = " ".join(
        str(value or "")
        for value in (
            spec.get("title") or event.get("title"),
            spec.get("category") or event.get("category"),
            spec.get("description") or event.get("description"),
            spec.get("rules") or event.get("rules"),
            event.get("market_ticker"),
            event.get("event_ticker"),
        )
    ).lower()

    lookup = _lookup_details(context)
    route_reason = str(context.get("route_reason") or "")
    route_was_corrected = "llm verifier corrected" in route_reason
    ambiguous_route = (
        template_name in {GENERIC_NEWS_UNIQUE, MARKET_PRIOR_BASELINE}
        or float(context.get("template_confidence") or 0.0) < 0.78
        or _mixed_domain_keyword_count(text) >= 2
    )
    has_weird_flow = _has_weird_flow_features(context)
    no_supported_lookup = not lookup
    supported_lookup_n = int(lookup.get("n") or 0) if lookup else 0
    lookup_is_weak = 0 < supported_lookup_n < 50
    lookup_near_floor_or_ceiling = prior <= 0.03 or prior >= 0.97
    generic_threshold_words = any(word in text for word in ("above", "below", "over", "under", "exceed", "at least", "at most"))

    if lookup_near_floor_or_ceiling and not (route_was_corrected or has_weird_flow or ambiguous_route):
        return False
    if lookup and supported_lookup_n >= 100 and not (route_was_corrected or has_weird_flow or ambiguous_route):
        return False
    if no_supported_lookup:
        return True
    if route_was_corrected or has_weird_flow or ambiguous_route:
        return True
    if lookup_is_weak and 0.15 <= prior <= 0.85:
        return True
    if generic_threshold_words and 0.15 <= prior <= 0.85:
        return True
    return False


def _should_run_factual_resolution_override(
    template_name: str,
    context: dict[str, Any],
    prior: float,
) -> bool:
    import os

    if os.environ.get("FACTUAL_RESOLUTION_OVERRIDE_ENABLED") == "0":
        return False
    if os.environ.get("FACTUAL_RESOLUTION_FORCE") == "1":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("OPENAI_API_KEY"):
        return False

    event = context.get("event") or {}
    spec = context.get("spec") or {}
    text = " ".join(
        str(value or "")
        for value in (
            spec.get("title") or event.get("title"),
            spec.get("category") or event.get("category"),
            spec.get("description") or event.get("description"),
            spec.get("rules") or event.get("rules"),
        )
    ).lower()
    if event.get("resolved_outcome") is not None:
        return True
    if prior <= 0.08 or prior >= 0.92:
        return True
    factual_terms = (
        "who won",
        "winner",
        "named",
        "announced",
        "released",
        "appointed",
        "resigned",
        "elected",
        "launched",
        "signed",
        "passed",
        "confirmed",
        "reported",
    )
    if any(term in text for term in factual_terms):
        return True
    return template_name in {
        POLITICS_ELECTIONS_POLICY,
        COMPANY_TECH_ANNOUNCEMENT,
        CULTURE_AWARDS_ENTERTAINMENT,
        GENERIC_NEWS_UNIQUE,
    } and _lookup_details(context) == {}


def _lookup_details(context: dict[str, Any]) -> dict[str, Any]:
    prior_details = context.get("prior_details")
    if isinstance(prior_details, dict):
        lookup = prior_details.get("sector_bucket_lookup") or prior_details.get("tennis_lookup")
        if isinstance(lookup, dict):
            return lookup
    return {}


def _has_weird_flow_features(context: dict[str, Any]) -> bool:
    features = context.get("features")
    if not isinstance(features, dict):
        return False
    return (
        abs(float(features.get("recent_price_change_1h") or 0.0)) >= 0.08
        or abs(float(features.get("recent_price_change_24h") or 0.0)) >= 0.15
        or float(features.get("volume_zscore") or 0.0) >= 4.0
        or float(features.get("large_trade_flag") or 0.0) > 0.0
    )


def _mixed_domain_keyword_count(text: str) -> int:
    domains = (
        ("btc", "bitcoin", "eth", "crypto", "stock", "nasdaq", "oil"),
        ("cpi", "inflation", "gdp", "fed", "jobs", "unemployment"),
        ("nba", "mlb", "nfl", "nhl", "tennis", "soccer", "game", "match"),
        ("weather", "temperature", "hurricane", "rain", "snow"),
        ("election", "president", "senate", "congress", "bill", "government"),
        ("oscar", "emmy", "grammy", "movie", "album", "box office"),
    )
    return sum(1 for domain in domains if any(token in text for token in domain))

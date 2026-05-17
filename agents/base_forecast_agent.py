"""Base class for modular Prophet Arena forecast agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import agents.feature_registry as feature_registry
import agents.llm_tools as llm_tools
import agents.logging_utils as logging_utils
import agents.prior as prior
import agents.template_models as template_models
from agents.blending import BlendResult, blend
from agents.prior import PriorEstimate
from agents.template_models import ModelOutput
from agents.templates import MARKET_PRIOR_BASELINE, TemplateRoute


class BaseForecastAgent(ABC):
    """Base class for modular router-based binary forecasting agents."""

    log_path = "forecast_logs.jsonl"

    def predict(self, event: dict[str, Any]) -> dict[str, Any]:
        """Run the full forecast pipeline for a single event."""
        trace: dict[str, Any] = {
            "timestamp": logging_utils.utc_now_iso(),
            "event_ticker": event.get("event_ticker"),
            "market_ticker": event.get("market_ticker"),
            "title": event.get("title"),
            "raw_event_category": event.get("category"),
            "routed_template": MARKET_PRIOR_BASELINE,
            "template_confidence": 0.0,
            "prior": 0.50,
            "prior_source": "default_0.50",
            "feature_plan": {},
            "structured_features": {},
            "model_outputs": [],
            "final_p_yes": 0.50,
            "rationale": "",
            "errors": [],
            "fallback_flags": [],
        }

        spec: dict[str, Any] = {}
        route = TemplateRoute(MARKET_PRIOR_BASELINE, 0.0, "default initialization")
        prior_estimate = PriorEstimate(0.50, "default_0.50")
        feature_plan: dict[str, Any] = {}
        raw_data: dict[str, Any] = {}
        structured_features: dict[str, float] = {}
        model_outputs: list[ModelOutput] = []
        blend_result = BlendResult(0.50, 0.0, 0.0, diagnostics={})

        try:
            self.validate_event(event)
            spec = self.parse_event_with_llm_or_rules(event)
            route = self.route_template(event, spec)
            prior_estimate = self.estimate_prior(event, spec, route)
            feature_plan = self.build_feature_plan(event, spec, route, prior_estimate)
            raw_data = self.collect_data(event, spec, route, feature_plan, prior_estimate)
            structured_features = self.compute_structured_features(
                event,
                spec,
                route,
                prior_estimate,
                feature_plan,
                raw_data,
            )
            model_outputs = self.run_template_model_or_reasoner(
                event,
                spec,
                route,
                prior_estimate,
                structured_features,
                raw_data,
            )
            blend_result = self.blend_with_prior(
                prior_estimate,
                model_outputs,
                route,
                structured_features,
            )
            final_p_yes = self.calibrate_and_clamp(
                prior_estimate,
                blend_result,
                route,
                structured_features,
            )
            rationale = self.make_rationale(
                event=event,
                spec=spec,
                route=route,
                prior_estimate=prior_estimate,
                feature_plan=feature_plan,
                raw_data=raw_data,
                structured_features=structured_features,
                model_outputs=model_outputs,
                blend_result=blend_result,
                final_p_yes=final_p_yes,
            )

            trace.update(
                {
                    "title": spec.get("title") or event.get("title"),
                    "raw_event_category": spec.get("category") or event.get("category"),
                    "routed_template": route.template_name,
                    "template_confidence": route.template_confidence,
                    "prior": prior_estimate.probability,
                    "prior_source": prior_estimate.prior_source,
                    "feature_plan": feature_plan,
                    "structured_features": structured_features,
                    "model_outputs": model_outputs,
                    "final_p_yes": final_p_yes,
                    "rationale": rationale,
                }
            )
            self.log_full_trace(trace)
            return {"p_yes": final_p_yes, "rationale": rationale}
        except Exception as exc:
            fallback = self.safe_prior_mode(event, spec, route, prior_estimate, exc)
            trace.update(
                {
                    "title": spec.get("title") or event.get("title"),
                    "raw_event_category": spec.get("category") or event.get("category"),
                    "routed_template": route.template_name,
                    "template_confidence": route.template_confidence,
                    "prior": prior_estimate.probability,
                    "prior_source": prior_estimate.prior_source,
                    "feature_plan": feature_plan,
                    "structured_features": structured_features,
                    "model_outputs": model_outputs,
                    "final_p_yes": fallback["p_yes"],
                    "rationale": fallback["rationale"],
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "fallback_flags": ["safe_prior_mode"],
                }
            )
            self.log_full_trace(trace)
            return fallback

    def validate_event(self, event: dict[str, Any]) -> None:
        """Require only a title and tolerate missing optional fields."""
        if not isinstance(event, dict):
            raise TypeError("event must be a dict")
        title = event.get("title") or event.get("question")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("event.title is required")

    def parse_event_with_llm_or_rules(self, event: dict[str, Any]) -> dict[str, Any]:
        """Parse the event using rule-based LLM stubs."""
        return llm_tools.parse_event_with_stub_llm_or_rules(event)

    @abstractmethod
    def route_template(self, event: dict[str, Any], spec: dict[str, Any]) -> TemplateRoute:
        """Choose the best reusable template for the event."""

    def estimate_prior(
        self,
        event: dict[str, Any],
        spec: dict[str, Any],
        route: TemplateRoute,
    ) -> PriorEstimate:
        """Estimate the market/base-rate prior."""
        return prior.estimate_prior(event, spec, route)

    def build_feature_plan(
        self,
        event: dict[str, Any],
        spec: dict[str, Any],
        route: TemplateRoute,
        prior_estimate: PriorEstimate,
    ) -> dict[str, Any]:
        """Build a template-specific feature plan."""
        return feature_registry.build_feature_plan(event, spec, route, prior_estimate)

    def collect_data(
        self,
        event: dict[str, Any],
        spec: dict[str, Any],
        route: TemplateRoute,
        feature_plan: dict[str, Any],
        prior_estimate: PriorEstimate,
    ) -> dict[str, Any]:
        """Collect raw data, without any external API calls in v1."""
        return feature_registry.collect_data(event, spec, route, feature_plan, prior_estimate)

    def compute_structured_features(
        self,
        event: dict[str, Any],
        spec: dict[str, Any],
        route: TemplateRoute,
        prior_estimate: PriorEstimate,
        feature_plan: dict[str, Any],
        raw_data: dict[str, Any],
    ) -> dict[str, float]:
        """Compute flat numeric features."""
        return feature_registry.compute_structured_features(
            event,
            spec,
            route,
            prior_estimate,
            feature_plan,
            raw_data,
        )

    def run_template_model_or_reasoner(
        self,
        event: dict[str, Any],
        spec: dict[str, Any],
        route: TemplateRoute,
        prior_estimate: PriorEstimate,
        structured_features: dict[str, float],
        raw_data: dict[str, Any],
    ) -> list[ModelOutput]:
        """Run the template model stack."""
        context = {
            "event": event,
            "spec": spec,
            "raw_data": raw_data,
            "prior_source": prior_estimate.prior_source,
            "prior_details": prior_estimate.details,
            "template_confidence": route.template_confidence,
            "route_reason": route.reason,
            "features": structured_features,
        }
        return template_models.evaluate_template_models(
            route.template_name,
            structured_features,
            prior_estimate.probability,
            context,
        )

    def blend_with_prior(
        self,
        prior_estimate: PriorEstimate,
        model_outputs: list[ModelOutput],
        route: TemplateRoute,
        structured_features: dict[str, float],
    ) -> BlendResult:
        """Blend the model outputs conservatively with the prior."""
        data_quality_hint = 0.20 + (0.30 * structured_features.get("credibility_score", 0.0))
        return blend(
            prior=prior_estimate.probability,
            model_outputs=model_outputs,
            template_confidence=route.template_confidence,
            data_quality_hint=data_quality_hint,
        )

    def calibrate_and_clamp(
        self,
        prior_estimate: PriorEstimate,
        blend_result: BlendResult,
        route: TemplateRoute,
        structured_features: dict[str, float],
    ) -> float:
        """Apply a final conservative calibration step and clamp to valid bounds."""
        prior_value = prior.clamp_probability(prior_estimate.probability)
        candidate = prior.clamp_probability(blend_result.probability)
        max_move = 0.04 + (0.10 * route.template_confidence * max(0.25, blend_result.data_quality))
        delta = candidate - prior_value
        if delta > max_move:
            candidate = prior_value + max_move
        elif delta < -max_move:
            candidate = prior_value - max_move
        if route.template_confidence < 0.55 or structured_features.get("already_priced_score", 0.0) > 0.7:
            candidate = (0.88 * prior_value) + (0.12 * candidate)
        return prior.clamp_probability(candidate)

    def safe_prior_mode(
        self,
        event: dict[str, Any],
        spec: dict[str, Any],
        route: TemplateRoute,
        prior_estimate: PriorEstimate,
        exc: Exception,
    ) -> dict[str, Any]:
        """Return a safe prior-anchored prediction on any failure."""
        if prior_estimate.prior_source == "default_0.50" and not spec:
            route = TemplateRoute(MARKET_PRIOR_BASELINE, 0.0, "fallback without parsed spec")
        p_yes = prior.clamp_probability(prior_estimate.probability)
        rationale = (
            f"Fallback prior-mode prediction due to {type(exc).__name__}; "
            f"template={route.template_name}; prior_source={prior_estimate.prior_source}; "
            f"p_yes={p_yes:.2f}."
        )
        return {"p_yes": p_yes, "rationale": rationale}

    def make_rationale(
        self,
        *,
        event: dict[str, Any],
        spec: dict[str, Any],
        route: TemplateRoute,
        prior_estimate: PriorEstimate,
        feature_plan: dict[str, Any],
        raw_data: dict[str, Any],
        structured_features: dict[str, float],
        model_outputs: list[ModelOutput],
        blend_result: BlendResult,
        final_p_yes: float,
    ) -> str:
        """Create a short user-facing rationale."""
        _ = (event, spec, feature_plan, raw_data, structured_features)
        lookup_phrase = _lookup_rationale(prior_estimate)
        fast_phrase = _model_delta_phrase(
            model_outputs,
            prior_estimate.probability,
            {"trained_order_flow_residual"},
            "fast_model",
        )
        llm_phrase = _model_delta_phrase(
            model_outputs,
            prior_estimate.probability,
            {"llm_conviction_nudge"},
            "llm_conviction",
        )
        other_outputs = [
            output.explanation
            for output in model_outputs
            if output.model_name not in {"baseline_market_prior", "trained_order_flow_residual", "llm_conviction_nudge"}
            and output.confidence > 0.0
        ]
        other_phrase = f"; other={' | '.join(other_outputs[:2])}" if other_outputs else ""
        return (
            f"Template={route.template_name}; prior={prior_estimate.probability:.2f} "
            f"({prior_estimate.prior_source}); model_weight={blend_result.weight_on_models:.2f}; "
            f"final={final_p_yes:.2f}. {lookup_phrase}; {fast_phrase}; {llm_phrase}{other_phrase}"
        )

    def log_full_trace(self, trace: dict[str, Any]) -> None:
        """Append a JSONL trace row. Logging failures never break prediction."""
        try:
            logging_utils.append_jsonl(self.log_path, trace)
        except Exception:
            return


def _lookup_rationale(prior_estimate: PriorEstimate) -> str:
    details = prior_estimate.details or {}
    lookup = details.get("sector_bucket_lookup") or details.get("tennis_lookup")
    if not isinstance(lookup, dict):
        return "lookup=no supported bucket"
    key = lookup.get("lookup_key")
    n = lookup.get("n")
    bucket = f"{lookup.get('bucket_low')}-{lookup.get('bucket_high')}"
    edge = lookup.get("capped_adjustment")
    raw_edge = lookup.get("uncapped_adjustment")
    granularity = lookup.get("fallback_steps")
    try:
        edge_text = f"{float(edge):+.3f}"
    except (TypeError, ValueError):
        edge_text = "n/a"
    try:
        raw_edge_text = f"{float(raw_edge):+.3f}"
    except (TypeError, ValueError):
        raw_edge_text = "n/a"
    return (
        f"lookup={key} bucket={bucket} n={n} fallback_steps={granularity} "
        f"raw_edge={raw_edge_text} capped_edge={edge_text}"
    )


def _model_delta_phrase(
    model_outputs: list[ModelOutput],
    prior_probability: float,
    model_names: set[str],
    label: str,
) -> str:
    output = next((item for item in model_outputs if item.model_name in model_names), None)
    if output is None:
        return f"{label}=not run"
    delta = output.p_model - prior_probability
    if output.confidence <= 0.0 or output.data_quality <= 0.0:
        return f"{label}=no nudge ({output.explanation})"
    return (
        f"{label}_nudge={delta:+.3f} "
        f"confidence={output.confidence:.2f} data_quality={output.data_quality:.2f} "
        f"({output.explanation})"
    )

"""Tiny economics router for prior-anchored forecasts.

Runtime economics forecasting is intentionally simple:
market prior + a small, explicit fast-data adjustment.

No LLM calls, no web scraping, no heavy model artifacts, and no broad JPMaQS
feature soup are used here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agents.economics.base_model import EconomicModelOutput
from agents.economics.economic_feature_collector import (
    collect_economic_public_features,
    public_signal_adjustment,
)
from agents.economics.event_parser import parse_economic_event
from agents.prior import clamp_probability


class EconomicModelRouter:
    """Apply only trusted fast-data nudges to economics events."""

    def predict(
        self,
        *,
        event: dict[str, Any],
        spec: dict[str, Any],
        prior: float,
        context: dict[str, Any],
    ) -> EconomicModelOutput:
        event_spec = parse_economic_event(event, spec)
        as_of = _parse_as_of(context.get("forecast_as_of"))
        features, diagnostics = collect_economic_public_features(
            event=event,
            spec=spec,
            event_spec=event_spec,
            as_of=as_of,
        )
        adjustment, signal_confidence, signal_reason = public_signal_adjustment(event_spec, features)
        p_model = clamp_probability(prior + adjustment)
        if adjustment == 0.0:
            explanation = "Economics runtime used market prior only; no trusted fast-data nudge matched."
        else:
            explanation = f"Economics runtime used market prior plus capped fast-data nudge ({adjustment:+.3f}). {signal_reason}"
        return EconomicModelOutput(
            p_model=p_model,
            confidence=min(0.18, 0.05 + signal_confidence),
            explanation=explanation,
            data_quality=min(0.30, signal_confidence + 0.05),
            model_name="economics_fast_nudge",
            diagnostics={
                "event_spec": event_spec.__dict__,
                "fast_features": features,
                "fast_feature_diagnostics": diagnostics,
                "adjustment": adjustment,
                "prior": clamp_probability(prior),
            },
        )


def _parse_as_of(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None

"""LLM-assisted discovery of numeric unstructured evidence features."""

from __future__ import annotations

import json
import os
from typing import Any

from agents.evidence.catalog import get_template_spec
from agents.evidence.schemas import EvidenceTemplateSpec, NumericFeatureSpec


SYSTEM_PROMPT = """\
You discover candidate features for prediction-market residual models.

Rules:
- Do not output probabilities.
- Propose only numeric features that can be measured historically.
- Each feature must be available from public sources before event close/as_of_time.
- Reject vague narrative features that cannot become numbers.
- Prefer repeated features across many similar events.
- Return only JSON.
"""


def discover_feature_spec(
    template: str,
    examples: list[dict[str, Any]] | None = None,
    max_new_features: int = 8,
) -> EvidenceTemplateSpec:
    """Ask the LLM to refine/propose numeric features for a template.

    If OpenAI is unavailable, returns the built-in default spec.
    """
    base_spec = get_template_spec(template)
    if not os.environ.get("OPENAI_API_KEY"):
        return base_spec

    try:
        payload = _call_openai(base_spec, examples or [], max_new_features)
    except Exception:
        return base_spec

    proposed = _parse_feature_payload(payload)
    if not proposed:
        return base_spec

    merged = _merge_specs(base_spec, proposed)
    return EvidenceTemplateSpec(
        template=base_spec.template,
        description=base_spec.description,
        preferred_source_types=base_spec.preferred_source_types,
        features=tuple(merged),
    )


def _call_openai(
    base_spec: EvidenceTemplateSpec,
    examples: list[dict[str, Any]],
    max_new_features: int,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("EVIDENCE_DISCOVERY_MODEL", os.environ.get("FORECAST_MODEL", "gpt-4o-mini"))
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "template": base_spec.to_dict(),
                        "examples": examples[:20],
                        "max_new_features": max_new_features,
                        "required_json_shape": {
                            "features": [
                                {
                                    "name": "snake_case_numeric_feature",
                                    "description": "what is measured",
                                    "direction": "higher supports YES|higher supports NO|uncertainty|base_rate",
                                    "min_value": -1.0,
                                    "max_value": 1.0,
                                    "default_value": 0.0,
                                    "required_source_types": ["official_source"],
                                    "historical_measurement": "how to reconstruct before as_of_time",
                                    "leakage_risk": "low|medium|high",
                                }
                            ]
                        },
                    },
                    ensure_ascii=True,
                ),
            },
        ],
    )
    return json.loads(response.choices[0].message.content or "{}")


def _parse_feature_payload(payload: dict[str, Any]) -> list[NumericFeatureSpec]:
    features: list[NumericFeatureSpec] = []
    raw_features = payload.get("features")
    if not isinstance(raw_features, list):
        return features
    for raw in raw_features:
        if not isinstance(raw, dict):
            continue
        spec = _feature_from_dict(raw)
        if spec is not None:
            features.append(spec)
    return features


def _feature_from_dict(raw: dict[str, Any]) -> NumericFeatureSpec | None:
    name = str(raw.get("name") or "").strip().lower()
    if not name or not name.replace("_", "").isalnum():
        return None
    try:
        min_value = float(raw.get("min_value"))
        max_value = float(raw.get("max_value"))
    except (TypeError, ValueError):
        return None
    if min_value >= max_value:
        return None
    measurement = str(raw.get("historical_measurement") or "").strip()
    if len(measurement) < 12:
        return None
    return NumericFeatureSpec(
        name=name,
        description=str(raw.get("description") or name).strip(),
        direction=str(raw.get("direction") or "unknown").strip(),
        min_value=min_value,
        max_value=max_value,
        default_value=_float_or_default(raw.get("default_value"), 0.0),
        required_source_types=tuple(str(item) for item in raw.get("required_source_types") or ()),
        historical_measurement=measurement,
        leakage_risk=str(raw.get("leakage_risk") or "medium").strip().lower(),
    )


def _merge_specs(
    base_spec: EvidenceTemplateSpec,
    proposed: list[NumericFeatureSpec],
) -> list[NumericFeatureSpec]:
    by_name = {feature.name: feature for feature in base_spec.features}
    for feature in proposed:
        by_name.setdefault(feature.name, feature)
    return list(by_name.values())


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


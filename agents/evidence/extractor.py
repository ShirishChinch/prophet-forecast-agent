"""LLM extraction of numeric evidence features from timestamped sources."""

from __future__ import annotations

import json
import os
from typing import Any

from agents.evidence.catalog import get_template_spec
from agents.evidence.schemas import (
    EvidenceSource,
    EvidenceTemplateSpec,
    ExtractedEvidenceFeatures,
    clamp_feature,
    clamp_unit,
)


SYSTEM_PROMPT = """\
You extract numeric features from timestamped public evidence.

Rules:
- Do not forecast probability.
- Do not invent facts absent from sources.
- Use only the provided sources.
- Prefer sources published at or before as_of_time.
- If source timing is missing/after as_of_time, increase temporal_leakage_risk.
- All feature values must be numeric.
- Return only JSON.
"""


def extract_evidence_features(
    *,
    event: dict[str, Any],
    template: str,
    as_of_time: str,
    sources: list[EvidenceSource],
    spec: EvidenceTemplateSpec | None = None,
) -> ExtractedEvidenceFeatures:
    """Extract numeric evidence features for one event/template."""
    feature_spec = spec or get_template_spec(template)
    if not os.environ.get("OPENAI_API_KEY"):
        return _neutral_features(event, feature_spec, as_of_time, sources, "OPENAI_API_KEY not set")

    try:
        payload = _call_openai(event, feature_spec, as_of_time, sources)
    except Exception as exc:
        return _neutral_features(event, feature_spec, as_of_time, sources, f"{type(exc).__name__}: {exc}")

    return _payload_to_extracted(event, feature_spec, as_of_time, sources, payload)


def _call_openai(
    event: dict[str, Any],
    spec: EvidenceTemplateSpec,
    as_of_time: str,
    sources: list[EvidenceSource],
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("EVIDENCE_EXTRACTION_MODEL", os.environ.get("FORECAST_MODEL", "gpt-4o-mini"))
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
                        "event": {
                            "event_ticker": event.get("event_ticker"),
                            "market_ticker": event.get("market_ticker"),
                            "title": event.get("title"),
                            "rules": event.get("rules"),
                            "outcomes": event.get("outcomes"),
                            "category": event.get("category"),
                            "close_time": event.get("close_time"),
                        },
                        "template": spec.to_dict(),
                        "as_of_time": as_of_time,
                        "sources": [source.to_dict() for source in sources[:12]],
                        "required_output": {
                            "features": {feature.name: feature.default_value for feature in spec.features},
                            "source_urls": [],
                            "extraction_confidence": 0.0,
                            "temporal_leakage_risk": 0.0,
                            "short_rationale": "one sentence; no probability",
                        },
                    },
                    ensure_ascii=True,
                ),
            },
        ],
    )
    return json.loads(response.choices[0].message.content or "{}")


def _payload_to_extracted(
    event: dict[str, Any],
    spec: EvidenceTemplateSpec,
    as_of_time: str,
    sources: list[EvidenceSource],
    payload: dict[str, Any],
) -> ExtractedEvidenceFeatures:
    raw_features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    features = {
        feature.name: clamp_feature(raw_features.get(feature.name), feature)
        for feature in spec.features
    }
    usable_sources = [source for source in sources if source.is_known_before(as_of_time)]
    source_urls = [
        str(url)
        for url in payload.get("source_urls", [])
        if isinstance(url, str)
    ][:12]
    if not source_urls:
        source_urls = [source.url for source in usable_sources if source.url][:12]
    return ExtractedEvidenceFeatures(
        template=spec.template,
        event_ticker=str(event.get("event_ticker") or ""),
        market_ticker=str(event.get("market_ticker") or event.get("event_ticker") or ""),
        as_of_time=as_of_time,
        features=features,
        source_urls=source_urls,
        source_count=len(sources),
        usable_source_count=len(usable_sources),
        extraction_confidence=clamp_unit(payload.get("extraction_confidence")),
        temporal_leakage_risk=clamp_unit(payload.get("temporal_leakage_risk"), default=1.0),
        short_rationale=str(payload.get("short_rationale") or "").strip(),
    )


def _neutral_features(
    event: dict[str, Any],
    spec: EvidenceTemplateSpec,
    as_of_time: str,
    sources: list[EvidenceSource],
    reason: str,
) -> ExtractedEvidenceFeatures:
    return ExtractedEvidenceFeatures(
        template=spec.template,
        event_ticker=str(event.get("event_ticker") or ""),
        market_ticker=str(event.get("market_ticker") or event.get("event_ticker") or ""),
        as_of_time=as_of_time,
        features={feature.name: feature.default_value for feature in spec.features},
        source_urls=[],
        source_count=len(sources),
        usable_source_count=0,
        extraction_confidence=0.0,
        temporal_leakage_risk=1.0,
        short_rationale=f"Neutral extraction: {reason}",
    )


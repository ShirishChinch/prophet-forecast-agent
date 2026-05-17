"""Runtime LLM extractor for curated edge features.

The LLM only fills numeric feature values. It is explicitly not allowed to
produce a forecast probability or betting recommendation.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from typing import Any
from urllib.parse import urlparse

from agents.evidence.edge_feature_schema import (
    CURATED_EDGE_FEATURES,
    default_llm_feature_row,
    normalize_llm_feature_payload,
)


SYSTEM_PROMPT = """\
You extract numeric public-evidence features for a prediction-market residual
model.

Hard rules:
- Do not forecast probability.
- Do not recommend trades.
- Do not use private information.
- Do not identify individual traders.
- Use only public information available at or before as_of_time.
- Return 0 when evidence cannot be measured reliably.
- Every feature is signed: positive supports YES, negative supports NO.
- Source URLs must be real public pages from search results, not placeholders
  or generic made-up domains.
- Do not use low-quality generated prediction pages, SEO pages, random blogs,
  unverifiable odds mirrors, or generic domains. Prefer official league/event
  pages, major sportsbooks, established odds aggregators, official stats pages,
  and reputable news wires.
- If you cannot cite at least one specific trusted source URL for a measured
  feature, set that feature to 0.
- Return JSON only.
"""


TRUSTED_SOURCE_SUFFIXES = (
    "atptour.com",
    "wtatennis.com",
    "itftennis.com",
    "tennis.com",
    "espn.com",
    "flashscore.com",
    "sofascore.com",
    "oddschecker.com",
    "draftkings.com",
    "fanduel.com",
    "betmgm.com",
    "pinnacle.com",
    "covers.com",
    "actionnetwork.com",
    "rotowire.com",
    "nfl.com",
    "nba.com",
    "mlb.com",
    "nhl.com",
    "pgatour.com",
    "weather.gov",
    "noaa.gov",
    "cmegroup.com",
    "tradingeconomics.com",
    "investing.com",
    "fred.stlouisfed.org",
    "atlantafed.org",
    "clevelandfed.org",
    "reuters.com",
    "bloomberg.com",
    "sec.gov",
    "coinbase.com",
    "binance.com",
    "coinmarketcap.com",
    "coingecko.com",
    "coinglass.com",
)


def extract_llm_edge_features(
    event: dict[str, Any],
    template: str,
    *,
    as_of_time: str | None = None,
    use_web_search: bool | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return curated LLM features and raw payload.

    Missing API key, disabled LLM mode, or API failures all return neutral
    features. Prediction reliability should never depend on this call working.
    """
    if os.environ.get("EVIDENCE_LLM_ENABLED") != "1":
        return default_llm_feature_row(), {"skipped": "EVIDENCE_LLM_ENABLED is not 1"}
    if not os.environ.get("OPENAI_API_KEY"):
        return default_llm_feature_row(), {"skipped": "OPENAI_API_KEY is not set"}

    try:
        payload = _call_openai(event, template, as_of_time, use_web_search)
    except Exception as exc:
        return default_llm_feature_row(), {"error": f"{type(exc).__name__}: {exc}"}
    if not _has_valid_sources(payload):
        payload["_source_validation_error"] = "missing_or_invalid_source_urls"
        return default_llm_feature_row(), payload
    return normalize_llm_feature_payload(payload, template), payload


def _call_openai(
    event: dict[str, Any],
    template: str,
    as_of_time: str | None,
    use_web_search: bool | None,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("EVIDENCE_EDGE_MODEL", os.environ.get("FORECAST_MODEL", "gpt-5.5"))
    prompt = _build_prompt(event, template, as_of_time)
    web_enabled = use_web_search if use_web_search is not None else os.environ.get("EVIDENCE_OPENAI_WEB_SEARCH") == "1"

    if web_enabled:
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search_preview"}],
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = getattr(response, "output_text", "") or "{}"
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or "{}"

    payload = json.loads(_extract_json(text))
    payload["_web_search_used"] = bool(web_enabled)
    return payload


def _build_prompt(event: dict[str, Any], template: str, as_of_time: str | None) -> str:
    feature_specs = [
        {
            "name": spec.name,
            "description": spec.description,
            "preferred_sources": list(spec.preferred_sources),
            "measurement": spec.measurement,
            "range": [spec.range_min, spec.range_max],
        }
        for spec in CURATED_EDGE_FEATURES
        if _template_matches(template, spec.template)
    ]
    if not feature_specs:
        feature_specs = [
            {
                "name": spec.name,
                "description": spec.description,
                "preferred_sources": list(spec.preferred_sources),
                "measurement": spec.measurement,
                "range": [spec.range_min, spec.range_max],
            }
            for spec in CURATED_EDGE_FEATURES
            if spec.template == "generic"
        ]
    as_of = as_of_time or datetime.now(UTC).isoformat()
    compact_event = {
        "title": event.get("title"),
        "market_ticker": event.get("market_ticker") or event.get("ticker"),
        "event_ticker": event.get("event_ticker"),
        "category": event.get("category"),
        "rules": event.get("rules") or event.get("description"),
        "outcomes": event.get("outcomes"),
        "close_time": event.get("close_time") or event.get("finish_time") or event.get("expiration_time"),
        "prior": event.get("prior") or event.get("midpoint") or event.get("yes_price"),
    }
    return json.dumps(
        {
            "task": "Fill current numeric values for these reusable evidence features. Do not forecast probability.",
            "as_of_time": as_of,
            "template": template,
            "event": compact_event,
            "features_to_measure": feature_specs,
            "required_json_shape": {
                "features": {
                    "feature_name": 0.0
                },
                "source_urls": ["public URL used"],
                "source_timestamps": ["YYYY-MM-DDTHH:MM:SSZ or unknown"],
                "short_rationale": "one sentence, no probability",
                "temporal_leakage_risk": 0.0,
            },
        },
        ensure_ascii=True,
    )


def _template_matches(route_template: str, spec_template: str) -> bool:
    route = route_template.lower()
    if route == spec_template:
        return True
    aliases = {
        "price_threshold": "crypto_price",
        "macro_release": "macro",
        "politics_elections_policy": "politics",
        "company_tech_announcement": "company",
        "culture_awards_entertainment": "culture",
        "generic_news_unique": "generic",
        "market_prior_baseline": "generic",
    }
    return aliases.get(route) == spec_template


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return "{}"


def _has_valid_sources(payload: dict[str, Any]) -> bool:
    urls = payload.get("source_urls")
    if not isinstance(urls, list) or not urls:
        return False
    return any(_valid_url(url) for url in urls)


def _valid_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value.strip())
    host = (parsed.netloc or "").lower()
    if not parsed.scheme.startswith("http") or not host:
        return False
    if host in {"example.com", "www.example.com", "localhost"}:
        return False
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in TRUSTED_SOURCE_SUFFIXES)

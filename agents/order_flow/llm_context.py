"""OpenAI-backed context extraction for suspicious Kalshi order-flow moves.

The LLM never returns a probability. It compresses public evidence around a
market move into numeric features that a residual regression can learn from.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from typing import Any
from urllib.parse import urlparse


CONTEXT_FEATURE_DEFAULTS: dict[str, float] = {
    "llm_context_available": 0.0,
    "llm_direction_score": 0.0,
    "llm_evidence_strength": 0.0,
    "llm_source_quality": 0.0,
    "llm_freshness_score": 0.0,
    "llm_cross_market_confirmation": 0.0,
    "llm_stale_market_correction": 0.0,
    "llm_near_resolution": 0.0,
    "llm_already_priced_likelihood": 0.0,
    "llm_temporal_leakage_risk": 1.0,
    "llm_source_count": 0.0,
    "llm_pre_trade_context": 0.0,
    "llm_effective_direction_score": 0.0,
}


SYSTEM_PROMPT = """\
You convert public context around a prediction-market move into structured
features for a residual regression model.

Hard rules:
- Do not output a probability forecast.
- Do not claim to know who traded or identify any trader.
- Use only public information.
- For historical rows, prefer sources published at or before the trade time.
- If source timing is uncertain or after the trade time, mark temporal_leakage_risk high.
- Only mark context_found true when the evidence would plausibly have been
  publicly knowable at the trade timestamp.
- Return only JSON with the requested fields.
"""


@dataclass(frozen=True)
class SuspiciousMove:
    """Market move that may justify LLM context extraction."""

    market_ticker: str
    title: str
    timestamp: str
    prior: float
    price_change_1: float = 0.0
    price_change_24: float = 0.0
    volume_zscore_24: float = 0.0
    trade_size: float = 0.0
    trade_size_percentile: float = 0.0
    category: str = ""


def should_review_with_llm(features: dict[str, float]) -> bool:
    """Gate expensive LLM calls to genuinely suspicious flow."""
    if features.get("llm_context_available", 0.0) > 0.0:
        return False
    if abs(features.get("price_change_1", 0.0)) >= 0.08:
        return True
    if abs(features.get("price_change_24", 0.0)) >= 0.15:
        return True
    if features.get("volume_zscore_24", 0.0) >= 4.0:
        return True
    if features.get("trade_size_percentile", 0.0) >= 0.99:
        return True
    prior = features.get("prior", 0.50)
    return prior >= 0.95 or prior <= 0.05


def extract_context_features(move: SuspiciousMove) -> tuple[dict[str, float], dict[str, Any]]:
    """Call OpenAI and return numeric features plus raw parsed payload.

    If the API key is missing or the call fails, return conservative zero
    features. This keeps prediction reliable and prevents hidden probability
    guesses from entering the system.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return dict(CONTEXT_FEATURE_DEFAULTS), {"error": "OPENAI_API_KEY not set"}

    try:
        payload = _call_openai(move)
    except Exception as exc:
        return dict(CONTEXT_FEATURE_DEFAULTS), {"error": f"{type(exc).__name__}: {exc}"}

    features = _payload_to_features(payload, move.timestamp)
    return features, payload


def _call_openai(move: SuspiciousMove) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("ORDER_FLOW_CONTEXT_MODEL", os.environ.get("FORECAST_MODEL", "gpt-4o-mini"))
    prompt = _build_prompt(move)

    web_error: str | None = None
    if os.environ.get("ORDER_FLOW_OPENAI_WEB_SEARCH") == "1":
        try:
            response = client.responses.create(
                model=model,
                tools=[{"type": "web_search_preview"}],
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = getattr(response, "output_text", "") or "{}"
            payload = json.loads(_extract_json_object(text))
            payload["_web_search_used"] = True
            return payload
        except Exception as exc:
            web_error = f"{type(exc).__name__}: {exc}"
            if os.environ.get("ORDER_FLOW_ALLOW_NO_WEB_FALLBACK") != "1":
                raise RuntimeError(f"OpenAI web search failed: {web_error}") from exc

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
    payload = json.loads(_extract_json_object(text))
    payload["_web_search_used"] = False
    if web_error:
        payload["_web_search_error"] = web_error
    return payload


def _build_prompt(move: SuspiciousMove) -> str:
    return f"""\
Market ticker: {move.market_ticker}
Question/title: {move.title}
Category: {move.category or "unknown"}
Trade timestamp UTC: {move.timestamp}
Market prior at move: {move.prior:.4f}
1-step price change: {move.price_change_1:+.4f}
24-step price change: {move.price_change_24:+.4f}
Volume z-score: {move.volume_zscore_24:.3f}
Trade size: {move.trade_size:.3f}
Trade size percentile: {move.trade_size_percentile:.4f}

Task:
Find the most plausible PUBLIC causal explanation for this move, if any.
Examples: injury/lineup update, official release, weather update, crypto/stock move,
sportsbook line move, score/game state, stale-market correction, near-resolution
mechanics, or no clear public explanation.

For every supporting source, include a publication date or state "unknown".
If your only evidence is a recap, final score, or article that appears after
the trade timestamp, set context_found=false and temporal_leakage_risk=1.0.

Return JSON exactly with:
{{
  "context_found": true/false,
  "causal_category": "none|injury|lineup|official_release|weather|macro|crypto_price|stock_price|sportsbook_line|game_state|near_resolution|stale_market_correction|news|other",
  "direction": "YES|NO|neutral",
  "evidence_strength": 0.0-1.0,
  "source_quality": 0.0-1.0,
  "freshness_score": 0.0-1.0,
  "cross_market_confirmation": 0.0-1.0,
  "stale_market_correction": 0.0-1.0,
  "near_resolution": 0.0-1.0,
  "already_priced_likelihood": 0.0-1.0,
  "temporal_leakage_risk": 0.0-1.0,
  "source_urls": ["..."],
  "source_published_dates": ["YYYY-MM-DDTHH:MM:SSZ|unknown"],
  "short_rationale": "one sentence; no probability"
}}
"""


def _payload_to_features(payload: dict[str, Any], as_of_time: str | None = None) -> dict[str, float]:
    features = dict(CONTEXT_FEATURE_DEFAULTS)
    features["llm_context_available"] = 1.0 if bool(payload.get("context_found")) else 0.0
    direction = str(payload.get("direction") or "neutral").strip().upper()
    if direction == "YES":
        features["llm_direction_score"] = 1.0
    elif direction == "NO":
        features["llm_direction_score"] = -1.0
    else:
        features["llm_direction_score"] = 0.0
    features["llm_evidence_strength"] = _unit(payload.get("evidence_strength"))
    features["llm_source_quality"] = _unit(payload.get("source_quality"))
    features["llm_freshness_score"] = _unit(payload.get("freshness_score"))
    features["llm_cross_market_confirmation"] = _unit(payload.get("cross_market_confirmation"))
    features["llm_stale_market_correction"] = _unit(payload.get("stale_market_correction"))
    features["llm_near_resolution"] = _unit(payload.get("near_resolution"))
    features["llm_already_priced_likelihood"] = _unit(payload.get("already_priced_likelihood"))
    features["llm_temporal_leakage_risk"] = _unit(payload.get("temporal_leakage_risk"))
    source_urls = payload.get("source_urls")
    if isinstance(source_urls, list):
        valid_urls = [_valid_source_url(url) for url in source_urls]
        features["llm_source_count"] = float(min(sum(1 for valid in valid_urls if valid), 5))
        if source_urls and not any(valid_urls):
            features["llm_source_quality"] = 0.0
            features["llm_temporal_leakage_risk"] = 1.0
            features["llm_context_available"] = 0.0

    if as_of_time:
        source_dates = payload.get("source_published_dates")
        timing = _source_timing_quality(source_dates, as_of_time)
        if timing == "no_valid_pre_trade_source":
            features["llm_temporal_leakage_risk"] = 1.0
            features["llm_context_available"] = 0.0
        elif timing == "mixed_or_unknown":
            features["llm_temporal_leakage_risk"] = max(features["llm_temporal_leakage_risk"], 0.75)

    pre_trade_context = (
        features["llm_context_available"] > 0.0
        and features["llm_temporal_leakage_risk"] <= 0.25
        and features["llm_source_quality"] >= 0.5
    )
    features["llm_pre_trade_context"] = 1.0 if pre_trade_context else 0.0
    evidence_multiplier = (
        features["llm_evidence_strength"]
        * features["llm_source_quality"]
        * features["llm_freshness_score"]
        * (1.0 - features["llm_temporal_leakage_risk"])
        * (1.0 - 0.5 * features["llm_already_priced_likelihood"])
    )
    features["llm_effective_direction_score"] = features["llm_direction_score"] * evidence_multiplier
    return features


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start:end + 1]
    return "{}"


def _unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _valid_source_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value.strip())
    host = (parsed.netloc or "").lower()
    if not parsed.scheme.startswith("http") or not host:
        return False
    blocked_hosts = {"example.com", "www.example.com", "localhost"}
    return host not in blocked_hosts


def _source_timing_quality(source_dates: Any, as_of_time: str) -> str:
    """Classify whether cited source dates are usable for historical training."""
    as_of = _parse_datetime(as_of_time)
    if as_of is None:
        return "mixed_or_unknown"
    if not isinstance(source_dates, list) or not source_dates:
        return "no_valid_pre_trade_source"

    valid_pre_trade = 0
    unknown_or_after = 0
    for value in source_dates:
        published_at = _parse_datetime(str(value))
        if published_at is None:
            unknown_or_after += 1
            continue
        if published_at <= as_of:
            valid_pre_trade += 1
        else:
            unknown_or_after += 1

    if valid_pre_trade <= 0:
        return "no_valid_pre_trade_source"
    if unknown_or_after > 0:
        return "mixed_or_unknown"
    return "all_pre_trade"


def _parse_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text or text.lower() == "unknown":
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

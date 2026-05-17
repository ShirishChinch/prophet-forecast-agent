"""Optional LLM verification for high-level forecast template routing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from agents.templates import ALL_TEMPLATES


SYSTEM_PROMPT = """\
You verify high-level routing for a prediction-market forecasting agent.

Hard rules:
- Do not forecast probability.
- Do not recommend trades.
- Do not use private information.
- Only classify the event into one of the allowed templates.
- Prefer the safer broader template when uncertain.
- Return JSON only.
"""


@dataclass(frozen=True)
class TemplateVerification:
    """Template-route verification result."""

    template_name: str
    confidence: float
    agree: bool
    used: bool
    reason: str
    raw: dict[str, Any]


def verify_template_route(
    event: dict[str, Any],
    spec: dict[str, Any],
    *,
    proposed_template: str,
    proposed_confidence: float,
) -> TemplateVerification:
    """Verify a template route with an LLM when enabled.

    Set `TEMPLATE_ROUTE_LLM_VERIFY=1` and `OPENAI_API_KEY` to enable.
    Failures return the proposed route.
    """
    if os.environ.get("TEMPLATE_ROUTE_LLM_VERIFY") != "1":
        return _fallback(proposed_template, "disabled")
    if not os.environ.get("OPENAI_API_KEY"):
        return _fallback(proposed_template, "OPENAI_API_KEY is not set")
    if not _should_verify(event, spec, proposed_template, proposed_confidence):
        return _fallback(proposed_template, "not_ambiguous_enough")

    try:
        payload = _call_openai(
            event,
            spec,
            proposed_template=proposed_template,
            proposed_confidence=proposed_confidence,
        )
    except Exception as exc:
        return _fallback(proposed_template, f"{type(exc).__name__}: {exc}")

    template_name = _clean_template(payload.get("corrected_template"), proposed_template)
    confidence = _clip(_to_float(payload.get("confidence")) or 0.0, 0.0, 1.0)
    agree = bool(payload.get("is_route_correct")) and template_name == proposed_template
    reason = str(payload.get("reason") or "")
    if confidence < 0.80:
        return TemplateVerification(
            template_name=proposed_template,
            confidence=confidence,
            agree=agree,
            used=False,
            reason=f"low_confidence_llm_kept_rules: {reason}".strip(),
            raw=payload,
        )

    return TemplateVerification(
        template_name=template_name,
        confidence=confidence,
        agree=agree,
        used=template_name != proposed_template,
        reason=reason,
        raw=payload,
    )


def _should_verify(
    event: dict[str, Any],
    spec: dict[str, Any],
    proposed_template: str,
    proposed_confidence: float,
) -> bool:
    if os.environ.get("TEMPLATE_ROUTE_LLM_VERIFY_ALL") == "1":
        return True
    text = _event_text(event, spec)
    threshold_words = ("above", "below", "over", "under", "exceed", "at least", "at most")
    has_threshold_word = any(word in text for word in threshold_words)
    if proposed_confidence < 0.85:
        return True
    if has_threshold_word:
        return True
    if _mixed_domain_keyword_count(text) >= 2:
        return True
    if proposed_template in {"GENERIC_NEWS_UNIQUE", "MARKET_PRIOR_BASELINE"}:
        return True
    return False


def _call_openai(
    event: dict[str, Any],
    spec: dict[str, Any],
    *,
    proposed_template: str,
    proposed_confidence: float,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("TEMPLATE_ROUTE_LLM_MODEL", os.environ.get("FORECAST_MODEL", "gpt-5.5"))
    compact_event = {
        "title": spec.get("title") or event.get("title"),
        "market_ticker": spec.get("market_ticker") or event.get("market_ticker") or event.get("ticker"),
        "event_ticker": spec.get("event_ticker") or event.get("event_ticker"),
        "category": spec.get("category") or event.get("category"),
        "description": spec.get("description") or event.get("description"),
        "rules": spec.get("rules") or event.get("rules"),
        "outcomes": spec.get("outcomes") or event.get("outcomes"),
        "close_time": spec.get("close_time") or event.get("close_time") or event.get("expiration_time"),
    }
    prompt = {
        "task": "Verify whether the proposed high-level template route is correct.",
        "event": compact_event,
        "proposed_template": proposed_template,
        "proposed_confidence": proposed_confidence,
        "allowed_templates": list(ALL_TEMPLATES),
        "template_guidance": {
            "PRICE_THRESHOLD": "asset/financial/crypto/stock/commodity/index price threshold, not weather or sports totals",
            "MACRO_RELEASE": "CPI, GDP, jobs, unemployment, Fed/FOMC/rates macro release",
            "SPORTS": "matches, games, teams, athletes, player/team props, sports totals",
            "WEATHER": "temperature, rainfall, snow, hurricanes, official weather outcomes",
            "POLITICS_ELECTIONS_POLICY": "elections, bills, policy, appointments, government action",
            "COMPANY_TECH_ANNOUNCEMENT": "company announcements, product launches, earnings/guidance",
            "CULTURE_AWARDS_ENTERTAINMENT": "awards, music, movies, box office, culture",
            "GENERIC_NEWS_UNIQUE": "one-off news question that does not fit better elsewhere",
            "MARKET_PRIOR_BASELINE": "only when no reusable template fits",
        },
        "required_json_shape": {
            "is_route_correct": True,
            "corrected_template": proposed_template,
            "market_kind": "short kind label, no probability",
            "confidence": 0.0,
            "reason": "short reason, no probability",
        },
    }

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
        ],
    )
    return json.loads(_extract_json(getattr(response, "output_text", "") or "{}"))


def _fallback(template_name: str, reason: str) -> TemplateVerification:
    return TemplateVerification(
        template_name=template_name,
        confidence=0.0,
        agree=True,
        used=False,
        reason=reason,
        raw={},
    )


def _clean_template(value: Any, fallback: str) -> str:
    template = str(value or "").strip().upper()
    return template if template in set(ALL_TEMPLATES) else fallback


def _event_text(event: dict[str, Any], spec: dict[str, Any]) -> str:
    return " ".join(
        str((spec.get(key) if key in spec else event.get(key)) or "")
        for key in ("title", "category", "description", "rules", "market_ticker", "event_ticker")
    ).lower()


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


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return "{}"


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))

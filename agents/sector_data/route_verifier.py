"""Optional LLM verification for sector-bucket routing.

The verifier only checks classification. It is not allowed to forecast a
probability or recommend a trade.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any


ALLOWED_SECTORS = {
    "sports_tennis",
    "sports_baseball",
    "sports_basketball",
    "sports_hockey",
    "sports_football",
    "sports_soccer",
    "sports_golf",
    "sports_combat",
    "crypto_price",
    "weather",
    "macro",
    "politics",
    "financials",
    "culture",
    "generic",
}

SYSTEM_PROMPT = """\
You verify routing for a prediction-market empirical-prior lookup.

Hard rules:
- Do not forecast probability.
- Do not recommend trades.
- Do not infer private information.
- Only classify the event's reusable sector/subtype.
- Prefer a broader safer sector when uncertain.
- Return JSON only.
"""


@dataclass(frozen=True)
class RouteVerification:
    """LLM sector verification result."""

    sector: str
    subtype: str
    confidence: float
    agree: bool
    used: bool
    reason: str
    raw: dict[str, Any]


def verify_sector_route(
    event: dict[str, Any],
    *,
    proposed_sector: str,
    proposed_subtype: str,
    template_family: str | None,
) -> RouteVerification:
    """Verify a sector route with an LLM when explicitly enabled.

    Set `SECTOR_ROUTE_LLM_VERIFY=1` and `OPENAI_API_KEY` to enable this layer.
    Failures return the proposed deterministic route.
    """
    if os.environ.get("SECTOR_ROUTE_LLM_VERIFY") != "1":
        return _fallback(proposed_sector, proposed_subtype, "disabled")
    if not os.environ.get("OPENAI_API_KEY"):
        return _fallback(proposed_sector, proposed_subtype, "OPENAI_API_KEY is not set")

    try:
        payload = _call_openai(
            event,
            proposed_sector=proposed_sector,
            proposed_subtype=proposed_subtype,
            template_family=template_family,
        )
    except Exception as exc:
        return _fallback(proposed_sector, proposed_subtype, f"{type(exc).__name__}: {exc}")

    sector = _clean_sector(payload.get("corrected_sector"), proposed_sector)
    subtype = _clean_subtype(payload.get("corrected_subtype"), sector, proposed_subtype)
    confidence = _clip(_to_float(payload.get("confidence")) or 0.0, 0.0, 1.0)
    agree = bool(payload.get("is_route_correct")) and sector == proposed_sector
    reason = str(payload.get("reason") or "")

    if confidence < 0.75:
        return RouteVerification(
            sector=proposed_sector,
            subtype=proposed_subtype,
            confidence=confidence,
            agree=agree,
            used=False,
            reason=f"low_confidence_llm_kept_rules: {reason}".strip(),
            raw=payload,
        )

    return RouteVerification(
        sector=sector,
        subtype=subtype,
        confidence=confidence,
        agree=agree,
        used=(sector != proposed_sector or subtype != proposed_subtype),
        reason=reason,
        raw=payload,
    )


def _call_openai(
    event: dict[str, Any],
    *,
    proposed_sector: str,
    proposed_subtype: str,
    template_family: str | None,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("SECTOR_ROUTE_LLM_MODEL", os.environ.get("FORECAST_MODEL", "gpt-5.5"))
    compact_event = {
        "title": event.get("title"),
        "market_ticker": event.get("market_ticker") or event.get("ticker"),
        "event_ticker": event.get("event_ticker"),
        "category": event.get("category"),
        "description": event.get("description"),
        "rules": event.get("rules"),
        "outcomes": event.get("outcomes"),
        "close_time": event.get("close_time") or event.get("expiration_time") or event.get("expected_expiration_time"),
    }
    prompt = {
        "task": "Verify whether the proposed empirical-prior sector route is correct.",
        "event": compact_event,
        "template_family": template_family,
        "proposed_sector": proposed_sector,
        "proposed_subtype": proposed_subtype,
        "allowed_sectors": sorted(ALLOWED_SECTORS),
        "subtype_guidance": {
            "sports_tennis": ["atp", "wta", "itf", "all"],
            "crypto_price": ["btc", "eth", "xrp", "solana", "all"],
            "sports_*": ["baseball", "basketball", "hockey", "football", "soccer", "golf", "combat"],
            "other": ["all"],
        },
        "required_json_shape": {
            "is_route_correct": True,
            "corrected_sector": proposed_sector,
            "corrected_subtype": proposed_subtype,
            "market_kind": "match_winner | price_threshold | macro_release | politics_policy | weather_threshold | culture | generic",
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
    text = getattr(response, "output_text", "") or "{}"
    return json.loads(_extract_json(text))


def _fallback(sector: str, subtype: str, reason: str) -> RouteVerification:
    return RouteVerification(
        sector=sector,
        subtype=subtype,
        confidence=0.0,
        agree=True,
        used=False,
        reason=reason,
        raw={},
    )


def _clean_sector(value: Any, fallback: str) -> str:
    sector = str(value or "").strip().lower()
    return sector if sector in ALLOWED_SECTORS else fallback


def _clean_subtype(value: Any, sector: str, fallback: str) -> str:
    subtype = str(value or "").strip().lower() or fallback
    if sector == "sports_tennis" and subtype in {"atp", "wta", "itf", "all"}:
        return subtype
    if sector == "crypto_price" and subtype in {"btc", "eth", "xrp", "solana", "all"}:
        return subtype
    if sector.startswith("sports_"):
        expected = sector.replace("sports_", "")
        return subtype if subtype == expected else expected
    return "all"


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

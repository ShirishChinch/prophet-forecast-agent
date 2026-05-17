"""Simple LLM evidence conviction nudge.

This layer deliberately does not ask for a probability. It asks three
questions:

1. Is there public evidence pointing YES or NO?
2. How confident are we that the evidence is accurate?
3. How confident are we that the evidence is not already priced in?

Only when all checks are high confidence does it return a bounded residual nudge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from typing import Any

from agents.prior import clamp_probability


SYSTEM_PROMPT = """\
You evaluate public evidence for a prediction-market event.

Hard rules:
- Do not forecast an independent probability.
- Do not recommend trades.
- Do not use private information.
- Do not identify individual traders.
- Use credible public sources only.
- Prefer official sources, major news wires, regulated market data, official
  league/stat pages, official weather/macro releases, reputable odds pages, or
  primary company/government pages.
- Before deciding, actively look for unstructured public edge signals: recent
  news, expert commentary, beat-reporter updates, official social posts, X/Twitter
  posts from credible primary accounts, Reddit/forum discussion, local/industry
  niche sources, injury/lineup/weather/model-update chatter, market-moving rumors,
  and cross-market odds movement. Treat these as leads to verify, not as truth.
- Avoid SEO pages, random blogs, hallucinated URLs, unverifiable screenshots,
  low-quality prediction aggregators, and social chatter unless it is only
  supporting context.
- Do not let social chatter or rumors trigger a nudge unless corroborated by a
  credible primary source, reputable reporter, official account, or strong
  cross-market confirmation.
- If evidence is weak, stale, or likely already priced in, return neutral.
- Return JSON only.
"""

DEFAULT_NUDGE_SIZE = 0.05
MIN_EVIDENCE_CONFIDENCE = 0.75
MIN_ACCURACY_CONFIDENCE = 0.75
MIN_NOT_PRICED_CONFIDENCE = 0.70


@dataclass(frozen=True)
class ConvictionNudge:
    """LLM conviction nudge result."""

    p_model: float
    nudge: float
    applied: bool
    confidence: float
    data_quality: float
    explanation: str
    payload: dict[str, Any]


def evaluate_llm_conviction_nudge(
    event: dict[str, Any],
    *,
    template_name: str,
    prior: float,
) -> ConvictionNudge:
    """Return a bounded nudge when LLM evidence checks are all strong."""
    prior_value = clamp_probability(prior)
    if os.environ.get("LLM_CONVICTION_NUDGE_ENABLED") == "0":
        return _neutral(prior_value, "disabled")
    if not os.environ.get("OPENAI_API_KEY"):
        return _neutral(prior_value, "OPENAI_API_KEY is not set")

    try:
        payload = _call_openai(event, template_name=template_name, prior=prior_value)
    except Exception as exc:
        return _neutral(prior_value, f"{type(exc).__name__}: {exc}")

    direction = str(payload.get("direction") or "neutral").strip().upper()
    evidence_confidence = _clip(_to_float(payload.get("evidence_confidence")) or 0.0, 0.0, 1.0)
    accuracy_confidence = _clip(_to_float(payload.get("accuracy_confidence")) or 0.0, 0.0, 1.0)
    not_priced_confidence = _clip(_to_float(payload.get("not_priced_in_confidence")) or 0.0, 0.0, 1.0)
    source_quality = _clip(_to_float(payload.get("source_quality")) or accuracy_confidence, 0.0, 1.0)
    reason = str(payload.get("short_rationale") or payload.get("reason") or "")

    checks_pass = (
        direction in {"YES", "NO"}
        and evidence_confidence >= MIN_EVIDENCE_CONFIDENCE
        and accuracy_confidence >= MIN_ACCURACY_CONFIDENCE
        and not_priced_confidence >= MIN_NOT_PRICED_CONFIDENCE
    )
    if not checks_pass:
        return ConvictionNudge(
            p_model=prior_value,
            nudge=0.0,
            applied=False,
            confidence=0.0,
            data_quality=0.0,
            explanation=(
                "LLM conviction nudge neutral: "
                f"direction={direction}, evidence={evidence_confidence:.2f}, "
                f"accuracy={accuracy_confidence:.2f}, not_priced={not_priced_confidence:.2f}. "
                f"{reason}"
            ).strip(),
            payload=payload,
        )

    nudge_size = _configured_nudge_size()
    nudge = nudge_size if direction == "YES" else -nudge_size
    confidence = min(evidence_confidence, accuracy_confidence, not_priced_confidence)
    data_quality = min(source_quality, accuracy_confidence)
    return ConvictionNudge(
        p_model=clamp_probability(prior_value + nudge),
        nudge=nudge,
        applied=True,
        confidence=confidence,
        data_quality=data_quality,
        explanation=(
            f"LLM conviction nudge {nudge:+.2f}: evidence={evidence_confidence:.2f}, "
            f"accuracy={accuracy_confidence:.2f}, not_priced={not_priced_confidence:.2f}. "
            f"{reason}"
        ).strip(),
        payload=payload,
    )


def _call_openai(event: dict[str, Any], *, template_name: str, prior: float) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("LLM_CONVICTION_MODEL", os.environ.get("FORECAST_MODEL", "gpt-5.5"))
    web_enabled = os.environ.get("LLM_CONVICTION_WEB_SEARCH", "1") != "0"
    prompt = _build_prompt(
        event,
        template_name=template_name,
        prior=prior,
        nudge_size=_configured_nudge_size(),
    )

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
    return json.loads(_extract_json(text))


def _build_prompt(event: dict[str, Any], *, template_name: str, prior: float, nudge_size: float) -> str:
    compact_event = {
        "title": event.get("title"),
        "market_ticker": event.get("market_ticker") or event.get("ticker"),
        "event_ticker": event.get("event_ticker"),
        "category": event.get("category"),
        "description": event.get("description"),
        "rules": event.get("rules"),
        "outcomes": event.get("outcomes"),
        "close_time": event.get("close_time") or event.get("expiration_time") or event.get("expected_expiration_time"),
        "current_market_probability": prior,
    }
    return json.dumps(
        {
            "task": (
                f"Answer only these checks for a fixed +/-{nudge_size:.2f} residual nudge. "
                "Do not output a forecast probability."
            ),
            "as_of_time": datetime.now(UTC).isoformat(),
            "template_name": template_name,
            "event": compact_event,
            "questions": [
                "What weird/unstructured public evidence exists beyond obvious market odds: news, experts, X/Twitter, Reddit/forums, official social posts, niche/local sources, injury/lineup chatter, weather/model updates, or cross-market movement?",
                "Is there credible public evidence that odds should move YES or NO versus current market odds?",
                "What is your confidence that this evidence is accurate?",
                "What is your confidence that this evidence has not already been priced into the market?",
            ],
            "source_search_order": [
                "official/primary source pages and official social accounts",
                "major news wires and reputable beat reporters/experts",
                "domain-specific sources such as league/stat/weather/macro/filing/odds pages",
                "public X/Twitter, Reddit, forums, Discord mirrors, or niche local/industry sources only as leads",
                "cross-market odds or sportsbook movement for confirmation",
            ],
            "decision_rule": (
                "Return direction YES or NO only if evidence_confidence>=0.75, "
                "accuracy_confidence>=0.75, and not_priced_in_confidence>=0.70. "
                "Social chatter, rumors, or weird unstructured signals must be corroborated before passing. "
                "Otherwise return neutral."
            ),
            "required_json_shape": {
                "direction": "YES | NO | neutral",
                "evidence_confidence": 0.0,
                "accuracy_confidence": 0.0,
                "not_priced_in_confidence": 0.0,
                "source_quality": 0.0,
                "weird_unstructured_signals_checked": ["short labels"],
                "evidence_summary": "one sentence",
                "already_priced_reasoning": "one sentence",
                "source_urls": ["real public URL used"],
                "short_rationale": "one sentence, no probability",
            },
        },
        ensure_ascii=True,
    )


def _neutral(prior: float, reason: str) -> ConvictionNudge:
    return ConvictionNudge(
        p_model=prior,
        nudge=0.0,
        applied=False,
        confidence=0.0,
        data_quality=0.0,
        explanation=f"LLM conviction nudge skipped: {reason}.",
        payload={"skipped": reason},
    )


def _configured_nudge_size() -> float:
    value = _to_float(os.environ.get("LLM_CONVICTION_NUDGE_SIZE"))
    if value is None:
        return DEFAULT_NUDGE_SIZE
    return _clip(value, 0.0, 0.15)


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

"""Router-based modular forecast agent."""

from __future__ import annotations

import re
from typing import Any

from agents.base_forecast_agent import BaseForecastAgent
import agents.llm_tools as llm_tools
from agents.template_route_verifier import verify_template_route
from agents.templates import (
    COMPANY_TECH_ANNOUNCEMENT,
    CULTURE_AWARDS_ENTERTAINMENT,
    GENERIC_NEWS_UNIQUE,
    MACRO_RELEASE,
    MARKET_PRIOR_BASELINE,
    POLITICS_ELECTIONS_POLICY,
    PRICE_THRESHOLD,
    SPORTS,
    TemplateRoute,
    WEATHER,
)


class RouterForecastAgent(BaseForecastAgent):
    """Forecast agent that routes events into reusable template families."""

    def route_template(self, event: dict[str, Any], spec: dict[str, Any]) -> TemplateRoute:
        route = self._rule_route_template(event, spec)
        verification = verify_template_route(
            event,
            spec,
            proposed_template=route.template_name,
            proposed_confidence=route.template_confidence,
        )
        if not verification.used:
            if verification.reason not in {"disabled", "not_ambiguous_enough"}:
                return TemplateRoute(
                    route.template_name,
                    route.template_confidence,
                    f"{route.reason}; llm_verifier_kept_route={verification.reason}",
                    closest_template=route.closest_template,
                )
            return route
        corrected_confidence = max(0.55, min(route.template_confidence, verification.confidence))
        return TemplateRoute(
            verification.template_name,
            corrected_confidence,
            f"llm verifier corrected {route.template_name}: {verification.reason}",
            closest_template=route.template_name,
        )

    def _rule_route_template(self, event: dict[str, Any], spec: dict[str, Any]) -> TemplateRoute:
        title = str(spec.get("title") or event.get("title") or "").lower()
        category = str(spec.get("category") or event.get("category") or "").lower()
        outcomes = spec.get("outcomes") or []
        threshold_value = spec.get("threshold_value")

        if _has_any(category, ("weather", "climate")):
            return TemplateRoute(WEATHER, 0.86, "weather/climate category match")
        if _has_any(category, ("economics", "economy", "macro")):
            return TemplateRoute(MACRO_RELEASE, 0.86, "economics category match")
        if _has_any(category, ("election", "politics", "government")):
            return TemplateRoute(POLITICS_ELECTIONS_POLICY, 0.84, "politics/elections category match")
        if _has_any(category, ("entertainment", "culture")):
            return TemplateRoute(CULTURE_AWARDS_ENTERTAINMENT, 0.80, "culture/entertainment category match")

        if _has_any(title, ("btc", "bitcoin", "eth", "ethereum", "crypto")):
            return TemplateRoute(PRICE_THRESHOLD, 0.90, "crypto asset keyword match")
        if _looks_like_financial_price_threshold(title, threshold_value):
            return TemplateRoute(PRICE_THRESHOLD, 0.84, "financial price/threshold match")
        if _has_any(title, ("cpi", "inflation", "fed", "fomc", "unemployment", "jobs", "gdp", "rates", "yield")):
            return TemplateRoute(MACRO_RELEASE, 0.88, "macro keyword match")
        if _category_has_sports(category) or _has_sports_signal(title):
            return TemplateRoute(SPORTS, 0.85, "sports keyword/category match")
        if _has_any(title, ("weather", "hurricane", "temperature", "rainfall", "snow", "storm", "forecast", "degrees", "fahrenheit", "celsius", "95f", "95°f")):
            return TemplateRoute(WEATHER, 0.84, "weather keyword match")
        if _has_any(title, ("election", "president", "senate", "congress", "bill", "policy", "trump", "biden", "democrat", "republican", "government", "pope", "prime minister", "successor", "leader of")):
            return TemplateRoute(POLITICS_ELECTIONS_POLICY, 0.82, "politics/policy keyword match")
        if _has_any(title, ("award", "emmy", "grammy", "oscar", "eurovision", "anime", "chart", "box office", "billboard", "hot 100")):
            return TemplateRoute(CULTURE_AWARDS_ENTERTAINMENT, 0.76, "culture/awards keyword match")
        if _has_any(title, ("launch", "announce", "product", "earnings", "guidance", "openai", "apple", "google", "meta", "tesla", "microsoft", "spacex", "starship")):
            return TemplateRoute(COMPANY_TECH_ANNOUNCEMENT, 0.78, "company/tech keyword match")
        if threshold_value is not None and any(word in title for word in ("above", "below", "over", "under", "exceed")):
            return TemplateRoute(PRICE_THRESHOLD, 0.75, "threshold-style question match")
        if len(outcomes) == 2 and str(outcomes[0]).lower() not in {"yes", "no"} and str(outcomes[1]).lower() not in {"yes", "no"}:
            return TemplateRoute(SPORTS, 0.55, "two-sided named outcomes suggest head-to-head event")

        llm_guess = llm_tools.classify_event_with_stub_llm(spec)
        closest_template = str(llm_guess.get("closest_template") or GENERIC_NEWS_UNIQUE)
        confidence = float(llm_guess.get("template_confidence") or 0.40)
        reason = str(llm_guess.get("reason") or "stub llm fallback")

        if closest_template == GENERIC_NEWS_UNIQUE:
            return TemplateRoute(
                GENERIC_NEWS_UNIQUE,
                confidence,
                reason,
                closest_template=MARKET_PRIOR_BASELINE,
            )
        return TemplateRoute(
            closest_template,
            confidence,
            reason,
            closest_template=closest_template,
        )


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Match words/phrases without accidental substring hits like bill/Billie."""
    lowered = text.lower()
    for pattern in patterns:
        if " " in pattern or not pattern.replace("$", "").replace("°", "").isalnum():
            if pattern in lowered:
                return True
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", lowered):
            return True
    return False


def _category_has_sports(category: str) -> bool:
    return bool(re.search(r"(?<![a-z0-9])sports?(?![a-z0-9])", category.lower()))


def _has_sports_signal(title: str) -> bool:
    """Detect sports markets while avoiding generic uses of game/match/title."""
    if _has_any(
        title,
        (
            "nba",
            "wnba",
            "mlb",
            "nfl",
            "nhl",
            "atp",
            "wta",
            "itf",
            "ufc",
            "mma",
            "pga",
            "tennis",
            "soccer",
            "football",
            "baseball",
            "basketball",
            "hockey",
            "golf",
            "league of legends",
            "esports",
            "playoff",
            "premier league",
            "champions league",
        ),
    ):
        return True
    return _has_any(title, ("match", "game")) and _has_any(title, ("beat", "win", "vs", "over", "spread", "map"))


def _looks_like_financial_price_threshold(title: str, threshold_value: Any) -> bool:
    """Route tradable asset price questions before company-news questions."""
    if threshold_value is None:
        return False
    return _has_any(
        title,
        (
            "share",
            "shares",
            "stock",
            "trade",
            "close",
            "price",
            "nasdaq",
            "s&p",
            "spx",
            "dow",
            "oil",
            "gold",
            "silver",
            "eur/usd",
            "usd",
            "tesla",
            "nvda",
            "nvidia",
        ),
    )

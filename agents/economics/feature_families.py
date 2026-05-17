"""Economic feature-family rules for JPMaQS model inputs."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class FeatureFamilyRule:
    """One allowed family of JPMaQS features."""

    name: str
    keywords: tuple[str, ...]
    max_features: int
    prefer_fast: bool = True


MODEL_FAMILY_RULES: dict[str, tuple[FeatureFamilyRule, ...]] = {
    "inflation": (
        FeatureFamilyRule("inflation_history", ("cpi", "infl", "core", "pce"), 24, False),
        FeatureFamilyRule("inflation_expectations", ("infexp", "inflation expectation", "breakeven"), 16),
        FeatureFamilyRule("wages_labor", ("wage", "earn", "labor", "labour", "unemp", "jobs"), 18, False),
        FeatureFamilyRule("commodities_energy", ("oil", "gas", "energy", "commodity", "food"), 18),
        FeatureFamilyRule("fx_import_prices", ("fx", "xrate", "import", "trade weighted"), 16),
        FeatureFamilyRule("rates_conditions", ("rate", "yield", "curve", "fincond"), 18),
        FeatureFamilyRule("activity_demand", ("gdp", "output", "sales", "production", "demand"), 16, False),
    ),
    "growth": (
        FeatureFamilyRule("activity", ("gdp", "output", "production", "industrial", "sales"), 28, False),
        FeatureFamilyRule("surveys_confidence", ("pmi", "survey", "confidence", "sentiment"), 20),
        FeatureFamilyRule("labor", ("labor", "labour", "jobs", "unemp", "employment"), 20, False),
        FeatureFamilyRule("credit_financial_conditions", ("credit", "spread", "fincond", "lending"), 20),
        FeatureFamilyRule("markets", ("equity", "stock", "yield", "rate", "fx"), 20),
        FeatureFamilyRule("trade_global", ("export", "import", "trade", "global"), 16, False),
    ),
    "yield": (
        FeatureFamilyRule("yield_curve", ("yield", "gb", "bond", "curve"), 28),
        FeatureFamilyRule("policy_short_rates", ("policy", "stir", "rate", "money market"), 24),
        FeatureFamilyRule("inflation", ("cpi", "infl", "breakeven"), 20),
        FeatureFamilyRule("growth", ("gdp", "output", "pmi", "production"), 18, False),
        FeatureFamilyRule("risk_credit", ("spread", "credit", "vol", "equity", "risk"), 20),
        FeatureFamilyRule("fx_global_rates", ("fx", "xrate", "global", "usd"), 16),
    ),
    "policy": (
        FeatureFamilyRule("policy_rates", ("policy", "central bank", "rate", "stir"), 24),
        FeatureFamilyRule("inflation_gap", ("cpi", "infl", "core", "target"), 24),
        FeatureFamilyRule("labor_growth", ("unemp", "jobs", "labor", "labour", "gdp", "output"), 24, False),
        FeatureFamilyRule("front_end_market", ("yield", "curve", "bond", "fx"), 20),
        FeatureFamilyRule("financial_conditions", ("spread", "credit", "equity", "fincond"), 18),
    ),
}

FAST_MOVING_KEYWORDS = (
    "yield",
    "rate",
    "fx",
    "xrate",
    "equity",
    "stock",
    "spread",
    "credit",
    "oil",
    "gas",
    "commodity",
    "vol",
)

GLOBAL_CIDS = {"USD", "EUR", "GLB"}


def allowed_feature_score(model_type: str, ticker: str, country_code: str | None) -> tuple[bool, float, str]:
    """Score whether a JPMaQS ticker is eligible for a model."""
    cid, xcat = split_jpmaqs_ticker(ticker)
    if country_code and cid not in {country_code, *GLOBAL_CIDS}:
        return False, 0.0, "wrong_country"

    rules = MODEL_FAMILY_RULES.get(model_type, ())
    normalized = f"{ticker} {xcat}".lower()
    best_score = 0.0
    best_family = ""
    for rule in rules:
        if any(keyword in normalized for keyword in rule.keywords):
            score = 0.55
            if rule.prefer_fast and any(keyword in normalized for keyword in FAST_MOVING_KEYWORDS):
                score += 0.20
            if cid == country_code:
                score += 0.15
            elif cid in GLOBAL_CIDS:
                score += 0.05
            if score > best_score:
                best_score = score
                best_family = rule.name
    return best_score > 0.0, min(1.0, best_score), best_family


def split_jpmaqs_ticker(ticker: str) -> tuple[str, str]:
    """Split a JPMaQS ticker into cid and xcat-ish parts."""
    value = str(ticker or "").strip()
    if "_" not in value:
        return "", value
    cid, xcat = value.split("_", 1)
    return cid.upper(), xcat.upper()


def feature_family_for_ticker(model_type: str, ticker: str) -> str:
    normalized = ticker.lower()
    for rule in MODEL_FAMILY_RULES.get(model_type, ()):
        if any(re.search(rf"\b{re.escape(keyword)}", normalized) for keyword in rule.keywords):
            return rule.name
    return "unknown"

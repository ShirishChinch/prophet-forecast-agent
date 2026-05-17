"""Default numeric unstructured-evidence feature specs by template."""

from __future__ import annotations

from agents.evidence.schemas import EvidenceTemplateSpec, NumericFeatureSpec


def get_default_catalog() -> dict[str, EvidenceTemplateSpec]:
    """Return the built-in feature checklist library."""
    return {
        "sports": EvidenceTemplateSpec(
            template="sports",
            description="Team/player events where injury, lineup, rest, and sportsbook context can matter.",
            preferred_source_types=("official_injury_report", "team_report", "sportsbook_odds", "beat_report", "box_score"),
            features=(
                _f("sports__key_player_absence_score", "Total importance of relevant unavailable players on YES side minus NO side; negative hurts YES.", -1.0, 1.0, "higher supports YES"),
                _f("sports__minutes_or_usage_share_out", "Estimated share of minutes/usage unavailable on YES side minus opponent side.", -1.0, 1.0, "higher supports YES"),
                _f("sports__lineup_confirmation_score", "Confirmed lineup/starting roster advantage for YES side.", -1.0, 1.0, "higher supports YES"),
                _f("sports__rest_travel_advantage", "Rest/travel/schedule advantage for YES side.", -1.0, 1.0, "higher supports YES"),
                _f("sports__sportsbook_line_move", "Recent reputable sportsbook movement toward YES after market prior timestamp.", -1.0, 1.0, "higher supports YES"),
                _f("sports__motivation_or_incentive_score", "Playoff/motivation incentive advantage for YES side.", -1.0, 1.0, "higher supports YES"),
            ),
        ),
        "macro": EvidenceTemplateSpec(
            template="macro",
            description="Economic releases and central-bank decisions where consensus and official nowcasts matter.",
            preferred_source_types=("official_release_calendar", "economist_poll", "nowcast", "rates_market", "central_bank"),
            features=(
                _f("macro__consensus_gap_to_yes_threshold", "Consensus estimate minus YES threshold, normalized so positive supports YES.", -5.0, 5.0, "higher supports YES"),
                _f("macro__consensus_dispersion", "Dispersion/uncertainty around expert estimates.", 0.0, 1.0, "uncertainty"),
                _f("macro__official_nowcast_gap", "Official/credible nowcast minus YES threshold, normalized.", -5.0, 5.0, "higher supports YES"),
                _f("macro__market_implied_policy_gap", "OIS/FedWatch/rates-implied gap supporting YES.", -1.0, 1.0, "higher supports YES"),
                _f("macro__recent_indicator_momentum", "High-frequency public indicators supporting YES.", -1.0, 1.0, "higher supports YES"),
            ),
        ),
        "weather": EvidenceTemplateSpec(
            template="weather",
            description="Weather threshold events where official forecasts and model agreement matter.",
            preferred_source_types=("official_weather", "forecast_model", "radar", "historical_weather"),
            features=(
                _f("weather__official_forecast_gap", "Official forecast value minus threshold in natural units; positive supports YES for above-threshold events.", -50.0, 50.0, "higher supports YES"),
                _f("weather__forecast_model_agreement", "Agreement among credible forecast sources.", 0.0, 1.0, "confidence"),
                _f("weather__forecast_confidence", "Confidence stated or implied by forecast source.", 0.0, 1.0, "confidence"),
                _f("weather__recent_track_or_radar_shift", "Recent official/radar shift toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("weather__seasonal_base_rate", "Historical seasonal base rate for the condition.", 0.0, 1.0, "base_rate"),
            ),
        ),
        "politics": EvidenceTemplateSpec(
            template="politics",
            description="Political/election/policy events where polls, institutional constraints, and credible reporting matter.",
            preferred_source_types=("polling_average", "official_calendar", "court_docket", "bill_tracker", "credible_reporter"),
            features=(
                _f("politics__polling_or_vote_gap", "Polling/whip/vote-count gap toward YES.", -20.0, 20.0, "higher supports YES"),
                _f("politics__institutional_constraint_score", "Institutional constraints that make YES harder; positive supports YES after sign adjustment.", -1.0, 1.0, "higher supports YES"),
                _f("politics__credible_report_direction", "Credible reporter/source direction toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("politics__source_agreement", "Agreement among independent sources.", 0.0, 1.0, "confidence"),
                _f("politics__source_conflict", "Conflict/disagreement among sources.", 0.0, 1.0, "uncertainty"),
            ),
        ),
        "company": EvidenceTemplateSpec(
            template="company",
            description="Company/tech announcement and product-release events.",
            preferred_source_types=("official_blog", "filing", "earnings_call", "credible_leak", "app_store", "github"),
            features=(
                _f("company__official_signal_score", "Official company/filing signal toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("company__credible_leak_score", "Credible leak/reporting signal toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("company__release_cadence_support", "Historical release cadence/deadline support for YES.", -1.0, 1.0, "higher supports YES"),
                _f("company__product_surface_evidence", "App store, code, docs, or deployment evidence toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("company__deadline_pressure", "Whether timing before deadline is plausible.", -1.0, 1.0, "higher supports YES"),
            ),
        ),
        "culture": EvidenceTemplateSpec(
            template="culture",
            description="Awards, entertainment, charts, releases, and cultural outcomes.",
            preferred_source_types=("bookmaker_odds", "critic_predictions", "award_precursor", "chart_data", "industry_outlet"),
            features=(
                _f("culture__bookmaker_gap", "Bookmaker/odds-implied edge toward YES versus Kalshi prior.", -1.0, 1.0, "higher supports YES"),
                _f("culture__critic_consensus_direction", "Critic/expert consensus direction toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("culture__precursor_awards_score", "Precursor awards/nomination evidence toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("culture__chart_or_rank_momentum", "Chart/ranking/streaming momentum toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("culture__social_signal_strength", "Broad public/social signal; weak unless strong and fresh.", -1.0, 1.0, "higher supports YES"),
            ),
        ),
        "crypto_price": EvidenceTemplateSpec(
            template="crypto_price",
            description="Crypto price threshold events where spot, volatility, options/funding, and news matter.",
            preferred_source_types=("spot_price", "exchange_data", "options_data", "funding_rates", "market_news"),
            features=(
                _f("crypto__spot_distance_to_threshold", "Spot price minus threshold divided by threshold; positive supports YES for above-threshold events.", -1.0, 1.0, "higher supports YES"),
                _f("crypto__realized_volatility_score", "Recent realized volatility normalized.", 0.0, 1.0, "uncertainty"),
                _f("crypto__options_or_funding_signal", "Options/funding signal toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("crypto__etf_flow_or_macro_signal", "ETF flow or macro risk-on/off signal toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("crypto__market_dislocation_signal", "Liquidation/exchange outage/dislocation signal.", -1.0, 1.0, "higher supports YES"),
            ),
        ),
        "generic": EvidenceTemplateSpec(
            template="generic",
            description="Fallback for weird one-off events. Extract only repeated, measurable public evidence.",
            preferred_source_types=("official_source", "credible_news", "cross_market", "public_dataset"),
            features=(
                _f("generic__official_confirmation_score", "Official/public confirmation toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("generic__credible_news_direction", "Credible news direction toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("generic__cross_market_confirmation", "Related-market confirmation toward YES.", -1.0, 1.0, "higher supports YES"),
                _f("generic__evidence_conflict", "Evidence conflict/ambiguity.", 0.0, 1.0, "uncertainty"),
            ),
        ),
    }


def get_template_spec(template: str) -> EvidenceTemplateSpec:
    """Fetch a default template spec, falling back to generic."""
    catalog = get_default_catalog()
    return catalog.get(template, catalog["generic"])


def _f(
    name: str,
    description: str,
    min_value: float,
    max_value: float,
    direction: str,
) -> NumericFeatureSpec:
    return NumericFeatureSpec(
        name=name,
        description=description,
        direction=direction,
        min_value=min_value,
        max_value=max_value,
        default_value=0.0,
        historical_measurement="Must be reconstructable from timestamped public sources before as_of_time.",
        leakage_risk="medium",
    )


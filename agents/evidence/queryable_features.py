"""Runtime-queryable unstructured feature catalog.

This file is deliberately separate from the generic LLM evidence extractor.
These are concrete feature contracts: what to query, how to measure it, how to
avoid leakage, and whether it is worth backtesting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QueryableFeature:
    """One unstructured feature that can be extracted into a number."""

    template: str
    name: str
    priority: int
    description: str
    numeric_definition: str
    range_min: float
    range_max: float
    default_value: float
    direction: str
    runtime_queries: tuple[str, ...]
    preferred_sources: tuple[str, ...]
    source_freshness: str
    historical_backtest_plan: str
    leakage_controls: tuple[str, ...]
    expected_coverage: str
    expected_signal: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_queryable_feature_catalog() -> list[QueryableFeature]:
    """Return the v1 list of usable LLM-queryable features.

    Priority is local to each template. Lower is more important.
    """
    rows: list[QueryableFeature] = []

    rows.extend(
        [
            _f(
                "sports",
                "sports__sportsbook_prob_gap",
                1,
                "Difference between reputable sportsbook implied probability and Kalshi prior.",
                "sportsbook_implied_yes_probability - kalshi_prior, normalized to [-1, 1]",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{event teams players date} sportsbook odds moneyline spread total current",
                    "{event teams date} odds movement line history",
                ),
                ("DraftKings/FanDuel/BetMGM/Circa odds pages", "OddsAPI-style feeds", "odds comparison pages"),
                "minutes to hours; must be current at prediction time",
                "For past games, scrape timestamped odds snapshots if available; otherwise use archived odds feeds only.",
                ("Do not use final score pages.", "Require odds timestamp <= as_of_time.", "Remove sportsbook vig before comparing."),
                "high",
                "high",
            ),
            _f(
                "sports",
                "sports__key_player_absence_impact",
                2,
                "Estimated net impact of missing/questionable key players on YES side versus opponent.",
                "sum(player_importance * status_weight) for opponent injuries minus same for YES side; clamp [-1, 1]",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{teams date} injury report starters questionable out",
                    "{team player injury status expected lineup today",
                    "{player} usage rate minutes share value over replacement",
                ),
                ("official injury reports", "team beat reporters", "Rotowire/Underdog/NBA/NFL/MLB injury pages", "player usage/minutes pages"),
                "same day; lineups can change within minutes",
                "Backtest with archived injury reports/statuses plus pre-game player minutes/usage from prior games.",
                ("Only statuses published <= as_of_time.", "Use prior-season/current-to-date importance, not final-game stats."),
                "medium",
                "high",
            ),
            _f(
                "sports",
                "sports__confirmed_lineup_edge",
                3,
                "Whether confirmed starting lineup/roster news improves YES side relative to expectation.",
                "LLM maps lineup news to [-1, 1] using player importance known before event.",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{teams date} confirmed lineup starters",
                    "{team} starting lineup confirmed {date}",
                    "{event} lineup news beat reporter",
                ),
                ("official team lineups", "league gamebooks", "trusted beat reporters"),
                "minutes before event",
                "Backtest only where confirmed lineup timestamps exist before market snapshot.",
                ("Reject recap pages.", "Require source timestamp.", "Use only pre-event player importance."),
                "medium",
                "medium-high",
            ),
            _f(
                "sports",
                "sports__live_game_state_edge",
                4,
                "For in-play markets, live score/clock/game state edge versus current Kalshi price.",
                "model-free score state measure in [-1, 1], e.g. lead/time remaining/possession mapped by rules.",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{event} live score play by play clock",
                    "{league} live box score {teams date}",
                ),
                ("official live score/play-by-play", "ESPN/NBA/MLB/NHL gamecast"),
                "seconds to minutes",
                "Use play-by-play events with timestamps; join to Kalshi snapshots by time.",
                ("Never use final score after as_of_time.", "Require play timestamp <= as_of_time."),
                "medium",
                "high for live markets",
            ),
        ]
    )

    rows.extend(
        [
            _f(
                "crypto_price",
                "crypto__spot_threshold_gap",
                1,
                "Current spot price distance to the market threshold.",
                "(spot_price - threshold) / threshold; sign adjusted for above/below markets; clamp [-1, 1]",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{asset} current spot price Coinbase Binance Kraken",
                    "{asset} price now USD",
                ),
                ("Coinbase/Binance/Kraken public price APIs", "CoinMarketCap/CoinGecko"),
                "seconds to minutes",
                "Historical exchange candles are easy to join point-in-time to Kalshi snapshots.",
                ("Use candle close/open before as_of_time.", "Do not use later high/low for same interval."),
                "high",
                "high",
            ),
            _f(
                "crypto_price",
                "crypto__short_horizon_momentum",
                2,
                "Recent spot momentum over 5m/15m/1h relative to threshold direction.",
                "weighted return over 5m, 15m, 1h; sign adjusted; clamp [-1, 1]",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{asset} 1 minute candles historical",
                    "{asset} current price chart 5m 15m 1h",
                ),
                ("exchange candle APIs", "CoinGecko/Coinbase candles"),
                "minutes",
                "Fully backtestable from historical spot candles.",
                ("Only candles ending <= as_of_time.",),
                "high",
                "medium-high",
            ),
            _f(
                "crypto_price",
                "crypto__derivatives_pressure",
                3,
                "Funding/open-interest/liquidation pressure that can move near-term crypto thresholds.",
                "signed composite of funding z-score, OI change, liquidation imbalance; clamp [-1, 1]",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{asset} funding rate open interest liquidation data",
                    "{asset} futures funding open interest current",
                ),
                ("Coinglass", "exchange futures APIs", "Deribit/Binance futures data"),
                "minutes to hours",
                "Backtest from timestamped futures/funding/OI histories if available.",
                ("No forward funding intervals.", "Use only published/current values before as_of_time."),
                "medium",
                "medium",
            ),
        ]
    )

    rows.extend(
        [
            _f(
                "weather",
                "weather__official_forecast_threshold_gap",
                1,
                "Official forecast value versus market threshold.",
                "forecast_value - threshold in natural units, sign adjusted and scaled by typical error",
                -5.0,
                5.0,
                0.0,
                "positive supports YES",
                (
                    "{location date} official forecast high temperature rainfall snow",
                    "NWS forecast {location} {date} temperature precipitation",
                ),
                ("NOAA/NWS", "National Weather Service API", "official meteorological agencies"),
                "hours; update cycles matter",
                "Use archived NWS forecast grids/text products with issuance time <= as_of_time.",
                ("Require forecast issue time.", "Do not use observed weather after event."),
                "medium-high",
                "high",
            ),
            _f(
                "weather",
                "weather__forecast_revision_momentum",
                2,
                "Recent forecast revisions toward or away from YES.",
                "latest_forecast_gap - previous_forecast_gap, scaled and clamped [-1, 1]",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{location date} forecast previous update",
                    "NWS forecast discussion {location} trend warmer colder wetter drier",
                ),
                ("NWS forecast discussions", "forecast archive APIs"),
                "hours",
                "Backtest from sequential forecast updates.",
                ("Use only revisions known before as_of_time.",),
                "medium",
                "medium-high",
            ),
            _f(
                "weather",
                "weather__model_agreement_confidence",
                3,
                "Agreement across official/model forecasts.",
                "1 - normalized dispersion across forecasts; range [0, 1]",
                0.0,
                1.0,
                0.5,
                "confidence multiplier",
                (
                    "{location date} forecast model ensemble agreement",
                    "{location date} NBM GFS ECMWF forecast spread",
                ),
                ("NOAA/NBM", "weather model pages", "forecast discussions"),
                "hours",
                "Backtest if archived ensemble/model forecasts are stored.",
                ("Require issue time <= as_of_time.",),
                "medium",
                "medium",
            ),
        ]
    )

    rows.extend(
        [
            _f(
                "macro",
                "macro__consensus_threshold_gap",
                1,
                "Economist consensus versus the market threshold/range boundary.",
                "consensus - threshold, scaled by historical release surprise std; sign adjusted",
                -5.0,
                5.0,
                0.0,
                "positive supports YES",
                (
                    "{country indicator release date} economist consensus forecast",
                    "{indicator} consensus estimate {release date} Reuters Bloomberg Trading Economics",
                ),
                ("Reuters/Bloomberg/Trading Economics/Investing/Econoday consensus pages"),
                "daily to intraday before release",
                "Need stored pre-release consensus snapshots; current web alone is not enough for old releases.",
                ("Consensus timestamp must be <= as_of_time.", "Do not use actual release value."),
                "medium",
                "high if feed is available",
            ),
            _f(
                "macro",
                "macro__nowcast_threshold_gap",
                2,
                "Official or respected nowcast versus threshold.",
                "nowcast - threshold, scaled by historical error; sign adjusted",
                -5.0,
                5.0,
                0.0,
                "positive supports YES",
                (
                    "{indicator country date} nowcast latest",
                    "Cleveland Fed inflation nowcast latest CPI PCE",
                    "Atlanta Fed GDPNow latest GDP estimate",
                ),
                ("Cleveland Fed nowcast", "Atlanta Fed GDPNow", "central bank nowcasts"),
                "daily/weekly updates",
                "Backtest from nowcast release archives with publication dates.",
                ("Only nowcast vintage before as_of_time.",),
                "medium",
                "high for CPI/GDP families",
            ),
            _f(
                "macro",
                "macro__expert_revision_momentum",
                3,
                "Whether expert estimates have recently revised toward YES.",
                "latest_consensus - previous_consensus, scaled by surprise std; sign adjusted",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{indicator release date} forecast revised consensus previous",
                    "{indicator} economist poll revision latest previous",
                ),
                ("consensus forecast feeds", "economic calendar archives"),
                "daily",
                "Requires snapshots across time, not just final consensus.",
                ("Only compare two snapshots both <= as_of_time.",),
                "low-medium until feed exists",
                "medium",
            ),
            _f(
                "macro",
                "macro__rates_market_implied_gap",
                4,
                "Rates market-implied decision/release signal versus threshold.",
                "market-implied policy probability or yield move, sign adjusted to YES; clamp [-1, 1]",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "FedWatch probabilities current target rate meeting",
                    "OIS implied central bank rate decision {meeting date}",
                    "{country} front end yield change before release",
                ),
                ("CME FedWatch", "OIS/yield data", "central bank futures"),
                "minutes to daily",
                "Backtest from historical futures/OIS/yield data.",
                ("Use market data timestamp <= as_of_time.",),
                "medium",
                "medium-high for policy events",
            ),
        ]
    )

    rows.extend(
        [
            _f(
                "company",
                "company__official_artifact_signal",
                1,
                "Concrete official artifact suggesting a product/release/announcement will happen.",
                "LLM-scored artifact strength in [-1, 1] from docs, app store, filings, blog, changelog.",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{company product event} official blog docs changelog app store filing",
                    "site:{company_domain} {product keywords} launch release changelog",
                ),
                ("company blog/docs", "SEC filings", "app stores", "GitHub/changelog"),
                "hours to days",
                "Backtest by storing dated official artifacts before market snapshots.",
                ("Require page publish/update timestamp.", "Do not infer from post-deadline articles."),
                "medium",
                "medium-high",
            ),
            _f(
                "company",
                "company__credible_reporter_signal",
                2,
                "Credible reporting/leaks about the event.",
                "source-credibility-weighted direction in [-1, 1]",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{company event} reported expected launch before deadline",
                    "{company product} credible leak reporter",
                ),
                ("major tech/business outlets", "known beat reporters", "regulatory filings"),
                "hours to days",
                "Backtest with timestamped articles and a source reliability table.",
                ("Reject rumor reposts.", "Require publication timestamp <= as_of_time."),
                "medium",
                "medium",
            ),
        ]
    )

    rows.extend(
        [
            _f(
                "culture",
                "culture__bookmaker_or_expert_gap",
                1,
                "Bookmaker/expert-implied probability versus Kalshi prior.",
                "external_implied_yes_probability - kalshi_prior; clamp [-1, 1]",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{award event nominee} odds predictions",
                    "{contest event} expert predictions odds winner",
                ),
                ("bookmaker odds", "GoldDerby/expert panels", "industry prediction sites"),
                "daily to weekly",
                "Backtest from archived odds/prediction snapshots.",
                ("Require timestamped odds/predictions.", "Remove vig where relevant."),
                "medium",
                "high if odds feed exists",
            ),
            _f(
                "culture",
                "culture__precursor_awards_score",
                2,
                "Evidence from prior awards/precursors/critic groups.",
                "weighted precursor wins/nominations supporting YES in [0, 1] or signed [-1, 1] for head-to-head.",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{award nominee} precursor awards wins nominations",
                    "{movie artist nominee} guild awards critics awards",
                ),
                ("official award bodies", "industry databases", "critic association results"),
                "days to weeks",
                "Backtest from event calendars and precursor result dates.",
                ("Only precursor results known before as_of_time.",),
                "medium",
                "medium",
            ),
        ]
    )

    rows.extend(
        [
            _f(
                "generic",
                "generic__official_confirmation_signal",
                1,
                "Official source confirms, denies, schedules, or materially constrains the event.",
                "LLM maps official text to [-1, 1] with citation.",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{event title keywords} official announcement statement schedule",
                    "site:gov OR site:edu OR site:org {event keywords}",
                ),
                ("official pages", "government/court/regulatory pages", "company/league pages"),
                "minutes to days",
                "Backtest only for repeated event families with archived official pages.",
                ("Require official source timestamp <= as_of_time.",),
                "medium",
                "medium-high when source exists",
            ),
            _f(
                "generic",
                "generic__credible_news_consensus",
                2,
                "Consensus direction from credible independent news sources.",
                "source-quality-weighted direction in [-1, 1], with conflict penalty.",
                -1.0,
                1.0,
                0.0,
                "positive supports YES",
                (
                    "{event title keywords} latest news reported expected",
                    "{event keywords} Reuters AP Bloomberg official",
                ),
                ("Reuters/AP/Bloomberg/major outlets", "specialist beat outlets"),
                "hours to days",
                "Backtest requires archived timestamped articles and source lists.",
                ("Require publication timestamps.", "Penalize conflicting or single-source claims."),
                "medium",
                "medium",
            ),
        ]
    )

    return sorted(rows, key=lambda row: (row.template, row.priority, row.name))


def top_features_by_template(max_priority: int = 4) -> dict[str, list[QueryableFeature]]:
    """Return the highest-priority features grouped by template."""
    grouped: dict[str, list[QueryableFeature]] = {}
    for feature in get_queryable_feature_catalog():
        if feature.priority <= max_priority:
            grouped.setdefault(feature.template, []).append(feature)
    return grouped


def _f(
    template: str,
    name: str,
    priority: int,
    description: str,
    numeric_definition: str,
    range_min: float,
    range_max: float,
    default_value: float,
    direction: str,
    runtime_queries: tuple[str, ...],
    preferred_sources: tuple[str, ...],
    source_freshness: str,
    historical_backtest_plan: str,
    leakage_controls: tuple[str, ...],
    expected_coverage: str,
    expected_signal: str,
    notes: str = "",
) -> QueryableFeature:
    return QueryableFeature(
        template=template,
        name=name,
        priority=priority,
        description=description,
        numeric_definition=numeric_definition,
        range_min=range_min,
        range_max=range_max,
        default_value=default_value,
        direction=direction,
        runtime_queries=runtime_queries,
        preferred_sources=preferred_sources,
        source_freshness=source_freshness,
        historical_backtest_plan=historical_backtest_plan,
        leakage_controls=leakage_controls,
        expected_coverage=expected_coverage,
        expected_signal=expected_signal,
        notes=notes,
    )

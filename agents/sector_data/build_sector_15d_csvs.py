"""Build sector-specific 15-day empirical odds snapshot CSVs.

The builder emits raw observed market prices near day offsets before market
close. It does not smooth, impute, regress, or manufacture missing prices.
Kalshi rows come from local downloaded historical files. Polymarket rows are
only fetched for sectors whose local Kalshi sample is weak.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"
USER_AGENT = "ai-prophet-sector-dataset/0.1"


@dataclass(frozen=True)
class PricePoint:
    timestamp: datetime
    price: float


@dataclass(frozen=True)
class SectorConfig:
    name: str
    poly_tag_ids: tuple[int, ...]
    matcher: Callable[[str, str], bool]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/sector_15d")
    parser.add_argument(
        "--kalshi-data-dir",
        action="append",
        default=[],
        help="Local Kalshi folder with markets.csv and trades.csv. Repeatable.",
    )
    parser.add_argument("--day-start", type=int, default=15)
    parser.add_argument("--day-end", type=int, default=0)
    parser.add_argument("--target-tolerance-hours", type=float, default=18.0)
    parser.add_argument("--min-local-observed-before-polymarket", type=int, default=1000)
    parser.add_argument("--max-polymarket-markets-per-sector", type=int, default=250)
    parser.add_argument("--polymarket-scan-markets-per-tag", type=int, default=900)
    parser.add_argument("--sleep", type=float, default=0.04)
    parser.add_argument("--skip-polymarket", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    kalshi_dirs = args.kalshi_data_dir or [
        "data/kalshi_order_flow_labeled_no_llm",
        "data/kalshi_order_flow_historical",
        "data/kalshi_order_flow_global",
        "data/kalshi_sports_more",
    ]

    configs = sector_configs()
    local_rows = build_kalshi_sector_rows(
        data_dirs=[Path(path) for path in kalshi_dirs],
        configs=configs,
        day_start=args.day_start,
        day_end=args.day_end,
        tolerance_hours=args.target_tolerance_hours,
    )

    rows_by_sector: dict[str, list[dict[str, Any]]] = {config.name: [] for config in configs}
    for row in local_rows:
        rows_by_sector.setdefault(str(row["sector"]), []).append(row)

    polymarket_counts: dict[str, int] = {}
    if not args.skip_polymarket:
        for config in configs:
            observed_local = sum(1 for row in rows_by_sector.get(config.name, []) if int(row.get("has_observation") or 0) == 1)
            if observed_local >= args.min_local_observed_before_polymarket or not config.poly_tag_ids:
                continue
            print(f"[polymarket] {config.name}: local observed={observed_local}, fetching tags={config.poly_tag_ids}")
            markets = discover_polymarket_markets(
                config=config,
                max_markets=args.max_polymarket_markets_per_sector,
                scan_markets_per_tag=args.polymarket_scan_markets_per_tag,
                raw_path=raw_dir / f"polymarket_{config.name}_markets.jsonl",
                sleep=args.sleep,
            )
            poly_rows = build_polymarket_rows(
                sector=config.name,
                markets=markets,
                day_start=args.day_start,
                day_end=args.day_end,
                tolerance_hours=args.target_tolerance_hours,
                cache_dir=raw_dir / "polymarket_price_history" / config.name,
                sleep=args.sleep,
            )
            polymarket_counts[config.name] = len(poly_rows)
            rows_by_sector.setdefault(config.name, []).extend(poly_rows)

    summary_rows: list[dict[str, Any]] = []
    for config in configs:
        rows = rows_by_sector.get(config.name, [])
        frame = pd.DataFrame(rows)
        sector_dir = out_dir / config.name
        sector_dir.mkdir(parents=True, exist_ok=True)
        grid_path = sector_dir / f"{config.name}_15d_snapshot_grid.csv"
        observed_path = sector_dir / f"{config.name}_15d_observed_snapshots.csv"
        frame.to_csv(grid_path, index=False)
        observed = frame[frame["has_observation"] == 1].copy() if "has_observation" in frame.columns else frame
        observed.to_csv(observed_path, index=False)
        summary_rows.append(
            {
                "sector": config.name,
                "grid_rows": int(len(frame)),
                "observed_rows": int(len(observed)),
                "markets": int(frame["market_ticker"].nunique()) if "market_ticker" in frame.columns and not frame.empty else 0,
                "kalshi_rows": int((frame["source"] == "kalshi").sum()) if "source" in frame.columns else 0,
                "polymarket_rows": int((frame["source"] == "polymarket").sum()) if "source" in frame.columns else 0,
                "polymarket_rows_added": int(polymarket_counts.get(config.name, 0)),
                "grid_csv": str(grid_path),
                "observed_csv": str(observed_path),
            }
        )

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "day_start": args.day_start,
        "day_end": args.day_end,
        "target_tolerance_hours": args.target_tolerance_hours,
        "kalshi_data_dirs": kalshi_dirs,
        "min_local_observed_before_polymarket": args.min_local_observed_before_polymarket,
        "max_polymarket_markets_per_sector": args.max_polymarket_markets_per_sector,
        "sectors": summary_rows,
        "notes": [
            "Raw nearest observed prices only; no smoothing or synthetic filling.",
            "Polymarket is fetched only for sectors below the local-observed threshold.",
            "Rows are one binary outcome side, even for multi-outcome markets.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print(json.dumps(summary, indent=2))


def sector_configs() -> list[SectorConfig]:
    return [
        SectorConfig("sports_tennis", (864,), lambda text, ticker: _has_any(text, ticker, ("tennis", "itf", "wta", "atp"))),
        SectorConfig("sports_baseball", (100381, 102627), lambda text, ticker: _has_any(text, ticker, ("mlb", "baseball", "kxmlb"))),
        SectorConfig("sports_basketball", (745, 100254, 102037), lambda text, ticker: _has_any(text, ticker, ("nba", "wnba", "basketball", "ncaamb", "ncaawb"))),
        SectorConfig("sports_hockey", (102044, 104595), lambda text, ticker: _has_any(text, ticker, ("nhl", "hockey", "kxnhl"))),
        SectorConfig("sports_football", (450, 1453), lambda text, ticker: _has_any(text, ticker, ("nfl", "football", "ncaafb"))),
        SectorConfig("sports_soccer", (1,), lambda text, ticker: _has_any(text, ticker, ("soccer", "epl", "uefa", "champions league", "premier league", "laliga"))),
        SectorConfig("sports_golf", (1,), lambda text, ticker: _has_any(text, ticker, ("golf", "pga", "masters", "pgatour"))),
        SectorConfig("sports_combat", (104362,), lambda text, ticker: _has_any(text, ticker, ("ufc", "mma", "boxing", "fight"))),
        SectorConfig("crypto_price", (21, 1312, 235, 39, 102321, 102322), lambda text, ticker: _has_any(text, ticker, ("btc", "bitcoin", "eth", "ethereum", "crypto", "xrp", "solana"))),
        SectorConfig("weather", (84, 1474, 85, 102023), lambda text, ticker: _has_any(text, ticker, ("weather", "temperature", "hurricane", "rain", "snow", "storm"))),
        SectorConfig("macro", (101250, 102000, 101249, 101261, 101248, 100196, 101701, 702, 100328), lambda text, ticker: _has_any(text, ticker, ("cpi", "inflation", "gdp", "fed", "fomc", "unemployment", "jobs report", "payroll", "rate cut", "rate hike"))),
        SectorConfig("politics", (2, 188, 573, 339, 102786), lambda text, ticker: _has_any(text, ticker, ("election", "president", "senate", "congress", "trump", "biden", "democrat", "republican", "government"))),
        SectorConfig("financials", (604, 602, 101305, 102321, 102322), lambda text, ticker: _has_any(text, ticker, ("nasdaq", "s&p", "spx", "stock", "yield", "treasury", "oil", "gas price", "dollar", "above $", "below $"))),
        SectorConfig("culture", (596, 100, 53, 1000, 315, 101), lambda text, ticker: _has_any(text, ticker, ("eurovision", "survivor", "oscar", "emmy", "grammy", "box office", "album", "song", "movie", "music"))),
        SectorConfig("generic", (), lambda text, ticker: True),
    ]


def build_kalshi_sector_rows(
    *,
    data_dirs: list[Path],
    configs: list[SectorConfig],
    day_start: int,
    day_end: int,
    tolerance_hours: float,
) -> list[dict[str, Any]]:
    market_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for root in data_dirs:
        markets_path = root / "markets.csv"
        trades_path = root / "trades.csv"
        if markets_path.exists():
            market_frames.append(pd.read_csv(markets_path))
        if trades_path.exists():
            trade_frames.append(pd.read_csv(trades_path))
    if not market_frames or not trade_frames:
        return []

    markets = _normalize_columns(pd.concat(market_frames, ignore_index=True))
    trades = _normalize_columns(pd.concat(trade_frames, ignore_index=True))
    labels = _kalshi_labels(markets, configs)
    snapshots = _kalshi_trade_snapshots(trades)
    if labels.empty or snapshots.empty:
        return []

    snapshots = snapshots.sort_values(["market_ticker", "timestamp"])
    history_by_ticker = {
        ticker: [PricePoint(row.timestamp, float(row.prior)) for row in group.itertuples(index=False)]
        for ticker, group in snapshots.groupby("market_ticker")
    }

    rows: list[dict[str, Any]] = []
    for idx, market in enumerate(labels.itertuples(index=False), start=1):
        if idx == 1 or idx % 500 == 0:
            print(f"[kalshi] processing market {idx}/{len(labels)}")
        history = history_by_ticker.get(str(market.market_ticker), [])
        rows.extend(
            _rows_from_history(
                source="kalshi",
                sector=str(market.sector),
                subtype=str(market.subtype),
                platform_market_id=str(market.market_ticker),
                platform_event_id=str(market.event_ticker),
                market_ticker=str(market.market_ticker),
                title=str(market.title),
                outcome_name=str(market.outcome_name),
                outcome_index=0,
                resolved_yes=float(market.resolved_yes),
                close_time=market.close_time,
                history=history,
                day_start=day_start,
                day_end=day_end,
                tolerance_hours=tolerance_hours,
                extra={"token_id": "", "volume": market.volume},
            )
        )
    return rows


def discover_polymarket_markets(
    *,
    config: SectorConfig,
    max_markets: int,
    scan_markets_per_tag: int,
    raw_path: Path,
    sleep: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tag_id in config.poly_tag_ids:
        offset = 0
        while offset < scan_markets_per_tag:
            payload = _get_json(
                f"{POLY_GAMMA}/markets",
                {"limit": 100, "offset": offset, "tag_id": tag_id, "closed": "true", "related_tags": "true"},
            )
            markets = payload if isinstance(payload, list) else payload.get("markets", [])
            if not markets:
                break
            for market in markets:
                if not isinstance(market, dict):
                    continue
                market_id = str(market.get("id") or "")
                if not market_id or market_id in seen:
                    continue
                text = f"{market.get('question', '')} {market.get('slug', '')} {market.get('description', '')}".lower()
                if config.name.startswith("sports_") and config.name != "sports_tennis" and not config.matcher(text, ""):
                    continue
                seen.add(market_id)
                rows.append(market)
            offset += 100
            time.sleep(sleep)
        if len(rows) >= max_markets:
            break
    rows = sorted(rows, key=_polymarket_sort_time, reverse=True)[:max_markets]
    _write_jsonl(raw_path, rows)
    return rows


def build_polymarket_rows(
    *,
    sector: str,
    markets: list[dict[str, Any]],
    day_start: int,
    day_end: int,
    tolerance_hours: float,
    cache_dir: Path,
    sleep: float,
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for market_index, market in enumerate(markets, start=1):
        if market_index == 1 or market_index % 50 == 0:
            print(f"[polymarket] {sector} market {market_index}/{len(markets)}")
        close_time = _parse_datetime(
            market.get("gameStartTime")
            or market.get("startTime")
            or market.get("closedTime")
            or market.get("endDate")
            or market.get("endDateIso")
        )
        if close_time is None:
            continue
        created = _parse_datetime(market.get("acceptingOrdersTimestamp") or market.get("createdAt"))
        history_start = close_time - timedelta(days=day_start + 1)
        if created is not None and created > history_start:
            history_start = created - timedelta(hours=1)

        outcomes = _json_list(market.get("outcomes"))
        token_ids = _json_list(market.get("clobTokenIds"))
        final_prices = [_to_float(value) for value in _json_list(market.get("outcomePrices"))]
        if len(outcomes) != len(token_ids) or not outcomes:
            continue
        include_indexes = _polymarket_outcome_indexes(outcomes)
        for outcome_index in include_indexes:
            if outcome_index >= len(token_ids) or outcome_index >= len(outcomes):
                continue
            resolved_yes = _resolved_from_final_price(final_prices[outcome_index] if outcome_index < len(final_prices) else None)
            if resolved_yes is None:
                continue
            token_id = str(token_ids[outcome_index])
            history = _load_polymarket_history(
                token_id=token_id,
                start=history_start,
                end=close_time + timedelta(hours=1),
                cache_dir=cache_dir,
            )
            time.sleep(sleep)
            rows.extend(
                _rows_from_history(
                    source="polymarket",
                    sector=sector,
                    subtype=_polymarket_subtype(sector, market),
                    platform_market_id=str(market.get("id") or ""),
                    platform_event_id=str(market.get("eventSlug") or market.get("slug") or ""),
                    market_ticker=str(market.get("id") or ""),
                    title=str(market.get("question") or ""),
                    outcome_name=str(outcomes[outcome_index]),
                    outcome_index=int(outcome_index),
                    resolved_yes=resolved_yes,
                    close_time=close_time,
                    history=history,
                    day_start=day_start,
                    day_end=day_end,
                    tolerance_hours=tolerance_hours,
                    extra={"token_id": token_id, "volume": market.get("volume") or market.get("volumeNum")},
                )
            )
    return rows


def _rows_from_history(
    *,
    source: str,
    sector: str,
    subtype: str,
    platform_market_id: str,
    platform_event_id: str,
    market_ticker: str,
    title: str,
    outcome_name: str,
    outcome_index: int,
    resolved_yes: float,
    close_time: datetime,
    history: list[PricePoint],
    day_start: int,
    day_end: int,
    tolerance_hours: float,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day_before_close in range(day_start, day_end - 1, -1):
        target = close_time - timedelta(days=day_before_close)
        point = _nearest_point(history, target, tolerance_hours=tolerance_hours)
        if point is None:
            continue
        price = max(0.01, min(0.99, point.price))
        rows.append(
            {
                "source": source,
                "sector": sector,
                "subtype": subtype,
                "platform_market_id": platform_market_id,
                "platform_event_id": platform_event_id,
                "market_ticker": market_ticker,
                "title": title,
                "outcome_name": outcome_name,
                "outcome_index": outcome_index,
                "token_id": extra.get("token_id", ""),
                "close_time": close_time.isoformat(),
                "target_time": target.isoformat(),
                "observed_time": point.timestamp.isoformat(),
                "day_before_close": day_before_close,
                "has_observation": 1,
                "hours_from_target": abs((point.timestamp - target).total_seconds()) / 3600.0,
                "p_yes": price,
                "odds_cent": int(round(price * 100.0)),
                "resolved_yes": resolved_yes,
                "volume": extra.get("volume", ""),
            }
        )
    return rows


def _kalshi_labels(markets: pd.DataFrame, configs: list[SectorConfig]) -> pd.DataFrame:
    ticker_col = _first_existing(markets, ["ticker", "market_ticker"])
    result_col = _first_existing(markets, ["result", "resolution", "outcome", "settlement_value"])
    close_col = _first_existing(markets, ["expected_expiration_time", "close_time", "expiration_time", "latest_expiration_time"])
    if ticker_col is None or result_col is None or close_col is None:
        return pd.DataFrame()
    frame = markets.copy()
    frame["market_ticker"] = frame[ticker_col].astype(str)
    frame["event_ticker"] = frame["event_ticker"].astype(str) if "event_ticker" in frame.columns else ""
    frame["title"] = frame["title"].astype(str) if "title" in frame.columns else frame["market_ticker"]
    frame["outcome_name"] = frame["yes_sub_title"].astype(str) if "yes_sub_title" in frame.columns else ""
    frame["resolved_yes"] = frame[result_col].map(_resolved_yes)
    frame["close_time"] = pd.to_datetime(frame[close_col], utc=True, errors="coerce")
    volume_col = _first_existing(frame, ["volume_fp", "volume_24h_fp", "liquidity_dollars"])
    frame["volume"] = frame[volume_col].map(_to_float) if volume_col else None
    frame = frame.dropna(subset=["market_ticker", "resolved_yes", "close_time"])
    routed = frame.apply(lambda row: _route_kalshi(row, configs), axis=1, result_type="expand")
    frame["sector"] = routed[0]
    frame["subtype"] = routed[1]
    frame = frame.drop_duplicates("market_ticker")
    return frame[["market_ticker", "event_ticker", "title", "outcome_name", "resolved_yes", "close_time", "volume", "sector", "subtype"]]


def _kalshi_trade_snapshots(trades: pd.DataFrame) -> pd.DataFrame:
    ticker_col = _first_existing(trades, ["ticker", "market_ticker"])
    time_col = _first_existing(trades, ["created_time", "timestamp", "time"])
    price_col = _first_existing(trades, ["yes_price_dollars", "yes_price", "price", "price_dollars"])
    if ticker_col is None or time_col is None or price_col is None:
        return pd.DataFrame()
    frame = trades.copy()
    frame["market_ticker"] = frame[ticker_col].astype(str)
    frame["timestamp"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    frame["prior"] = frame[price_col].map(_to_probability)
    return frame[["market_ticker", "timestamp", "prior"]].dropna().drop_duplicates(["market_ticker", "timestamp", "prior"])


def _route_kalshi(row: pd.Series, configs: list[SectorConfig]) -> tuple[str, str]:
    ticker = str(row.get("market_ticker") or "").lower()
    text = f"{row.get('title', '')} {row.get('outcome_name', '')} {ticker}".lower()
    for config in configs:
        if config.name == "generic":
            continue
        if config.matcher(text, ticker):
            return config.name, _subtype_from_text(config.name, text, ticker)
    return "generic", "generic"


def _subtype_from_text(sector: str, text: str, ticker: str) -> str:
    if sector == "sports_tennis":
        if "wta" in text or "kxwtamatch" in ticker:
            return "wta"
        if "atp" in text or "kxatpmatch" in ticker:
            return "atp"
        if "itf" in text or "kxitf" in ticker:
            return "itf"
    if sector.startswith("sports_"):
        return sector.replace("sports_", "")
    if sector == "crypto_price":
        for asset in ("btc", "bitcoin", "eth", "ethereum", "xrp", "solana"):
            if asset in text:
                return {"bitcoin": "btc", "ethereum": "eth"}.get(asset, asset)
    return "all"


def _polymarket_subtype(sector: str, market: dict[str, Any]) -> str:
    text = f"{market.get('question', '')} {market.get('slug', '')}".lower()
    return _subtype_from_text(sector, text, "")


def _polymarket_outcome_indexes(outcomes: list[Any]) -> list[int]:
    labels = [str(outcome).strip().lower() for outcome in outcomes]
    if len(labels) == 2 and labels == ["yes", "no"]:
        return [0]
    return list(range(len(labels)))


def _load_polymarket_history(*, token_id: str, start: datetime, end: datetime, cache_dir: Path) -> list[PricePoint]:
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    path = cache_dir / f"{token_id}_{start_ts}_{end_ts}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            payload = _get_json(
                f"{POLY_CLOB}/prices-history",
                {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 60},
            )
        except Exception as exc:
            payload = {"history": [], "error": f"{type(exc).__name__}: {exc}"}
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    history = payload.get("history", []) if isinstance(payload, dict) else []
    points = [_price_point(row) for row in history]
    return sorted([point for point in points if point is not None], key=lambda point: point.timestamp)


def _nearest_point(history: list[PricePoint], target: datetime, *, tolerance_hours: float) -> PricePoint | None:
    if not history:
        return None
    best = min(history, key=lambda point: abs((point.timestamp - target).total_seconds()))
    if abs((best.timestamp - target).total_seconds()) <= tolerance_hours * 3600.0:
        return best
    return None


def _price_point(row: dict[str, Any]) -> PricePoint | None:
    ts = _parse_unix(row.get("t"))
    price = _to_probability(row.get("p"))
    if ts is None or price is None:
        return None
    return PricePoint(ts, price)


def _polymarket_sort_time(market: dict[str, Any]) -> datetime:
    return (
        _parse_datetime(market.get("gameStartTime") or market.get("startTime") or market.get("closedTime") or market.get("endDate"))
        or datetime(1970, 1, 1, tzinfo=UTC)
    )


def _has_any(text: str, ticker: str, tokens: tuple[str, ...]) -> bool:
    blob = f"{text} {ticker}".lower()
    return any(token in blob for token in tokens)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    return result


def _first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    full_url = url if not params else f"{url}?{urlencode({k: v for k, v in params.items() if v is not None})}"
    request = Request(full_url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in {400, 401, 403, 404}:
                raise
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
        except (URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"GET failed: {full_url}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=str))
            handle.write("\n")


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except ValueError:
            return []
    return []


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_unix(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (TypeError, ValueError, OSError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_probability(value: Any) -> float | None:
    number = _to_float(value)
    if number is None:
        return None
    if number > 1.0:
        number /= 100.0
    if 0.0 <= number <= 1.0:
        return number
    return None


def _resolved_from_final_price(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 0.99:
        return 1.0
    if value <= 0.01:
        return 0.0
    return None


def _resolved_yes(value: Any) -> float | None:
    text = str(value).strip().lower()
    if text == "yes":
        return 1.0
    if text == "no":
        return 0.0
    if text in {"1", "1.0", "true"}:
        return 1.0
    if text in {"0", "0.0", "false"}:
        return 0.0
    return None


if __name__ == "__main__":
    main()

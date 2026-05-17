"""Build a tennis-only 15-day pre-match odds snapshot CSV.

Rows are observed market prices for one outcome side near a target day offset
before scheduled match start. This intentionally does not smooth, regress, or
fill missing prices.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"
USER_AGENT = "ai-prophet-tennis-dataset/0.1"


@dataclass(frozen=True)
class PricePoint:
    timestamp: datetime
    price: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/tennis_15d")
    parser.add_argument("--max-polymarket-matches", type=int, default=1000)
    parser.add_argument("--polymarket-scan-markets", type=int, default=5000)
    parser.add_argument("--max-kalshi-pages", type=int, default=80)
    parser.add_argument("--day-start", type=int, default=15)
    parser.add_argument("--day-end", type=int, default=0)
    parser.add_argument("--target-tolerance-hours", type=float, default=18.0)
    parser.add_argument("--sleep", type=float, default=0.04)
    parser.add_argument("--observed-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    polymarket_markets = discover_polymarket_tennis_matches(
        max_matches=args.max_polymarket_matches,
        scan_markets=args.polymarket_scan_markets,
        raw_path=raw_dir / "polymarket_tennis_markets.jsonl",
        sleep=args.sleep,
    )
    kalshi_markets = discover_kalshi_tennis_match_sides(
        max_pages=args.max_kalshi_pages,
        raw_path=raw_dir / "kalshi_tennis_markets.jsonl",
        sleep=args.sleep,
    )

    rows: list[dict[str, Any]] = []
    rows.extend(
        build_polymarket_rows(
            polymarket_markets,
            day_start=args.day_start,
            day_end=args.day_end,
            tolerance_hours=args.target_tolerance_hours,
            emit_missing=not args.observed_only,
            cache_dir=raw_dir / "polymarket_price_history",
            sleep=args.sleep,
        )
    )
    rows.extend(
        build_kalshi_rows(
            kalshi_markets,
            day_start=args.day_start,
            day_end=args.day_end,
            tolerance_hours=args.target_tolerance_hours,
            emit_missing=not args.observed_only,
            cache_dir=raw_dir / "kalshi_trade_history",
            sleep=args.sleep,
        )
    )

    output = pd.DataFrame(rows)
    out_path = out_dir / ("tennis_15d_observed_snapshots.csv" if args.observed_only else "tennis_15d_snapshot_grid.csv")
    output.to_csv(out_path, index=False)
    observed_path = out_dir / "tennis_15d_observed_snapshots.csv"
    if not output.empty and "has_observation" in output.columns and not args.observed_only:
        output[output["has_observation"] == 1].to_csv(observed_path, index=False)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "day_start": args.day_start,
        "day_end": args.day_end,
        "target_tolerance_hours": args.target_tolerance_hours,
        "polymarket_match_markets": len(polymarket_markets),
        "kalshi_match_side_markets": len(kalshi_markets),
        "rows": int(len(output)),
        "observed_rows": int(output["has_observation"].sum()) if "has_observation" in output.columns else int(len(output)),
        "csv": str(out_path),
        "observed_csv": str(observed_path),
        "notes": [
            "No smoothing and no synthetic filling.",
            "Polymarket uses CLOB prices-history for each outcome token.",
            "Kalshi uses public historical trades for discovered tennis match-side markets.",
            "Rows are emitted only when an observed price is within tolerance of the target day offset.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def discover_polymarket_tennis_matches(
    *,
    max_matches: int,
    scan_markets: int,
    raw_path: Path,
    sleep: float,
) -> list[dict[str, Any]]:
    """Discover closed Polymarket tennis moneyline match markets."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    offset = 0
    while offset < scan_markets:
        payload = _get_json(
            f"{POLY_GAMMA}/markets",
            {
                "limit": 100,
                "offset": offset,
                "tag_id": 864,
                "closed": "true",
            },
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
            if not _is_polymarket_match_market(market):
                continue
            seen.add(market_id)
            rows.append(market)
        offset += 100
        time.sleep(sleep)

    rows = sorted(rows, key=_polymarket_sort_time, reverse=True)[:max_matches]
    _write_jsonl(raw_path, rows)
    return rows


def discover_kalshi_tennis_match_sides(
    *,
    max_pages: int,
    raw_path: Path,
    sleep: float,
) -> list[dict[str, Any]]:
    """Discover settled Kalshi tennis match-side markets from historical sports pages."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor: str | None = None
    for _page in range(max_pages):
        params: dict[str, Any] = {"limit": 1000, "status": "settled", "category": "Sports"}
        if cursor:
            params["cursor"] = cursor
        payload = _get_json(f"{KALSHI_BASE}/historical/markets", params)
        markets = payload.get("markets", [])
        for market in markets:
            ticker = str(market.get("ticker") or "")
            if not ticker or ticker in seen:
                continue
            if _is_kalshi_tennis_match_side(market):
                seen.add(ticker)
                rows.append(market)
        cursor = payload.get("cursor")
        if not cursor:
            break
        time.sleep(sleep)
    _write_jsonl(raw_path, rows)
    return rows


def build_polymarket_rows(
    markets: list[dict[str, Any]],
    *,
    day_start: int,
    day_end: int,
    tolerance_hours: float,
    emit_missing: bool,
    cache_dir: Path,
    sleep: float,
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for market_index, market in enumerate(markets, start=1):
        if market_index == 1 or market_index % 100 == 0:
            print(f"[polymarket] processing match {market_index}/{len(markets)}")
        match_start = _parse_datetime(
            market.get("gameStartTime")
            or market.get("startTime")
            or market.get("endDate")
            or market.get("endDateIso")
        )
        if match_start is None:
            continue
        market_created = _parse_datetime(market.get("acceptingOrdersTimestamp") or market.get("createdAt"))
        history_start = match_start - timedelta(days=day_start + 1)
        if market_created is not None and market_created > history_start:
            history_start = market_created - timedelta(hours=1)
        outcomes = _json_list(market.get("outcomes"))
        token_ids = _json_list(market.get("clobTokenIds"))
        final_prices = [_to_float(value) for value in _json_list(market.get("outcomePrices"))]
        if len(outcomes) != len(token_ids) or len(outcomes) < 2:
            continue
        primary_history = _load_polymarket_history(
            token_id=str(token_ids[0]),
            start=history_start,
            end=match_start + timedelta(hours=1),
            cache_dir=cache_dir,
        )
        time.sleep(sleep)
        histories = [
            primary_history,
            [
                PricePoint(point.timestamp, max(0.0, min(1.0, 1.0 - point.price)))
                for point in primary_history
            ],
        ]
        for idx, (outcome, token_id) in enumerate(zip(outcomes[:2], token_ids[:2])):
            resolved_yes = _resolved_from_final_price(final_prices[idx] if idx < len(final_prices) else None)
            if resolved_yes is None:
                continue
            rows.extend(
                _rows_from_history(
                    source="polymarket",
                    platform_market_id=str(market.get("id") or ""),
                    platform_event_id=str(market.get("conditionId") or market.get("id") or ""),
                    market_ticker=str(market.get("slug") or market.get("id") or ""),
                    title=str(market.get("question") or ""),
                    outcome_name=str(outcome),
                    outcome_index=idx,
                    resolved_yes=resolved_yes,
                    match_start=match_start,
                    history=histories[idx],
                    day_start=day_start,
                    day_end=day_end,
                    tolerance_hours=tolerance_hours,
                    emit_missing=emit_missing,
                    extra={
                        "series": _polymarket_series(market),
                        "token_id": str(token_id),
                        "volume": market.get("volumeNum") or market.get("volume"),
                    },
                )
            )
    return rows


def build_kalshi_rows(
    markets: list[dict[str, Any]],
    *,
    day_start: int,
    day_end: int,
    tolerance_hours: float,
    emit_missing: bool,
    cache_dir: Path,
    sleep: float,
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for market_index, market in enumerate(markets, start=1):
        if market_index == 1 or market_index % 50 == 0:
            print(f"[kalshi] processing side {market_index}/{len(markets)}")
        match_start = _parse_datetime(market.get("expected_expiration_time") or market.get("close_time"))
        if match_start is None:
            continue
        ticker = str(market.get("ticker") or "")
        resolved_yes = _resolved_yes(market.get("result"))
        if not ticker or resolved_yes is None:
            continue
        history = _load_kalshi_trade_history(ticker=ticker, cache_dir=cache_dir)
        time.sleep(sleep)
        rows.extend(
            _rows_from_history(
                source="kalshi",
                platform_market_id=ticker,
                platform_event_id=str(market.get("event_ticker") or ""),
                market_ticker=ticker,
                title=str(market.get("title") or ""),
                outcome_name=str(market.get("yes_sub_title") or _kalshi_outcome_from_title(market)),
                outcome_index=0,
                resolved_yes=resolved_yes,
                match_start=match_start,
                history=history,
                day_start=day_start,
                day_end=day_end,
                tolerance_hours=tolerance_hours,
                emit_missing=emit_missing,
                extra={
                    "series": _kalshi_series_from_ticker(ticker),
                    "token_id": "",
                    "volume": market.get("volume_fp") or market.get("volume_24h_fp"),
                },
            )
        )
    return rows


def _rows_from_history(
    *,
    source: str,
    platform_market_id: str,
    platform_event_id: str,
    market_ticker: str,
    title: str,
    outcome_name: str,
    outcome_index: int,
    resolved_yes: float,
    match_start: datetime,
    history: list[PricePoint],
    day_start: int,
    day_end: int,
    tolerance_hours: float,
    emit_missing: bool,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day_before_close in range(day_start, day_end - 1, -1):
        target = match_start - timedelta(days=day_before_close)
        point = _nearest_point(history, target, tolerance_hours=tolerance_hours)
        if point is None and not emit_missing:
            continue
        price = max(0.01, min(0.99, point.price)) if point is not None else None
        rows.append(
            {
                "source": source,
                "platform_market_id": platform_market_id,
                "platform_event_id": platform_event_id,
                "market_ticker": market_ticker,
                "title": title,
                "series": extra.get("series", ""),
                "outcome_name": outcome_name,
                "outcome_index": outcome_index,
                "token_id": extra.get("token_id", ""),
                "match_start": match_start.isoformat(),
                "target_time": target.isoformat(),
                "observed_time": point.timestamp.isoformat() if point is not None else "",
                "day_before_close": day_before_close,
                "has_observation": 1 if point is not None else 0,
                "hours_from_target": abs((point.timestamp - target).total_seconds()) / 3600.0 if point is not None else "",
                "p_yes": price if price is not None else "",
                "odds_cent": int(round(price * 100.0)) if price is not None else "",
                "resolved_yes": resolved_yes,
                "volume": extra.get("volume", ""),
            }
        )
    return rows


def _load_polymarket_history(
    *,
    token_id: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
) -> list[PricePoint]:
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    path = cache_dir / f"{token_id}_{start_ts}_{end_ts}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            payload = _get_json(
                f"{POLY_CLOB}/prices-history",
                {
                    "market": token_id,
                    "startTs": start_ts,
                    "endTs": end_ts,
                    "fidelity": 60,
                },
            )
        except Exception as exc:
            payload = {"history": [], "error": f"{type(exc).__name__}: {exc}"}
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    history = payload.get("history", []) if isinstance(payload, dict) else []
    return [_price_point(row) for row in history if _price_point(row) is not None]


def _load_kalshi_trade_history(*, ticker: str, cache_dir: Path) -> list[PricePoint]:
    path = cache_dir / f"{ticker}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        trades: list[dict[str, Any]] = []
        cursor: str | None = None
        for _page in range(3):
            params: dict[str, Any] = {"ticker": ticker, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            try:
                payload_page = _get_json(f"{KALSHI_BASE}/historical/trades", params)
            except Exception:
                break
            trades.extend(payload_page.get("trades", []))
            cursor = payload_page.get("cursor")
            if not cursor:
                break
        payload = {"trades": trades}
        path.write_text(json.dumps(payload, ensure_ascii=True, default=str), encoding="utf-8")
    rows = payload.get("trades", []) if isinstance(payload, dict) else []
    points: list[PricePoint] = []
    for row in rows:
        ts = _parse_datetime(row.get("created_time") or row.get("timestamp"))
        price = _to_probability(row.get("yes_price_dollars") or row.get("yes_price") or row.get("price"))
        if ts is not None and price is not None:
            points.append(PricePoint(ts, price))
    return sorted(points, key=lambda point: point.timestamp)


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


def _is_polymarket_match_market(market: dict[str, Any]) -> bool:
    question = str(market.get("question") or "").lower()
    slug = str(market.get("slug") or "").lower()
    description = str(market.get("description") or "").lower()
    sports_market_type = str(market.get("sportsMarketType") or "").lower()
    outcomes = _json_list(market.get("outcomes"))
    if len(outcomes) != 2:
        return False
    if outcomes == ["Yes", "No"]:
        return False
    if sports_market_type and sports_market_type != "moneyline":
        return False
    reject_text = f"{question} {slug}"
    if any(token in reject_text for token in ("set 1", "set 2", "set 3", "total", "o/u", "games", "spread")):
        return False
    if any(str(outcome).lower() in {"over", "under", "over 2.5", "under 2.5"} for outcome in outcomes):
        return False
    if description and "will resolve to" in description and "advances against" not in description:
        return False
    text = f"{question} {slug}"
    return bool(re.search(r"\bvs\.?\b| v ", text))


def _polymarket_sort_time(market: dict[str, Any]) -> datetime:
    return (
        _parse_datetime(
            market.get("gameStartTime")
            or market.get("startTime")
            or market.get("closedTime")
            or market.get("endDate")
        )
        or datetime(1970, 1, 1, tzinfo=UTC)
    )


def _is_kalshi_tennis_match_side(market: dict[str, Any]) -> bool:
    text = f"{market.get('ticker', '')} {market.get('event_ticker', '')} {market.get('title', '')}".lower()
    if not any(token in text for token in ("tennis", "itf", "wta", "atp")):
        return False
    if "set winner" in text or " set " in text:
        return False
    if str(market.get("result") or "").lower() not in {"yes", "no"}:
        return False
    return bool(re.search(r"\bmatch\??\b", text))


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
            time.sleep(0.35 * (attempt + 1))
        except (URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            time.sleep(0.35 * (attempt + 1))
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
    return None


def _polymarket_series(market: dict[str, Any]) -> str:
    events = market.get("events")
    if isinstance(events, list) and events:
        series = events[0].get("series") if isinstance(events[0], dict) else None
        if isinstance(series, list) and series and isinstance(series[0], dict):
            return str(series[0].get("slug") or series[0].get("ticker") or "")
    slug = str(market.get("slug") or "").lower()
    if slug.startswith("wta-"):
        return "wta"
    if slug.startswith("atp-"):
        return "atp"
    return ""


def _kalshi_series_from_ticker(ticker: str) -> str:
    return ticker.split("-")[0] if ticker else ""


def _kalshi_outcome_from_title(market: dict[str, Any]) -> str:
    title = str(market.get("title") or "")
    match = re.match(r"Will (.+?) win ", title)
    return match.group(1) if match else ""


if __name__ == "__main__":
    main()

"""Collect a live short-horizon cohort for queryable feature testing.

This is for the parallel LLM/search feature system. It snapshots markets that
are expected to resolve soon, attaches queryable feature contracts, and computes
directly available features where possible. Correlations are computed later
after the cohort resolves.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from agents.evidence.queryable_features import get_queryable_feature_catalog
from agents.order_flow.features import classify_market_template
from agents.order_flow.kalshi_client import KalshiPublicClient


THRESHOLD_RE = re.compile(r"-(?:T|B)(?P<threshold>\d+(?:\.\d+)?)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-hours", type=float, default=4.0)
    parser.add_argument("--max-per-category", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--out-dir", default="reports/live_feature_experiment")
    parser.add_argument("--min-volume", type=float, default=0.0)
    parser.add_argument("--prefer-liquid", action="store_true", default=True)
    parser.add_argument("--single-leg-only", action="store_true", help="Drop MVE/cross-category parlay-style markets.")
    args = parser.parse_args()

    now = datetime.now(UTC)
    end = now + timedelta(hours=args.horizon_hours)
    out_dir = Path(args.out_dir)
    run_dir = out_dir / now.strftime("%Y%m%d_%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    markets = _fetch_open_markets(args.max_pages)
    candidates = _build_candidates(markets, now, end, args.min_volume, args.single_leg_only)
    selected = _select_by_category(candidates, args.max_per_category, args.prefer_liquid)
    feature_rows = _build_feature_rows(selected, now)
    feature_plan = _feature_plan_by_template()

    _write_csv(run_dir / "live_events.csv", selected)
    _write_csv(run_dir / "live_feature_rows.csv", feature_rows)
    (run_dir / "feature_plan_by_template.json").write_text(json.dumps(feature_plan, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(_summary(now, end, markets, candidates, selected, run_dir), indent=2),
        encoding="utf-8",
    )

    # Keep a latest pointer for convenience.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    print((run_dir / "summary.json").read_text(encoding="utf-8"))


def _fetch_open_markets(max_pages: int) -> list[dict[str, Any]]:
    client = KalshiPublicClient(timeout=15)
    rows: list[dict[str, Any]] = []
    for params in (
        {"status": "open"},
        {"status": "open", "category": "Sports"},
        {"status": "open", "category": "Crypto"},
        {"status": "open", "category": "Financials"},
        {"status": "open", "category": "Economics"},
        {"status": "open", "category": "Weather"},
        {"status": "open", "category": "Politics"},
        {"status": "open", "category": "Entertainment"},
    ):
        try:
            rows.extend(client.paginated("/markets", params=params, item_key="markets", limit=1000, max_pages=max_pages))
        except Exception:
            continue
    return _dedupe(rows, "ticker")


def _build_candidates(
    markets: list[dict[str, Any]],
    now: datetime,
    end: datetime,
    min_volume: float,
    single_leg_only: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market in markets:
        finish = _finish_time(market)
        if finish is None or not (now <= finish <= end):
            continue
        prior, prior_source, spread = _market_prior(market)
        if prior is None:
            continue
        volume = _to_float(market.get("volume_fp") or market.get("volume") or 0.0)
        if volume < min_volume:
            continue
        title = str(market.get("title") or "")
        ticker = str(market.get("ticker") or market.get("market_ticker") or "")
        if single_leg_only and _looks_multileg(ticker, title):
            continue
        template = _route_template(title, ticker, str(market.get("category") or ""))
        rows.append(
            {
                "snapshot_time": now.isoformat(),
                "finish_time": finish.isoformat(),
                "hours_to_finish": (finish - now).total_seconds() / 3600.0,
                "template": template,
                "ticker": ticker,
                "event_ticker": market.get("event_ticker"),
                "title": title,
                "category": market.get("category"),
                "yes_sub_title": market.get("yes_sub_title"),
                "no_sub_title": market.get("no_sub_title"),
                "prior": prior,
                "prior_source": prior_source,
                "spread": spread,
                "yes_bid_dollars": market.get("yes_bid_dollars"),
                "yes_ask_dollars": market.get("yes_ask_dollars"),
                "last_price_dollars": market.get("last_price_dollars"),
                "volume_fp": volume,
                "volume_24h_fp": _to_float(market.get("volume_24h_fp") or 0.0),
                "open_interest_fp": _to_float(market.get("open_interest_fp") or 0.0),
                "liquidity_dollars": _to_float(market.get("liquidity_dollars") or 0.0),
                "close_time": market.get("close_time"),
                "expected_expiration_time": market.get("expected_expiration_time"),
                "expiration_time": market.get("expiration_time"),
                "status": market.get("status"),
            }
        )
    return rows


def _looks_multileg(ticker: str, title: str) -> bool:
    text = f"{ticker} {title}".lower()
    if ticker.upper().startswith(("KXMVE", "KXMVECROSSCATEGORY")):
        return True
    return title.count(",") >= 2 or text.count("yes ") >= 3 or text.count("no ") >= 3


def _route_template(title: str, ticker: str, category: str) -> str:
    template = classify_market_template(title, ticker)
    category_l = category.lower()
    text = f"{title} {ticker}".lower()
    sports_tokens = (
        "cleveland",
        "cincinnati",
        "chicago ws",
        "chicago c",
        "milwaukee",
        "minnesota",
        "houston",
        "texas",
        "atlanta",
        "boston",
        "new york y",
        "new york m",
        "seattle",
        "san diego",
        "los angeles d",
        "los angeles a",
        "tampa bay",
        "miami",
        "detroit",
        "toronto",
        "wins by over",
        "runs scored",
        "points scored",
        "fight ends",
        "round ",
    )
    if "weather" in category_l or any(token in text for token in ("temperature", "snow", "rain", "hurricane")):
        return "weather"
    if "sport" in category_l:
        return "sports"
    if "economic" in category_l:
        return "macro"
    if any(token in text for token in sports_tokens):
        return "sports"
    if "target price" in text or (" price:" in text and "$" in text):
        return "crypto_price"
    return template


def _select_by_category(
    candidates: list[dict[str, Any]],
    max_per_category: int,
    prefer_liquid: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    frame = pd.DataFrame(candidates)
    if frame.empty:
        return []
    if prefer_liquid:
        frame["quality_score"] = (
            frame["volume_fp"].astype(float)
            + frame["open_interest_fp"].astype(float)
            + 100.0 * frame["liquidity_dollars"].astype(float)
            - 1000.0 * frame["spread"].astype(float).fillna(1.0)
        )
    else:
        frame["quality_score"] = -frame["hours_to_finish"].astype(float)
    for _, group in frame.sort_values("quality_score", ascending=False).groupby("template", sort=True):
        selected.extend(group.head(max_per_category).drop(columns=["quality_score"]).to_dict(orient="records"))
    return sorted(selected, key=lambda row: (str(row["template"]), float(row["hours_to_finish"])))


def _build_feature_rows(events: list[dict[str, Any]], snapshot_time: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    crypto_prices = _load_crypto_prices(events, snapshot_time)
    catalog = _feature_names_by_template()
    for event in events:
        row: dict[str, Any] = {
            "snapshot_time": event["snapshot_time"],
            "finish_time": event["finish_time"],
            "template": event["template"],
            "ticker": event["ticker"],
            "event_ticker": event["event_ticker"],
            "title": event["title"],
            "prior": event["prior"],
        }
        for feature_name in catalog.get(str(event["template"]), []):
            row[feature_name] = 0.0
        if event["template"] == "crypto_price":
            row.update(_crypto_features(event, crypto_prices, snapshot_time))
        row["feature_extraction_status"] = _status_for_template(str(event["template"]))
        rows.append(row)
    return rows


def _crypto_features(
    event: dict[str, Any],
    prices: dict[str, pd.DataFrame],
    snapshot_time: datetime,
) -> dict[str, float]:
    asset = _asset_from_ticker(str(event["ticker"]))
    threshold = _threshold_from_ticker(str(event["ticker"]))
    if asset is None or threshold is None:
        return {}
    spot = _nearest_price(prices, asset, snapshot_time)
    if spot is None:
        return {}
    ret_5m = _return_since(prices, asset, snapshot_time, 5)
    ret_15m = _return_since(prices, asset, snapshot_time, 15)
    ret_60m = _return_since(prices, asset, snapshot_time, 60)
    return {
        "crypto__spot_threshold_gap": _clamp((spot - threshold) / threshold, -1.0, 1.0),
        "crypto__short_horizon_momentum": _weighted_momentum(ret_5m, ret_15m, ret_60m),
        "crypto__return_5m": ret_5m or 0.0,
        "crypto__return_15m": ret_15m or 0.0,
        "crypto__return_60m": ret_60m or 0.0,
    }


def _status_for_template(template: str) -> str:
    if template == "crypto_price":
        return "direct_features_computed_from_public_spot_candles_when_threshold_parseable"
    return "feature_plan_only_collect_sources_before_labeling"


def _feature_plan_by_template() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for feature in get_queryable_feature_catalog():
        grouped.setdefault(feature.template, []).append(feature.to_dict())
    return grouped


def _feature_names_by_template() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for feature in get_queryable_feature_catalog():
        grouped.setdefault(feature.template, []).append(feature.name)
    # Add direct crypto returns used in this experiment.
    grouped.setdefault("crypto_price", []).extend(["crypto__return_5m", "crypto__return_15m", "crypto__return_60m"])
    return grouped


def _finish_time(market: dict[str, Any]) -> datetime | None:
    # expected_expiration_time is the closest proxy for "will finish soon" on
    # multileg sports markets where close_time can be days after component games.
    for key in ("expected_expiration_time", "expiration_time", "latest_expiration_time", "close_time"):
        parsed = _parse_time(market.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _market_prior(market: dict[str, Any]) -> tuple[float | None, str, float | None]:
    bid = _prob(market.get("yes_bid_dollars") or market.get("yes_bid"))
    ask = _prob(market.get("yes_ask_dollars") or market.get("yes_ask"))
    if bid is not None and ask is not None and ask >= bid:
        return _clamp((bid + ask) / 2.0, 0.01, 0.99), "yes_bid_ask_midpoint", ask - bid
    last = _prob(market.get("last_price_dollars") or market.get("last_price"))
    if last is not None:
        return _clamp(last, 0.01, 0.99), "last_price", None
    return None, "missing", None


def _prob(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1.0:
        number /= 100.0
    if 0.0 <= number <= 1.0:
        return number
    return None


def _load_crypto_prices(events: list[dict[str, Any]], snapshot_time: datetime) -> dict[str, pd.DataFrame]:
    assets = sorted({asset for event in events if (asset := _asset_from_ticker(str(event["ticker"])))})
    prices: dict[str, pd.DataFrame] = {}
    for asset in assets:
        try:
            prices[asset] = _fetch_coinbase_candles(asset, snapshot_time - timedelta(hours=2), snapshot_time)
        except Exception:
            prices[asset] = pd.DataFrame(columns=["timestamp", "close"])
    return prices


def _asset_from_ticker(ticker: str) -> str | None:
    text = ticker.upper()
    if "BTC" in text:
        return "BTC"
    if "ETH" in text:
        return "ETH"
    return None


def _threshold_from_ticker(ticker: str) -> float | None:
    match = THRESHOLD_RE.search(ticker.upper())
    if not match:
        return None
    return _to_float(match.group("threshold"))


def _fetch_coinbase_candles(asset: str, start: datetime, end: datetime) -> pd.DataFrame:
    params = urlencode(
        {
            "start": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "end": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "granularity": 60,
        }
    )
    url = f"https://api.exchange.coinbase.com/products/{asset}-USD/candles?{params}"
    request = Request(url, headers={"User-Agent": "ai-prophet-live-feature-experiment/0.1"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    time.sleep(0.15)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=["timestamp", "close"])
    frame = pd.DataFrame(payload, columns=["epoch", "low", "high", "open", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["epoch"], unit="s", utc=True)
    return frame.sort_values("timestamp")[["timestamp", "close"]].reset_index(drop=True)


def _nearest_price(prices: dict[str, pd.DataFrame], asset: str, timestamp: datetime) -> float | None:
    frame = prices.get(asset)
    if frame is None or frame.empty:
        return None
    ts = pd.Timestamp(timestamp).tz_convert("UTC")
    eligible = frame[frame["timestamp"] <= ts]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1]["close"])


def _return_since(prices: dict[str, pd.DataFrame], asset: str, timestamp: datetime, minutes: int) -> float | None:
    current = _nearest_price(prices, asset, timestamp)
    previous = _nearest_price(prices, asset, timestamp - timedelta(minutes=minutes))
    if current is None or previous is None or previous <= 0:
        return None
    return (current - previous) / previous


def _weighted_momentum(return_5m: float | None, return_15m: float | None, return_60m: float | None) -> float:
    parts = [(0.5, return_5m), (0.3, return_15m), (0.2, return_60m)]
    total = sum(weight for weight, value in parts if value is not None)
    if total <= 0:
        return 0.0
    value = sum(weight * float(value) for weight, value in parts if value is not None) / total
    return _clamp(value * 100.0, -1.0, 1.0)


def _summary(
    now: datetime,
    end: datetime,
    markets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "snapshot_time": now.isoformat(),
        "horizon_end": end.isoformat(),
        "markets_fetched": len(markets),
        "candidates_in_horizon": len(candidates),
        "selected_events": len(selected),
        "selected_by_template": _counts(selected, "template"),
        "candidate_by_template": _counts(candidates, "template"),
        "run_dir": str(run_dir),
        "files": {
            "events": str(run_dir / "live_events.csv"),
            "feature_rows": str(run_dir / "live_feature_rows.csv"),
            "feature_plan": str(run_dir / "feature_plan_by_template.json"),
        },
        "next_step": "After finish_time passes, fetch resolved market labels and compute correlations against resolved_yes - prior.",
    }


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _dedupe(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        value = str(row.get(key) or "")
        if value and value in seen:
            continue
        if value:
            seen.add(value)
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


if __name__ == "__main__":
    main()

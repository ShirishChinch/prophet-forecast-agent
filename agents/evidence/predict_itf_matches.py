"""Predict clean Kalshi ITF tennis winner markets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from agents.data_sources.kalshi_itf import fetch_event_markets, fetch_itf_match_markets
from my_agent import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--max-markets", type=int, default=80)
    parser.add_argument("--event-ticker", action="append", default=[])
    parser.add_argument("--out", default="reports/itf_match_predictions.csv")
    args = parser.parse_args()

    markets: list[dict[str, Any]] = []
    for event_ticker in args.event_ticker:
        markets.extend(fetch_event_markets(event_ticker))
    if not markets:
        markets = fetch_itf_match_markets(horizon_hours=args.horizon_hours, max_pages=args.max_pages)
    markets = markets[: args.max_markets]

    rows = [_predict_market(market) for market in markets]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_path, rows)
    print(json.dumps({"n_predictions": len(rows), "out": str(out_path)}, indent=2))
    for row in rows[:30]:
        print(
            f"{row['finish_time']} | {row['ticker']} | prior={row['market_prior']:.3f} "
            f"p_yes={row['p_yes']:.3f} | {row['title']}"
        )


def _predict_market(market: dict[str, Any]) -> dict[str, Any]:
    bid = _probability(market.get("yes_bid_dollars") or market.get("yes_bid"))
    ask = _probability(market.get("yes_ask_dollars") or market.get("yes_ask"))
    last = _probability(market.get("last_price_dollars") or market.get("last_price"))
    if bid is not None and ask is not None and ask >= bid:
        prior = (bid + ask) / 2.0
    elif last is not None:
        prior = last
    else:
        prior = 0.5
    event = {
        "event_ticker": market.get("event_ticker"),
        "market_ticker": market.get("ticker"),
        "title": market.get("title"),
        "description": market.get("rules_secondary") or market.get("rules_primary") or market.get("title"),
        "rules": market.get("rules_primary") or market.get("rules_secondary"),
        "category": "Sports",
        "close_time": market.get("expected_expiration_time") or market.get("close_time") or market.get("expiration_time"),
        "outcomes": ["Yes", "No"],
        "best_bid": bid,
        "best_ask": ask,
        "last_price_dollars": last,
        "volume_fp": market.get("volume_fp"),
        "open_interest_fp": market.get("open_interest_fp"),
    }
    output = predict(event)
    return {
        "finish_time": market.get("expected_expiration_time") or market.get("finish_time") or market.get("close_time"),
        "event_ticker": market.get("event_ticker"),
        "ticker": market.get("ticker"),
        "yes_sub_title": market.get("yes_sub_title"),
        "title": market.get("title"),
        "market_prior": prior,
        "yes_bid": bid,
        "yes_ask": ask,
        "last_price": last,
        "volume_fp": _float(market.get("volume_fp")),
        "open_interest_fp": _float(market.get("open_interest_fp")),
        "p_yes": float(output["p_yes"]),
        "edge_vs_prior": float(output["p_yes"]) - prior,
        "rationale": output.get("rationale"),
    }


def _probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1.0:
        number /= 100.0
    if number < 0.0 or number > 1.0:
        return None
    return number


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

"""Download public Kalshi market/trade/candle data for residual backtests.

Examples:
    python -m agents.order_flow.download_kalshi_history --status settled --max-markets 500
    python -m agents.order_flow.download_kalshi_history --category Economics --days 90
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
from typing import Any

from agents.order_flow.kalshi_client import KalshiPublicClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/kalshi_order_flow")
    parser.add_argument("--category", default=None)
    parser.add_argument("--status", default="settled", choices=["open", "closed", "settled", "all"])
    parser.add_argument("--market-source", default="live", choices=["live", "historical", "both"])
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--max-markets", type=int, default=1000)
    parser.add_argument("--trade-pages-per-market", type=int, default=2)
    parser.add_argument("--global-trade-pages", type=int, default=0)
    parser.add_argument("--label-top-trade-tickers", type=int, default=0)
    parser.add_argument("--include-candles", action="store_true")
    parser.add_argument("--period-minutes", type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = KalshiPublicClient()

    params: dict[str, Any] = {"status": None if args.status == "all" else args.status}
    if args.category:
        params["category"] = args.category

    market_limit = min(max(args.max_markets, 1), 1000)
    max_pages = max(1, math.ceil(args.max_markets / market_limit))
    markets: list[dict[str, Any]] = []
    if args.market_source in {"live", "both"}:
        markets.extend(
            client.paginated(
                "/markets",
                params=params,
                item_key="markets",
                limit=market_limit,
                max_pages=max_pages,
            )
        )
    if args.market_source in {"historical", "both"} or (not markets and args.status in {"settled", "closed"}):
        markets.extend(client.paginated(
            "/historical/markets",
            params=params,
            item_key="markets",
            limit=market_limit,
            max_pages=max_pages,
        ))
    markets = _dedupe_rows(markets, key="ticker")
    markets = markets[: args.max_markets]

    _write_jsonl(out_dir / "markets.jsonl", markets)
    _write_csv(out_dir / "markets.csv", markets)

    tickers = [
        str(market.get("ticker") or market.get("market_ticker") or "")
        for market in markets
        if market.get("ticker") or market.get("market_ticker")
    ]
    trades: list[dict[str, Any]] = []
    if args.global_trade_pages > 0:
        try:
            trades.extend(
                client.paginated("/markets/trades", item_key="trades", limit=1000, max_pages=args.global_trade_pages)
            )
        except Exception:
            pass
        try:
            trades.extend(
                client.paginated(
                    "/historical/trades",
                    item_key="trades",
                    limit=1000,
                    max_pages=args.global_trade_pages,
                )
            )
        except Exception:
            pass

    if args.trade_pages_per_market > 0:
        for ticker in tickers:
            ticker_trades: list[dict[str, Any]] = []
            try:
                ticker_trades.extend(
                    client.paginated(
                        "/markets/trades",
                        params={"ticker": ticker},
                        item_key="trades",
                        limit=1000,
                        max_pages=args.trade_pages_per_market,
                    )
                )
            except Exception:
                pass
            try:
                ticker_trades.extend(
                    client.paginated(
                        "/historical/trades",
                        params={"ticker": ticker},
                        item_key="trades",
                        limit=1000,
                        max_pages=args.trade_pages_per_market,
                    )
                )
            except Exception:
                pass
            trades.extend(ticker_trades)
    trades = _dedupe_rows(trades, key="trade_id")

    if args.label_top_trade_tickers > 0 and trades:
        existing_tickers = {
            str(market.get("ticker") or market.get("market_ticker") or "")
            for market in markets
        }
        ranked_tickers = _rank_trade_tickers(trades)
        missing_tickers = [
            ticker for ticker in ranked_tickers
            if ticker and ticker not in existing_tickers
        ][: args.label_top_trade_tickers]
        fetched_labels: list[dict[str, Any]] = []
        for ticker in missing_tickers:
            market = _fetch_market_label(client, ticker)
            if market is not None:
                fetched_labels.append(market)
        if fetched_labels:
            markets = _dedupe_rows([*markets, *fetched_labels], key="ticker")

    _write_jsonl(out_dir / "trades.jsonl", trades)
    _write_csv(out_dir / "trades.csv", trades)
    _write_jsonl(out_dir / "markets.jsonl", markets)
    _write_csv(out_dir / "markets.csv", markets)

    if args.include_candles and tickers:
        end_ts = int(datetime.now(UTC).timestamp())
        start_ts = int((datetime.now(UTC) - timedelta(days=args.days)).timestamp())
        candles: list[dict[str, Any]] = []
        for batch in _chunks(tickers, 100):
            try:
                candles.extend(
                    client.get_candlesticks(
                        batch,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        period_interval=args.period_minutes,
                    )
                )
            except Exception:
                continue
        _write_jsonl(out_dir / "candlesticks.jsonl", candles)

    print(f"Downloaded {len(markets)} markets and {len(trades)} trades into {out_dir}")


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _dedupe_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        row_key = str(row.get(key) or "")
        if row_key and row_key in seen:
            continue
        if row_key:
            seen.add(row_key)
        output.append(row)
    return output


def _rank_trade_tickers(trades: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for trade in trades:
        ticker = str(trade.get("ticker") or trade.get("market_ticker") or "")
        if ticker:
            counts[ticker] = counts.get(ticker, 0) + 1
    return [
        ticker for ticker, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]


def _fetch_market_label(client: KalshiPublicClient, ticker: str) -> dict[str, Any] | None:
    for path in (f"/markets/{ticker}", f"/historical/markets/{ticker}"):
        try:
            payload = client.get_json(path)
        except Exception:
            continue
        market = payload.get("market")
        if isinstance(market, dict):
            return market
    return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=str))
            handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

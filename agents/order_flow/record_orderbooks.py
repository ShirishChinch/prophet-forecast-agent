"""Record current public Kalshi order books for future order-flow training.

This is the piece that creates true book-history from now onward. Kalshi can
serve current order books, but historical full-depth book snapshots are not the
same thing as candles/trades. If we want depth/imbalance history, we record it.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

from agents.order_flow.kalshi_client import KalshiPublicClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", default=[])
    parser.add_argument("--events-json", default=None, help="Optional Prophet events.json to extract market_ticker values from.")
    parser.add_argument("--out", default="data/kalshi_order_flow/orderbook_snapshots.jsonl")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--depth", type=int, default=100)
    args = parser.parse_args()

    tickers = list(args.tickers)
    if args.events_json:
        tickers.extend(_tickers_from_events(Path(args.events_json)))
    tickers = sorted({ticker for ticker in tickers if ticker})
    if not tickers:
        raise SystemExit("No tickers provided.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = KalshiPublicClient()

    with out_path.open("a", encoding="utf-8") as handle:
        for iteration in range(args.iterations):
            timestamp = datetime.now(UTC).isoformat()
            for ticker in tickers:
                row: dict[str, Any] = {"timestamp": timestamp, "market_ticker": ticker}
                try:
                    row["payload"] = client.get_orderbook(ticker, depth=args.depth)
                    row["ok"] = True
                except Exception as exc:
                    row["ok"] = False
                    row["error"] = f"{type(exc).__name__}: {exc}"
                handle.write(json.dumps(row, ensure_ascii=True, default=str))
                handle.write("\n")
            handle.flush()
            if iteration + 1 < args.iterations:
                time.sleep(args.interval_seconds)

    print(f"Wrote orderbook snapshots for {len(tickers)} tickers to {out_path}")


def _tickers_from_events(path: Path) -> list[str]:
    events = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        return []
    tickers: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        ticker = event.get("market_ticker") or event.get("event_ticker")
        if ticker:
            tickers.append(str(ticker))
    return tickers


if __name__ == "__main__":
    main()


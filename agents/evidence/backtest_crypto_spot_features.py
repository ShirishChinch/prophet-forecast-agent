"""Backtest queryable crypto spot features against Kalshi residuals.

This is the first real historical test of the new feature catalog. It builds:

    crypto__spot_threshold_gap = signed distance from current spot to threshold

using public exchange candles known at or before each Kalshi snapshot.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from agents.order_flow.features import build_order_flow_dataset


THRESHOLD_RE = re.compile(r"-(?:T|B)(?P<threshold>\d+(?:\.\d+)?)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/kalshi_order_flow_labeled_no_llm")
    parser.add_argument("--out-dir", default="reports/queryable_features")
    parser.add_argument("--max-markets", type=int, default=0)
    args = parser.parse_args()

    dataset = build_order_flow_dataset(args.data_dir)
    frame = dataset.X.reset_index(drop=True).join(dataset.meta.reset_index(drop=True))
    frame["resolved_yes"] = dataset.resolved_yes.reset_index(drop=True).astype(float)
    frame["residual_target"] = dataset.y.reset_index(drop=True).astype(float)
    frame = frame[frame["template_family"].eq("crypto_price")].copy()
    frame["asset"] = frame["market_ticker"].map(_asset_from_ticker)
    frame["threshold"] = frame["market_ticker"].map(_threshold_from_ticker)
    frame = frame.dropna(subset=["asset", "threshold", "timestamp", "prior", "resolved_yes", "residual_target"])

    if args.max_markets > 0:
        keep = set(frame["market_ticker"].drop_duplicates().head(args.max_markets))
        frame = frame[frame["market_ticker"].isin(keep)].copy()

    if frame.empty:
        raise SystemExit("No crypto threshold rows with parseable thresholds found.")

    prices = _load_prices(frame)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        spot = _nearest_price(prices, str(row["asset"]), pd.Timestamp(row["timestamp"]).to_pydatetime())
        if spot is None:
            continue
        momentum_5m = _return_since(prices, str(row["asset"]), pd.Timestamp(row["timestamp"]).to_pydatetime(), minutes=5)
        momentum_15m = _return_since(prices, str(row["asset"]), pd.Timestamp(row["timestamp"]).to_pydatetime(), minutes=15)
        momentum_60m = _return_since(prices, str(row["asset"]), pd.Timestamp(row["timestamp"]).to_pydatetime(), minutes=60)
        threshold = float(row["threshold"])
        signed_gap = (spot - threshold) / threshold
        signed_momentum = _weighted_momentum(momentum_5m, momentum_15m, momentum_60m)
        rows.append(
            {
                "market_ticker": row["market_ticker"],
                "timestamp": row["timestamp"],
                "asset": row["asset"],
                "threshold": threshold,
                "spot": spot,
                "prior": float(row["prior"]),
                "resolved_yes": float(row["resolved_yes"]),
                "residual_target": float(row["residual_target"]),
                "crypto__spot_threshold_gap": max(-1.0, min(1.0, signed_gap)),
                "crypto__return_5m": momentum_5m,
                "crypto__return_15m": momentum_15m,
                "crypto__return_60m": momentum_60m,
                "crypto__short_horizon_momentum": signed_momentum,
            }
        )

    features = pd.DataFrame(rows)
    if features.empty:
        raise SystemExit("No rows could be joined to spot prices.")

    corr = _correlations(
        features,
        [
            "crypto__spot_threshold_gap",
            "crypto__short_horizon_momentum",
            "crypto__return_5m",
            "crypto__return_15m",
            "crypto__return_60m",
        ],
    )
    report = {
        "features": [
            "crypto__spot_threshold_gap",
            "crypto__short_horizon_momentum",
            "crypto__return_5m",
            "crypto__return_15m",
            "crypto__return_60m",
        ],
        "n_rows": int(len(features)),
        "n_markets": int(features["market_ticker"].nunique()),
        "assets": sorted(str(asset) for asset in features["asset"].dropna().unique()),
        "correlations": corr,
        "note": "Spot candles are joined at or before Kalshi snapshot timestamps.",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "crypto_spot_feature_rows.csv", index=False)
    pd.DataFrame(corr).to_csv(out_dir / "crypto_spot_feature_correlations.csv", index=False)
    (out_dir / "crypto_spot_feature_backtest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _asset_from_ticker(ticker: str) -> str | None:
    text = str(ticker).upper()
    if "BTC" in text:
        return "BTC"
    if "ETH" in text:
        return "ETH"
    return None


def _threshold_from_ticker(ticker: str) -> float | None:
    match = THRESHOLD_RE.search(str(ticker).upper())
    if not match:
        return None
    try:
        return float(match.group("threshold"))
    except ValueError:
        return None


def _load_prices(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for asset, group in frame.groupby("asset"):
        start = pd.Timestamp(group["timestamp"].min()).to_pydatetime() - timedelta(hours=2)
        end = pd.Timestamp(group["timestamp"].max()).to_pydatetime() + timedelta(minutes=5)
        prices[str(asset)] = _fetch_coinbase_candles(str(asset), start, end)
    return prices


def _fetch_coinbase_candles(asset: str, start: datetime, end: datetime) -> pd.DataFrame:
    product = f"{asset}-USD"
    rows: list[list[float]] = []
    cursor = start.astimezone(UTC)
    end = end.astimezone(UTC)
    while cursor < end:
        chunk_end = min(cursor + timedelta(minutes=300), end)
        params = urlencode(
            {
                "start": cursor.isoformat().replace("+00:00", "Z"),
                "end": chunk_end.isoformat().replace("+00:00", "Z"),
                "granularity": 60,
            }
        )
        url = f"https://api.exchange.coinbase.com/products/{product}/candles?{params}"
        request = Request(url, headers={"User-Agent": "ai-prophet-research/0.1"})
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, list):
            rows.extend(payload)
        cursor = chunk_end
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "close"])
    data = pd.DataFrame(rows, columns=["epoch", "low", "high", "open", "close", "volume"])
    data["timestamp"] = pd.to_datetime(data["epoch"], unit="s", utc=True)
    data = data.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return data[["timestamp", "close"]].reset_index(drop=True)


def _nearest_price(prices: dict[str, pd.DataFrame], asset: str, timestamp: datetime) -> float | None:
    frame = prices.get(asset)
    if frame is None or frame.empty:
        return None
    ts = pd.Timestamp(timestamp).tz_convert("UTC") if pd.Timestamp(timestamp).tzinfo else pd.Timestamp(timestamp, tz="UTC")
    eligible = frame[frame["timestamp"] <= ts]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1]["close"])


def _return_since(prices: dict[str, pd.DataFrame], asset: str, timestamp: datetime, minutes: int) -> float | None:
    current = _nearest_price(prices, asset, timestamp)
    previous = _nearest_price(prices, asset, timestamp - timedelta(minutes=minutes))
    if current is None or previous is None or previous <= 0:
        return None
    return float((current - previous) / previous)


def _weighted_momentum(return_5m: float | None, return_15m: float | None, return_60m: float | None) -> float:
    parts = [
        (0.50, return_5m),
        (0.30, return_15m),
        (0.20, return_60m),
    ]
    total_weight = sum(weight for weight, value in parts if value is not None)
    if total_weight <= 0:
        return 0.0
    value = sum(weight * float(value) for weight, value in parts if value is not None) / total_weight
    # Crypto minute returns are small; scale to a useful feature range without
    # letting rare jumps dominate.
    return max(-1.0, min(1.0, value * 100.0))


def _correlations(frame: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        x = pd.to_numeric(frame[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "n": int(x.notna().sum()),
                "corr_with_resolved_yes": _safe_corr(x, frame["resolved_yes"]),
                "corr_with_residual_target": _safe_corr(x, frame["residual_target"]),
                "mean": float(x.mean()),
                "std": float(x.std()),
            }
        )
    return rows


def _safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    value = left.corr(pd.to_numeric(right, errors="coerce"))
    if value is None or np.isnan(value):
        return None
    return float(value)


if __name__ == "__main__":
    main()

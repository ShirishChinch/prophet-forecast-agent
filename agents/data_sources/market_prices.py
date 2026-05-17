"""Public fast-moving market feature collection."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


YAHOO_FAST_SERIES = {
    # Yahoo treasury index values are quoted as yield * 10.
    "us_10y_yield": {"symbol": "^TNX", "scale": 0.10},
    "us_5y_yield": {"symbol": "^FVX", "scale": 0.10},
    "wti_oil": {"symbol": "CL=F", "scale": 1.0},
    "gasoline": {"symbol": "RB=F", "scale": 1.0},
    "dollar_index": {"symbol": "DX-Y.NYB", "scale": 1.0},
    "sp500": {"symbol": "^GSPC", "scale": 1.0},
    "credit_risk_hyg": {"symbol": "HYG", "scale": 1.0},
}


def collect_fast_market_features(as_of: datetime | None = None) -> tuple[dict[str, float], dict[str, Any]]:
    """Collect deterministic public market features from Yahoo chart data."""
    _ = as_of
    features: dict[str, float] = {}
    status: dict[str, Any] = {}
    for prefix, config in YAHOO_FAST_SERIES.items():
        symbol = str(config["symbol"])
        scale = float(config["scale"])
        snapshot = _yahoo_snapshot(symbol=symbol, scale=scale)
        features.update(_flatten_snapshot(prefix, snapshot))
        status[prefix] = {
            "symbol": symbol,
            "status": snapshot.get("status"),
            "latest_timestamp": snapshot.get("latest_timestamp"),
        }

    if "us_10y_yield_latest" in features and "us_5y_yield_latest" in features:
        features["us_10y_5y_slope"] = features["us_10y_yield_latest"] - features["us_5y_yield_latest"]
    if "us_10y_yield_change_30d" in features and "us_5y_yield_change_30d" in features:
        features["us_10y_5y_slope_change_30d"] = (
            features["us_10y_yield_change_30d"] - features["us_5y_yield_change_30d"]
        )

    return features, status


@lru_cache(maxsize=64)
def _download_yahoo_chart(symbol: str, range_: str = "10y") -> dict[str, Any]:
    encoded = quote(symbol, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?range={range_}&interval=1d"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=12.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return {}


def _yahoo_snapshot(symbol: str, scale: float) -> dict[str, Any]:
    payload = _download_yahoo_chart(symbol)
    try:
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError):
        return {"status": "missing"}

    points = [
        (int(ts), float(close) * scale)
        for ts, close in zip(timestamps, closes, strict=False)
        if close is not None and math.isfinite(float(close))
    ]
    if not points:
        return {"status": "empty"}
    points.sort(key=lambda item: item[0])
    latest_ts, latest = points[-1]
    return {
        "status": "ok",
        "latest": latest,
        "change_7d": _change_by_index(points, latest, 5),
        "change_30d": _change_by_index(points, latest, 21),
        "change_90d": _change_by_index(points, latest, 63),
        "pct_change_7d": _pct_change_by_index(points, latest, 5),
        "pct_change_30d": _pct_change_by_index(points, latest, 21),
        "pct_change_90d": _pct_change_by_index(points, latest, 63),
        "latest_timestamp": latest_ts,
    }


def _flatten_snapshot(prefix: str, snapshot: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in (
        "latest",
        "change_7d",
        "change_30d",
        "change_90d",
        "pct_change_7d",
        "pct_change_30d",
        "pct_change_90d",
    ):
        value = snapshot.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            out[f"{prefix}_{key}"] = float(value)
    return out


def _change_by_index(points: list[tuple[int, float]], latest: float, periods: int) -> float | None:
    if len(points) <= periods:
        return None
    return latest - points[-1 - periods][1]


def _pct_change_by_index(points: list[tuple[int, float]], latest: float, periods: int) -> float | None:
    if len(points) <= periods:
        return None
    prior = points[-1 - periods][1]
    if prior == 0:
        return None
    return (latest / prior) - 1.0

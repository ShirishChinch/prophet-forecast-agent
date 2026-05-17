"""Historical public market feature panels for model training."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agents.data_sources.market_prices import YAHOO_FAST_SERIES, _download_yahoo_chart


def load_fast_market_feature_panel(
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    cache_dir: str | Path = "logs/public_market_cache",
) -> pd.DataFrame:
    """Load historical fast-market features as a daily panel.

    Yahoo chart data is used as a lightweight public source. The output columns
    intentionally match the live feature names from `market_prices.py`, with a
    `public__` prefix added by the dataset builder.
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    for prefix, config in YAHOO_FAST_SERIES.items():
        symbol = str(config["symbol"])
        scale = float(config["scale"])
        series = _load_symbol_series(prefix=prefix, symbol=symbol, scale=scale, cache_root=cache_root)
        if series.empty:
            continue
        parts.append(_make_features(prefix, series))
    if not parts:
        return pd.DataFrame()

    panel = pd.concat(parts, axis=1).sort_index()
    if "us_10y_yield_latest" in panel.columns and "us_5y_yield_latest" in panel.columns:
        panel["us_10y_5y_slope"] = panel["us_10y_yield_latest"] - panel["us_5y_yield_latest"]
    if "us_10y_yield_change_30d" in panel.columns and "us_5y_yield_change_30d" in panel.columns:
        panel["us_10y_5y_slope_change_30d"] = (
            panel["us_10y_yield_change_30d"] - panel["us_5y_yield_change_30d"]
        )

    if start_date is not None:
        panel = panel[panel.index >= pd.Timestamp(start_date.date())]
    if end_date is not None:
        panel = panel[panel.index <= pd.Timestamp(end_date.date())]
    return panel.ffill()


def _load_symbol_series(prefix: str, symbol: str, scale: float, cache_root: Path) -> pd.Series:
    cache_file = cache_root / f"{_safe_name(prefix)}.parquet"
    if cache_file.exists():
        frame = pd.read_parquet(cache_file)
        if {"date", "value"}.issubset(frame.columns):
            return pd.Series(frame["value"].to_numpy(), index=pd.to_datetime(frame["date"]), name=prefix)

    payload = _download_yahoo_chart(symbol)
    try:
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError):
        return pd.Series(dtype=float, name=prefix)

    rows: list[tuple[pd.Timestamp, float]] = []
    for ts, close in zip(timestamps, closes, strict=False):
        if close is None:
            continue
        try:
            date = pd.to_datetime(int(ts), unit="s").normalize()
            value = float(close) * scale
        except (TypeError, ValueError):
            continue
        rows.append((date, value))
    if not rows:
        return pd.Series(dtype=float, name=prefix)
    frame = pd.DataFrame(rows, columns=["date", "value"]).drop_duplicates("date", keep="last")
    frame = frame.sort_values("date")
    frame.to_parquet(cache_file, index=False)
    return pd.Series(frame["value"].to_numpy(), index=pd.to_datetime(frame["date"]), name=prefix)


def _make_features(prefix: str, series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(index=series.index)
    frame[f"{prefix}_latest"] = series
    for label, periods in (("7d", 5), ("30d", 21), ("90d", 63)):
        frame[f"{prefix}_change_{label}"] = series - series.shift(periods)
        frame[f"{prefix}_pct_change_{label}"] = (series / series.shift(periods)) - 1.0
    return frame


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


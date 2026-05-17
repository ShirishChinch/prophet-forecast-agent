"""Build order-flow residual-model features from Kalshi market data files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_COLUMNS = [
    "prior",
    "spread",
    "price_change_1",
    "price_change_3",
    "price_change_12",
    "price_change_24",
    "realized_vol_24",
    "volume",
    "volume_zscore_24",
    "open_interest",
    "open_interest_change_24",
    "trade_count_24",
]

LLM_CONTEXT_FEATURE_COLUMNS = [
    "llm_context_available",
    "llm_direction_score",
    "llm_evidence_strength",
    "llm_source_quality",
    "llm_freshness_score",
    "llm_cross_market_confirmation",
    "llm_stale_market_correction",
    "llm_near_resolution",
    "llm_already_priced_likelihood",
    "llm_temporal_leakage_risk",
    "llm_source_count",
    "llm_pre_trade_context",
    "llm_effective_direction_score",
]

TEMPLATE_FAMILIES = [
    "sports",
    "crypto_price",
    "financials",
    "macro",
    "politics",
    "weather",
    "culture",
    "generic",
]


@dataclass(frozen=True)
class OrderFlowDataset:
    """Supervised residual dataset."""

    X: pd.DataFrame
    y: pd.Series
    prior: pd.Series
    resolved_yes: pd.Series
    meta: pd.DataFrame


def build_order_flow_dataset(data_dir: str | Path, snapshot_stride: int = 1) -> OrderFlowDataset:
    """Create a residual target from downloaded Kalshi CSV/JSONL data.

    Target is `resolved_yes - prior`, so a model learns the correction to the
    market midpoint, not the absolute event probability.
    """
    root = Path(data_dir)
    markets = _read_any(root, ["markets.csv", "markets.jsonl"])
    trades = _read_any(root, ["trades.csv", "trades.jsonl"])
    candles = _candles_to_frame(_read_any(root, ["candlesticks.csv", "candlesticks.jsonl"]))

    if candles.empty:
        candles = _trades_to_snapshots(trades)
    if candles.empty:
        raise ValueError(f"No usable candles/trades found in {root}")

    markets = _normalize_columns(markets)
    candles = _normalize_columns(candles)
    candles = _with_market_ticker(candles)
    candles = _with_timestamp(candles)
    candles = _with_prior(candles)
    candles = candles.dropna(subset=["market_ticker", "timestamp", "prior"])

    if snapshot_stride > 1:
        candles = candles.sort_values(["market_ticker", "timestamp"])
        candles = candles.groupby("market_ticker", group_keys=False).apply(
            lambda frame: frame.iloc[::snapshot_stride],
            include_groups=False,
        )

    labels = _market_labels(markets)
    frame = candles.merge(labels, on="market_ticker", how="inner")
    frame = frame.dropna(subset=["resolved_yes", "prior"])
    if frame.empty:
        raise ValueError("No resolved labels could be joined to market snapshots.")

    frame = _add_features(frame)
    feature_columns = list(FEATURE_COLUMNS)
    frame = _merge_llm_context_features(root, frame)
    if any(column in frame.columns for column in LLM_CONTEXT_FEATURE_COLUMNS):
        for column in LLM_CONTEXT_FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.0
        feature_columns.extend(LLM_CONTEXT_FEATURE_COLUMNS)

    frame = frame.dropna(subset=feature_columns + ["resolved_yes", "prior"])
    X = frame[feature_columns].astype(float)
    prior = frame["prior"].astype(float).clip(0.01, 0.99)
    resolved_yes = frame["resolved_yes"].astype(float)
    y = resolved_yes - prior
    if "template_family" not in frame.columns:
        frame["template_family"] = frame.apply(
            lambda row: classify_market_template(
                str(row.get("title") or ""),
                str(row.get("market_ticker") or ""),
            ),
            axis=1,
        )
    meta_cols = [column for column in ["market_ticker", "timestamp", "title", "category", "template_family"] if column in frame.columns]
    return OrderFlowDataset(X=X, y=y, prior=prior, resolved_yes=resolved_yes, meta=frame[meta_cols].copy())


def classify_market_template(title: str, ticker: str = "") -> str:
    """Route a Kalshi market into a coarse trainable residual family."""
    text = f"{title} {ticker}".lower()
    if any(token in text for token in ("btc", "bitcoin", "eth", "ethereum", "crypto")):
        return "crypto_price"
    if any(token in text for token in ("cpi", "inflation", "gdp", "fed", "fomc", "unemployment", "jobs report", "payroll", "rate cut", "rate hike")):
        return "macro"
    if any(token in text for token in ("temperature", "hurricane", "rain", "snow", "weather", "high temp", "low temp")):
        return "weather"
    if any(token in text for token in ("election", "president", "senate", "congress", "trump", "biden", "democrat", "republican", "truth social")):
        return "politics"
    if any(token in text for token in ("eurovision", "survivor", "oscar", "emmy", "grammy", "box office", "album", "song", "movie")):
        return "culture"
    if any(token in text for token in ("nasdaq", "s&p", "spx", "stock", "treasury", "yield", "oil", "gas price", "dollar", "price at", "above $", "below $")):
        return "financials"
    if any(
        token in text
        for token in (
            "nba",
            "nfl",
            "mlb",
            "nhl",
            "ufc",
            "pga",
            "wta",
            "atp",
            "soccer",
            "football",
            "baseball",
            "basketball",
            "tennis",
            "winner",
            "match",
            "game",
            "spread",
            "total runs",
            "points",
        )
    ):
        return "sports"
    return "generic"


def _read_any(root: Path, names: list[str]) -> pd.DataFrame:
    for name in names:
        path = root / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".jsonl":
            return pd.read_json(path, lines=True)
    return pd.DataFrame()


def _merge_llm_context_features(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Merge optional LLM context features generated for suspicious trades.

    The enrichment file is optional. If absent, training remains the pure
    order-flow regression. Historical enrichment must be generated using only
    sources known before each row's timestamp; the leakage-risk feature lets us
    later filter weak rows.
    """
    path = root / "llm_context_features.csv"
    if not path.exists() or path.stat().st_size == 0:
        return frame

    llm_features = pd.read_csv(path)
    llm_features = _normalize_columns(llm_features)
    if "market_ticker" not in llm_features.columns or "timestamp" not in llm_features.columns:
        return frame

    llm_features["timestamp"] = pd.to_datetime(llm_features["timestamp"], utc=True, errors="coerce")
    keep = ["market_ticker", "timestamp"]
    for column in LLM_CONTEXT_FEATURE_COLUMNS:
        if column in llm_features.columns:
            llm_features[column] = llm_features[column].map(_to_float).fillna(0.0)
            keep.append(column)
    if len(keep) <= 2:
        return frame

    merged = frame.merge(
        llm_features[keep].drop_duplicates(["market_ticker", "timestamp"]),
        on=["market_ticker", "timestamp"],
        how="left",
    )
    for column in LLM_CONTEXT_FEATURE_COLUMNS:
        if column in merged.columns:
            merged[column] = merged[column].fillna(0.0)
    return merged


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in result.columns
    ]
    return result


def _candles_to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "candlesticks" not in raw.columns:
        return raw
    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        ticker = row.get("market_ticker") or row.get("ticker")
        candles = row.get("candlesticks")
        if not isinstance(candles, list):
            continue
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            flat = {"market_ticker": ticker}
            flat.update(_flatten_candle(candle))
            rows.append(flat)
    return pd.DataFrame(rows)


def _flatten_candle(candle: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in candle.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def _trades_to_snapshots(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    frame = _normalize_columns(trades)
    frame = _with_market_ticker(frame)
    frame = _with_timestamp(frame)
    price_col = _first_existing(frame, ["yes_price", "price", "price_dollars", "yes_price_dollars"])
    if price_col is None:
        return pd.DataFrame()
    frame["price"] = frame[price_col].map(_to_probability)
    volume_col = _first_existing(frame, ["count", "quantity", "contracts", "volume"])
    frame["volume"] = frame[volume_col].map(_to_float) if volume_col else 1.0
    return frame[["market_ticker", "timestamp", "price", "volume"]].dropna(subset=["price"])


def _with_market_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    ticker_col = _first_existing(result, ["market_ticker", "ticker", "market_id"])
    if ticker_col and ticker_col != "market_ticker":
        result["market_ticker"] = result[ticker_col]
    return result


def _with_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    ts_col = _first_existing(result, ["timestamp", "ts", "created_time", "created_ts", "end_period_ts", "time"])
    if ts_col is None:
        result["timestamp"] = pd.NaT
        return result
    values = result[ts_col]
    if pd.api.types.is_numeric_dtype(values):
        result["timestamp"] = pd.to_datetime(values, unit="s", utc=True, errors="coerce")
    else:
        result["timestamp"] = pd.to_datetime(values, utc=True, errors="coerce")
    return result


def _with_prior(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    bid_col = _first_existing(result, ["yes_bid_close_dollars", "yes_bid", "best_bid"])
    ask_col = _first_existing(result, ["yes_ask_close_dollars", "yes_ask", "best_ask"])
    price_col = _first_existing(result, ["price_close_dollars", "price", "yes_price", "last_price"])
    if bid_col and ask_col:
        result["prior"] = (result[bid_col].map(_to_probability) + result[ask_col].map(_to_probability)) / 2.0
        result["spread"] = (result[ask_col].map(_to_probability) - result[bid_col].map(_to_probability)).abs()
    elif price_col:
        result["prior"] = result[price_col].map(_to_probability)
        result["spread"] = 0.0
    else:
        result["prior"] = pd.NA
        result["spread"] = 0.0
    return result


def _market_labels(markets: pd.DataFrame) -> pd.DataFrame:
    if markets.empty:
        return pd.DataFrame(columns=["market_ticker", "resolved_yes"])
    frame = _with_market_ticker(_normalize_columns(markets))
    frame["resolved_yes"] = frame.apply(_resolved_yes_from_row, axis=1)
    keep = ["market_ticker", "resolved_yes"]
    for col in ["title", "category"]:
        if col in frame.columns:
            keep.append(col)
    labels = frame[keep].dropna(subset=["market_ticker", "resolved_yes"]).drop_duplicates("market_ticker")
    labels["template_family"] = labels.apply(
        lambda row: classify_market_template(
            str(row.get("title") or ""),
            str(row.get("market_ticker") or ""),
        ),
        axis=1,
    )
    return labels


def _resolved_yes_from_row(row: pd.Series) -> float | None:
    for col in ["settlement_value", "result", "resolved_outcome", "resolution", "outcome"]:
        if col not in row or pd.isna(row[col]):
            continue
        value = str(row[col]).strip().lower()
        if value in {"yes", "y", "true", "1", "yes_won", "settled_yes"}:
            return 1.0
        if value in {"no", "n", "false", "0", "no_won", "settled_no"}:
            return 0.0
    for col in ["yes_settlement_price", "settlement_price", "final_price"]:
        if col in row and not pd.isna(row[col]):
            prob = _to_probability(row[col])
            if prob is not None:
                return 1.0 if prob >= 0.5 else 0.0
    return None


def _add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["market_ticker", "timestamp"]).copy()
    result["volume"] = _numeric_or_default(result, ["volume_fp", "volume"], 0.0)
    result["open_interest"] = _numeric_or_default(result, ["open_interest_fp", "open_interest"], 0.0)
    grouped = result.groupby("market_ticker", group_keys=False)
    for lag in [1, 3, 12, 24]:
        result[f"price_change_{lag}"] = grouped["prior"].diff(lag).fillna(0.0)
    result["realized_vol_24"] = grouped["prior"].diff().rolling(24, min_periods=3).std().reset_index(level=0, drop=True).fillna(0.0)
    volume_mean = grouped["volume"].rolling(24, min_periods=3).mean().reset_index(level=0, drop=True)
    volume_std = grouped["volume"].rolling(24, min_periods=3).std().reset_index(level=0, drop=True)
    result["volume_zscore_24"] = ((result["volume"] - volume_mean) / volume_std.replace(0.0, pd.NA)).fillna(0.0)
    result["open_interest_change_24"] = grouped["open_interest"].diff(24).fillna(0.0)
    result["trade_count_24"] = grouped["prior"].rolling(24, min_periods=1).count().reset_index(level=0, drop=True)
    return result


def _numeric_or_default(frame: pd.DataFrame, candidates: list[str], default: float) -> pd.Series:
    col = _first_existing(frame, candidates)
    if col is None:
        return pd.Series(default, index=frame.index)
    return frame[col].map(_to_float).fillna(default)


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _to_probability(value: Any) -> float | None:
    number = _to_float(value)
    if number is None:
        return None
    if number > 1.0:
        number = number / 100.0
    if 0.0 <= number <= 1.0:
        return number
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

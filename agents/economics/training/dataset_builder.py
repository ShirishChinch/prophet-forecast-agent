"""Build compact supervised datasets from local JPMaQS parquet chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from agents.data_sources.market_history import load_fast_market_feature_panel
from agents.economics.feature_families import allowed_feature_score, split_jpmaqs_ticker


POLICY_RATE_CIDS = ("AUD", "DEM", "ESP", "GBP", "JPY", "USD", "CAD")
POLICY_RATE_XCAT_PREFERENCE = (
    "RPCBRATE_NSA",
    "EQXR_NSA",
    "ONRATE_NSA",
    "CASHRATE_NSA",
    "PCREDITGDP_SA",
    "PCREDITGDP_SJA_D1M1ML12",
    "GB02YYLD_NSA",
    "GB02YRYLD_NSA",
)


@dataclass(frozen=True)
class SupervisedDataset:
    """Feature matrix, target vector, and metadata."""

    X: pd.DataFrame
    y: pd.Series
    current_series: pd.Series
    target_series: pd.Series
    target_tickers: list[str]
    diagnostics: dict[str, object]


TARGET_XCATS = {
    "yield": {"GB10YYLD_NSA"},
    "inflation": {"CPIH_SA_P1M1ML12", "CPIC_SA_P1M1ML12"},
    "growth": {"RGDP_SA_P1Q1QL4", "RGDP_SA_P1Q1QL1AR"},
    "policy": {"GB02YYLD_NSA", "GB02YRYLD_NSA", "GB10YYLD_NSA"},
}


def build_supervised_dataset(
    *,
    data_dir: str | Path,
    model_type: str,
    horizon_days: int,
    sample_freq: str,
    max_features: int,
) -> SupervisedDataset:
    """Build a point-in-time supervised dataset for one model family."""
    root = Path(data_dir)
    catalogue = pd.read_csv(root / "jpmaqs_catalogue.csv")
    wide = load_wide_panel(root)
    target_tickers = choose_target_tickers(catalogue, model_type)
    if not target_tickers:
        raise ValueError(f"No target tickers found for model_type={model_type}")

    feature_names = choose_feature_names(
        catalogue=catalogue,
        model_type=model_type,
        target_tickers=target_tickers,
        max_features=max_features,
    )
    if not feature_names:
        raise ValueError(f"No feature tickers found for model_type={model_type}")

    missing_targets = [ticker for ticker in target_tickers if ticker not in wide.columns]
    target_tickers = [ticker for ticker in target_tickers if ticker in wide.columns]
    if not target_tickers:
        raise ValueError(f"Target tickers are absent from parquet data: {missing_targets}")

    feature_names = [name for name in feature_names if name in wide.columns]
    target_panel = wide[target_tickers].ffill()
    feature_panel = wide[feature_names].ffill()
    policy_rate_panel, policy_rate_sources = build_policy_rate_feature_panel(wide) if model_type == "policy" else (pd.DataFrame(index=wide.index), {})
    public_panel = load_fast_market_feature_panel(
        start_date=feature_panel.index.min().to_pydatetime(),
        end_date=feature_panel.index.max().to_pydatetime(),
    )
    sampled_index = pd.date_range(
        feature_panel.index.min(),
        feature_panel.index.max() - pd.Timedelta(days=horizon_days),
        freq=sample_freq,
    )
    sampled_index = sampled_index.intersection(feature_panel.index)

    rows = []
    targets = []
    current_values = []
    target_values = []
    row_index = []
    for ticker in target_tickers:
        current = target_panel[ticker].reindex(sampled_index, method="ffill")
        future_dates = sampled_index + pd.Timedelta(days=horizon_days)
        future = target_panel[ticker].reindex(future_dates, method="ffill")
        future.index = sampled_index
        y = future - current
        valid = y.notna() & current.notna()
        if valid.sum() < 20:
            continue
        cid, _ = split_jpmaqs_ticker(ticker)
        country_features = feature_panel.reindex(sampled_index).ffill()
        selected_for_cid = _select_country_features(feature_names, cid)
        if ticker not in selected_for_cid and ticker in feature_panel.columns:
            selected_for_cid = [ticker, *selected_for_cid]
        X_part = country_features[selected_for_cid].loc[valid].copy()
        X_part.columns = [f"jpmaqs__{col}__latest" for col in X_part.columns]
        if model_type == "policy" and not policy_rate_panel.empty:
            policy_part = policy_rate_panel.reindex(X_part.index, method="ffill")
            X_part = pd.concat([X_part, policy_part], axis=1)
        if not public_panel.empty:
            public_part = public_panel.reindex(X_part.index, method="ffill").copy()
            public_part.columns = [f"public__{col}" for col in public_part.columns]
            X_part = pd.concat([X_part, public_part], axis=1)
        X_part[f"country__{cid}"] = 1.0
        rows.append(X_part)
        targets.append(y.loc[valid])
        current_values.append(current.loc[valid])
        target_values.append(future.loc[valid])
        row_index.extend((date, ticker) for date in y.loc[valid].index)

    if not rows:
        raise ValueError(f"Not enough target rows for model_type={model_type}")

    X = pd.concat(rows, axis=0).fillna(0.0)
    y = pd.concat(targets, axis=0).astype(float)
    current_series = pd.concat(current_values, axis=0).astype(float)
    target_series = pd.concat(target_values, axis=0).astype(float)
    X.index = pd.MultiIndex.from_tuples(row_index, names=["real_date", "target_ticker"])
    y.index = X.index
    current_series.index = X.index
    target_series.index = X.index
    X = X.reindex(sorted(X.columns), axis=1)
    X = X.sort_index()
    y = y.reindex(X.index)
    current_series = current_series.reindex(X.index)
    target_series = target_series.reindex(X.index)

    return SupervisedDataset(
        X=X,
        y=y,
        current_series=current_series,
        target_series=target_series,
        target_tickers=target_tickers,
        diagnostics={
            "model_type": model_type,
            "data_dir": str(root),
            "n_rows": int(len(X)),
            "n_features": int(X.shape[1]),
            "n_public_features": int(0 if public_panel.empty else public_panel.shape[1]),
            "target_tickers": target_tickers,
            "missing_targets": missing_targets,
            "policy_rate_feature_sources": policy_rate_sources,
            "horizon_days": horizon_days,
            "sample_freq": sample_freq,
        },
    )


def load_wide_panel(data_dir: str | Path) -> pd.DataFrame:
    """Load chunked JPMaQS parquet files into a daily wide dataframe."""
    root = Path(data_dir)
    parts = []
    for path in sorted(root.glob("jpmaqs_chunk_*.parquet")):
        frame = pd.read_parquet(path)
        frame = standardize_long_frame(frame)
        parts.append(frame)
    if not parts:
        raise ValueError(f"No jpmaqs_chunk_*.parquet files found in {root}")
    long = pd.concat(parts, ignore_index=True)
    long = long.dropna(subset=["real_date", "ticker", "value"])
    long = long.drop_duplicates(["real_date", "ticker"], keep="last")
    wide = long.pivot(index="real_date", columns="ticker", values="value")
    wide = wide.sort_index()
    return wide


def standardize_long_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common JPMaQS output columns to real_date, ticker, value."""
    out = frame.copy()
    rename = {}
    for column in out.columns:
        lowered = str(column).lower()
        if lowered in {"real_date", "date", "realdate"}:
            rename[column] = "real_date"
        elif lowered == "ticker":
            rename[column] = "ticker"
        elif lowered == "value":
            rename[column] = "value"
    out = out.rename(columns=rename)
    if "ticker" not in out.columns and {"cid", "xcat"}.issubset(set(out.columns)):
        out["ticker"] = out["cid"].astype(str) + "_" + out["xcat"].astype(str)
    out = out[["real_date", "ticker", "value"]].copy()
    out["real_date"] = pd.to_datetime(out["real_date"])
    return out


def build_policy_rate_feature_panel(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build policy-rate proxy level and change features from available JPMaQS columns."""
    features = pd.DataFrame(index=wide.index)
    sources: dict[str, str] = {}
    available = set(str(column) for column in wide.columns)
    for cid in POLICY_RATE_CIDS:
        source = _choose_policy_rate_source(cid, available)
        if source is None:
            continue
        series = wide[source].ffill().astype(float)
        prefix = f"jpmaqs__{cid}_policy_rate_level"
        features[f"{prefix}__latest"] = series
        features[f"{prefix}__d1m"] = series - series.reindex(series.index - pd.DateOffset(months=1), method="ffill").to_numpy()
        features[f"{prefix}__d3m"] = series - series.reindex(series.index - pd.DateOffset(months=3), method="ffill").to_numpy()
        sources[cid] = source
    return features, sources


def _choose_policy_rate_source(cid: str, available_tickers: set[str]) -> str | None:
    for xcat in POLICY_RATE_XCAT_PREFERENCE:
        ticker = f"{cid}_{xcat}"
        if ticker in available_tickers:
            return ticker
    return None


def choose_target_tickers(catalogue: pd.DataFrame, model_type: str) -> list[str]:
    """Choose target series from the compact JPMaQS catalogue."""
    target_xcats = TARGET_XCATS[model_type]
    tickers = []
    for _, row in catalogue.iterrows():
        ticker = str(row["ticker"])
        xcat = str(row["xcat"]).upper()
        if xcat in target_xcats:
            tickers.append(ticker)
    return sorted(set(tickers))


def choose_feature_names(
    *,
    catalogue: pd.DataFrame,
    model_type: str,
    target_tickers: Iterable[str],
    max_features: int,
) -> list[str]:
    """Choose allowed feature tickers, capped by manual relevance score."""
    scored = []
    target_cids = {split_jpmaqs_ticker(ticker)[0] for ticker in target_tickers}
    for _, row in catalogue.iterrows():
        ticker = str(row["ticker"])
        cid, _ = split_jpmaqs_ticker(ticker)
        country = cid if cid in target_cids else None
        allowed, score, family = allowed_feature_score(model_type, ticker, country)
        if allowed:
            scored.append((score, family, ticker))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [ticker for _, _, ticker in scored[:max_features]]


def _select_country_features(feature_names: list[str], cid: str) -> list[str]:
    selected = []
    for name in feature_names:
        feature_cid, _ = split_jpmaqs_ticker(name)
        if feature_cid == cid or feature_cid in {"USD", "EUR"}:
            selected.append(name)
    if not selected:
        return feature_names
    return selected

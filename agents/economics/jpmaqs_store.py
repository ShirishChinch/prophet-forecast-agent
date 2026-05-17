"""Lightweight JPMaQS point-in-time feature store facade.

This module intentionally avoids importing pandas unless data access is needed.
The forecast agent can run without JPMaQS downloads or pandas installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any

from agents.economics.feature_families import allowed_feature_score


DEFAULT_JPMAQS_DIRS = (
    "jpmaqs_relevant_download",
    "jpmaqs_full_download",
    "../jpmaqs_relevant_download",
    "../jpmaqs_full_download",
    "../../jpmaqs_relevant_download",
    "../../jpmaqs_full_download",
)

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

_SNAPSHOT_CACHE: dict[tuple[str, str, str, str, int], JPMaQSFeatureSnapshot] = {}


@dataclass(frozen=True)
class JPMaQSFeatureSnapshot:
    """Compact feature vector plus diagnostics."""

    features: dict[str, float]
    data_quality: float
    selected_tickers: list[str]
    diagnostics: dict[str, Any]


class JPMaQSFeatureStore:
    """Read point-in-time JPMaQS features from local parquet/csv outputs."""

    def __init__(self, root: str | Path | None = None, max_raw_features: int = 600) -> None:
        self.root = _resolve_root(root)
        self.max_raw_features = max_raw_features

    def available(self) -> bool:
        """Return True when JPMaQS output files are present."""
        return self.root is not None and self.root.exists()

    def get_snapshot(
        self,
        *,
        model_type: str,
        country_code: str | None,
        as_of: datetime | None,
    ) -> JPMaQSFeatureSnapshot:
        """Return compact, causal features as of a forecast date."""
        if not self.available():
            return JPMaQSFeatureSnapshot(
                features={},
                data_quality=0.0,
                selected_tickers=[],
                diagnostics={"status": "jpmaqs_data_not_found"},
            )

        try:
            cache_key = (
                str(self.root),
                model_type,
                country_code or "",
                as_of.date().isoformat() if as_of else "",
                self.max_raw_features,
            )
            if cache_key in _SNAPSHOT_CACHE:
                return _SNAPSHOT_CACHE[cache_key]
            snapshot = self._read_snapshot(model_type=model_type, country_code=country_code, as_of=as_of)
            _SNAPSHOT_CACHE[cache_key] = snapshot
            return snapshot
        except Exception as exc:
            return JPMaQSFeatureSnapshot(
                features={},
                data_quality=0.0,
                selected_tickers=[],
                diagnostics={"status": "jpmaqs_read_failed", "error": f"{type(exc).__name__}: {exc}"},
            )

    def _read_snapshot(
        self,
        *,
        model_type: str,
        country_code: str | None,
        as_of: datetime | None,
    ) -> JPMaQSFeatureSnapshot:
        import pandas as pd  # type: ignore

        assert self.root is not None
        catalogue = self._read_catalogue(pd)
        if catalogue is None or "ticker" not in catalogue.columns:
            return JPMaQSFeatureSnapshot({}, 0.0, [], {"status": "catalogue_missing"})

        candidate_rows = []
        for row in catalogue.to_dict("records"):
            ticker = str(row.get("ticker") or "")
            allowed, score, family = allowed_feature_score(model_type, ticker, country_code)
            if allowed:
                candidate_rows.append((ticker, score, family))
        candidate_rows.sort(key=lambda item: item[1], reverse=True)
        selected = candidate_rows[: self.max_raw_features]
        selected_tickers = [ticker for ticker, _, _ in selected]
        policy_rate_sources = _choose_policy_rate_sources(catalogue, model_type)
        for ticker in policy_rate_sources.values():
            if ticker not in selected_tickers:
                selected_tickers.append(ticker)
        if not selected_tickers:
            return JPMaQSFeatureSnapshot({}, 0.0, [], {"status": "no_allowed_features"})

        raw = self._read_latest_values(pd, selected_tickers, as_of)
        features: dict[str, float] = {}
        for ticker, value in raw.items():
            try:
                features[f"jpmaqs__{ticker}__latest"] = float(value)
            except (TypeError, ValueError):
                continue
        if model_type == "policy" and policy_rate_sources:
            features.update(self._read_policy_rate_features(pd, policy_rate_sources, as_of))

        quality = min(0.70, len(features) / max(100.0, len(selected_tickers)))
        return JPMaQSFeatureSnapshot(
            features=features,
            data_quality=quality,
            selected_tickers=selected_tickers,
            diagnostics={
                "status": "ok",
                "root": str(self.root),
                "candidate_count": len(candidate_rows),
                "selected_count": len(selected_tickers),
                "feature_count": len(features),
                "policy_rate_feature_sources": policy_rate_sources,
            },
        )

    def _read_catalogue(self, pd: Any) -> Any:
        assert self.root is not None
        for name in ("jpmaqs_catalogue.csv", "catalogue.csv", "lightweight_catalogue.csv"):
            path = self.root / name
            if path.exists():
                return pd.read_csv(path)
        return None

    def _read_latest_values(self, pd: Any, tickers: list[str], as_of: datetime | None) -> dict[str, float]:
        assert self.root is not None
        parquet_files = sorted(self.root.glob("jpmaqs_chunk_*.parquet"))
        if not parquet_files:
            return {}

        selected = set(tickers)
        latest: dict[str, tuple[Any, float]] = {}
        as_of_date = as_of.date() if as_of else None
        for path in parquet_files:
            chunk = pd.read_parquet(path)
            chunk = _standardize_frame(chunk)
            if "ticker" not in chunk.columns or "value" not in chunk.columns or "real_date" not in chunk.columns:
                continue
            chunk = chunk[chunk["ticker"].isin(selected)]
            if as_of_date is not None:
                dates = pd.to_datetime(chunk["real_date"], errors="coerce").dt.date
                chunk = chunk[dates <= as_of_date]
            if chunk.empty:
                continue
            chunk = chunk.sort_values("real_date")
            for row in chunk.groupby("ticker", sort=False).tail(1).to_dict("records"):
                value = row.get("value")
                if value is None:
                    continue
                latest[str(row["ticker"])] = (row.get("real_date"), value)
        return {ticker: value for ticker, (_, value) in latest.items()}

    def _read_policy_rate_features(
        self,
        pd: Any,
        sources: dict[str, str],
        as_of: datetime | None,
    ) -> dict[str, float]:
        assert self.root is not None
        parquet_files = sorted(self.root.glob("jpmaqs_chunk_*.parquet"))
        if not parquet_files:
            return {}

        selected = set(sources.values())
        parts = []
        as_of_date = as_of.date() if as_of else None
        for path in parquet_files:
            chunk = pd.read_parquet(path)
            chunk = _standardize_frame(chunk)
            if "ticker" not in chunk.columns or "value" not in chunk.columns or "real_date" not in chunk.columns:
                continue
            chunk = chunk[chunk["ticker"].isin(selected)]
            if as_of_date is not None:
                dates = pd.to_datetime(chunk["real_date"], errors="coerce").dt.date
                chunk = chunk[dates <= as_of_date]
            if not chunk.empty:
                parts.append(chunk[["real_date", "ticker", "value"]])
        if not parts:
            return {}

        long = pd.concat(parts, ignore_index=True)
        long["real_date"] = pd.to_datetime(long["real_date"], errors="coerce")
        features: dict[str, float] = {}
        for cid, ticker in sources.items():
            series = (
                long[long["ticker"] == ticker]
                .dropna(subset=["real_date", "value"])
                .drop_duplicates("real_date", keep="last")
                .sort_values("real_date")
                .set_index("real_date")["value"]
                .astype(float)
            )
            if series.empty:
                continue
            latest_date = series.index.max()
            latest = float(series.loc[latest_date])
            one_month = latest_date - pd.DateOffset(months=1)
            three_months = latest_date - pd.DateOffset(months=3)
            lag_1m = series.reindex([one_month], method="ffill").iloc[0]
            lag_3m = series.reindex([three_months], method="ffill").iloc[0]
            prefix = f"jpmaqs__{cid}_policy_rate_level"
            features[f"{prefix}__latest"] = latest
            features[f"{prefix}__d1m"] = float(latest - lag_1m)
            features[f"{prefix}__d3m"] = float(latest - lag_3m)
        return features


def _resolve_root(root: str | Path | None) -> Path | None:
    explicit = root or os.environ.get("JPMAQS_DATA_DIR")
    candidates = [Path(explicit)] if explicit else [Path(value) for value in DEFAULT_JPMAQS_DIRS]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    return None


def _choose_policy_rate_sources(catalogue: Any, model_type: str) -> dict[str, str]:
    if model_type != "policy" or "ticker" not in catalogue.columns:
        return {}
    available = set(str(ticker) for ticker in catalogue["ticker"].dropna().tolist())
    sources: dict[str, str] = {}
    for cid in POLICY_RATE_CIDS:
        for xcat in POLICY_RATE_XCAT_PREFERENCE:
            ticker = f"{cid}_{xcat}"
            if ticker in available:
                sources[cid] = ticker
                break
    return sources


def _standardize_frame(frame: Any) -> Any:
    rename = {}
    for column in frame.columns:
        lowered = str(column).lower()
        if lowered in {"cid_xcat", "ticker"}:
            rename[column] = "ticker"
        elif lowered in {"real_date", "date"}:
            rename[column] = "real_date"
        elif lowered == "value":
            rename[column] = "value"
    out = frame.rename(columns=rename)
    if "ticker" not in out.columns and {"cid", "xcat"}.issubset(set(out.columns)):
        out = out.copy()
        out["ticker"] = out["cid"].astype(str) + "_" + out["xcat"].astype(str)
    return out

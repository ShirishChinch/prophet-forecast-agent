"""Label LLM-discovered feature vectors and compute correlations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.order_flow.kalshi_client import KalshiPublicClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--min-n", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    feature_path = Path(args.features) if args.features else run_dir / "live_llm_discovered_feature_vectors.csv"
    if not feature_path.exists() or feature_path.stat().st_size == 0:
        raise SystemExit(f"Missing discovered features: {feature_path}")

    long = pd.read_csv(feature_path)
    labels = _fetch_labels(long["ticker"].dropna().astype(str).unique().tolist())
    labeled = long.merge(labels, on="ticker", how="left").dropna(subset=["resolved_yes"])
    if labeled.empty:
        raise SystemExit("No discovered-feature markets are resolved yet.")

    labeled["prior"] = pd.to_numeric(labeled["prior"], errors="coerce").clip(0.01, 0.99)
    labeled["resolved_yes"] = pd.to_numeric(labeled["resolved_yes"], errors="coerce")
    labeled["residual_target"] = labeled["resolved_yes"] - labeled["prior"]
    labeled["feature_value"] = pd.to_numeric(labeled["feature_value"], errors="coerce").fillna(0.0)
    labeled["confidence"] = pd.to_numeric(labeled.get("confidence", 0.0), errors="coerce").fillna(0.0)
    labeled["temporal_leakage_risk"] = pd.to_numeric(labeled.get("temporal_leakage_risk", 1.0), errors="coerce").fillna(1.0)
    labeled["quality_weighted_value"] = labeled["feature_value"] * labeled["confidence"] * (1.0 - labeled["temporal_leakage_risk"])

    corr_rows = _correlations(labeled, args.min_n)
    labeled_path = run_dir / "live_llm_discovered_feature_vectors_labeled.csv"
    corr_path = run_dir / "live_llm_discovered_feature_correlations.csv"
    report_path = run_dir / "live_llm_discovered_feature_label_report.json"
    labeled.to_csv(labeled_path, index=False)
    _write_csv(corr_path, corr_rows)

    report = {
        "run_dir": str(run_dir),
        "n_labeled_long_rows": int(len(labeled)),
        "n_labeled_markets": int(labeled["ticker"].nunique()),
        "min_n": args.min_n,
        "top_features_by_abs_residual_corr": sorted(
            corr_rows,
            key=lambda row: abs(float(row.get("corr_with_residual_target") or 0.0)),
            reverse=True,
        )[:50],
        "files": {
            "labeled_rows": str(labeled_path),
            "correlations": str(corr_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _fetch_labels(tickers: list[str]) -> pd.DataFrame:
    client = KalshiPublicClient(timeout=15)
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        market = _fetch_market(client, ticker)
        result = str((market or {}).get("result") or "").strip().lower()
        rows.append(
            {
                "ticker": ticker,
                "result": result or None,
                "resolved_yes": _resolved_yes(result, market or {}),
                "label_status": (market or {}).get("status"),
                "settlement_ts": (market or {}).get("settlement_ts"),
            }
        )
    return pd.DataFrame(rows)


def _fetch_market(client: KalshiPublicClient, ticker: str) -> dict[str, Any] | None:
    for path in (f"/markets/{ticker}", f"/historical/markets/{ticker}"):
        try:
            payload = client.get_json(path)
        except Exception:
            continue
        market = payload.get("market")
        if isinstance(market, dict):
            return market
    return None


def _resolved_yes(result: str, market: dict[str, Any]) -> float | None:
    if result in {"yes", "y", "true", "1"}:
        return 1.0
    if result in {"no", "n", "false", "0"}:
        return 0.0
    settlement = market.get("settlement_value_dollars") or market.get("settlement_value")
    try:
        value = float(settlement)
    except (TypeError, ValueError):
        return None
    if value >= 0.99:
        return 1.0
    if value <= 0.01:
        return 0.0
    return None


def _correlations(frame: pd.DataFrame, min_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (template, feature), group in frame.groupby(["template", "feature_name"]):
        if len(group) < min_n:
            continue
        for value_col in ("feature_value", "quality_weighted_value"):
            x = pd.to_numeric(group[value_col], errors="coerce").fillna(0.0)
            if x.nunique(dropna=False) <= 1:
                continue
            rows.append(
                {
                    "template": template,
                    "feature_name": feature,
                    "value_column": value_col,
                    "n": int(len(group)),
                    "nonzero_share": float((x.abs() > 1e-12).mean()),
                    "mean_value": float(x.mean()),
                    "mean_confidence": float(group["confidence"].mean()),
                    "mean_temporal_leakage_risk": float(group["temporal_leakage_risk"].mean()),
                    "corr_with_resolved_yes": _safe_corr(x, group["resolved_yes"]),
                    "corr_with_residual_target": _safe_corr(x, group["residual_target"]),
                }
            )
    return rows


def _safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    value = left.corr(pd.to_numeric(right, errors="coerce"))
    if value is None or np.isnan(value):
        return None
    return float(value)


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


if __name__ == "__main__":
    main()

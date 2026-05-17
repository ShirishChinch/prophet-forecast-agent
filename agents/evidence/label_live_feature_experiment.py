"""Label a live feature experiment after markets resolve and compute correlations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.order_flow.kalshi_client import KalshiPublicClient


META_COLUMNS = {
    "snapshot_time",
    "finish_time",
    "template",
    "ticker",
    "event_ticker",
    "title",
    "prior",
    "feature_extraction_status",
    "resolved_yes",
    "result",
    "residual_target",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--base-dir", default="reports/live_feature_experiment")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run(Path(args.base_dir))
    feature_path = run_dir / "live_feature_rows.csv"
    if not feature_path.exists():
        raise SystemExit(f"Missing feature rows: {feature_path}")

    features = pd.read_csv(feature_path)
    labels = _fetch_labels(features["ticker"].dropna().astype(str).unique().tolist())
    labeled = features.merge(labels, on="ticker", how="left")
    labeled = labeled.dropna(subset=["resolved_yes"])
    if labeled.empty:
        raise SystemExit("No selected markets are resolved yet. Run this after finish_time/settlement.")

    labeled["prior"] = pd.to_numeric(labeled["prior"], errors="coerce").clip(0.01, 0.99)
    labeled["resolved_yes"] = pd.to_numeric(labeled["resolved_yes"], errors="coerce")
    labeled["residual_target"] = labeled["resolved_yes"] - labeled["prior"]

    corr_rows = _correlations(labeled)
    labeled_path = run_dir / "live_feature_rows_labeled.csv"
    corr_path = run_dir / "live_feature_correlations.csv"
    report_path = run_dir / "label_report.json"
    labeled.to_csv(labeled_path, index=False)
    _write_csv(corr_path, corr_rows)
    report = {
        "run_dir": str(run_dir),
        "n_labeled_rows": int(len(labeled)),
        "n_labeled_markets": int(labeled["ticker"].nunique()),
        "labeled_by_template": labeled.groupby("template").size().astype(int).to_dict(),
        "top_correlations_by_abs_residual": sorted(
            corr_rows,
            key=lambda row: abs(float(row.get("corr_with_residual_target") or 0.0)),
            reverse=True,
        )[:25],
        "files": {
            "labeled_rows": str(labeled_path),
            "correlations": str(corr_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _latest_run(base_dir: Path) -> Path:
    pointer = base_dir / "latest_run.txt"
    if pointer.exists():
        return Path(pointer.read_text(encoding="utf-8").strip())
    runs = [path for path in base_dir.iterdir() if path.is_dir()]
    if not runs:
        raise SystemExit(f"No live experiment runs found in {base_dir}")
    return sorted(runs)[-1]


def _fetch_labels(tickers: list[str]) -> pd.DataFrame:
    client = KalshiPublicClient(timeout=15)
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        market = _fetch_market(client, ticker)
        if market is None:
            rows.append({"ticker": ticker, "result": None, "resolved_yes": None})
            continue
        result = str(market.get("result") or market.get("settlement_value") or "").strip().lower()
        resolved_yes = _resolved_yes(result, market)
        rows.append(
            {
                "ticker": ticker,
                "result": result or None,
                "resolved_yes": resolved_yes,
                "label_status": market.get("status"),
                "settlement_ts": market.get("settlement_ts"),
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
    if result in {"yes", "y", "1", "true"}:
        return 1.0
    if result in {"no", "n", "0", "false"}:
        return 0.0
    settlement = market.get("settlement_value_dollars") or market.get("settlement_value")
    try:
        value = float(settlement)
    except (TypeError, ValueError):
        return None
    if value <= 0.01:
        return 0.0
    if value >= 0.99:
        return 1.0
    return None


def _correlations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_columns = [
        column for column in frame.columns
        if column not in META_COLUMNS and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    for template, group in frame.groupby("template"):
        for feature in feature_columns:
            if group[feature].isna().all():
                continue
            x = pd.to_numeric(group[feature], errors="coerce").fillna(0.0)
            if x.nunique(dropna=False) <= 1:
                continue
            rows.append(
                {
                    "template": template,
                    "feature": feature,
                    "n": int(len(group)),
                    "nonzero_share": float((x.abs() > 1e-12).mean()),
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

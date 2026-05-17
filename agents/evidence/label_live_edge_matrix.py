"""Label a live edge feature matrix after Kalshi markets resolve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.evidence.label_canonical_feature_matrix import _fetch_labels


META_COLUMNS = {
    "snapshot_time",
    "finish_time",
    "template",
    "ticker",
    "event_ticker",
    "title",
    "prior",
    "result",
    "resolved_yes",
    "residual_target",
    "label_status",
    "settlement_ts",
    "llm_extraction_status",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--matrix", default=None)
    parser.add_argument("--min-n", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    matrix_path = Path(args.matrix) if args.matrix else run_dir / "live_edge_feature_matrix.csv"
    matrix = pd.read_csv(matrix_path)
    labels = _fetch_labels(matrix["ticker"].dropna().astype(str).unique().tolist())
    labeled = matrix.merge(labels, on="ticker", how="left").dropna(subset=["resolved_yes"])
    if labeled.empty:
        raise SystemExit("No live edge matrix markets are resolved yet.")

    labeled["prior"] = pd.to_numeric(labeled["prior"], errors="coerce").clip(0.01, 0.99)
    labeled["resolved_yes"] = pd.to_numeric(labeled["resolved_yes"], errors="coerce")
    labeled["residual_target"] = labeled["resolved_yes"] - labeled["prior"]
    corr_rows = _correlations(labeled, args.min_n)

    labeled_path = run_dir / "live_edge_feature_matrix_labeled.csv"
    corr_path = run_dir / "live_edge_feature_correlations.csv"
    report_path = run_dir / "live_edge_label_report.json"
    labeled.to_csv(labeled_path, index=False)
    _write_csv(corr_path, corr_rows)
    report = {
        "run_dir": str(run_dir),
        "matrix": str(matrix_path),
        "n_labeled_rows": int(len(labeled)),
        "n_labeled_markets": int(labeled["ticker"].nunique()),
        "labeled_by_template": labeled.groupby("template").size().astype(int).to_dict(),
        "min_n": args.min_n,
        "top_features_by_abs_residual_corr": sorted(
            corr_rows,
            key=lambda row: abs(float(row.get("corr_with_residual_target") or 0.0)),
            reverse=True,
        )[:50],
        "files": {
            "labeled_matrix": str(labeled_path),
            "correlations": str(corr_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _correlations(frame: pd.DataFrame, min_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_columns = [
        column
        for column in frame.columns
        if column not in META_COLUMNS and "__" in column and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    for template, group in frame.groupby("template"):
        if len(group) < min_n:
            continue
        rows.extend(_correlation_rows(str(template), group, feature_columns))
    if len(frame) >= min_n:
        rows.extend(_correlation_rows("__all__", frame, feature_columns))
    return rows


def _correlation_rows(template: str, frame: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        x = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
        if x.nunique(dropna=False) <= 1:
            continue
        rows.append(
            {
                "template": template,
                "feature": feature,
                "n": int(len(frame)),
                "nonzero_share": float((x.abs() > 1e-12).mean()),
                "mean_value": float(x.mean()),
                "corr_with_resolved_yes": _safe_corr(x, frame["resolved_yes"]),
                "corr_with_residual_target": _safe_corr(x, frame["residual_target"]),
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

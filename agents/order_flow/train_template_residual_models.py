"""Train small template-specific Kalshi residual regressions.

Each model predicts `resolved_yes - Kalshi_prior` for one coarse family. The
global order-flow model remains the fallback; these artifacts are only useful
when their own held-out Brier beats the prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.order_flow.features import TEMPLATE_FAMILIES, build_order_flow_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/kalshi_order_flow_labeled")
    parser.add_argument("--artifact-dir", default="agents/order_flow/artifacts/templates")
    parser.add_argument("--report-json", default="reports/order_flow_template_backtests.json")
    parser.add_argument("--report-csv", default="reports/order_flow_template_backtests.csv")
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--max-delta", type=float, default=0.05)
    args = parser.parse_args()

    dataset = build_order_flow_dataset(args.data_dir)
    frame = dataset.X.reset_index(drop=True).copy()
    frame["target"] = dataset.y.reset_index(drop=True)
    frame["prior"] = dataset.prior.reset_index(drop=True)
    frame["resolved_yes"] = dataset.resolved_yes.reset_index(drop=True)
    frame = frame.join(dataset.meta.reset_index(drop=True))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.sort_values("timestamp").dropna(subset=["timestamp", "template_family"])

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []

    family_masks = {
        family: frame["template_family"].eq(family)
        for family in TEMPLATE_FAMILIES
    }
    family_masks["near_resolution"] = (frame["prior"] >= 0.95) | (frame["prior"] <= 0.05)
    family_masks["liquid_sports"] = frame["template_family"].eq("sports") & (frame["trade_count_24"] >= 3)

    feature_columns = [
        column
        for column in dataset.X.columns
        if column not in {"target", "resolved_yes"}
    ]

    for family, mask in family_masks.items():
        subset = frame[mask].copy()
        if len(subset) < args.min_rows:
            reports.append({
                "family": family,
                "status": "skipped_too_few_rows",
                "n_rows": int(len(subset)),
            })
            continue

        report = _train_one(
            family=family,
            subset=subset,
            feature_columns=feature_columns,
            artifact_dir=artifact_dir,
            test_frac=args.test_frac,
            max_delta=args.max_delta,
        )
        reports.append(report)

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(reports).to_csv(args.report_csv, index=False)
    print(json.dumps(reports, indent=2, sort_keys=True))


def _train_one(
    *,
    family: str,
    subset: pd.DataFrame,
    feature_columns: list[str],
    artifact_dir: Path,
    test_frac: float,
    max_delta: float,
) -> dict[str, Any]:
    split = max(1, int(len(subset) * (1.0 - test_frac)))
    train = subset.iloc[:split]
    test = subset.iloc[split:]
    if len(test) < 50:
        return {"family": family, "status": "skipped_too_few_test_rows", "n_rows": int(len(subset)), "n_test": int(len(test))}

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-4, 3, 20))),
        ]
    )
    model.fit(train[feature_columns], train["target"])

    prior = test["prior"].to_numpy(dtype=float)
    outcome = test["resolved_yes"].to_numpy(dtype=float)
    raw_delta = model.predict(test[feature_columns])
    delta = np.clip(raw_delta, -max_delta, max_delta)
    pred = np.clip(prior + delta, 0.01, 0.99)
    baseline = np.clip(prior, 0.01, 0.99)

    baseline_brier = float(np.mean((baseline - outcome) ** 2))
    model_brier = float(np.mean((pred - outcome) ** 2))
    improvement = baseline_brier - model_brier
    report = {
        "family": family,
        "status": "trained",
        "n_rows": int(len(subset)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "baseline_brier": baseline_brier,
        "model_brier": model_brier,
        "brier_improvement": improvement,
        "relative_brier_improvement": improvement / baseline_brier if baseline_brier else 0.0,
        "mean_abs_delta": float(np.mean(np.abs(pred - baseline))),
        "max_delta": max_delta,
        "artifact": str(artifact_dir / f"{family}.joblib"),
    }

    joblib.dump(
        {
            "model": model,
            "family": family,
            "feature_columns": feature_columns,
            "max_delta": max_delta,
            "report": report,
        },
        artifact_dir / f"{family}.joblib",
    )
    return report


if __name__ == "__main__":
    main()


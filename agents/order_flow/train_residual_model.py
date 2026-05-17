"""Train and evaluate a Kalshi order-flow residual model.

The model predicts only `outcome - market_prior`. Final probabilities are
`prior + capped_delta`, so it can only win by improving on Kalshi's own odds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.order_flow.features import build_order_flow_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/kalshi_order_flow")
    parser.add_argument("--artifact", default="agents/order_flow/artifacts/order_flow_residual.joblib")
    parser.add_argument("--report", default="reports/order_flow_residual_backtest.json")
    parser.add_argument("--max-delta", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--snapshot-stride", type=int, default=1)
    args = parser.parse_args()

    dataset = build_order_flow_dataset(args.data_dir, snapshot_stride=args.snapshot_stride)
    if len(dataset.X) < 200:
        raise SystemExit(f"Need at least 200 labeled snapshots; found {len(dataset.X)}")

    split = max(1, int(len(dataset.X) * (1.0 - args.test_frac)))
    X_train, X_test = dataset.X.iloc[:split], dataset.X.iloc[split:]
    y_train, y_test = dataset.y.iloc[:split], dataset.y.iloc[split:]
    prior_test = dataset.prior.iloc[split:].to_numpy(dtype=float)
    outcome_test = dataset.resolved_yes.iloc[split:].to_numpy(dtype=float)
    feature_columns = list(dataset.X.columns)

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-4, 3, 20))),
        ]
    )
    model.fit(X_train, y_train)
    raw_delta = model.predict(X_test)
    delta = np.clip(raw_delta, -args.max_delta, args.max_delta)
    pred = np.clip(prior_test + delta, 0.01, 0.99)
    baseline = np.clip(prior_test, 0.01, 0.99)

    report = _metrics(outcome_test, baseline, pred)
    report.update(
        {
            "n_rows": int(len(dataset.X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "feature_columns": feature_columns,
            "max_delta": args.max_delta,
            "artifact": args.artifact,
        }
    )

    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "max_delta": args.max_delta,
            "report": report,
        },
        artifact_path,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def _metrics(outcome: np.ndarray, baseline: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    baseline_brier = float(np.mean((baseline - outcome) ** 2))
    model_brier = float(np.mean((pred - outcome) ** 2))
    return {
        "baseline_brier": baseline_brier,
        "model_brier": model_brier,
        "brier_improvement": baseline_brier - model_brier,
        "relative_brier_improvement": (baseline_brier - model_brier) / baseline_brier if baseline_brier else 0.0,
        "baseline_log_loss": _log_loss(outcome, baseline),
        "model_log_loss": _log_loss(outcome, pred),
        "mean_abs_delta": float(np.mean(np.abs(pred - baseline))),
    }


def _log_loss(outcome: np.ndarray, pred: np.ndarray) -> float:
    clipped = np.clip(pred, 0.01, 0.99)
    return float(-np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped)))


if __name__ == "__main__":
    main()

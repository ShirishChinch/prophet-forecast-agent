"""Backtest the lightweight economics fast-data nudge.

This is not a Kalshi Brier backtest. It tests whether the runtime fast-market
signal contains information about future JPMaQS target changes versus a
zero-change baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from agents.economics.economic_feature_collector import public_signal_adjustment
from agents.economics.event_parser import EconomicEventSpec
from agents.economics.training.dataset_builder import build_supervised_dataset
from agents.economics.training.train_common import MODEL_CONFIGS


MODEL_TYPES = ("yield", "inflation", "growth", "policy")


def backtest_fast_nudge(data_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Run one-feature chronological backtests for every economics family."""
    results: dict[str, dict[str, Any]] = {}
    for model_type in MODEL_TYPES:
        config = MODEL_CONFIGS[model_type]
        dataset = build_supervised_dataset(
            data_dir=data_dir,
            model_type=model_type,
            horizon_days=int(config["horizon_days"]),
            sample_freq=str(config["sample_freq"]),
            max_features=int(config["max_features"]),
        )
        X = dataset.X
        y = dataset.y.to_numpy(dtype=float)
        signals = np.array([_signal_for_row(model_type, row) for _, row in X.iterrows()], dtype=float)
        valid = np.isfinite(y) & np.isfinite(signals)
        y = y[valid]
        signals = signals[valid]
        if len(y) < 100:
            raise ValueError(f"Not enough rows for {model_type}: {len(y)}")

        split_idx = int(len(y) * 0.80)
        y_train, y_test = y[:split_idx], y[split_idx:]
        s_train, s_test = signals[:split_idx], signals[split_idx:]

        beta = _fit_no_intercept_beta(s_train, y_train)
        pred_train = beta * s_train
        pred_test = beta * s_test
        baseline_train = np.zeros_like(y_train)
        baseline_test = np.zeros_like(y_test)

        results[model_type] = {
            "n_rows": int(len(y)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "horizon_days": int(config["horizon_days"]),
            "sample_freq": str(config["sample_freq"]),
            "signal_nonzero_share": float(np.mean(np.abs(signals) > 0.0)),
            "signal_abs_mean": float(np.mean(np.abs(signals))),
            "fitted_beta": float(beta),
            "train_rmse": _rmse(y_train, pred_train),
            "test_rmse": _rmse(y_test, pred_test),
            "baseline_test_rmse": _rmse(y_test, baseline_test),
            "rmse_improvement_vs_zero_change": _relative_improvement(
                baseline_rmse=_rmse(y_test, baseline_test),
                model_rmse=_rmse(y_test, pred_test),
            ),
            "test_mae": _mae(y_test, pred_test),
            "baseline_test_mae": _mae(y_test, baseline_test),
            "test_corr_signal_target": _corr(s_test, y_test),
            "test_direction_accuracy": _direction_accuracy(s_test, y_test),
            "baseline_direction_accuracy": 0.5,
            "dataset_diagnostics": dataset.diagnostics,
        }
    return results


def _signal_for_row(model_type: str, row: Any) -> float:
    features = {
        str(key).removeprefix("public__"): float(value)
        for key, value in row.items()
        if str(key).startswith("public__") and _is_finite(value)
    }
    event_spec = EconomicEventSpec(
        model_type=model_type,
        country_code=None,
        variable=model_type,
        condition="above",
        threshold=0.0,
        bucket_width=None,
        target_date_text=None,
        yes_outcome=None,
        confidence=1.0,
    )
    adjustment, _, _ = public_signal_adjustment(event_spec, features)
    return float(adjustment)


def _fit_no_intercept_beta(signal: np.ndarray, target: np.ndarray) -> float:
    denom = float(np.dot(signal, signal))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(signal, target) / denom)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _relative_improvement(*, baseline_rmse: float, model_rmse: float) -> float:
    if baseline_rmse <= 0:
        return 0.0
    return float((baseline_rmse - model_rmse) / baseline_rmse)


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def _direction_accuracy(signal: np.ndarray, target: np.ndarray) -> float:
    mask = (np.abs(signal) > 0.0) & (np.abs(target) > 0.0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.sign(signal[mask]) == np.sign(target[mask])))


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def write_reports(results: dict[str, dict[str, Any]], out_json: str | Path, out_csv: str | Path) -> None:
    json_path = Path(out_json)
    csv_path = Path(out_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    fields = [
        "model_type",
        "n_rows",
        "n_test",
        "horizon_days",
        "signal_abs_mean",
        "fitted_beta",
        "test_rmse",
        "baseline_test_rmse",
        "rmse_improvement_vs_zero_change",
        "test_mae",
        "baseline_test_mae",
        "test_corr_signal_target",
        "test_direction_accuracy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model_type, metrics in results.items():
            writer.writerow({"model_type": model_type, **{field: metrics.get(field) for field in fields if field != "model_type"}})


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest economics fast-data nudge on JPMaQS target changes.")
    parser.add_argument("--data-dir", default="jpmaqs_relevant_download")
    parser.add_argument("--out-json", default="reports/fast_nudge_backtest.json")
    parser.add_argument("--out-csv", default="reports/fast_nudge_backtest.csv")
    args = parser.parse_args()
    results = backtest_fast_nudge(args.data_dir)
    write_reports(results, args.out_json, args.out_csv)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
